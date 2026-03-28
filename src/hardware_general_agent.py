from self_learning import self_learn
from autorater import auto_correct
from utils_agent import initiate_chat_with_retry

import os
from autogen.agentchat.contrib.capabilities import teachability
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from autogen import (
    GroupChat,
    GroupChatManager,
    AssistantAgent,
    ConversableAgent,
    UserProxyAgent,
    config_list_from_json,
    register_function,
)
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
# from autogen.agentchat.contrib.phi_image_agent import PhiVConversableAgent
from feature_graph_generation import train_feature_graph

import time
import tracemalloc
import openai
import diskcache
import sqlite3
import random

# from openai.error import OpenAIError

import re
from FVEval.fv_eval import utils as utils2
from iterative_assertion_generator import run_assertion_generation
from mcts_assertion_generation import run_mcts_assertion_generation
from assert_gen_with_pruner import run_assertion_generation_w_pruner
from saver import saver

print = saver.log_info

from config import FLAGS
from pprint import pformat
import fv_tools
import pickle


class RateLimitError(Exception):
    pass


# Normally user don't need to change the termination msg
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


class HardwareAgent:
    # Hardware agent has planner, react assistant, RAG assistant, and memory controlling unit
    # agent config
    """
    agent_configs = [
    {'type': 'UserProxyAgent',
     'tools': [],
     'is_termination_msg': lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
     'base_agent_config': { 'name': 'user',
                 'description': 'User proxy who ask questions and execute the tools with the provided input from Assistant.',
                 'args': ...
                 }
     },
     {'type': 'AssistantAgent',
     'tools': [],
     'teachable': {'args': {'reset_db': True, ...}},
     'is_termination_msg': lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
     'transform_message': { 'method': ['LLMSummary', 'HistoryLimit', 'TokenLimit'], 'args': [{'max_messages': 3, 'min_tokens': 10}, {}]},
     'base_agent_config': {'name': 'assistant',
                           'description': 'Assistant who reasons the problem and uses the provided tools to resolve the task.',
                            args...,
                          }
     }
      # reference https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_RAG/
      Only support for rag_chat in the start; Can only use one of UserProxyAgent or RetrieveUserProxyAgent.
      {'type': 'RetrieveUserProxyAgent',
      'tools': [],
      'is_termination_msg': lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
     'teachable': {'args': {'reset_db': True, ...}},
     'transform_message': { 'method': ['LLMSummary', 'HistoryLimit', 'TokenLimit'], 'args': [{'max_messages': 3, 'min_tokens': 10}, {}]},
     ''base_agent_config': { 'name': 'Retrieval Assistant',
                             'retrieve_config': {'taks': ...},
                             'description': 'Assistant who reasons the problem and uses the provided tools to resolve the task.',
                             'args': ...,
                            }
      }
    ]
    tool_configs = [{'function_call': function,
                    'executor': 'execute agent name',
                    'caller': 'caller agent name',
                    'name': '<name>',
                    'description: '<description>',
                    'tool_examples': 'examples for prompt'
                    }, ...],
    ###
    Reference of Assistant Agent
    # ReACT LLM
    AssistantAgent(
        name="Assistant",
        system_message="Only use the tools you have been provided with. Reply TERMINATE when the task is done.",
        llm_config={"config_list": config_list, "cache_seed": None},
        description="Assistant who reasons the problem and uses the provided tools to resolve the task.",
    )
    ###
    Reference to register a function
    register_function(
        timing_metric_calculation_tool,
        caller=assistant,
        executor=user_proxy,
        name="timing_metric_calculation_tool",
        description="Use this tool to calculate the change of one of WNS, TNS, or FEP\" timing metric of designs.\n"
                    "Input the metric name and the metric value vectors for calculating the change of a pair of settings in string format.",
    )
    llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0.8, "seed": 1234}
    """
    # Define the supported agent class list
    AGENTCLASSLIST = {
        'UserProxyAgent': UserProxyAgent,
        'AssistantAgent': AssistantAgent,
        'ConversableAgent': ConversableAgent,
        'RetrieveUserProxyAgent': RetrieveUserProxyAgent,
        'RetrieveAssistantAgent': RetrieveAssistantAgent,
        # 'PhiVConversableAgent': PhiVConversableAgent,  # Added image LLM support
    }
    TRANSFORMMESSAGELIST = {
        'LLMSummary': transform_messages.LLMTransformMessages,
        'HistoryLimit': transforms.MessageHistoryLimiter,
        'TokenLimit': transforms.MessageTokenLimiter,
    }

    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        tool_configs: List[Dict[str, Any]],
        group_chat_kwargs: Optional[Dict[str, Any]] = {
            'group_chat': {'speaker_selection_method': "auto", 'messages': [], 'max_round': 5},
            'chat_manager': {'llm_config': {"config_list": None, "cache_seed": None}},
        },
        example_rag_agent=None,
        suggestion_rag_agent=None,
    ):

        # Todo: need to check number of proxy agent
        self.num_proxy_agent = 0
        self.num_assistant_agent = 0
        self.num_rag_proxy_agent = 0
        self.agents = {}
        self.proxy_agent = None
        # Memories
        self.teachable_database = []
        self.transform_messages_handlers = []
        
        # Track whether suggestions were used in the current chat
        self.used_suggestions = False
        
        # Store RAG agents
        self.example_rag_agent = example_rag_agent
        self.suggestion_rag_agent = suggestion_rag_agent

        for agent_config in agent_configs:

            # print(f'\t Creating agent of config {pformat(agent_config)}')

            if agent_config['type'] not in self.AGENTCLASSLIST:
                print('[Error] The agent type ', agent_config['type'], ' is not supported!')
                continue

            # create agent
            if agent_config['type'] == 'UserProxyAgent':
                self.proxy_agent = self.AGENTCLASSLIST['UserProxyAgent'](
                    # is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
                    **agent_config['base_agent_config']
                )
                self.num_proxy_agent += 1
                self.agents[agent_config['base_agent_config']['name']] = self.proxy_agent
            elif agent_config['type'] == 'RetrieveUserProxyAgent':
                assert self.proxy_agent is None
                self.proxy_agent = self.AGENTCLASSLIST['RetrieveUserProxyAgent'](
                    # is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
                    **agent_config['base_agent_config']
                )
                self.num_rag_proxy_agent += 1
                self.agents[agent_config['base_agent_config']['name']] = self.proxy_agent
            else:
                # Should include llm_configs
                self.agents[agent_config['base_agent_config']['name']] = self.AGENTCLASSLIST[agent_config['type']](
                    # is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
                    **agent_config['base_agent_config']
                )
                self.num_assistant_agent += 1
            # short term memory
            if 'transform_message' in agent_config:
                self._add_transform_messages(
                    transform_message_config_list=agent_config['transform_message'],
                    agent=self.agents[agent_config['base_agent_config']['name']],
                    agent_name=agent_config['base_agent_config']['name'],
                )
            # long term memory capability
            if 'teachable' in agent_config:
                self._add_teachability(
                    teachable_config=agent_config['teachable']['args'],
                    agent=self.agents[agent_config['base_agent_config']['name']],
                    agent_name=agent_config['base_agent_config']['name'],
                )
        # end initializing the agent
        self.manager = None
        # if self.num_assistant_agent + self.num_proxy_agent + self.num_rag_proxy_agent > 2:
        agents_list = []
        for agent_name in self.agents:
            agents_list.append(self.agents[agent_name])
        """
        speaker_selection_method argument:
        https://microsoft.github.io/autogen/docs/reference/agentchat/groupchat/
        "auto": the next speaker is selected automatically by LLM.
        "manual": the next speaker is selected manually by user input.
        "random": the next speaker is selected randomly.
        "round_robin": the next speaker is selected in a round robin fashion, i.e., iterating in the same order as provided in agents.
        a customized speaker selection function (Callable): the function will be called to select the next speaker. The function should take the last speaker and the group chat as input and return one of the following:
        """
        # groupchat = GroupChat(agents=agents_list, **group_chat_kwargs['group_chat'])
        #     messages=[], max_round=15,
        #    speaker_selection_method=group_speaker_selection_method
        # )
        # self.manager = GroupChatManager(groupchat=groupchat, **group_chat_kwargs['chat_manager'])

        # register tools
        for tool in tool_configs:
            print(
                'register tool  '
                + str(tool['name'])
                + '  '
                + str(tool['function_call'])
                + '  caller:  '
                + str(self.agents[tool['caller']])
                + '  executer:  '
                + str(self.agents[tool['executor']])
            )
            register_function(
                tool['function_call'],
                caller=self.agents[tool['caller']],
                executor=self.agents[tool['executor']],
                name=tool['name'],
                description=tool['description'],
            )
        print(
            f"Hardware Agent Initialized %d proxy, %d rag proxy, %d assistants"
            % (self.num_proxy_agent, self.num_rag_proxy_agent, self.num_assistant_agent)
        )

    def _add_transform_messages(
        self,
        transform_message_config_list: Dict[str, List[Any]],
        agent: Union[UserProxyAgent, AssistantAgent, RetrieveUserProxyAgent] = None,
        agent_name: str = "",
    ):
        # Not support LLMSummary with other types of method
        if 'LLMSummary' in transform_message_config_list['method']:
            assert len(transform_message_config_list['method']) == 1
        if 'LLMSummary' in transform_message_config_list['method']:
            """
            condensed_config_list = config_list_from_json(env_or_file="chat_summary_llm_config.txt")
            context_handling = transform_messages.LLMTransformMessages(
                    llm_config={"config_list": config_list, "cache_seed": None}, max_token=1500)
            """
            self.transform_messages_handlers.append(
                {
                    'agent': agent_name,
                    'config': transform_message_config_list,
                    'obj': self.TRANSFORMMESSAGELIST['LLMSummary'](**transform_message_config_list['args'][0]),
                }
            )
        else:
            """
            context_handling = transform_messages.TransformMessages(
               transforms=[
                    transforms.MessageHistoryLimiter(max_messages=10),
                    transforms.MessageTokenLimiter(max_tokens=1000, max_tokens_per_message=50, min_tokens=500),
               ]
            )
            """
            transforms_list = []
            for t in range(len(transform_message_config_list['method'])):
                # print(f"@@@@@transform_message_config_list: {transform_message_config_list}")

                transforms_list.append(
                    self.TRANSFORMMESSAGELIST[transform_message_config_list['method'][t]](
                        **transform_message_config_list['args'][t]
                    )
                )
            self.transform_messages_handlers.append(
                {
                    'agent': agent_name,
                    'config': transform_message_config_list,
                    'obj': transform_messages.TransformMessages(transforms=transforms_list),
                }
            )
        self.transform_messages_handlers[-1]['obj'].add_to_agent(agent)
        return

    def _add_teachability(
        self,
        teachable_config: Dict[str, Any] = {'verbosity': 1, 'reset_db': True, 'path_to_db': "./tmp/"},
        agent: Union[UserProxyAgent, AssistantAgent, RetrieveUserProxyAgent] = None,
        agent_name: str = "",
    ):
        self.teachable_database.append(
            {'agent': agent_name, 'config': teachable_config, 'obj': teachability.Teachability(**teachable_config)}
        )
        self.teachable_database[-1]['obj'].add_to_agent(agent)
        return

    def reset_agents(self):
        for agent_name in self.agents:
            self.agents[agent_name].reset()

    # revalidate the llm for gateway chat
    
    def revalidate_llm_config(self):
        for agent_name in self.agents:
            if type(self.agents[agent_name]) == UserProxyAgent or \
                    type(self.agents[agent_name]) == RetrieveUserProxyAgent:
                continue
            self.agents[agent_name].revalidate_llm_config()

    # Mark: start the chat to proxy
    # Need to input the pure text question after using prompt formatting
    def initiate_chat(self, use_cache: bool = False, cache_seed: int = 43, **kwargs) -> str:

        if self.manager is None:
            # one assistant + one proxy
            # if FLAGS.agent_arch == "two_agents":
            if FLAGS.agent_arch == "three_agents":

                li = [
                    {
                        "recipient": self.agents["Coding"],
                        "message": kwargs.get('message'),
                        "max_turns": 1,
                        "summary_method": "last_msg",
                    },
                ]

                for i, helper in enumerate(FLAGS.helper_list):
                    li.append(
                        {
                            "recipient": self.agents[f"Helper{i+1}"],
                            "message": helper,
                            "max_turns": 1,
                            "summary_method": "last_msg",
                        },
                    )
                chat_results = self.agents["user"].initiate_chats(li)
                # return chat_results
            elif FLAGS.agent_arch == "four_agents":
                chat_results = self.agents["user"].initiate_chats(
                    [
                        {
                            "recipient": self.agents["Coding"],
                            "message": kwargs.get('message'),
                            "max_turns": 1,
                            "summary_method": "last_msg",
                        },
                        {
                            "recipient": self.agents["Helper1"],
                            "message": "Please double check whether this problem uses the sequential or combinatorial assertions and enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.",
                            "max_turns": 1,
                            "summary_method": "last_msg",
                        },
                        {
                            "recipient": self.agents["JasperGold"],
                            "message": """                         
                            
Respond to the human as helpfully and accurately as possible. You have access to the following tool: 
                                                                                                                                 
evaluate_jg:                                                                                             
        Use this tool to analyze the systemverilog assertion.                               
        Input the LLM_response for analyzing the assertion in string format.           

Use the following format:
Question: the input question you must answer                                                            
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)


Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!         


Begin! Reply TERMINATE when you provide the Final Answer..                                                                       
Question:

You are an experienced Formal Verification engineer who can look at design specification in natural language and reflect on the written assertions in SystemVerilog and evaluate the SVA using JasperGold.

Here are some tool examples:                                                                                                

[Tool Examples]


    Use JasperGold to check the syntax in the following way:
    Parse the system verilog and then call the tool evaluate_jg with the string input of the system verilog code. For example, if you see the following 
    
    '''

    Context: 
    ```systemverilog
    assert property(@(posedge clk)
        sig_D |=> s_eventually(sig_F)
    );
    ```
    ''',

    send the following string (python data type str) into the function/tool "evaluate_jg" with the argument "assert property(@(posedge clk)
        sig_D |=> s_eventually(sig_F)
    );". In other words, Arguments: {"LLM_response": "evaluate_jg" with the argument "assert property(@(posedge clk)
        sig_D |=> s_eventually(sig_F)
    );"}.
    

Do NOT execute the code directly. Instead, send the code to our function/tool. You cannot execute SystemVerilog, so do NOT even try to run/execute the code.

You need to use the provided tools to analyze the code! You are not good at executing the code!


                            """,
                            "max_turns": 3,
                            "summary_method": "last_msg",
                        },
                    ]
                )
            elif FLAGS.agent_arch == "no_agent":
                stant = None
                for agent_name in self.agents:
                    if (
                        type(self.agents[agent_name]) == UserProxyAgent
                        or type(self.agents[agent_name]) == RetrieveUserProxyAgent
                    ):
                        continue
                    assistant = self.agents[agent_name]
                assert assistant is not None
                # Cache LLM responses. To get different responses, change the cache_seed value.
                with Cache.disk(cache_seed=cache_seed) as cache:
                    if use_cache:
                        kwargs['cache'] = cache
                        print('[Info]: use cache seed ', cache_seed, ' for chat')
                    # Regular ReACT reasoning
                    if type(self.proxy_agent) == UserProxyAgent:
                        return self.proxy_agent.initiate_chat(
                            assistant, **kwargs
                        )  # Lily: changed from initiate_chat to initiate_chat_with_retry
                    # RAG proxy
                    elif type(self.proxy_agent) == RetrieveUserProxyAgent:
                        return initiate_chat_with_retry(
                            self.proxy_agent, assistant, message=self.proxy_agent.message_generator, **kwargs
                        )  # Lily: changed from initiate_chat to initiate_chat_with_retry
            elif FLAGS.agent_arch == "flexible_agents":
                # cache_path = f"./cache/43"
                # if os.path.exists(cache_path):
                #     os.remove(cache_path)

                # with Cache.disk(cache_seed=43) as cache:
                if "nl2sva" in FLAGS.task:
                    original_message = kwargs.get('message')
                    # appended_suggestions = None
                    if FLAGS.use_RAG:
                        # Retrieve relevant documents using RAGAgent
                        full_question = f"Question: Create a SVA assertion that checks: {kwargs.get('row').prompt}" # TODO: iterative add answers
                        testbench = kwargs.get('row').testbench
                        # print(f"@@@kwargs.get('row').prompt: {kwargs.get('row').prompt}")
                        original_message = kwargs.get('message')

                        if 'Examples' in FLAGS.RAG_content:
                            retrieved_docs = self.example_rag_agent.retrieve(full_question, top_k=FLAGS.Examples_top_k)
                            # breakpoint()
                            examples_list = []
                            message = kwargs.get('message')
                            for i, doc in enumerate(retrieved_docs, 1):
                                print(f"@@@Retrieved docs: {i}. {doc}")
                                formatted_doc = f"- {doc}\n"
                                examples_list.append(formatted_doc)
                            if FLAGS.prompting_instruction == 0:
                                message = (
                                    f"{message}\n\n"
                                    "Additional context from similar documents:\n"
                                    + "\n".join(f"{example}" for example in examples_list)
                                )
                            elif FLAGS.prompting_instruction == 1:
                                message = (
                                    f"{message}\n\n"
                                    "Use the following SVA assertion examples to understand the structure and logic of similar assertions. Pay attention to how conditions and logical operators are used to form the assertions. Based on this understanding, generate the requested SVA assertion accurately.\n "
                                    + "\n".join(f"- {example}" for example in examples_list)
                                )


                        # Initialize response and appended_suggestions variables
                        response = None
                        appended_suggestions = None  # Initialize for JG error correction later
                        
                        if 'Suggestions' in FLAGS.RAG_content:
                            # qtree_enhanced_retrieval returns:
                            # - 'suggestions': List of improvement hints/rules (NOT complete SVAs)
                            # - 'initial_sva': A complete SVA assertion generated by qtree
                            retrieved_result = self.suggestion_rag_agent.retrieve(full_question, top_k=FLAGS.Suggestions_top_k, testbench=testbench)
                            
                            # breakpoint()

                            # Handle new dict return format with initial_sva
                            if isinstance(retrieved_result, dict):
                                retrieved_docs = retrieved_result.get('suggestions', [])
                                initial_sva = retrieved_result.get('initial_sva', None)
                                operator_explanations = retrieved_result.get('operator_explanations', {})
                                
                                # Decision tree:
                                # 1. If we have suggestions → Use them to guide a second LLM call for better SVA
                                # 2. If no suggestions but have initial_sva → Use initial_sva directly (skip second LLM)
                                # 3. Otherwise → Fall back to normal LLM call
                                
                                if len(retrieved_docs) == 0 and initial_sva:
                                    # Case 2: No suggestions, but qtree generated initial_sva
                                    # Use it directly and skip to JG checking
                                    print(f"Initial SVA: {initial_sva}")
                                    if not FLAGS.operator_explanation:
                                        # Skip the suggestion processing below
                                        print(f"\n✅ No suggestions from qtree. Using initial SVA directly (skip second LLM call).")
                                        response = initial_sva
                                        appended_suggestions = None  # No suggestions to append
                                    elif FLAGS.operator_explanation and operator_explanations:
                                        message = f"{message}\n\n"
                                        message += f"Additional background information about operators:\n"
                                        for operator, explanation in operator_explanations.items():
                                            message += f"Operator: {operator}\nExplanation: {explanation}\n"
                                        response = initiate_chat_with_retry(
                                            self.agents["user"],
                                            self.agents["Coding"],
                                            message=message
                                        )
                                        appended_suggestions = None
                                    
                            else:
                                # Backward compatibility - handle list format
                                retrieved_docs = retrieved_result
                            
                            # Process suggestions if we haven't already set response
                            if response is None:
                                suggestions_list = []
                                seen_suggestions = set()
                                if FLAGS.baseline_LFM:
                                    message = kwargs.get('message')
                                    response = initiate_chat_with_retry(
                                        self.agents["user"],
                                        self.agents["Coding"],
                                        message=message
                                    )
                                    message = (
                                        "Please analyze the differences between [Your First Answer] and [Reference Answer] and use the analysis to answer the last question\n"                                 
                                        + "\n".join(retrieved_docs)
                                        + "Check whether the following a needs revision."
                                        + f"{message}\n\n"
                                        + f"[Your First Answer]:{response}\n"
                                        + f"[Your Second Answer]\n"
                                        + "Create another answer [Your Second Answer] different from the previous [Your First Answer] using the above analysis."
                                    )
                                else:
                                    if FLAGS.RAG_content == ['Suggestions']:
                                        message = kwargs.get('message')
                                    for i, doc in enumerate(retrieved_docs, 1):
                                        print(f"@@@Full question:{full_question}")
                                        print(f"@@@Retrieved docs: {i}. {doc}")
                                        # Remove the "Suggestions:" prefix if it exists
                                        doc = re.sub(r'^Suggestions:\s*', '', doc.strip())
                                        
                                        # Split the document into lines
                                        if FLAGS.deduplication:
                                            for suggestion in doc.split('\n'):
                                                if suggestion not in seen_suggestions:
                                                    suggestions_list.append(suggestion)
                                                    seen_suggestions.add(suggestion)
                                        else:
                                            suggestions_list += doc.split('\n')

                                    appended_suggestions = "\n".join(f"{suggestion}" for suggestion in suggestions_list)

                                    # Seeing the appended_suggestions
                                    # print(f"-----------------------\nappended_suggestions:\n {appended_suggestions}")
                                    # breakpoint()

                                    if FLAGS.use_RAG:
                                        if FLAGS.Suggestions_Reasoning:
                                            suggestions_summary = initiate_chat_with_retry(
                                                self.agents["user"],
                                                self.agents["Suggestions_Reasoning"],
                                                message=(
                                                    f"{original_message}\n\n"
                                                    "Task: Select the most relevant suggestions for improving LLM-generated SystemVerilog Assertions.\n\n"
                                                    "Instructions:\n"
                                                    "1. Review the suggestions below.\n"
                                                    "2. Reorder the suggestions, placing the most critical ones at the beginning.\n"
                                                    "3. Focus on selecting suggestions that directly enhance the functional correctness of assertion generation, particularly addressing error-prone operators that may appear in LLM responses.\n"
                                                    "4. Provide selected suggestions, each starting with '-'.\n"
                                                    "5. If none of the suggestions are considered helpful or relevant, return 'None'.\n\n"
                                                    "Suggestions to consider:\n"
                                                    f"{chr(10).join(f'{suggestion}' for suggestion in suggestions_list)}\n\n"
                                                    "Response format: List selected suggestions, each starting with '-', or return 'None' if no suggestions are helpful."
                                                )
                                            )
                                            suggestions_extraction = re.findall(r"^-.*", suggestions_summary, re.MULTILINE)
                                            post_suggestions_summary = "\n".join(suggestions_extraction)
                                            # Check if the response indicates no helpful suggestions
                                            if "none" in suggestions_summary.lower().strip() and not suggestions_extraction:
                                                appended_suggestions = None
                                            else:
                                                appended_suggestions = post_suggestions_summary

                                    # Seeing the appended_suggestions
                                    # print(f"-----------------------\nExtracted suggestions:\n {appended_suggestions}")

                                    # Track that suggestions were used if appended_suggestions is not empty
                                    if appended_suggestions and appended_suggestions.strip():
                                        self.used_suggestions = True
                                        # Record to saver.stats for later analysis
                                        saver.save_stats('used_suggestions', 1)
                                        message = (
                                            f"{message}\n\n"
                                            "Additional knowledge/suggestions to follow/obey:\n" 
                                            + appended_suggestions
                                        )
                                        
                                        # Add operator explanations if available and enabled
                                        if FLAGS.operator_explanation and operator_explanations:
                                            message += "\n\nAdditional background information about operators:\n"
                                            for operator, explanation in operator_explanations.items():
                                                message += f"Operator: {operator}\nExplanation: {explanation}\n\n"
                                    else:
                                        saver.save_stats('used_suggestions', 0)

                                # breakpoint()
                                # Case 1: Have suggestions - Second LLM call guided by suggestions
                                response = initiate_chat_with_retry(
                                    self.agents["user"],
                                    self.agents["Coding"],
                                    message=message
                                )
                        
                        # If response is still None, we need to call LLM
                        # This handles:
                        # - Examples-only mode (no Suggestions in RAG_content)
                        # - Any other case where response wasn't set above
                        if response is None:
                            response = initiate_chat_with_retry(
                                self.agents["user"],
                                self.agents["Coding"],
                                message=message
                            )

                        print(f"@@@DEBUG:assertion = {response}")

                        if FLAGS.debug:
                            print(f"enriched_prompt=\n{message}")

                                # else: # temporal: initial test shows it is not so useful
                                #     responses = []
                                #     for i, suggestion in enumerate(suggestions_list):
                                #         if i == 0:
                                #             initial_prompt = (
                                #                 f"{kwargs.get('message')}\n\n"
                                #                 "Additional knowledge and suggestion to incorporate:\n"
                                #                 f"{suggestion}"
                                #             )
                                #             response = initiate_chat_with_retry(
                                #                 self.agents["user"],
                                #                 self.agents["Coding"],
                                #                 message=initial_prompt
                                #             )
                                #         else:
                                #             # print(f"@@@DEBUG: response = {response}({type(responses)})")
                                #             last_response = utils2.parse_code_response(response)
                                #             # print(f"@@@DEBUG: last_response = {last_response}")
                                #             follow_up_prompt = (
                                #                 f"{kwargs.get('message')}\n\n"
                                #                 f"Previously generated assertion:\n{last_response}\n\n"
                                #                 f"Additional knowledge and suggestion to refine the previous assertion:\n"
                                #                 f"{suggestion}\n\n"
                                #                 "Please modify the previous assertion based on this new information."
                                #             )
                                # if 'Suggestions' in FLAGS.RAG_content:
                                #         response = initiate_chat_with_retry(
                                #             self.agents["user"],
                                #             self.agents["Coding"],
                                #             message=follow_up_prompt
                                #         )
                                #     responses.append(response)
                                # response = responses[-1]  # Final response after all iterations
                                # elif 'Examples' in FLAGS.RAG_content:

                        # Append retrieved documents to the prompt
                        # enriched_prompt = (
                        #     f"{kwargs.get('message')}\n\n"
                        #     "Additional context from similar documents:\n"
                        #     # "Additional knowledge/suggestions to follow/obey:\n"
                        #     # " Use the following SVA assertion examples to understand the structure and logic of similar assertions. Pay attention to how conditions and logical operators are used to form the assertions. Based on this understanding, generate the requested SVA assertion accurately.\n "
                        #     + "\n".join(f"{suggestion}" for suggestion in suggestions_list)
                        #     # + "-If the natural language query contains some signal names, just use the signals instead of adding more you imagine. For example, if the current natural language asks you to use a latency threshold, just extract one signal corresonding to that signal instead of overthinking and using additional signals. "
                        #     # + "-If the natural language already contains some signal names, for example, if the natural language query contains following sentence: 'Use the signals 'tb_gnt','last_gnt', 'hold', and 'tb_req'.'Then, do not use extra signals/variables such as NUM_OF_CLIENTS."
                        #     # + "-If the natural language already contains some signal names, for example, if the natural language query contains following sentence: 'Use the signals 'tb_gnt','last_gnt', 'hold', and 'tb_req'.'You may ignore/miss some variables such as hold, i.e. you don't have to use every signal in the query."
                        #     # + "-Double check if you really need to use the sequential logic containing latency operators such as '|-> ##1', i.e. only use such statement if simpler alternatives are not enough."
                        #     # + "-We prefer not using the implication operator '|->' or '|=>', i.e. only use '|->' or '|=>' if simpler alternatives are not enough. In such cases, directly write combinatorial logic using '&&', '!==', etc."
                        #     # + "-Do NOT use the implication operator '|->' if possible, i.e. only use '|->' if simpler alternatives are not enough. In such cases, directly write combinatorial logic using '&&', '!==', etc."
                        #     # + "-Simply do NOT use '|->' or '|=>' for this example!!!!!!!!!!'"
                        # )
                            

                        # response = initiate_chat_with_retry(
                        #     self.agents["user"],
                        #     self.agents["Coding"],
                        #     message=enriched_prompt
                        # )

                        if FLAGS.debug:
                            print(f"@@@response: {response}")
                    else: # use_RAG = False
                        # Record that no suggestions were used
                        saver.save_stats('used_suggestions', 0)
                        
                        if FLAGS.global_task == "inference" and FLAGS.baseline_emsemble and FLAGS.voting_tech == "LLM":
                            message = kwargs.get("message")
                            response_list = []
                            full_question = f"Question: Create a SVA assertion that checks: {kwargs.get('row').prompt}"
                            for i in range(FLAGS.passk):
                                # Change the temperature
                                random.seed(FLAGS.random_seed)
                                if "o" in FLAGS.llm_model:
                                    pass
                                else:
                                    temperature = random.random()
                                    self.agents["Coding"].llm_config["temperature"] = temperature
                                response = initiate_chat_with_retry(
                                    self.agents["user"],
                                    self.agents["Coding"],
                                    **kwargs
                                )
                                response_list.append(response)
                                # Construct the voting message
                            voting_message = f"""
                            Natural Language Description and contexts: {message}

                            Generated SVAs:
                            """

                            for idx, response in enumerate(response_list):
                                voting_message += f"\nSolution {idx+1}:\n{response}\n"

                            voting_message += """
                            Please evaluate each SVA based on the natural language description and the following criteria:

                            1. Correctness (0-4 points):
                            - Does the assertion correctly check the behavior described in the prompt?

                            2. Completeness (0-3 points):
                            - Does the assertion cover all aspects mentioned in the description?
                            - Are all necessary signals included?

                            3. Syntax (0-2 points):
                            - Is the SVA syntactically correct?
                            - Are all parentheses and operators properly placed?

                            4. Efficiency (0-1 point):
                            - Is the assertion written in a clear and concise manner?

                            Based on the above criteria, choose or vote for the only one best SVA from the SVA candidates. Provide your evaluation in the following format:
                            Enclose your SVA code with ```systemverilog and ```. Only output the best code snippet and do NOT output anything else.

                            Answer:
                            ```systemverilog
                            assert property(@(posedge clk)
                            (sig_A && !sig_B) |-> sig_C
                            );
                            ```
                            """
                            
                            # print(f"@@@DEBUG: voting_message = {voting_message}")
                            response = initiate_chat_with_retry(
                                self.agents["user"],
                                self.agents["voting"],
                                message = voting_message
                            )

                        else:
                            # DEBUG: 
                            # print(f'self.agents["user"]:{self.agents["user"]}')
                            # print(f'self.agents["Coding"]:{self.agents["Coding"]}')
                            # print(f'kwargs:{kwargs}')

                            response = initiate_chat_with_retry(
                                self.agents["user"],
                                self.agents["Coding"],
                                **kwargs
                            )
                        # if FLAGS.debug:
                        #     print(f"@@@response: {response}")


                    if FLAGS.global_task == 'inference' and FLAGS.use_autorater:
                        response = auto_correct(self.agents, response, kwargs.get('row'))

                    if FLAGS.use_JG and FLAGS.global_task == 'inference':
                        num_syntax_iter = 0
                        # print(f"@@@DEBUG: in JG")
                        while True:
                            if num_syntax_iter >= FLAGS.max_num_syntax_iter:
                                print(f'@@@JasperGold: Too many syntax iters')
                                break

                            try:
                                lm_response = response
                            except AttributeError as e:
                                print(f'@@@JasperGold: AttributeError encountered; response={response}')
                                response = None
                                num_syntax_iter += 1
                                continue
                            except Exception as e:
                                print(f'@@@JasperGold: Exception encountered; response={response}, error={str(e)}')
                                raise e
                            
                            # Start time and memory tracking
                            start_time = time.time()
                            tracemalloc.start()

                            if FLAGS.task == "nl2sva_machine" or FLAGS.task == "nl2sva_human":
                                jasper_out_str, syntax_error_match = fv_tools.evaluate_jg(lm_response)
                            elif FLAGS.task == "nl2sva_opencore":
                                jasper_out_str, syntax_error_match = fv_tools.evaluate_jg_opencore(lm_response)

                            # Lily0916: remove this
                            # print(f"@@@JasperGold: JasperGold output string = {jasper_out_str}")
                            
                            # Stop time and memory tracking
                            current, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            end_time = time.time()

                            # Calculate runtime and memory usage
                            runtime = end_time - start_time
                            memory_usage = peak - current
                            memory_usage_kb = memory_usage / (1024)  # Convert bytes to KB


                            # Save stats
                            saver.save_stats('JasperGold_runtime', runtime)
                            saver.save_stats('JasperGold_memory_KB', memory_usage_kb)

                            # print(f"JasperGold response:\n{jasper_out_str}" )
                            if not syntax_error_match:
                                print(f'@@@JasperGold: No syntax error detected :)')
                                if num_syntax_iter > 0:
                                    print(f'@@@JasperGold: Found a good example! num_syntax_iter={num_syntax_iter}')
                                break
                            else:
                                # print(
                                #     f'@@@JasperGold: Syntax error detected!\nlm_response={lm_response}\nnum_syntax_iter={num_syntax_iter}; jasper_out_str={jasper_out_str}'
                                # )
                                if FLAGS.task == "nl2sva_machine" or FLAGS.task == "nl2sva_human":
                                    error_lines = [line for line in re.findall(r'ERROR.*', jasper_out_str) if 'dummy' not in line]
                                elif FLAGS.task == "nl2sva_opencore":
                                    # Collect ERROR Messages after 'Summary of errors detected:' and with '[ERROR'
                                    summary_index = jasper_out_str.find('Summary of errors detected:')
                                    if summary_index != -1:
                                        errors_section = jasper_out_str[summary_index:]
                                        error_lines = [line for line in re.findall(r'\[ERROR.*', errors_section) if 'PEC' not in line]
                                    else:
                                        error_lines = []

                                # Join the lines into a single string
                                error_string = '\n'.join(error_lines)
                                questions = re.findall(r'Question:.*?(?=\n\nAnswer:|$)', kwargs.get("message"), re.DOTALL)
                                # questions_string = '\n\n'.join(questions)
                                last_question = questions[-1]
                                assertion_generation = utils2.parse_code_response(lm_response)
                                num_syntax_iter += 1
                                
                                # JG error correction with suggestions (if available)
                                if 'Suggestions' in FLAGS.RAG_content and appended_suggestions:
                                    JG_message = f'The natural language question for assertion generation is: \n {last_question}\n The previously generated assertion is: \n {assertion_generation}\n However, there is a syntax error. Please analyze the error message generated by JasperGold and re-generate the assertion to fix the syntax error: \n Error Message: \n """{error_string}"""\n\nAdditional knowledge/suggestions to follow/obey:\n{appended_suggestions}'
                                    
                                    # Add operator explanations if available and enabled
                                    if FLAGS.operator_explanation and operator_explanations:
                                        JG_message += "\n\nAdditional background information about operators:\n"
                                        for operator, explanation in operator_explanations.items():
                                            JG_message += f"Operator: {operator}\nExplanation: {explanation}\n\n"
                                    
                                    response = initiate_chat_with_retry(
                                        self.agents["user"],
                                        self.agents["Coding"],
                                        message=JG_message,
                                        # cache=cache,
                                    )

                                    print(f"@@@DEBUG: message: {JG_message}")
                                    print(f"@@@DEBUG: response: {response}")
                                    # breakpoint()

                                else:
                                    response = initiate_chat_with_retry(
                                        self.agents["user"],
                                        self.agents["Coding"],
                                        message=(
                                            f'The natural language question for assertion generation is: \n {last_question}\n'
                                            f'The previously generated assertion is: \n {assertion_generation}\n'
                                            f'However, there is a syntax error. Please analyze the error message generated by JasperGold and re-generate the assertion to fix the syntax error: \n'
                                            f'Error Message: \n """{error_string}"""\n'
                                        ),
                                        # cache=cache,
                                    )
                                # print(f"@@@New JG message: {message}")
                                # print(f"@@@New JG response: {response}")
                    elif (FLAGS.use_JG and FLAGS.global_task == 'train'):
                        num_syntax_iter = 0
                        # print(f"@@@DEBUG: in JG")
                        while True:
                            if num_syntax_iter >= FLAGS.max_num_syntax_iter:
                                print(f'@@@JasperGold: Too many syntax iters')
                                break

                            try:
                                lm_response = response
                            except AttributeError as e:
                                print(f'@@@JasperGold: AttributeError encountered; response={response}')
                                response = None
                                num_syntax_iter += 1
                                continue
                            except Exception as e:
                                print(f'@@@JasperGold: Exception encountered; response={response}, error={str(e)}')
                                raise e
                            
                            # Start time and memory tracking
                            start_time = time.time()
                            tracemalloc.start()

                            if FLAGS.task == "nl2sva_machine" or FLAGS.task == "nl2sva_human":
                                jasper_out_str, syntax_error_match = fv_tools.evaluate_jg(lm_response)
                            elif FLAGS.task == "nl2sva_opencore":
                                jasper_out_str, syntax_error_match = fv_tools.evaluate_jg_opencore(lm_response)

                            # print(f"@@@JasperGold: JasperGold output string = {jasper_out_str}")
                            # breakpoint()
                            
                            # Stop time and memory tracking
                            current, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            end_time = time.time()

                            # Calculate runtime and memory usage
                            runtime = end_time - start_time
                            memory_usage = peak - current
                            memory_usage_kb = memory_usage / (1024)  # Convert bytes to KB


                            # Save stats
                            saver.save_stats('JasperGold_runtime', runtime)
                            saver.save_stats('JasperGold_memory_KB', memory_usage_kb)

                            # print(f"JasperGold response:\n{jasper_out_str}" )
                            if not syntax_error_match:
                                print(f'@@@JasperGold: No syntax error detected :)')
                                if num_syntax_iter > 0:
                                    print(f'@@@JasperGold: Found a good example! num_syntax_iter={num_syntax_iter}')
                                break
                            else:
                                # print(
                                #     f'@@@JasperGold: Syntax error detected!\nlm_response={lm_response}\nnum_syntax_iter={num_syntax_iter}; jasper_out_str={jasper_out_str}'
                                # )
                                if FLAGS.task == "nl2sva_machine" or FLAGS.task == "nl2sva_human":
                                    error_lines = [line for line in re.findall(r'ERROR.*', jasper_out_str) if 'dummy' not in line]
                                elif FLAGS.task == "nl2sva_opencore":
                                    # Collect ERROR Messages after 'Summary of errors detected:' and with '[ERROR'
                                    summary_index = jasper_out_str.find('Summary of errors detected:')
                                    if summary_index != -1:
                                        errors_section = jasper_out_str[summary_index:]
                                        error_lines = [line for line in re.findall(r'\[ERROR.*', errors_section) if 'PEC' not in line]
                                    else:
                                        error_lines = []

                                # Join the lines into a single string
                                error_string = '\n'.join(error_lines)
                                questions = re.findall(r'Question:.*?(?=\n\nAnswer:|$)', kwargs.get("message"), re.DOTALL)
                                # questions_string = '\n\n'.join(questions)
                                last_question = questions[-1]
                                assertion_generation = utils2.parse_code_response(lm_response)
                                # postprocessing finetuning
                                num_syntax_iter += 1
                                response = initiate_chat_with_retry(
                                    self.agents["user"],
                                    self.agents["Coding"],
                                    message=f'The natural language question for assertion generation is: \n {last_question}\n The previously generated assertion is: \n {assertion_generation}\n However, there is a syntax error. Please analyze the error message generated by JasperGold and re-generate the assertion to fix the syntax error: \n Error Message: \n """{error_string}"""\n',
                                    # cache=cache,
                                )
                                # print(f"New JG message: {message}")

                    if FLAGS.global_task == 'train':
                        if not FLAGS.use_JG:
                            self_learn(self.agents, response, kwargs.get('message'), kwargs.get('row'))
                        elif FLAGS.use_JG:
                            self_learn(self.agents, response, original_message, kwargs.get('row'))
                    # , cache)
                    return response

                elif "design2sva" in FLAGS.task:
                    if FLAGS.global_task == 'inference':
                        if FLAGS.iterative_optimization:
                            response = run_assertion_generation(self.agents, kwargs.get('message'), kwargs.get('row'))
                        elif FLAGS.use_MCTS:
                            response = run_mcts_assertion_generation(self.agents, kwargs.get('message'), kwargs.get('row'))
                        elif FLAGS.use_multi_and_pruner:
                            response = run_assertion_generation_w_pruner(self.agents, kwargs.get('message'), kwargs.get('row'))
                    elif FLAGS.global_task == "train":
                        response = train_feature_graph(self.agents, kwargs.get('message'), kwargs.get('row'))
                    return response
                

            else:
                raise NotImplementedError()
            return chat_results

        elif self.manager is not None:
            # Group chat
            with Cache.disk(cache_seed=cache_seed) as cache:
                if use_cache:
                    kwargs['cache'] = cache
                    print('[Info]: use cache seed ', cache_seed, ' for chat')
                # Regular ReACT reasoning
                if type(self.proxy_agent) == UserProxyAgent:
                    return self.proxy_agent.initiate_chat(self.manager, **kwargs)
                # RAG proxy
                elif type(self.proxy_agent) == RetrieveUserProxyAgent:
                    return self.proxy_agent.initiate_chat(
                        self.manager, message=self.proxy_agent.message_generator, **kwargs
                    )
        # return "[Error]: Didn't complete chat! please check the agent config"


# Todo: RAG as a function call
class HardwareAgentWithRAGCall(HardwareAgent):
    # Call the RAG through function call during chats
    """
    # reference https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_RAG/
    {'type': 'RetrieveUserProxyAgent',
    'teachable': {'enable': True/False, 'args': {'reset_db': True, ...}},
    'MessageTransform': { 'method': ['LLMSummary', 'LastN', 'MaxToken'], 'args': {'max_messages': 3, 'min_tokens': 10}},
    ''base_agent_config': { 'name': 'Retrieval Assistant',
                            'retrieve_config': {'taks': ...},
                            'descriptions': 'Assistant who reasons the problem and uses the provided tools to resolve the task.',
                            'args': ...,
                           }
     }
    """

    def __init__(self):
        pass
