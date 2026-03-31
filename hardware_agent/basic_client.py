from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, Annotated
from autogen import config_list_from_json
from autogen.oai.client import ModelClient, OpenAIWrapper
######### perflab openai api #########
from openai import AzureOpenAI

# basic llm call without autogen agent
class BasicClient:
    
    def __init__(self,
                 llm_config: Optional[Union[Dict, Literal[False]]], # Client llm to control the llm calls in MCTS
                 ):
        self._llm_config = llm_config
        self._validate_llm_config(llm_config)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.system_message = None
        
    def _validate_llm_config(self, llm_config):
        assert llm_config in (None, False) or isinstance(
            llm_config, dict
        ), "llm_config must be a dict or False or None."
        if llm_config is None:
            llm_config = self.DEFAULT_CONFIG
        self.llm_config = self.DEFAULT_CONFIG if llm_config is None else llm_config
        # TODO: more complete validity check
        if self.llm_config in [{}, {"config_list": []}, {"config_list": [{"model": ""}]}]:
            raise ValueError(
                "When using OpenAI or Azure OpenAI endpoints, specify a non-empty 'model' either in 'llm_config' or in each config of 'config_list'."
            )
        self.client = None if self.llm_config is False else OpenAIWrapper(**self.llm_config)
    
    def resync_client(self):
        self._validate_llm_config(self._llm_config)
        
    # def ask_client(self, prompt: str, is_extract_response: bool=True) -> str:
    #     # print("prompt = ", prompt)
    #     self.resync_client()
    #     messages = [{"content": prompt, "role": "user"}]
    #     response = self.client.create(messages=messages)
    #     if is_extract_response:
    #         extracted_response = self.client.extract_text_or_completion_object(response)[0]
    #         return extracted_response
    #         # print("response = ", response)
    #         # if extracted_response is None:
    #         #    print("Object is none. Prompt is ", prompt)
    #         #    assert(False)
    #         #else:
    #         #    extracted_response = extracted_response.replace("\"", "")
    #         #    return extracted_response
    #     return response

    # Lily: add system message
    def ask_client(
        self,
        prompt: str,
        is_extract_response: bool = True,
        system_message: str | None = None,
    ) -> str:
        self.resync_client()

        sm = system_message if system_message is not None else self.system_message

        messages = []
        if sm:
            messages.append({"role": "system", "content": sm})
        messages.append({"role": "user", "content": prompt})

        response = self.client.create(messages=messages)

        if is_extract_response:
            extracted_response = self.client.extract_text_or_completion_object(response)[0]
            return extracted_response
        return response
    
    # directly call create using the list of messages
    def create(self, messages: List[Dict], 
               is_extract_response: bool=True,
               cancellation_token = None) -> str:
        self.resync_client()
        response = self.client.create(messages=messages) # Mark: not support cancellation token yet
        if is_extract_response:
            extracted_response = self.client.extract_text_or_completion_object(response)[0]
            return extracted_response
            # if extracted_response is None:
            #    print("Object is none. Prompt is ", prompt)
            #    assert(False)
            #else:
            #    extracted_response = extracted_response.replace("\"", "")
            #    return extracted_response
        return response
    
    def chat(self, prompt: str, n=1, stop=None) -> list:
        
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = self.ask_client(prompt=prompt, is_extract_response=False)
            # res here
            print(res)
            outputs.extend([choice.message.content for choice in res.choices]) # not sure why res has "choices"
            # log completion tokens
            self.completion_tokens += res.usage.completion_tokens
            self.prompt_tokens += res.usage.prompt_tokens
        return outputs
    
    def gpt_usage(self):
        model = self._llm_config["config_list"]["model"]
        if model == "gpt-4":
            cost = self.completion_tokens / 1000 * 0.06 + self.prompt_tokens / 1000 * 0.03
        elif model == "gpt-3.5-turbo":
            cost = self.completion_tokens / 1000 * 0.002 + self.prompt_tokens / 1000 * 0.0015
        elif model == "gpt-3.5-turbo-16k":
            cost = self.completion_tokens / 1000 * 0.004 + self.prompt_tokens / 1000 * 0.003
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}
    
class PerflabClient:
    
    def __init__(self, model_name: str="gpt-4o-20241120"):
                
        self.client = AzureOpenAI(
            api_version="2025-02-01-preview",
            azure_endpoint="https://llm-proxy.perflab.nvidia.com",
            api_key="<your API key>",
        )
        self.model_name = model_name # "gpt-4o-20241120" 
        # check https://confluence.nvidia.com/pages/viewpage.action?spaceKey=PERFLABGRP&title=Perflab+OneAPI
        
    # polumorphism
    def _validate_llm_config(self, llm_config):
        pass
    
    def resync_client(self):
        pass
        
    def ask_client(self, prompt: str, temperature: float=0.1, is_extract_response: bool=True) -> str:
        messages = [{"content": prompt, "role": "user", "temperature": temperature}]
        chat_completion = self.client.chat.completions.create(
            model=self.model_name, # model = "deployment_name".
            messages=messages
        )
        response = chat_completion.choices[0].message.content
        return response
    
    # directly call create using the list of messages
    def create(self, messages: List[Dict], 
               is_extract_response: bool=True,
               cancellation_token = None) -> str:
        
        chat_completion = self.client.chat.completions.create(
           model=self.model_name, # model = "deployment_name".
           messages=messages
        )
        response = chat_completion.choices[0].message.content
        
        response = self.client.choices[0].message.content
        return response
    
