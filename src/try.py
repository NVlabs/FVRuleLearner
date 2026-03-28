import sys

sys.path.insert(0, "/home/scratch.yunshengb_cpu/fv/nv_textgrad")


import textgrad as tg
from textgrad.autograd.string_based_ops import StringBasedFunction

llm_engine = tg.get_engine("mixtral_8x7b")
tg.set_backward_engine("mixtral_8x7b")

from textgrad.tasks import load_task

_, val_set, _, eval_fn = load_task("BBH_object_counting", llm_engine)
question_str, answer_str = val_set[0]
question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
print(f'question is {question}')
answer = tg.Variable(str(answer_str), role_description="answer to the question", requires_grad=False)
print(f'answer is {answer}')


system_prompt = tg.Variable(
    "You are a concise LLM. Think step by step.",
    requires_grad=True,
    role_description="system prompt to guide the LLM's reasoning strategy for accurate responses",
)

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
optimizer = tg.TGD(parameters=list(model.parameters()))

prediction = model(question)
print(f'prediction is {prediction}')

loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))
print(f'eval_fn is {eval_fn}')
print(f'ground_truth_answer is {answer}')
print(f'loss is {loss}')
stuff = loss.backward()
print(f'type(stuff) is {type(stuff)}')
print(f'stuff is {stuff}')

stuff2 = optimizer.step()
print(f'stuff2 is {stuff2}')

print(f'new question is {question}')

print(f'new answer is {answer}')
prediction = model(question)



print(f'new prediction is {prediction}')


# import os
# from copy import deepcopy
# from adlrchat import ADLRChat, ADLRCompletion, LLMGatewayChat, LLMGatewayCompletion
# default_llms = ['chipnemo_8b','nemo_8b']
# def library(model_name=None, name=None, **llm_kwargs):
#     if model_name is None:
#         if name:
#             model_name = name
#         else:
#             model_name = os.environ.get('CHIPNEMO_APP_DEFAULT_MODEL', 'chipnemo_43b_chat')
#     library_config = _library[model_name] if model_name in _library else dict( llm_class=ADLRChat, llm_kwargs=dict(model_name=model_name))
#     config = deepcopy(library_config)
#     config['llm_kwargs'] = {**config['llm_kwargs'], **llm_kwargs}
#     return config
# _library = {
#     'nemo_43b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_43b_chat_beta')),
#     'chipnemo_43b' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_43b_beta')),
#     'chipnemo_43b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_43b_chat_delta')),
#     'nemo_8b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_8b_chat_alpha')),
#     'chipnemo_8b' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_8b_beta')),
#     'chipnemo_8b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_8b_chat_beta')),
#     'starcoder' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='starcoder')),
#     'gpt-4' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4')),
#     'gpt-4-32k' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4-32k')),
#     'gpt-35-turbo' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-35-turbo'))
# }
# def get_llm(*args, **kwargs):
#     config = library(*args, **kwargs)
#     print(config)
#     return config['llm_class'](**config['llm_kwargs'])
# import sys
# llm_agent = get_llm(model_name='gpt-4', temperature=0.1)
# from langchain.agents import initialize_agent, AgentType
# from langchain.callbacks import get_openai_callback
# from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
# from langchain import PromptTemplate
# # Count the langchain token
# def count_tokens(chain, query):
#     with get_openai_callback() as cb:
#         result = chain.run(query)
#         print(cb)
#         print(f'Spent a total of {cb.total_tokens} tokens')
#         # if cb.total_tokens > 14000:
#         #    exit(1)
#     return result

# prompt_template = "Tell me a {adjective} joke"
# prompt = PromptTemplate(
#     input_variables=["adjective"], template=prompt_template
# )

# llm_chain = LLMChain(prompt=prompt, llm=llm_agent)
# result = count_tokens(llm_chain, query="Hi")
# print('\n\n=== AI: response === \n', result)

# folder_path = "/Users/yunshengb/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/software-gnn/file/yunsheng_mlcad 3/0501_atefeh_harp"
# folder_path = '/Users/yunshengb/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/software-gnn/file/yunsheng_mlcad 3/05-20_stage_1'

# result1 = []
# result2 = []

# for subfolder in os.listdir(folder_path):
#     kernel_name = extract_kernel_name(subfolder)
#     if kernel_name:
#         log_file = os.path.join(folder_path, subfolder, "log.txt")
#         min_perf = extract_min_perf(log_file)
#         result1.append(f"{kernel_name}")
#         result2.append(f"{min_perf}")
#     else:
#         print(f"Skipping folder: {subfolder}")

# print("\n".join(result1))
# print("\n".join(result2))
