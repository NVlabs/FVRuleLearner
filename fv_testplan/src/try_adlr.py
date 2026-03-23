import os
from copy import deepcopy
from adlrchat import ADLRChat, ADLRCompletion, LLMGatewayChat, LLMGatewayCompletion
default_llms = ['chipnemo_8b','nemo_8b']
def library(model_name=None, name=None, **llm_kwargs):
    if model_name is None:
        if name:
            model_name = name
        else:
            model_name = os.environ.get('CHIPNEMO_APP_DEFAULT_MODEL', 'chipnemo_43b_chat')
    library_config = _library[model_name] if model_name in _library else dict( llm_class=ADLRChat, llm_kwargs=dict(model_name=model_name))
    config = deepcopy(library_config)
    config['llm_kwargs'] = {**config['llm_kwargs'], **llm_kwargs}
    return config
_library = {
    'nemo_43b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_43b_chat_beta')),
    'chipnemo_43b' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_43b_beta')),
    'chipnemo_43b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_43b_chat_delta')),
    'nemo_8b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='nemo_8b_chat_alpha')),
    'chipnemo_8b' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='chipnemo_8b_beta')),
    'chipnemo_8b_chat' : dict( llm_class=ADLRChat, llm_kwargs=dict(model_name='chipnemo_8b_chat_beta')),
    'starcoder' : dict( llm_class=ADLRCompletion, llm_kwargs=dict(model_name='starcoder')),
    'gpt-4' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4')),
    'gpt-4-32k' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4-32k')),
    'gpt-35-turbo' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-35-turbo')),
    'gpt-4-turbo' : dict( llm_class=LLMGatewayChat, llm_kwargs=dict(model_name='gpt-4-turbo')),

}
def get_llm(*args, **kwargs):
    config = library(*args, **kwargs)
    print(config)
    return config['llm_class'](**config['llm_kwargs'])
import sys
llm_agent = get_llm(model_name='gpt-4-turbo') #, temperature=0.1)
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
from langchain import PromptTemplate
# Count the langchain token
def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(cb)
        print(f'Spent a total of {cb.total_tokens} tokens')
        # if cb.total_tokens > 14000:
        #    exit(1)
    return result

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"], template=prompt_template
)

llm_chain = LLMChain(prompt=prompt, llm=llm_agent)
result = count_tokens(llm_chain, query="Hi")
print('\n\n=== AI: response === \n', result)

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


























