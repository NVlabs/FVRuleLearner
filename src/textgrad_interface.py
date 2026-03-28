from saver import saver
print = saver.log_info


from utils import get_root_path
import sys, os

# print(f'get_root_path {get_root_path()}')
# exit()
nx_textgrad_dir = f"{get_root_path()}/../nv_textgrad"


# nx_textgrad_dir = '/home/scratch.yunshengb_cpu/fv/nv_textgrad'

sys.path.insert(0, nx_textgrad_dir)  # make sure to git clone here
if not os.listdir(nx_textgrad_dir):
    raise RuntimeError(
        f'To use nx_textgrad_dir, "git clone ssh://git@gitlab-master.nvidia.com:12051/shornaa/nv_textgrad.git" under/outside {nx_textgrad_dir} (make sure this dir is non-empty)'
    )


print(f'Adding nx_textgrad_dir={nx_textgrad_dir} into sys.path')
import textgrad as tg
from textgrad.autograd.string_based_ops import StringBasedFunction
from textgrad.tasks.big_bench_hard import BigBenchHard, string_based_equality_fn


def prompt_self_reflection_textgrad(user_agent, reflection_agent, row, response, sorted_suggestions):
    llm_engine = tg.get_engine("mixtral_8x7b") # TODO: change this to be more flexible
    tg.set_backward_engine("mixtral_8x7b") # TODO: change this to be more flexible

    # llm_engine = tg.get_engine("meta/llama3-70b-instruct") # TODO: change this to be more flexible
    # tg.set_backward_engine("meta/llama3-70b-instruct") # TODO: change this to be more flexible

    # from textgrad.tasks import load_task

    # _, val_set, _, eval_fn = load_task("BBH_object_counting", llm_engine)
    # question_str, answer_str = val_set[0]

    question_str = row.prompt
    answer_str = row.ref_solution

    question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
    # print(f'question is {question}')
    answer = tg.Variable(str(answer_str), role_description="answer to the question", requires_grad=False)
    # print(f'answer is {answer}')

    system_prompt = tg.Variable(
        "You are a concise LLM. Think step by step.",
        requires_grad=True,
        role_description="system prompt to guide the LLM's reasoning strategy for accurate responses",
    )

    model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
    optimizer = tg.TGD(parameters=list(model.parameters()))

    prediction = model(question)
    # print(f'prediction is {prediction}')

    fn_purpose = "The runtime of string-based function that checks if the prediction is correct."

    eval_fn = StringBasedFunction(string_based_equality_fn, fn_purpose)

    loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))

    # print(f'eval_fn is {eval_fn}')
    # print(f'ground_truth_answer is {answer}')
    # print(f'loss is {loss}')
    stuff = loss.backward()
    # print(f'type(stuff) is {type(stuff)}')
    # print(f'stuff is {stuff}')

    stuff2 = optimizer.step()
    # print(f'stuff2 is {stuff2}')

    # print(f'new question is {question}')

    # print(f'new answer is {answer}')
    # prediction = model(question)

    # print(f'new prediction is {prediction}')
    print(f'new system_prompt={system_prompt}')
    # print(f'type(system_prompt)={type(system_prompt)}')
    # print(f'type(system_prompt.value)={type(system_prompt.value)}')
    # exit()
    return system_prompt.value
