########################################################
# NVIDIA License
#
# 1. Definitions
#
# “Licensor” means any person or entity that distributes its Work.
# “Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
# The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
# Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.
#
# 2. License Grant
#
# 2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.
#
# 3. Limitations
#
# 3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.
#
# 3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.
#
# 3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.
#
# 3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.
#
# 3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.
#
# 3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.
#
# 4. Disclaimer of Warranty.
#
# THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 
#
# 5. Limitation of Liability.
#
# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
########################################################
# User write this file for different applications
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, Annotated
import re
import os
import shutil
import subprocess
import multiprocessing
from nltk.translate.bleu_score import sentence_bleu
from FVEval.fv_eval import utils as utils2

# from FVEval.fv_eval import evaluation
from FVEval.fv_eval import benchmark_launcher
from tqdm import tqdm
from pathlib import Path
import pickle
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'fv_eval'))) # Not sure whether it should be added
from FVEval.fv_eval import prompts_svagen_nl2sva as prompts_svagen_nl2sva
from FVEval.fv_eval.data import InputData, LMResult
from config import FLAGS

# using the react prompt
from saver import saver

print = saver.log_info


def sentence_bleu_score_calculation_tool(
    reference: Annotated[str, "The generated reference string for bleu score calculation"],
    candidate: Annotated[str, "The golden candidate string for bleu score calculation"],
) -> Annotated[str, "The calculated bleu score"]:
    reference_list = reference.split()
    candidate_list = candidate.split()

    bleu1_score = sentence_bleu([reference_list], candidate_list, weights=(1, 0, 0, 0))
    bleu2_score = sentence_bleu([reference_list], candidate_list, weights=(0, 1, 0, 0))
    bleu3_score = sentence_bleu([reference_list], candidate_list, weights=(0, 0, 1, 0))
    bleu4_score = sentence_bleu([reference_list], candidate_list, weights=(0, 0, 0, 1))
    string_output = (
        "Reference: "
        + reference
        + "\nCandidate: "
        + candidate
        + "\nThe bleu score of reference and candidate: "
        + "\n1-gram: "
        + str(bleu1_score)
        + "\n2-gram: "
        + str(bleu2_score)
        + "\n3-gram: "
        + str(bleu3_score)
        + "\n4-gram: "
        + str(bleu4_score)
    )

    return string_output


def write_design_sv(
    temp_dir: str, r: LMResult
):  # experiment_id -> r.experiment_id, task_id -> r.task_id, design_rtl -> r.design_rtl
    # For each result, write the packaged testbench to a SystemVerilog file
    # in the subdir directory
    # for r in results_list:
    with open(f"{temp_dir}/{r.experiment_id}_{r.task_id}.sv", "w") as f:
        f.write(r.design_rtl)


def write_testbench_sv(temp_dir: str, r: LMResult):
    tb = r.output_tb
    with open(f"{temp_dir}/{r.experiment_id}_{r.task_id}.sva", "w") as f:
        f.write(tb)


def launch_jg(tcl_file_path: str, sv_dir: str, experiment_id: str, task_id: str) -> str:
    tmp_jg_proj_dir = os.path.join(sv_dir, f"jg/{experiment_id}_{task_id}")
    if os.path.isdir(tmp_jg_proj_dir):
        shutil.rmtree(tmp_jg_proj_dir)
    jg_command = f"jg -fpv -batch -tcl {tcl_file_path} -define EXP_ID {experiment_id} -define TASK_ID {task_id} -define SV_DIR {sv_dir} -proj {tmp_jg_proj_dir}"

    with os.popen(jg_command) as process:
        output = process.read()

    return output.strip()


def launch_jg_opencore(
    tcl_file_path: str,
    sv_dir: str,
    experiment_id: str,
    task_id: str,
    lm_assertion_text: str,
    signal_list_text: str,
    output_queue: multiprocessing.Queue = None,
) -> None:
    tmp_jg_proj_dir = sv_dir + f"/jg/{experiment_id}_{task_id}"
    if os.path.isdir(tmp_jg_proj_dir):
        shutil.rmtree(tmp_jg_proj_dir)
    jg_command = [
        "jg",
        "-fpv",
        "-batch",
        "-tcl",
        tcl_file_path,
        "-define",
        "LM_ASSERT_TEXT",
        lm_assertion_text,
        "-define",
        "SIGNAL_LIST",
        signal_list_text,
        "-define",
        "EXP_ID",
        experiment_id,
        "-define",
        "TASK_ID",
        task_id,
        "-define",
        "SV_DIR",
        sv_dir,
        "-proj",
        tmp_jg_proj_dir,
        "-allow_unsupported_OS",
    ]

    FVEval_dir = os.path.abspath(os.path.join(os.path.dirname(tcl_file_path), '..'))
    print(f"Executing command in directory: {FVEval_dir}")
    result = subprocess.run(jg_command, cwd=FVEval_dir, capture_output=True, text=True)
    # result = subprocess.run(jg_command, capture_output=True, text=True)
    return result.stdout.strip()

def generate_question_prompt(prompt):
    return (
        prompts_svagen_nl2sva.SVAGEN_QUESTION_PREAMBLE + prompt + "\n" + prompts_svagen_nl2sva.SVAGEN_QUESTION_POSTAMBLE
    )


def package_testbench(
    prompt, ref_solution, testbench, lm_response: str
):  # used to be row.prompt, row.ref_solution, row.testbench
    question_prompt = generate_question_prompt(prompt)
    reference_assertion_text = ref_solution.replace("asrt", "reference")
    assertion_text = utils2.parse_code_response(lm_response)

    # retrieve question text
    commented_question_text = "\n//".join(question_prompt.split("\n"))
    testbench_text = testbench
    packaged_tb_text = (
        testbench_text.split("endmodule")[0]
        + "\n\n"
        + commented_question_text
        + "\n\n"
        + reference_assertion_text
        + "\n\n"
        + assertion_text
        + "\n\n"
        + "endmodule"
    )
    return packaged_tb_text


def evaluate_jg(
    LLM_response: Annotated[str, "Assertions output from LLM"]
    # result_list: Annotated[list[LMResult], "The list of the series of assertion results with one component."], 
) -> Annotated[str, "The feedback from JasperGold."]:

    # exit()
    # raise NotImplemented()

    LLM_response = utils2.parse_code_response(LLM_response)
    file_path = Path(saver.logdir) / f'var_temp.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # with_rtl_design = FLAGS.with_rtl_design
    with_rtl_design = False  # Release mainline uses the NL2SVA path.
    # setup temp and save directories
    temp_dir = Path(saver.logdir + "/temp")
    if not os.path.isdir(temp_dir):
        utils2.mkdir_p(temp_dir)
    # For each result, write the packaged testbench to a SystemVerilog file
    # in the temp directory
    packaged_tb_text = package_testbench(data['user_prompt'], data['ref_solution'], data['testbench'], LLM_response)

    # Save outside
    response = LMResult(
        experiment_id=data['experiment_id'],
        task_id=data['task_id'],
        model_name=data['model_name'],
        response=LLM_response,
        ref_solution=data['ref_solution'],
        user_prompt=data['user_prompt'],
        output_tb=packaged_tb_text,
        design_rtl=data['design_rtl'],
        cot_response=data['cot_response'],
    )

    if with_rtl_design:
        write_design_sv(temp_dir, response)
    write_testbench_sv(temp_dir, response)

    uid = f"{response.experiment_id}_{response.task_id}"

    jasper_out_str = launch_jg(
        tcl_file_path=FLAGS.tcl_file_path,
        sv_dir=temp_dir,
        experiment_id=response.experiment_id,
        task_id=response.task_id,
    )

    # print(f'@@@@@@@@@@@@@@@@@@@@@@@ evaluate_jg')
    if FLAGS.debug:
        print(jasper_out_str)

    if FLAGS.cleanup_temp_files:
        shutil.rmtree(temp_dir)

    syntax_error_match = re.findall(r"syntax error", jasper_out_str)
    
    # if not syntax_error_match: # No syntax error
    #     os.remove(file_path)

    return jasper_out_str, syntax_error_match


def evaluate_jg_opencore(
    LLM_response: Annotated[str, "Assertions output from LLM"]
    # result_list: Annotated[list[LMResult], "The list of the series of assertion results with one component."], 
) -> Annotated[str, "The feedback from JasperGold."]:

    # exit()
    # raise NotImplemented()

    # Check if LLM_response is None
    if LLM_response is None:
        print("Warning: LLM_response is None. Skipping evaluation.")
        return "", []  # Skip further processing and return empty results

    lm_assertion_text = (
        utils2.parse_code_response(LLM_response)
        .strip()
        .replace("\n", "")
    )
    file_path = Path(saver.logdir) / f'var_temp.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # with_rtl_design = FLAGS.with_rtl_design
    with_rtl_design = False  # Release mainline uses the NL2SVA path.
    # setup temp and save directories
    # temp_dir = Path(saver.logdir + "/temp")
    # if not os.path.isdir(temp_dir):
    #     utils2.mkdir_p(temp_dir)
    # For each result, write the packaged testbench to a SystemVerilog file
    # in the temp directory
    temp_dir = os.path.join(saver.logdir , "tmp_sva")

    # Save outside
    post_ref_solution = f"assert property({data['ref_solution']});"
    safe_task_id = data['task_id'].replace('/', '_')
    response = LMResult(
        experiment_id=data['experiment_id'],
        task_id=safe_task_id,
        model_name=data['model_name'],
        response=lm_assertion_text,
        ref_solution=post_ref_solution,
        user_prompt=data['user_prompt'],
        output_tb="\n",
        design_rtl="\n",
        cot_response="cot_response\n",
    )

    uid = f"{response.experiment_id}_{response.task_id}"

    lm_assertion_text = utils2.extract_assertion_formula(lm_assertion_text)

    # print(f"@@@DEBUG: lm_assertion_text = {lm_assertion_text}")
    # print(f"@@@DEBUG: ref_assertion_text = {ref_assertion_text}")
    
    if FLAGS.llm_model == "gpt-4":
        signal_list = re.findall(r"'([a-zA-Z_]\w*)'", data['user_prompt'])
    else:   
        signal_list = re.findall(r"[\'`\"]([a-zA-Z_]\w*)[\'`\"]", data['user_prompt'])
    # Filter out any invalid signal names
    valid_signal_list = [sig for sig in signal_list if re.match(r"^[a-zA-Z_]\w*$", sig)]
    # Deduplicate signal names
    valid_signal_list = list(set(valid_signal_list))
    # Join the signal names into a comma-separated string
    signal_list_text = ",".join(valid_signal_list)

    jasper_out_str = launch_jg_opencore(
        tcl_file_path=FLAGS.tcl_file_path,
        sv_dir=temp_dir,
        experiment_id=response.experiment_id,
        task_id = safe_task_id,
        lm_assertion_text = lm_assertion_text,
        signal_list_text = signal_list_text,
    )

    # print(f'@@@@@@@@@@@@@@@@@@@@@@@ evaluate_jg')
    if FLAGS.debug:
        print(jasper_out_str)

    if FLAGS.cleanup_temp_files:
        # shutil.rmtree(temp_dir)
        utils2.safe_rmtree(temp_dir)

    syntax_error_match = re.findall(r"syntax error", jasper_out_str)
    
    # if not syntax_error_match: # No syntax error
    #     os.remove(file_path)

    return jasper_out_str, syntax_error_match

# def launch_jg_custom_equiv_check(
#     tcl_file_path: str,
#     lm_assertion_text: str,
#     # ref_assertion_text: str,
#     # signal_list_text: str,
#     sv_dir: str,
#     # experiment_id: str,
#     # task_id: str,
# ) -> str:
#     tmp_jg_proj_dir = sv_dir + f"/jg/test"
#     if os.path.isdir(tmp_jg_proj_dir):
#         shutil.rmtree(tmp_jg_proj_dir)
#     jg_command = [
#         "jg",
#         "-fpv",
#         "-batch",
#         "-tcl",
#         tcl_file_path,
#         # "-define",
#         # "EXP_ID",
#         # experiment_id,
#         # "-define",
#         # "TASK_ID",
#         # task_id,
#         "-define",
#         "SV_DIR",
#         sv_dir,
#         "-proj",
#         tmp_jg_proj_dir,
#     ]
#     # Run the command
#     os.system(jg_command)

#     log_file_path = os.path.join(tmp_jg_proj_dir, "output.log")
#     # Read the output from the generated log file
#     with open(log_file_path, "r") as log_file:
#         result = log_file.read().strip()

#     return result


def extract_uncovered_lines(feedback):
    """
    Extracts uncovered or unchecked code lines from JasperGold feedback.

    Parameters:
    - feedback (str): The JasperGold feedback text.

    Returns:
    - List[Dict]: A list of dictionaries containing details about each uncovered or unchecked code line.
    """
    # Remove leading and trailing braces
    feedback = feedback.strip('{}')
    
    # Split the feedback into individual entries
    entries = feedback.split('} {')
    
    entries_list = []
    
    for entry in entries:
        # Ensure the entry is properly formatted
        entry = entry.strip()
        if not entry.startswith('{'):
            entry = '{' + entry
        if not entry.endswith('}'):
            entry = entry + '}'
        
        # Use regex to extract key-value pairs
        pattern = r'(\S+)\s*\{(.*?)\}'
        pairs = re.findall(pattern, entry)
        
        # Build a dictionary for each entry
        entry_dict = {}
        for key, value in pairs:
            entry_dict[key] = value
        entries_list.append(entry_dict)
    
    # Extract entries where formal_status is not 'Covered and Checked'
    uncovered_entries = []
    for entry in entries_list:
        formal_status = entry.get('formal_status', '')
        if formal_status != 'Covered and Checked':
            uncovered_entries.append(entry)
    
    # Return the list of uncovered entries
    return uncovered_entries