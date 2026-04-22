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
import argparse
from dataclasses import asdict
import glob
import json
import os
import pathlib

from typing import Iterable, Dict

import pandas as pd

from fv_eval.data import read_sv_from_dir, InputData



def read_reference_solutions(solutions_dir: str, dut_name_list: Iterable[str]) -> Dict:
    benchmark_reference_outputs = {}
    for dut_name in dut_name_list:
        svafile_glob = glob.glob(solutions_dir + f"/{dut_name}/*.sva")
        if not svafile_glob:
            print(f"No reference solution found for:{dut_name}")
            continue
        benchmark_reference_outputs[dut_name] = {}
        for svafile in svafile_glob:
            with open(svafile, "r") as f:
                svatext = f.read()
            task_id = os.path.basename(svafile).split(".sva")[0]
            benchmark_reference_outputs[dut_name].update({task_id: svatext.strip()})
    return benchmark_reference_outputs


def read_svagen_raw_problems(problems_dir: str, dut_name_list: Iterable[str]) -> Dict:
    benchmark_inputs = {}
    for dut_name in dut_name_list:
        jsonfile_glob = glob.glob(problems_dir + f"/{dut_name}/*.jsonl")
        if not jsonfile_glob:
            print(f"No input prompt found for:{dut_name}")
            continue
        jsonlfile = jsonfile_glob[0]
        benchmark_inputs[dut_name] = []
        with open(jsonlfile, "r") as f:
            for line in list(f):
                json_object = json.loads(line)
                benchmark_inputs[dut_name].append(json_object)
    return benchmark_inputs


def preprocess_svagen_data(
    problems_dir: str,
    solutions_dir: str,
    testbench_dir: str,
    save_dir: str,
    debug: bool = False,
):
    """
    Preprocesses context (DUT, Testbench), input prompts, reference solutions, and in-context examples
    necessary for the svagen benchmark suite.
    Packages all necessary information into a single .jsonl file that run scripts
    can later read into administer evaluation of LMs

    Params:
        problems_dir: (str) source directory of svagen promp prompts
        solutions_dir: (str) source directory of svagen reference solutions
        testbench_dir: (str) source directory of svagen context testbenches
        save_dir: (str) save path
    """
    testbenches_dict = read_sv_from_dir(data_dir=testbench_dir)
    dut_names = list(testbenches_dict.keys())
    benchmark_inputs = read_svagen_raw_problems(
        problems_dir=problems_dir,
        dut_name_list=dut_names,
    )
    benchmark_reference_outputs = read_reference_solutions(
        solutions_dir=solutions_dir,
        dut_name_list=dut_names,
    )
    if debug:
        dut_names = [dut_names[0]]
    # Package into single dictionary
    full_dataset = []
    for dut_name in dut_names:
        for problem_dict in benchmark_inputs[dut_name]:
            task_id = problem_dict["task_id"]
            prompt = problem_dict["prompt"]
            testbench_context = testbenches_dict[dut_name]
            ref_solution = benchmark_reference_outputs[dut_name][task_id]
            full_dataset.append(
                InputData(
                    design_name=dut_name,
                    task_id=task_id,
                    prompt=prompt,
                    ref_solution=ref_solution,
                    testbench=testbench_context
                )
            )
    if debug:
        pd.DataFrame([asdict(d) for d in full_dataset]).to_csv(
            save_dir + f"/nl2sva_human_debug.csv", sep=",", index=False
        )
    else:
        pd.DataFrame([asdict(d) for d in full_dataset]).to_csv(
            save_dir + f"/nl2sva_human.csv", sep=",", index=False
        )


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Run LLM Inference for the FVEval-SVAGen Benchmark")
    parser.add_argument(
        "--svagen_nl2sva_input_dir",
        type=str,
        help="path to raw NL2SVA input dataset",
        default=ROOT / "annotated_instructions",
    ) 
    parser.add_argument(
        "--svagen_nl2sva_tb_dir",
        type=str,
        help="path to raw NL2SVA testbench (TB) SystemVerilog files",
        default=ROOT / "annotated_tb",
    )
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        help="path to save directory, where processed datasets for SVAGen tasks will be saved",
        default=ROOT / "data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
    )

    args = parser.parse_args()
    nl2sva_prompt_dir = args.svagen_nl2sva_input_dir.as_posix()
    nl2sva_tb_dir = args.svagen_nl2sva_tb_dir.as_posix()
    save_dir = args.save_dir.as_posix()

    preprocess_svagen_data(
        testbench_dir=nl2sva_tb_dir,
        problems_dir=nl2sva_prompt_dir,
        solutions_dir=nl2sva_prompt_dir,
        save_dir=save_dir,
    )