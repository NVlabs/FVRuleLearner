# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import glob
import os
import pathlib
from typing import Dict, Optional


@dataclass
class TextSimilarityEvaluationResult:
    experiment_id: str
    task_id: str
    model_name: str
    bleu: float
    rouge: float
    exact_match: float


@dataclass
class JGEvaluationResult:
    experiment_id: str
    task_id: str
    model_name: str
    syntax: float
    functionality: float
    func_relaxed: float
    coverage_stimuli_statement: Optional[float] = 0.0
    coverage_stimuli_branch: Optional[float] = 0.0
    coverage_stimuli_functional: Optional[float] = 0.0
    coverage_stimuli_toggle: Optional[float] = 0.0
    coverage_stimuli_expression: Optional[float] = 0.0
    coverage_coi_statement: Optional[float] = 0.0
    coverage_coi_branch: Optional[float] = 0.0
    coverage_coi_functional: Optional[float] = 0.0
    coverage_coi_toggle: Optional[float] = 0.0
    coverage_coi_expression: Optional[float] = 0.0
    # bound_improve: int


@dataclass
class LMResult:
    experiment_id: str
    task_id: str
    model_name: str
    response: str
    ref_solution: str
    design_rtl: str
    output_tb: str
    user_prompt: str
    cot_response: str


@dataclass
class InputData:
    design_name: str
    prompt: str
    ref_solution: str
    task_id: Optional[str] = None
    testbench: Optional[str] = None
    module_interface: Optional[str] = None
    assumption: Optional[str] = None


ROOT = pathlib.Path(__file__).parent.parent

"""
    Read SystemVerilog files from given data directory
    args: directory path to SV files (str)
    returns: Dict of (SV file descripter, raw SV text)
"""


def read_sv_from_dir(data_dir: str, is_testbench: bool = False) -> dict[str, str]:
    dut_texts = {}
    file_suffix = ".sva" if is_testbench else ".sv"
    for dut_sv in glob.glob(data_dir + f"/*{file_suffix}"):
        svtext = ""
        with open(dut_sv, "r") as f:
            svtext = f.read()
        dut_name = os.path.basename(dut_sv).split(file_suffix)[0]
        dut_texts.update({dut_name: svtext.strip()})
    return dut_texts


"""
    Read dataset CSV files from given data directory
    args: directory path to SV files (str)
    returns: Dict of (SV file descripter, raw SV text)
"""


def read_datasets_from_dir(data_dir: str) -> dict[str, str]:
    file_suffix = ".csv"
    return glob.glob(data_dir + f"/*{file_suffix}")
