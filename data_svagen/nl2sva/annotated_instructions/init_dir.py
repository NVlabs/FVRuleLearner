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
import gzip
import json
import os
import pathlib

from typing import Iterable, Dict

def write_jsonl(filepath: pathlib.Path, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    Skipping None in data
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    if not filepath.exists():
        filepath.touch()
    if filepath.as_posix().endswith(".gz"):
        fp = filepath.open(mode)
        with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
            for x in data:
                if x:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        fp = filepath.open(mode)
        for x in data:
            if x:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def parse_args():
    parser = argparse.ArgumentParser(
        description="initialize FV eval directory for a suite of DUT"
    )

    parser.add_argument(
        "-n","--name",
        type=str,
        help="name of DUT suite"
    )
    parser.add_argument(
        "-i","--num_test_cases",
        type=int,
        default=10,
        help="Number of test cases (assertions)"
    )
    return parser.parse_args()

def create_input_dir(name:str, num_test_cases:int):
    dir_path = pathlib.Path(__file__).parent / name
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    # create jsonl
    sample_jsons = []
    for i in range(num_test_cases):
        json =  {"task_id": f"{name}_{i}", "prompt": "replace"}
        sample_jsons.append(json)
        sva_file = dir_path / f"{name}_{i}.sva"
        sva_file.touch()
    
    write_jsonl(dir_path / f"{name}.jsonl", data=sample_jsons, append=True)

    
if __name__ == "__main__":
    args = parse_args()
    create_input_dir(args.name, args.num_test_cases)