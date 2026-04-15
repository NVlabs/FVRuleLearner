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
def extract_timing_table(data: str):
    for i in range(len(data)):
        if data[i] == "=":
            break
    data = data[i:]
    return data


TimingTables = """==========================================================================================================================================================================================================================
|                                                                                               |                                        |     SlacK Distribution from less than 0 to less than -0.050                   |
| corner                      end_clk  datecode                                                 | WNS      TNS        FEP    WID         |  0.000 -0.005 -0.010 -0.015 -0.020 -0.025 -0.030 -0.035 -0.040 -0.045 -0.050  |
==========================================================================================================================================================================================================================
| ssg_0c_0p6v_eoc3_max_si     gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.460   -8.155     63     13:4267     |      9      1      4      2      2      4      0      1      2      1     37  |
| ssg_0c_0p6v_eoc3_max_si     gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.432   -7.861     49     38:5546     |      5      1      3      0      2      0      2      0      0      0     36  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| ssg_105c_0p55v_eoc3_max_si  gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.280   -1.453     22     15:5436     |      0      3      0      2      0      0      1      2      1      0     13  |
| ssg_105c_0p55v_eoc3_max_si  gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.254   -1.970     22     40:6801     |      1      0      0      1      0      4      0      1      2      0     13  |
==========================================================================================================================================================================================================================
"""


system_prompt = """
You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.
Do not add code to output an error message string.
Enclose your SVA code with <CODE> and </CODE>. Only output the code snippet and do NOT output anything else.

{FV_prompt}
"""


DesignerHints = """Let’s think step by step. Firstly, analyze the change of WNS, and TNS for all designs in all PVT corners tables of these two "datecode" settings and explain with numbers using tools.
Secondly, comparing "FEP" of two datecode settings in all PVT corners tables.
Thirdly, analyze the slack distribution to identify the distribution of "slack less than 0" paths.
Finally, summarize the analysis in the following aspects:
1. Provide key takeaways, comparison, and suggestions with bullet points of the two "datecode" settings based on previous steps.
2. Identify the corner which still suffers many timing violations if any.

You need to use the provided tools to analyze the timing metrics! You are not good at math!
"""


ToolExample = """
[Tool Examples]

[Example Table]
==========================================================================================================================================================================================================================
|                                                                                               |                                        |     SlacK Distribution from less than 0 to less than -0.050                   |
| corner                      end_clk  datecode                                                 | WNS      TNS        FEP    WID         |  0.000 -0.005 -0.010 -0.015 -0.020 -0.025 -0.030 -0.035 -0.040 -0.045 -0.050  |
==========================================================================================================================================================================================================================
| ssg_0c_0p6v_eoc3_max_si     gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.460   -8.155     63     13:4267     |      9      1      4      2      2      4      0      1      2      1     37  |
| ssg_0c_0p6v_eoc3_max_si     gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.432   -7.861     49     38:5546     |      5      1      3      0      2      0      2      0      0      0     36  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| ssg_105c_0p55v_eoc3_max_si  gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.280   -1.453     23     15:5436     |      0      3      0      2      0      0      1      2      1      0     13  |
| ssg_105c_0p55v_eoc3_max_si  gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.254   -1.970     22     40:6801     |      1      0      0      1      0      4      0      1      2      0     13  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[Example 1]: Calculatation of the improvement or degradation of WNS of two datecode settings from 2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std to 2024Mar29_revG40_set1_dtpc_nortskew_fullSM in the example table.
Action: Call timing_metric_calculation_tool tool to calculate the change of WNS of two datecode settings from 2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std to 2024Mar29_revG40_set1_dtpc_nortskew_fullSM.
Arguments: {"timing_metric_name": "WNS","metric_values": "[-0.460, -0.280]-[-0.432, -0.254]"}

[Example 2]: Calculatation of the change of "FEP" of two datecode settings from the 2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std to 2024Mar29_revG40_set1_dtpc_nortskew_fullSM in the example table.
Action: Call timing_metric_calculation_tool tool to calculate the change of FEP of two datecode settings from 2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std to 2024Mar29_revG40_set1_dtpc_nortskew_fullSM.
Arguments: {"timing_metric_name": "FEP","metric_values": "[63, 23]-[49, 22]"}

[Example 3]: Calculate the change of slack distribution of corner ssg_0c_0p6v_eoc3_max_si from the 2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std to 2024Mar29_revG40_set1_dtpc_nortskew_fullSM in the example table.
Action: Call slack_distribution_calculation_tool tool to calculate the change of slack distribution of two datecode settings from 2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std to 2024Mar29_revG40_set1_dtpc_nortskew_fullSM.
Arguments: {"metric_values": "[9, 1, 4, 2, 2, 4, 0, 1, 2, 1, 37]-[5, 1, 3, 0, 2, 0, 2, 0, 0, 0, 36]"}'

[Tool Examples End]
"""
