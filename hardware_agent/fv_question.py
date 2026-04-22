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