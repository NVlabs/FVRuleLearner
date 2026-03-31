
# neglect the command

def extract_timing_table(data: str):
    for i in range (len(data)):
        if data[i] == "=":
            break
    data = data[i:]
    return data

# data table example
TimingTables = """==========================================================================================================================================================================================================================
|                                                                                               |                                        |     SlacK Distribution from less than 0 to less than -0.050                   |
| corner                      end_clk  datecode                                                 | WNS      TNS        FEP    WID         |  0.000 -0.005 -0.010 -0.015 -0.020 -0.025 -0.030 -0.035 -0.040 -0.045 -0.050  |
==========================================================================================================================================================================================================================
| ssg_0c_0p6v_eoc3_max_si     gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.460   -8.155     63     13:4267     |      9      1      4      2      2      4      0      1      2      1     37  |
| ssg_0c_0p6v_eoc3_max_si     gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.432   -7.861     49     38:5546     |      5      1      3      0      2      0      2      0      0      0     36  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| ssg_105c_0p55v_eoc3_max_si  gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.280   -1.453     22     15:5436     |      0      3      0      2      0      0      1      2      1      0     13  |
| ssg_105c_0p55v_eoc3_max_si  gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.254   -1.970     22     40:6801     |      1      0      0      1      0      4      0      1      2      0     13  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| ssg_105c_0p67v_eoc5_max_si  gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.466   -132.982   3086   17:9190     |    325    269    257    258    224    171    196    158    146    138    944  |
| ssg_105c_0p67v_eoc5_max_si  gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.530   -59.561    1668   43:10654    |    241    191    188    141    127    114    112    116     65     54    319  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| ssg_105c_0p94v_eoc3_max_si  gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.357   -312.388   6515   19:161520   |    426    374    473    410    359    414    417    422    347    353   2520  |
| ssg_105c_0p94v_eoc3_max_si  gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.466   -203.904   5457   45:170766   |    557    494    492    444    412    376    359    321    313    237   1452  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| ssg_m40c_0p67v_eoc5_max_si  gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.432   -29.172    708    21:5731     |     97     79     63     66     62     49     45     30     28     19    170  |
| ssg_m40c_0p67v_eoc5_max_si  gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.453   -14.032    226    48:6949     |     35     25     25     23     17      9     10     10      6      5     61  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| tt_105c_0p55v_eoc3_max_si   gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.534   -36.614    763    25:2340897  |     79     78     83     48     57     59     41     46     32     28    212  |
| tt_105c_0p55v_eoc3_max_si   gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.539   -17.279    236    52:2279207  |     37     28     18     13     22     10     13     12      8      2     73  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| tt_105c_0p67v_eoc3_max_si   gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.529   -207.751   4546   27:2980193  |    364    344    310    344    340    315    262    287    228    196   1556  |
| tt_105c_0p67v_eoc3_max_si   gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.552   -108.402   3085   54:2915875  |    367    362    303    287    265    205    191    176    143    131    655  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| tt_105c_0p87v_eoc3_max_si   gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.451   -338.673   6874   28:201370   |    468    412    376    455    395    393    409    428    399    335   2804  |
| tt_105c_0p87v_eoc3_max_si   gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.472   -226.000   5902   55:211799   |    582    531    507    478    443    414    355    355    307    298   1632  |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| tt_105c_0p94v_eoc3_max_si   gpcclk   2024Mar13_revG40_set1_ipo0_SOL_CA_fullSM_std             | -0.376   -362.648   7275   29:392984   |    470    465    394    451    448    375    431    405    429    386   3021  |
| tt_105c_0p94v_eoc3_max_si   gpcclk   2024Mar29_revG40_set1_dtpc_nortskew_fullSM               | -0.464   -251.370   6379   57:414509   |    560    573    526    485    512    429    404    370    321    328   1871  |
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
