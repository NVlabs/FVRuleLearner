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
"""
Prompts used for Design2SVA
Subject to modifications and improvements
"""

SVAGEN_HEADER = """You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to generate a SystemVerilog assertion for the design-under-test provided.
"""

ASSUMPGEN_HEADER = """You are an AI assistant specializing in the formal verification of register transfer level (RTL) designs. Your primary task is to generate SystemVerilog assumptions that must hold true for the design-under-test (DUT) comment_PH. Ensure the assumptions are consistent with the design's expected behavior and constraints, and help guide the verification process by making sure the assumptions are neither too weak nor too strong.
"""

SVAGEN_DUT_PREAMBLE = """Here is the design RTL to generate assertions for:\n"""

ASSUMPGEN_DUT_PREAMBLE = """Here is the design RTL to generate assumptions for:\n"""

SVAGEN_TB_PREAMBLE = """Here is a partial testbench for you to work on:\n"""

ASSUMPGEN_TB_PREAMBLE = """Here is a partial testbench for you to work on:\n"""

def get_design2sva_planning_prompt(with_assumptions, num_assertions=None) -> str:
  if with_assumptions:
    if num_assertions:
      return f"""Question: generate all possible SVA assertions for the feature you listed.
If necessary, include any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else. Remember to output {num_assertions} assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```.

Only output the code snippet and do NOT output anything else. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```.

The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para

Answer:"""
    else:
      return f"""The following are constraints you need to satisfy in completing the task:
Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.
When implementing the assertion, generate a concurrent SVA assertion and do not add code to output an error message string.
Enclose each of your SystemVerilog code with ```systemverilog and ```. 

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else. The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para

Question: in words, describe {num_assertions} feature(s) of the design that should be verified."""
    
  elif not with_assumptions:
    return f"""The following are constraints you need to satisfy in completing the task:
Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.
When implementing the assertion, generate a concurrent SVA assertion and do not add code to output an error message string.
Enclose each of your SystemVerilog code with ```systemverilog and ``` individually.

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else.

Question: in words, describe {num_assertions} feature(s) of the design that should be verified.
Answer:"""


SVAGEN_MODELING_QUESTION = """Question: for the feature you listed, implement modeling code, including wires, registers, and their assignments,
that is necessary for creating assertions.
Answer:"""


ASSUMPGEN_MODELING_QUESTION = """Question: for the feature you listed, implement modeling code, including wires, registers, and their assignments,
that is necessary for creating assumptions.
Answer:"""


def get_design2sva_question_prompt(with_assumptions , num_assertions=None) -> str:
  assumptions_paragraph = """The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para
"""

  if num_assertions:
    question_prompt = f"""Question: generate {num_assertions} SVA assertion(s) for the feature you listed. 
If necessary, include any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assertions, implement as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else. Remember to output {num_assertions} assertion(s), and each assertion should be wrapped individually in systemverilog and .
"""
  else: 
    question_prompt = """Question: generate all possible SVA assertions for the feature you listed. If necessary, include any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added. Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification. Do NOT instantiate the design module inside the testbench.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string. Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```.
"""
  # Append the assumptions paragraph if with_assumptions is True
  if with_assumptions:
    question_prompt += assumptions_paragraph

  return question_prompt


def get_design2sva_direct_question_prompt(with_assumptions, num_assertions = None) -> str:
  if with_assumptions:
    if num_assertions:
        return f"""Question: generate {num_assertions} SVA assertion(s) for the given design RTL that is most important to verify.
If necessary, produce any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assertion, generate {num_assertions} concurrent SVA assertion(s) and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```. 

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else. Remember to output {num_assertions} assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```.

The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para

Answer:"""
    else:
        return f"""Question: generate all possible SVA assertions for the given design RTL that are important to verify.
If necessary, produce any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assertion, generate all possible SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```. 

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```.

The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para

Answer:"""
    
  elif not with_assumptions:
    return f"""Question: generate a single SVA assertion for the feature you listed. 
If necessary, include any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the te  stbench.

When implementing the assertions, implement as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog code with ```systemverilog and ```. 

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else.

Answer:"""

def get_design2sva_pma_prompt(with_assumptions, use_inter_param, all_signals_in_one_shot, num_assertions=None) -> str:
  if with_assumptions:
    if use_inter_param and not all_signals_in_one_shot:
      # With assumptions + Add a number of assertions for generation + Use intermediate parameters
      return f"""Question: generate all possible SVA assertions for the feature you listed.
If necessary, include any extra code, including wires, registers, and their assignments.

- Do **NOT** use signals from the design RTL. Only use the module input signals or any internal signals you have added in the testbench.
- Do **NOT** use any 'initial' blocks. This testbench is for formal verification, not for running RTL simulation.
- Do **NOT** instantiate the design module inside the testbench.
- When implementing assertions, generate concurrent SVA assertions without adding code to output error message strings.
- Enclose each of your SystemVerilog code with ```systemverilog and ``` individually.
- Each feature's code should be self-contained. Ensure that any intermediate signals (e.g., state, in_A, in_B, etc.) are declared and assigned within the feature's code block.
- Ensure that the assertion’s reset condition matches the reset polarity of the design.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
  (a && b) != 1'b1
);
```

The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para

Question: In words, describe {num_assertions} feature(s) of the design that should be verified. For the feature you listed, implement modeling code, including wires, registers, and their assignments, that is necessary for creating assertions in the same SystemVerilog block. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```. signal_rule_PH

Answer:"""

    elif use_inter_param and all_signals_in_one_shot:
      return f"""Question: generate all possible SVA assertions for the feature you listed.
If necessary, include any extra code, including wires, registers, and their assignments.

- Do **NOT** use signals from the design RTL. Only use the module input signals or any internal signals you have added in the testbench.
- Do **NOT** use any 'initial' blocks. This testbench is for formal verification, not for running RTL simulation.
- Do **NOT** instantiate the design module inside the testbench.
- When implementing assertions, generate concurrent SVA assertions without adding code to output error message strings.
- Enclose each of your SystemVerilog code with ```systemverilog and ``` individually.
- Each feature's code should be self-contained. Ensure that any intermediate signals (e.g., state, in_A, in_B, etc.) are declared and assigned within the feature's code block.
- Ensure that the assertion’s reset condition matches the reset polarity of the design.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
  (a && b) != 1'b1
);
```

The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para

Question: In words, describe overall signal_size_PH feature(s) of the design that should be verified. For the feature you listed, implement modeling code, including wires, registers, and their assignments, that is necessary for creating assertions in the same SystemVerilog block. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```. signal_rule_PH

Answer:"""

    elif not use_inter_param:
      # With assumptions + Not use intermediate parameters
      return f"""Question: generate all possible SVA assertions for the feature you listed.

- Do **NOT** use signals from the design RTL. Only use the module input signals or any internal signals added in the testbench.
- Do **NOT** use any 'initial' blocks. This testbench is for formal verification, not for running RTL simulation.
- Do **NOT** instantiate the design module inside the testbench.
- Do **NOT** generate the intermediate parameters assignments.
- When implementing assertions, generate concurrent SVA assertions without adding code to output error message strings.
- Enclose each of your SystemVerilog code with ```systemverilog and ``` individually.
- Ensure that the assertion’s reset condition matches the reset polarity of the design.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
assert property (@(posedge clk) disable iff (tb_reset)
  (a && b) != 1'b1
);
```

The following assumptions and partial assertions should be considered when creating additional assertions. These assumptions define the expected behavior and constraints of the design and constraints of the design and partial existing assertions:
assumption_para

Question: In words, describe {num_assertions} feature(s) of the design that should be verified. Do not use intermediate parameter LLM generated to generate the assertion. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```. signal_rule_PH

Answer:"""
    
  elif not with_assumptions:
    if use_inter_param and not all_signals_in_one_shot:
      return f"""Question: generate all possible SVA assertions for the feature you listed.
If necessary, include any extra code, including wires, registers, and their assignments.

- Do **NOT** use signals from the design RTL. Only use the module input signals or any internal signals you have added in the testbench.
- Do **NOT** use any 'initial' blocks. This testbench is for formal verification, not for running RTL simulation.
- Do **NOT** instantiate the design module inside the testbench.
- When implementing assertions, generate concurrent SVA assertions without adding code to output error message strings.
- Enclose each of your SystemVerilog code with ```systemverilog and ``` individually.
- Each feature's code should be self-contained. Ensure that any intermediate signals (e.g., state, in_A, in_B, etc.) are declared and assigned within the feature's code block.
- Ensure that the assertion’s reset condition matches the reset polarity of the design.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
  (a && b) != 1'b1
);
```

Question: In words, describe {num_assertions} feature(s) of the design that should be verified. For the feature you listed, implement modeling code, including wires, registers, and their assignments, that is necessary for creating assertions in the same SystemVerilog block. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```. signal_rule_PH

Answer:"""

    elif use_inter_param and all_signals_in_one_shot:
      return f"""Question: generate all possible SVA assertions for the feature you listed.
If necessary, include any extra code, including wires, registers, and their assignments.

- Do **NOT** use signals from the design RTL. Only use the module input signals or any internal signals you have added in the testbench.
- Do **NOT** use any 'initial' blocks. This testbench is for formal verification, not for running RTL simulation.
- Do **NOT** instantiate the design module inside the testbench.
- When implementing assertions, generate concurrent SVA assertions without adding code to output error message strings.
- Enclose each of your SystemVerilog code with ```systemverilog and ``` individually.
- Each feature's code should be self-contained. Ensure that any intermediate signals (e.g., state, in_A, in_B, etc.) are declared and assigned within the feature's code block.
- Ensure that the assertion’s reset condition matches the reset polarity of the design.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
  (a && b) != 1'b1
);
```

Question: In words, describe signal_size_PH feature(s) of the design that should be verified. For the feature you listed, implement modeling code, including wires, registers, and their assignments, that is necessary for creating assertions in the same SystemVerilog block. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```. signal_rule_PH

Answer:"""

    elif not use_inter_param:
      return f"""Question: generate all possible SVA assertions for the feature you listed.

- Do **NOT** use signals from the design RTL. Only use the module input signals or any internal signals added in the testbench.
- Do **NOT** use any 'initial' blocks. This testbench is for formal verification, not for running RTL simulation.
- Do **NOT** instantiate the design module inside the testbench.
- Do **NOT** generate the intermediate parameters.
- When implementing assertions, generate concurrent SVA assertions without adding code to output error message strings.
- Enclose each of your SystemVerilog code with ```systemverilog and ``` individually.
- Ensure that the assertion’s reset condition matches the reset polarity of the design.

When implementing the assertions, implement them as concurrent SVA assertions and do not add code to output an error message string.
Enclose each of your SystemVerilog assertion code with ```systemverilog and ```.

For example:
```systemverilog
assert property (@(posedge clk) disable iff (tb_reset)
  (a && b) != 1'b1
);
```

Question: In words, describe overall signal_size_PH feature(s) of the design that should be verified. Do not use intermediate parameter LLM generated to generate the assertion. Remember to output assertion(s), and each assertion should be wrapped individually in ```systemverilog and ```. signal_rule_PH

Answer:"""


def get_design2sva_assumption_planning_prompt() -> str:
  return f"""The following are constraints you need to satisfy in completing the task:
Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.
When implementing the assumptions, generate concurrent SVA assumptions and do not add code to output an error message string.
Enclose each of your SystemVerilog code with ```systemverilog and ```.

For example:
```systemverilog
asum_example_property: assume property (@(posedge clk) disable iff (!reset)
    (signal_condition |-> strong(##[1:$] !signal_condition)));
```

Only output the code snippet and do NOT output anything else.

Answer:"""

def get_design2sva_assumption_question_prompt() -> str:
    return f"""Question: generate all possible SVA assumptions for the feature you listed. 
If necessary, include any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assumptions, implement them as concurrent SVA assumptions and do not add code to output an error message string.
Enclose all your SystemVerilog code with ```systemverilog and ```.

For example:
```systemverilog
asum_example_property: assume property (@(posedge clk) disable iff (!reset)
    (signal_condition |-> strong(##[1:$] !signal_condition)));
```
Only output the code snippet and do NOT output anything else.

Answer:"""

def get_design2sva_assumption_direct_question_prompt() -> str:
    return f"""Question: generate all possible SVA assumptions for the given design RTL that are important to verify.
If necessary, produce any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assumptions, generate concurrent SVA assumptions and do not add code to output an error message string.
Enclose all your SystemVerilog code with ```systemverilog and ```.

For example:
```systemverilog
asum_example_property: assume property (@(posedge clk) disable iff (!reset)
    (signal_condition |-> strong(##[1:$] !signal_condition)));
```
Only output the code snippet and do NOT output anything else.

Answer:"""

# arbiter_v = """
# // *****************************************************************************************************
# // SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# // SPDX-License-Identifier: LicenseRef-NvidiaProprietary
# //
# // NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# // property and proprietary rights in and to this material, related
# // documentation and any modifications thereto. Any use, reproduction,
# // disclosure or distribution of this material and related documentation
# // without an express license agreement from NVIDIA CORPORATION or
# // its affiliates are strictly prohibited.
# // *****************************************************************************************************
# //-----------------------------------------------------
# // Design Name : arbiter
# // File Name   : arbiter.v
# // Function    : Arbitration among several client requests
# //--------------------------------------------
# module fv_arbiter_example(
# clk,
# rst,
# req,
# gnt_busy,
# gnt
# );

# parameter NUM_OF_CLIENTS = 4;

# //Declaring ports
# input clk;
# input rst;
# input req;
# input gnt_busy;
# output gnt;

# //Declaring registers and wires
# wire [NUM_OF_CLIENTS-1:0] req;
# wire gnt_busy;
# reg  [NUM_OF_CLIENTS-1:0] gnt;

# //internal variables
# reg  [NUM_OF_CLIENTS-1:0] gnt_pre;
# reg  [NUM_OF_CLIENTS-1:0] rr_gnt;

# always_comb begin
#     gnt = {NUM_OF_CLIENTS{!gnt_busy}} & gnt_pre;
# end

# always_comb begin
#       gnt_pre = 4'b0000;
#         case (rr_gnt)
#             4'b0000 : begin
#                  if      (req[0]) begin
#                      gnt_pre = 4'b0001;
#                  end
#                  else if (req[1]) begin
#                      gnt_pre = 4'b0010;
#                  end
#                  else if (req[2]) begin
#                      gnt_pre = 4'b0100;
#                  end
#                  else if (req[3]) begin
#                      gnt_pre = 4'b1000;
#                  end
#             end
#             4'b0001 : begin
#                  if      (req[1]) begin
#                      gnt_pre = 4'b0010;
#                  end
#                  else if      (req[2]) begin
#                      gnt_pre = 4'b0100;
#                  end
#                  else if (req[3]) begin
#                      gnt_pre = 4'b1000;
#                  end
#                  else if (req[0]) begin
#                      gnt_pre = 4'b0001;
#                  end
#             end
#             4'b0010 : begin
#                  if      (req[2]) begin
#                      gnt_pre = 4'b0100;
#                  end
#                  else if (req[3]) begin
#                      gnt_pre = 4'b1000;
#                  end
#                  else if (req[0]) begin
#                      gnt_pre = 4'b0001;
#                  end
#                  else if (req[1]) begin
#                      gnt_pre = 4'b0010;
#                  end
#             end
#             4'b0100 : begin
#                  if (req[3]) begin
#                      gnt_pre = 4'b1000;
#                  end
#                  else if      (req[0]) begin
#                      gnt_pre = 4'b0001;
#                  end
#                  else if (req[1]) begin
#                      gnt_pre = 4'b0010;
#                  end
#                  else if (req[2]) begin
#                      gnt_pre = 4'b0100;
#                  end
#             end
#             4'b1000 : begin
#                  if      (req[0]) begin
#                      gnt_pre = 4'b0001;
#                  end
#                  else if (req[1]) begin
#                      gnt_pre = 4'b0010;
#                  end
#                  else if (req[2]) begin
#                      gnt_pre = 4'b0100;
#                  end
#                  else if (req[3]) begin
#                      gnt_pre = 4'b1000;
#                  end
#             end
#             default : begin 
#                         gnt_pre[NUM_OF_CLIENTS-1:0] = {NUM_OF_CLIENTS{1'b0}};
#                       end  
#         endcase
# end

# always_ff @(posedge clk or negedge rst) begin
#   if (!rst) begin
#     rr_gnt <= {NUM_OF_CLIENTS{1'b0}};
#   end else begin
#     if (!gnt_busy & req != {NUM_OF_CLIENTS{1'b0}}) begin
#         rr_gnt <= gnt;
#     end 
#   end
# end
# endmodule
# """

# fifo_v = """
# //-----------------------------------------------------
# // Design Name : fifo
# // File Name   : fifo.v
# // Function    : Synchronous (single clock) FIFO
# //--------------------------------------------
# module fv_fifo_example (
# clk,
# rst,
# wr_vld,
# wr_data,
# wr_ready,
# rd_vld,
# rd_data,
# rd_ready,
# fifo_full,
# fifo_empty
# );    
 
# // FIFO constants
# parameter DATA_WIDTH = 8;
# parameter RAM_DEPTH = 4; 

# // Port Declarations
# input clk;
# input rst;
# output wr_ready;
# input [DATA_WIDTH-1:0] wr_data;
# input wr_vld;
# output rd_vld;
# output [DATA_WIDTH-1:0] rd_data;
# input rd_ready;
# output fifo_empty,fifo_full;

# //-----------Internal variables-------------------
# reg [1:0] wr_pointer;
# reg [1:0] rd_pointer;
# reg [2:0] status_cnt;
# wire [7:0] rd_data ;
# reg [RAM_DEPTH-1:0][DATA_WIDTH-1:0] stored;
# reg fifo_full,fifo_empty;

# //-----------Variable assignments---------------
# assign fifo_full = (status_cnt == RAM_DEPTH);
# assign fifo_empty = (status_cnt == 0);
# assign wr_ready = !fifo_full;
# assign rd_vld = !fifo_empty;

# //-----------Code Start---------------------------
# always @ (posedge clk or posedge !rst)
# begin : WRITE_POINTER
#   if (!rst) begin
#     wr_pointer <= 0;
#   end else if ((!fifo_full) && wr_vld ) begin
#    stored[wr_pointer][7:0] <= wr_data;
#    wr_pointer <= wr_pointer + 1;
#    end
# end

# assign rd_data = ((!fifo_empty) && rd_ready ) ? stored[rd_pointer][7:0] : 8'b0;

# always @ (posedge clk or posedge !rst)
# begin : READ_POINTER
#   if (!rst) begin
#     rd_pointer <= 0;
#   end else if ((!fifo_empty) && rd_ready ) begin
#     rd_pointer <= rd_pointer + 1;
#   end
# end

# always @ (posedge clk or posedge !rst)
# begin : STATUS_COUNTER
#   if (!rst) begin
#     status_cnt <= 0;
#   // Read but no write.
#   end else if ((rd_ready && rd_ready) && !(wr_vld && wr_ready) 
#                 && (status_cnt > 0)) begin
#     status_cnt <= status_cnt - 1;
#       // Write but no read.
#   end else if ((wr_vld && wr_ready) && !(rd_ready && rd_ready) 
#                && (status_cnt < 4 )) begin
#     status_cnt <= status_cnt + 1;
#   end

# end

# endmodule
# """

# feature_tree_generation_with_assumptions = """
# Please analyze the following design and generate a detailed 5-level feature tree that captures the design features, assumptions, and assertions in JSON format. Each node should be a key, and suggestions connecting assumptions to assertions should be represented as edges in the JSON. The structure should be suitable for easy extraction and conversion into a graph.

# - **Level 0 (Design Description):** Provide an overview of the design, including its purpose, scope, and high-level behavior. This level sets the context for understanding the features, assumptions, and assertions.
#   - **Level 3 (Design Features):** Identify and list the main features of the design. There can be multiple features at this level, each representing a high-level functionality of the design.
#     - **Level 2 (Block Assumptions):** For each design feature at Level 3, list the assumption blocks related to that feature. These blocks group together specific assumptions.
#       - **Level 1 (Specific Assumptions):** For each assumption block at Level 2, provide specific assumptions with natural language descriptions.
#     - **Level 4 (Block Assertions):** For each design feature at Level 3, list the assertion blocks related to that feature. These blocks group together specific assertions.
#       - **Level 5 (Specific Assertions):** For each assertion block at Level 4, provide specific assertions with natural language descriptions.
#     - **Suggestions (Edges):** Include any suggestions (rules/guidelines) connecting specific assumptions and assertions as edges.

# **Design Module:** design_PH

# **Output format:** 
# - Present the feature tree in JSON format.
# - Include 'edges' as a separate JSON key at the top level, which captures the relationships (suggestions) between assumptions and assertions.
# - The tree structure should be hierarchical and indented accordingly to reflect the design's features and assumptions.


# **Example:**
# Level 0: Design
#     Description: A Simple Light Switch that controls a light bulb.

#     Level 3: Feature 1: Toggle Light
#         Description: Allows turning the light on and off using a switch.

#         Level 2: Assumption Block for Toggle Light
#             Description: The switch will reliably respond when pressed, and no unintended toggles will occur.

#             Level 1: Specific Assumption 1
#                 Description: The switch provides a stable signal without bouncing.

#         Level 4: Assertion Block for Toggle Light
#             Description: The light will correctly reflect the state of the switch, turning on or off in sync with the toggle.

#             Level 5: Specific Assertion 1
#                 Description: The light turns on when the switch is in the 'on' position.
# """

# feature_tree_generation_wo_assumptions = """
# Please analyze the following design and generate a detailed 3-level feature tree in JSON format that captures the design features and assertions. Each node should be a key, and suggestions connecting specific assertions should be represented as edges in the JSON. The structure should be suitable for easy extraction and conversion into a graph.

# - **Level 0 (Design Description):** Provide an overview of the design, including its purpose, scope, and high-level behavior. This level sets the context for understanding the features and assertions.
#   - **Level 1 (Design Features):** Identify and list the main features of the design. There can be multiple features at this level, each representing a high-level functionality of the design.
#     - **Level 2 (Specific Assertions):** For each design feature at Level 1, provide specific assertions with natural language descriptions.
#     - **Suggestions (Edges):** Include any suggestions (rules/guidelines) connecting specific assertions as edges.

# **Design Module:** design_PH

# **Output format:** 
# - Present the feature tree in JSON format.
# - Include 'edges' as a separate JSON key at the top level, which captures the relationships (suggestions) between specific assertions.
# - The tree structure should be hierarchical and indented accordingly to reflect the design's features and assertions.

# **Example Output:**

# {
#   "nodes": [
#     {
#       "id": "Design_Description",
#       "label": "Design Description",
#       "description": "A Simple Light Switch that controls a light bulb."
#     },
#     {
#       "id": "Feature_1",
#       "label": "Feature 1: Toggle Light",
#       "description": "Allows turning the light on and off using a switch."
#     },
#     {
#       "id": "Assertion_1",
#       "label": "Assertion 1",
#       "description": "The light turns on when the switch is in the 'on' position."
#     },
#     {
#       "id": "Assertion_2",
#       "label": "Assertion 2",
#       "description": "The light turns off when the switch is in the 'off' position."
#     }
#   ],
#   "edges": [
#     {
#       "from": "Design_Description",
#       "to": "Feature_1",
#       "description": "The design feature that controls the light."
#     },
#     {
#       "from": "Feature_1",
#       "to": "Assertion_1",
#       "description": "The light behavior when the switch is in the 'on' position."
#     },
#     {
#       "from": "Feature_1",
#       "to": "Assertion_2",
#       "description": "The light behavior when the switch is in the 'off' position."
#     }
#   ]
# }
# """

# training_feature_tree_generation_wo_assumptions = """
# Please analyze the following design module with the golden testbench and generate a detailed 3-level feature tree in JSON format that captures the design features and assertions. The generated graph should align with the golden model to ensure consistency and correctness. Each node should be a key, and suggestions connecting specific assertions should be represented as edges in the JSON. Ensure that the leaf nodes are the assertions. The structure should be suitable for easy extraction and conversion into a graph for further use.

# Level 0 (Design Description): Provide an overview of the design, including its purpose, scope, and high-level behavior. This level sets the context for understanding the features and assertions.
# Level 1 (Design Features): Identify and list the main features of the design, ensuring they align with the functionalities specified in the golden testbench. There can be multiple features at this level, each representing a high-level functionality of the design.
# Level 2 (Specific Assertions): For each design feature at Level 1, provide specific assertions with natural language descriptions that correspond to the behaviors verified in the golden testbench. These assertions should be the leaf nodes of the tree.
# Suggestions (Edges): Include any suggestions (rules/guidelines) connecting specific assertions as edges, ensuring they reflect the relationships defined in the golden testbench.

# Design Module: design_PH

# Golden Testbench: testbench_PH

# Output format:
# 1. Present the feature tree in JSON format.
# 2. Include 'edges' as a separate JSON key at the top level, which captures the relationships (suggestions) between specific assertions.
# 3. Ensure the output graph aligns with the golden model and is formatted for easy use in further processing.
# 4. The tree structure should be hierarchical and indented accordingly to reflect the design's features and assertions.
# 5. Ensure that the leaf nodes are the assertions.

# Example Output:
# ```json
# {
#   "nodes": [
#     {
#       "id": "DesignDescription",
#       "type": "description",
#       "description": "An overview of the design's purpose and behavior."
#     },
#     {
#       "id": "Feature1",
#       "type": "feature",
#       "description": "This feature allows toggling the light on and off."
#     },
#     {
#       "id": "Feature2",
#       "type": "feature",
#       "description": "This feature adjusts the brightness of the light."
#     },
#     {
#       "id": "Assertion1",
#       "type": "assertion",
#       "description": "The light turns on when the switch is in the 'on' position."
#     },
#     {
#       "id": "Assertion2",
#       "type": "assertion",
#       "description": "The brightness increases when the up button is pressed."
#     }
#   ],
#   "edges": [
#     {
#       "from": "DesignDescription",
#       "to": "Feature1",
#       "reason": "Feature of toggling light derived from design purpose."
#     },
#     {
#       "from": "DesignDescription",
#       "to": "Feature2",
#       "reason": "Feature of adjusting brightness derived from design purpose."
#     },
#     {
#       "from": "Feature1",
#       "to": "Assertion1",
#       "reason": "Assertion verifying light toggling functionality."
#     },
#     {
#       "from": "Feature2",
#       "to": "Assertion2",
#       "reason": "Assertion verifying brightness adjustment functionality."
#     }
#   ]
# }
# ```
# """