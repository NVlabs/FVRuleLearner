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
Prompts used for NL2SVA-Machine
Subject to modifications and improvements
"""

# NL2SVA-Machine Question Prompts
SVAGEN_HEADER = """You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.
"""

SVAGEN_HEADER_TEXTGRAD = """You are a concise LLM that carefully considers all conditions provided in the input and explicitly addresses whether each condition is met in your response. Think step by step, ensuring that each condition is met in your response. If you cannot fulfill a condition, provide alternative explanations or insights related to the input.
"""

SVAGEN_HEADER_HYBRID = """You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.

Here are important SystemVerilog Assertion operators and their meanings:

**Temporal Implications:**
- |-> : Overlapping implication. The consequent is evaluated in the same clock cycle as the end of the antecedent sequence.
- |=> : Non-overlapping implication. The consequent is evaluated in the next clock cycle after the antecedent sequence completes.

**Temporal Qualifiers:**
- always : The property must hold at every current and future clock cycle.
- eventually / s_eventually : The property must hold at some current or future clock cycle.
- until / s_until : Property p holds until property q holds.
- until_with / s_until_with : Property p holds until and including when property q holds.
- next / nexttime : The property holds in the next clock cycle.

**Delay Operators:**
- ## : Delays evaluation by a specified number of clock cycles.
- ##N : Delays by exactly N clock cycles.
- ##[N:M] : Delays by N to M clock cycles.
- ##[N:$] : Delays by at least N clock cycles (unbounded).

**Repetition Operators:**
- [*N] : Consecutive repetition N times.
- [*N:M] : Consecutive repetition from N to M times.
- [->N] : Goto repetition - waits for N occurrences (not necessarily consecutive).
- [=N] : Non-consecutive repetition - N occurrences anywhere in the sequence.

**Property Control:**
- strong() : Strong sequence operator - the sequence must complete.
- weak() : Weak sequence operator - allows sequence to not complete.
- disable iff : Disables the assertion when the condition is true (typically used for reset).

**Comparison Operators:**
- === : Case equality (includes X and Z states).
- !== : Case inequality (includes X and Z states).
- == / != : Logical equality/inequality (X and Z are not handled properly).

**System Functions:**
- $onehot() : Checks if exactly one bit is high.
- $onehot0() : Checks if zero or one bit is high.
- $countones() : Counts the number of '1' bits.
- $rose() : Detects 0-to-1 transition.
- $fell() : Detects 1-to-0 transition.
- $stable() : Detects no change in value.
- $past() : References past value of a signal.
"""

SVAGEN_MACHINE_ICL_EXAMPLE_1 = """As an example:

Question: Create a SVA assertion that checks: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
```
"""


SVAGEN_MACHINE_ICL_EXAMPLE_2 = """As an example:

Question: Create a SVA assertion that checks: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
```

Question: Create a SVA assertion that checks: If sig_C contains at least one '1' bit or sig_D is not equal to sig_A, then sig_F must eventually be true
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (|sig_C || (sig_D !== sig_A )) |=> s_eventually(sig_F)
);
```
"""

SVAGEN_MACHINE_ICL_EXAMPLE_3 = """As an example:

Question: Create a SVA assertion that checks: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
```

Question: Create a SVA assertion that checks: If sig_C contains at least one '1' bit or sig_D is not equal to sig_A, then sig_F must eventually be true
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (|sig_C || (sig_D !== sig_A )) |=> s_eventually(sig_F)
);
```

Question: Create a SVA assertion that checks: Whenever the value of sig_J is less than the result of the XOR operation between sig_C and the negation of the bitwise negation of sig_H, and this result is equal to the result of the OR operation between the identity comparison of sig_A and the negation of sig_J and sig_B, the assertion is true
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
	((sig_J < (sig_B == (sig_C ^ ~|sig_H))) == ((|sig_A === !sig_J) || sig_B))
);
```
"""

SVAGEN_TB_PREAMBLE = """Now here is the testbench to perform your translation:\n"""

SVAGEN_IC_EX_PREAMBLE = """\n\nMore detailed examples of correct translations from description into an SVA assertion:"""


# NL2SVA-Machine Question Prompts
SVAGEN_QUESTION_PREAMBLE = "\nQuestion: Create a SVA assertion that checks: "
SVAGEN_QUESTION_POSTAMBLE = """
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:"""

SVAGEN_QUESTION_POSTAMBLE_ZERO_SHOT = """
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
assert property (@(posedge clk)
    (sig_A && sig_B) != 1'b1
);
```
Answer:"""

OPENCORE_SVAGEN_MODULE_PREAMBLE = "As an example, consider the following SystemVerilog module interface:"