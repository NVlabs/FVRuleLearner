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