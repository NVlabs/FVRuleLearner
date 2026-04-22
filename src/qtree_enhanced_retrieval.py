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
Enhanced Q-Tree Retrieval Module
Supports three modes:
1. 'prompt': Original retrieval based on prompt similarity
2. 'initial_sva_generation': Generate initial SVA, then rank with hybrid (LLM + operator) scoring and prune
3. 'rule_generation': Generate rules from Q-Trees instead of retrieval
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import FLAGS
from saver import saver
from utils_agent import llm_inference
from FVEval.fv_eval import prompts_nl2sva_human, prompts_nl2sva_machine

print = saver.log_info


class Rule:
    """Represents a rule generated from the qtree"""
    def __init__(self, id, text, node_id, node_question, reference_sva,
                 prompt, parent_chain=None, source_task_id=None, source_design=None, source_metrics=None):
        self.id = id
        self.text = text
        self.node_id = node_id
        self.node_question = node_question
        self.reference_sva = reference_sva
        self.prompt = prompt
        self.parent_chain = parent_chain or []
        self.source_task_id = source_task_id
        self.source_design = source_design
        self.source_metrics = source_metrics or {}


class EnhancedQTreeRetrieval:
    """Enhanced Q-Tree retrieval with multiple modes"""
    
    # SVA Operator Explanations - maps operator patterns to their semantic meanings
    SVA_OPERATOR_EXPLANATIONS = {
        # Temporal implications
        "|->": "Implication with overlapped timing. For every match of the sequence expression <left_expr> beginning at the start point, the evaluation of property expression <right_expr> beginning in the current clock cycle at the end point of the match succeeds and returns 1.",
        "|=>": "Implication with non-overlapped timing. For every match of the sequence expression <left_expr> beginning at the start point, the evaluation of property expression <right_expr> beginning in the next clock cycle at the end point of the match succeeds and returns 1.",
        
        # Temporal qualifiers
        "always": "Returns 1 if property expression p is 1 at every current and future clock cycle.",
        "eventually": "Returns 1 if there exists a current or future clock cycle at which property expression p is 1.",
        "next": "Returns 1 if the property expression p is 1 in the next clock cycle.",
        "until": "Returns 1 if property expression p holds until property expression q holds.",
        "s_until": "Returns 1 if property expression p holds at every cycle until property expression q holds.",
        "until_with": "Returns 1 if property expression p holds until and including when property expression q holds.",
        "s_until_with": "Returns 1 if property expression p holds at every cycle until and including when property expression q holds.",
        "implies": "Logical implication - if the left expression is true, then the right expression must be true. Equivalent to (!left_expr || right_expr).",
        
        # Delay operators
        "##": "The evaluation of sequence expression s is delayed by a specified number of clock cycles.",
        "##N": "The evaluation of sequence expression s is delayed by N clock cycles.",
        "##[N:M]": "The evaluation of sequence expression s is delayed by N to M clock cycles (range delay).",
        "##[N:$]": "The evaluation of sequence expression s is delayed by N or more clock cycles (unbounded delay).",
        
        # System functions - temporal sampling
        "$rose": "Returns 1 if the LSB of sequence expression s changed to 1. Otherwise, returns 0.",
        "$fell": "Returns 1 if the LSB of sequence expression s changed to 0. Otherwise, returns 0.",
        "$past": "Returns the value of sequence expression s in a N clock cycle step prior to the current one.",
        "$stable": "Returns 1 if the value of sequence expression s did not change. Otherwise, returns 0.",
        "$changed": "Returns 1 if the value of sequence expression s changed. Otherwise, returns 0.",
        "$sampled": "Returns the sampled value of the expression in the preponed region.",
        
        # System functions - encoding checks
        "$onehot": "Returns 1 if exactly one bit of sequence expression s is 1. Otherwise, returns 0.",
        "$onehot0": "Returns 1 if no more than one bit of sequence expression s is 1. Otherwise, returns 0.",
        "$isunknown": "Returns 1 if any bit of the expression is X or Z. Otherwise, returns 0.",
        "$countbits": "Returns the number of bits set to a specified value.",
        "$countones": "Returns the number of bits set to 1.",
        
        # Repetition operators
        "[*": "Consecutive repetition operator - matches the sequence a specified number of times consecutively.",
        "[+": "Consecutive repetition operator - matches the sequence one or more times consecutively.",
        "[*N]": "The sequence is repeated exactly N times consecutively.",
        "[*N:M]": "The sequence is repeated between N and M times consecutively.",
        "[*N:$]": "The sequence is repeated N or more times consecutively (unbounded).",
        "[->": "Goto repetition operator - the sequence must match N times, but not necessarily consecutively.",
        "[=": "Non-consecutive repetition operator - the sequence matches N times, ending at the Nth match.",
        "[->N]": "The sequence must match exactly N times (goto repetition), not necessarily consecutively.",
        "[->N:M]": "The sequence must match between N and M times (goto repetition).",
        "[=N]": "The sequence matches exactly N times, ending at the Nth match (non-consecutive).",
        "[=N:M]": "The sequence matches between N and M times, ending at the last match.",
        
        # Sequence combinators
        "throughout": "Boolean expression must hold throughout the entire sequence evaluation.",
        "within": "One sequence must complete within another sequence.",
        "intersect": "Both sequences must match over the same time interval.",
        "first_match": "Returns the first match of a sequence expression.",
        
        # Property control
        "strong": "Strong property - must eventually complete (no infinite waiting).",
        "weak": "Weak property - allowed to wait indefinitely without completion.",
    }
    
    # SVA Operators organized by semantic purpose
    SVA_OPERATORS = {
        # 1) Temporal semantics (what happens over time)
        "temporal": {
            "implication": ["|->", "|=>"],  # sequence->property, overlapped/non-overlapped
            "qualifiers": [                  # property/sequence temporal modalities
                "always", "s_always",
                "eventually", "s_eventually",
                "until", "s_until", "until_with", "s_until_with",
                "nexttime", "s_nexttime",
                "throughout", "within", "intersect",
                "first_match"
            ],
            "concatenation_and_delay": ["##", "##0", "##1", "##N", "##[N:M]", "##[N:$]"],  # Fixed, range, unbounded
        },

        # 2) Repetition (counting occurrences)
        "repetition": {
            "consecutive": [               # consecutive repetition [*n], [+n]
                "[*", "[+", "[*N]", "[*N:M]", "[*N:$]"
            ],
            "goto": [                      # goto repetition [->n], [=n]
                "[->", "[=", "[->N]", "[->N:M]", "[=N]", "[=N:M]"
            ]
        },

        # 3) Property-level control (strength, aborts, clock/reset guards)
        "property_control": {
            "strength_modifiers": ["strong", "weak"],
            "abort_operators": [
                "accept_on", "reject_on",
                "sync_accept_on", "sync_reject_on",
                "async_accept_on", "async_reject_on"
            ],
            "guards": [                    # no posedge/negedge variants here
                "@", "@@",                # single/multi-clock event specs
                "disable iff", "iff", "if", "case"
            ]
        },

        # 4) Boolean / arithmetic-like expressions (inside sequences/properties)
        "expressions": {
            "logical": ["&&", "||", "!", "and", "or", "not", "->", "<->"],
            "comparison": ["===", "!==", "==", "!=", "<", ">", "<=", ">=", "==?", "!=?"],
            "bitwise": ["&", "|", "~", "^", "~&", "~|", "^~", "~^", "<<", ">>", "<<<", ">>>"],
            "reduction": ["&", "~&", "|", "~|", "^", "~^", "^~"]
        },

        # 5) Assertion kinds & binding
        "assertion_forms": ["assert", "assume", "cover", "restrict", "expect"],
        "binding": ["bind"],

        # 6) System functions (grouped by purpose)
        "system_functions": {
            "encoding_checks": ["$onehot", "$onehot0", "$isunknown"],
            "counters": ["$countbits", "$countones"],
            "temporal_sampling": ["$past", "$rose", "$fell", "$stable", "$changed", "$sampled"],
            "control_and_reporting": [
                "$assertoff", "$asserton", "$assertkill", "$assertcontrol",
                "$assertpasson", "$assertpassoff", "$assertfailon", "$assertfailoff",
                "$assertnonvacuouson", "$assertnonvacuousoff",
                "$error", "$fatal", "$warning", "$info"
            ]
        },

        # 7) Sequence-only combinators (beyond plain delays)
        "sequence_combinators": ["throughout", "within", "intersect", "first_match"],

        # 8) Local variables / helpers
        "local_constructs": ["let"],

        # 9) Multiclock indicators (kept minimal; no edge qualifiers)
        "multiclock": ["@", "@@"],

        # 10) Non-standard or project-specific extensions (kept separate)
        "extensions_nonstandard": ["#-#", "#=#", "&&&", "|||"]
    }
    
    def __init__(self, qtree_data: List[Dict], csv_df=None):
        """
        Initialize enhanced retrieval.
        
        Args:
            qtree_data: List of Q-Tree entries
            csv_df: Optional pandas DataFrame with training examples
        """
        self.qtree_data = qtree_data
        self.csv_df = csv_df
        self.embedding_model = None
        
        # Initialize lookup structures (copied from QTreeRAGTester)
        self.qtree_entries: List[Dict[str, Any]] = []
        self.qtree_prompt_texts: List[str] = []
        self.qtree_task_ids: List[str] = []
        self.qtree_prompt_embeddings: Optional[Dict[str, Any]] = None
        self.qtree_sva_texts: List[str] = []
        self.qtree_sva_embeddings: Optional[Dict[str, Any]] = None
        self.rules_index: Dict[str, Dict[str, List[Rule]]] = {}
        self.parent_chains_index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        # Prepare Q-Tree index
        self._prepare_qtree_index()
        
        # Load training examples if available
        self.training_examples = []
        if csv_df is not None and FLAGS.use_RAG and 'Examples' in FLAGS.RAG_content:
            self._load_training_examples()
        
        # Collect training rules
        self.all_training_rules = self.collect_all_training_rules()
        
    def _get_embedding_model(self):
        """Get or initialize embedding model"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"Initialized SentenceTransformer for similarity calculation")
        return self.embedding_model
    
    def get_operator_explanations(self, operators: List[str], normalize_keys: bool = True) -> Dict[str, str]:
        """
        Get explanations for a list of operators.
        
        Args:
            operators: List of operator strings
            normalize_keys: If True, normalize operator keys (##1 -> ##N, etc.) in the output
            
        Returns:
            Dictionary mapping operators (possibly normalized) to their explanations
        """
        explanations = {}
        
        for op in operators:
            # Normalize the operator key if requested (e.g., ##1 -> ##N)
            op_key = self.normalize_operator(op) if normalize_keys else op
            
            # Skip if we already have this normalized operator
            if op_key in explanations:
                continue
            
            # Normalize operator for case-insensitive lookup
            normalized_op = op.lower()
            
            # Try exact match first
            if op in self.SVA_OPERATOR_EXPLANATIONS:
                explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS[op]
            # Try case-insensitive match
            elif any(k.lower() == normalized_op for k in self.SVA_OPERATOR_EXPLANATIONS):
                matching_key = [k for k in self.SVA_OPERATOR_EXPLANATIONS if k.lower() == normalized_op][0]
                explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS[matching_key]
            # Try to match normalized patterns (e.g., ##2 -> ##N)
            else:
                # Check if it's a delay pattern
                if re.match(r'##\d+', op):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("##N", "Delay operator")
                elif re.match(r'##\[\d+:\d+\]', op):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("##[N:M]", "Range delay operator")
                elif re.match(r'##\[\d+:\$\]', op):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("##[N:$]", "Unbounded delay operator")
                # Check if it's a repetition pattern
                elif re.match(r'\[\*\d+\]', op):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("[*N]", "Consecutive repetition operator")
                elif re.match(r'\[\*\d+:\d+\]', op):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("[*N:M]", "Range consecutive repetition")
                elif re.match(r'\[\*\d+:\$\]', op):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("[*N:$]", "Unbounded consecutive repetition")
                elif re.match(r'\[->\d+\]', op) or op.startswith('[->'):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("[->N]", "Goto repetition operator")
                elif re.match(r'\[=\d+\]', op) or op.startswith('[='):
                    explanations[op_key] = self.SVA_OPERATOR_EXPLANATIONS.get("[=N]", "Non-consecutive repetition operator")
        
        return explanations
    
    def extract_operators_from_sva(self, sva_code: str) -> List[str]:
        """
        Extract temporal operators from SVA code.
        
        Returns a list of operators found in the SVA code.
        """
        if not sva_code:
            return []
        
        # Define common SystemVerilog temporal operators
        temporal_operators = [
            # Delay operators
            r'##\d+',  # Fixed delay (##1, ##2, etc.)
            r'##\[\d+:\d+\]',  # Range delay (##[1:3])
            r'##\[\d+:\$\]',  # Unbounded delay (##[1:$])
            
            # Sequence operators
            r'\|->',  # Overlapping implication
            r'\|=>',  # Non-overlapping implication
            r'#-#',  # Throughout
            r'#=#',  # Throughout with non-overlapping
            
            # Property operators
            r'\balways\b',
            r'\beventually\b',
            r'\bnext\b',
            r'\buntil\b',
            r'\bs_until\b',
            r'\buntil_with\b',
            r'\bs_until_with\b',
            r'\bimplies\b',
            r'\bstrong\b',
            r'\bweak\b',
            
            # Repetition operators
            r'\[\*\d+\]',  # Fixed repetition [*3]
            r'\[\*\d+:\d+\]',  # Range repetition [*1:3]
            r'\[\*\d+:\$\]',  # Unbounded repetition [*1:$]
            r'\[->',  # Goto repetition
            r'\[=',  # Non-consecutive repetition

            r'\blet\b',
            
            # Boolean operators in property context
            # r'\band\b',
            # r'\bor\b',
            # r'\bnot\b',
            
            # Other property operators
            r'\$rose',
            r'\$fell',
            r'\$stable',
            r'\$changed',
            r'\$past',
            r'\$sampled',
            
            # Sequence methods
            r'\.first_match',
            r'\.throughout',
            r'\.within',
            r'\.intersect',

            r'\b$onehot\b',
            r'\b$onehot0\b',
            
            # Edge operators
            # r'\bposedge\b',
            # r'\bnegedge\b',
            # r'\bedge\b'
        ]
        
        operators_found = []
        for op_pattern in temporal_operators:
            matches = re.findall(op_pattern, sva_code, re.IGNORECASE)
            operators_found.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_operators = []
        for op in operators_found:
            if op not in seen:
                seen.add(op)
                unique_operators.append(op)
        
        return unique_operators
    
    def _flatten_operators(self, ops_dict, parent_key=''):
        """Flatten nested operator dictionary into list with category paths."""
        items = []
        for key, value in ops_dict.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(self._flatten_operators(value, new_key))
            elif isinstance(value, list):
                for op in value:
                    items.append((op, new_key))
            else:
                items.append((value, new_key))
        
        return items
    
    def _get_operator_category(self, operator: str) -> List[str]:
        """
        Find which category/categories an operator belongs to.
        Returns list of category paths like ["temporal.implication"]
        """
        flat_operators = self._flatten_operators(self.SVA_OPERATORS)
        # Case-insensitive operator matching
        categories = [cat_path for op, cat_path in flat_operators if op.lower() == operator.lower()]
        return categories
    
    def _are_operators_similar(self, op1: str, op2: str) -> Tuple[bool, float, str]:
        """
        Check if two operators are similar with semantic scoring.
        
        Similarity scoring:
        - Exact match: 1.0
        - Same subcategory (e.g., both temporal.implication): 0.7
        - Same main category (e.g., both temporal.*): 0.5
        - Different categories: 0.0
        
        Returns: (is_similar, score, reason)
        """
        # Case-insensitive comparison for exact match
        if op1.lower() == op2.lower():
            return (True, 1.0, "exact_match")
        
        categories1 = self._get_operator_category(op1)
        categories2 = self._get_operator_category(op2)
        
        if not categories1 or not categories2:
            return (False, 0.0, "unknown_operator")
        
        # Check for same subcategory
        for cat1 in categories1:
            if cat1 in categories2:
                return (True, 0.7, f"same_subcategory:{cat1}")
        
        # Check for same main category
        main_cats1 = set(cat.split('.')[0] for cat in categories1)
        main_cats2 = set(cat.split('.')[0] for cat in categories2)
        
        shared_main = main_cats1 & main_cats2
        if shared_main:
            return (True, 0.5, f"same_main_category:{list(shared_main)[0]}")
        
        return (False, 0.0, "different_category")
    
    def normalize_operator(self, op: str) -> str:
        """
        Normalize an operator by replacing specific numeric values with generic patterns.
        
        Examples:
            ##1, ##2 -> ##N
            ##[1:3], ##[2:5] -> ##[N:M]
            [*3], [*5] -> [*N]
            [->2], [->5] -> [->N]
        
        Args:
            op: Operator string to normalize
            
        Returns:
            Normalized operator string
        """
        # Delay operators
        op = re.sub(r'##\d+', '##N', op)
        op = re.sub(r'##\[\d+:\d+\]', '##[N:M]', op)
        op = re.sub(r'##\[\d+:\$\]', '##[N:$]', op)
        
        # Consecutive repetition operators
        op = re.sub(r'\[\*\d+\]', '[*N]', op)
        op = re.sub(r'\[\*\d+:\d+\]', '[*N:M]', op)
        op = re.sub(r'\[\*\d+:\$\]', '[*N:$]', op)
        
        # Goto repetition operators
        op = re.sub(r'\[->\d+\]', '[->N]', op)
        op = re.sub(r'\[->\d+:\d+\]', '[->N:M]', op)
        
        # Non-consecutive repetition operators
        op = re.sub(r'\[=\d+\]', '[=N]', op)
        op = re.sub(r'\[=\d+:\d+\]', '[=N:M]', op)
        
        # Don't convert to lowercase - preserve case for display (##N not ##n)
        return op
    
    def normalize_operators_in_text(self, text: str) -> str:
        """
        Normalize all operators found in a text string.
        
        Args:
            text: Text containing operators
            
        Returns:
            Text with normalized operators
        """
        if not text:
            return text
        
        # Apply all operator normalization patterns
        normalized = text
        
        # Delay operators (order matters - more specific patterns first)
        normalized = re.sub(r'##\[\d+:\$\]', '##[N:$]', normalized)
        normalized = re.sub(r'##\[\d+:\d+\]', '##[N:M]', normalized)
        normalized = re.sub(r'##\d+', '##N', normalized)
        
        # Consecutive repetition operators
        normalized = re.sub(r'\[\*\d+:\$\]', '[*N:$]', normalized)
        normalized = re.sub(r'\[\*\d+:\d+\]', '[*N:M]', normalized)
        normalized = re.sub(r'\[\*\d+\]', '[*N]', normalized)
        
        # Goto repetition operators
        normalized = re.sub(r'\[->\d+:\d+\]', '[->N:M]', normalized)
        normalized = re.sub(r'\[->\d+\]', '[->N]', normalized)
        
        # Non-consecutive repetition operators
        normalized = re.sub(r'\[=\d+:\d+\]', '[=N:M]', normalized)
        normalized = re.sub(r'\[=\d+\]', '[=N]', normalized)
        
        return normalized
    
    def calculate_operator_alignment_score(self, operators1: List[str], operators2: List[str]) -> float:
        """
        Calculate alignment score between two sets of operators.
        
        Returns a score between 0 and 1 indicating how well the operators align.
        """
        if not operators1 and not operators2:
            return 0
        if not operators1 or not operators2:
            return 0
        
        # Normalize operators for comparison using the class method
        normalized1 = [self.normalize_operator(op) for op in operators1]
        normalized2 = [self.normalize_operator(op) for op in operators2]
        
        # Remove duplicates while preserving order
        unique1 = []
        seen1 = set()
        for op in normalized1:
            if op not in seen1:
                unique1.append(op)
                seen1.add(op)
        
        unique2 = []
        seen2 = set()
        for op in normalized2:
            if op not in seen2:
                unique2.append(op)
                seen2.add(op)
        
        # Calculate FUZZY JACCARD SCORE with semantic operator similarity
        # 
        # Traditional Jaccard: |intersection| / |union|
        # Fuzzy Jaccard: fuzzy_intersection / |union|
        #
        # Where fuzzy_intersection accounts for partial similarity:
        # - For each operator, find its best match and add that similarity score
        # - Use bipartite matching to avoid double-counting
        
        all_ops_set = set(unique1) | set(unique2)
        union_size = len(all_ops_set)
        
        if union_size == 0:
            return 0.0
        
        # Build similarity matrix between all operators
        # For efficiency, we only compute similarities for the union of operators
        all_ops = list(all_ops_set)
        op_to_idx = {op: idx for idx, op in enumerate(all_ops)}
        
        # Track which operators from unique1 and unique2
        ops1_indices = [op_to_idx[op] for op in unique1]
        ops2_indices = [op_to_idx[op] for op in unique2]
        
        # Calculate fuzzy intersection using greedy bipartite matching
        # This ensures each operator is matched at most once
        fuzzy_intersection = 0.0
        matched_from_2 = set()
        
        # Sort ops1 by their best match score (descending) for greedy matching
        ops1_with_scores = []
        for op1 in unique1:
            best_score = 0.0
            best_op2 = None
            for op2 in unique2:
                is_similar, score, reason = self._are_operators_similar(op1, op2)
                if score > best_score:
                    best_score = score
                    best_op2 = op2
            ops1_with_scores.append((op1, best_op2, best_score))
        
        # Sort by score descending to match high-scoring pairs first
        ops1_with_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy matching: assign each op1 to its best available op2
        for op1, best_op2, score in ops1_with_scores:
            if best_op2 and best_op2 not in matched_from_2:
                fuzzy_intersection += score
                matched_from_2.add(best_op2)
            elif best_op2 is None:
                # No match found - this op1 has no similar op2
                fuzzy_intersection += 0.0
        
        # Fuzzy Jaccard Score = fuzzy_intersection / union_size
        # This is guaranteed to be in [0, 1]:
        # - fuzzy_intersection is at most min(len(unique1), len(unique2)) * 1.0
        # - union_size is at least max(len(unique1), len(unique2))
        # - So ratio is at most 1.0
        fuzzy_jaccard = fuzzy_intersection / union_size
        
        # High-quality match bonus for focused rules
        # Problem: Rules focusing on 1-2 operators get low scores due to union normalization
        # Solution: Give bonus for high-quality matches (score >= 0.7)
        #
        # Example:
        #   Rule ops: ["|=>"]  (focused on implication issue)
        #   Target ops: ["|->", "##", "==="]
        #   Match: |=> vs |-> = 0.7 (same subcategory)
        #   Base Jaccard: 0.7 / 3 = 0.23 (too low!)
        #   With bonus: 0.23 + 0.3 * (0.7 / 1) = 0.44 (better!)
        
        high_quality_matches = 0
        high_quality_scores_sum = 0.0
        
        for op1, best_op2, score in ops1_with_scores:
            if score >= 0.7:  # Exact match (1.0) or same subcategory (0.7)
                high_quality_matches += 1
                high_quality_scores_sum += score
        
        # Bonus calculation: reward precision when rules are focused
        precision_bonus = 0.0
        if len(unique1) > 0 and high_quality_matches > 0:
            # Precision: how many of trace's operators are high-quality matches
            precision = high_quality_matches / len(unique1)
            # Average quality of those matches
            avg_quality = high_quality_scores_sum / high_quality_matches
            # Bonus: up to 0.3 for perfect precision with perfect matches
            precision_bonus = 0.3 * precision * avg_quality
        
        # Final score: base Jaccard + precision bonus
        # Still guaranteed in [0, 1]: fuzzy_jaccard ∈ [0,1], precision_bonus ∈ [0, 0.3]
        # Max possible: 1.0 (perfect match) + 0.0 (no bonus needed) = 1.0
        # Focused rule: 0.23 + 0.3 = 0.53 (much better!)
        final_score = min(1.0, fuzzy_jaccard + precision_bonus)
        
        return final_score

    def extract_rules_from_trace_text(self, trace_text: str) -> str:
        """
        Extract only the rule text from a trace_text string.
        
        Args:
            trace_text: Full trace text containing Q&A and rules
            
        Returns:
            String containing only the rules (lines starting with "=> Rule")
        """
        if not trace_text:
            return ""
        
        rule_lines = []
        for line in trace_text.split('\n'):
            # Extract lines that contain rules
            if '=> Rule' in line:
                # Extract the rule content after "=> Rule N:"
                if ':' in line:
                    rule_content = line.split(':', 1)[1].strip()
                    rule_lines.append(rule_content)
                else:
                    rule_lines.append(line.strip())
        
        return '\n'.join(rule_lines)
    
    def _score_traces_by_operators_only(self, traces, initial_sva):
        """
        Score traces by operators only, extracting operators from rules instead of full trace text.
        """
        target_ops = self.extract_operators_from_sva(initial_sva)
        print(f"Target operators from initial SVA: {target_ops}")
        for t in traces:
            trace_text = t.get('trace_text', '') or ''
            rules_only = self.extract_rules_from_trace_text(trace_text)
            
            # print(f"Rules extracted: {rules_only}")  # Show first 200 chars
            
            # Extract operators from rules only (not the entire trace)
            trace_ops = self.extract_operators_from_sva(rules_only)
            # print(f"Trace operators from rules: {trace_ops}")
            # breakpoint()
            
            score = self.calculate_operator_alignment_score(target_ops, trace_ops)
            t['trace_operators'] = trace_ops
            t['operator_alignment_score'] = score

    def generate_initial_sva(self, prompt: str, testbench: str, examples: List[Dict] = None) -> str:
        """
        Generate an initial SVA for a test case.
        
        Args:
            prompt: Natural language description
            testbench: Testbench code
            examples: Optional list of example dicts with 'prompt' and 'reference_sva'
            
        Returns:
            Generated SVA string
        """
        print(f"\n=== Generating Initial SVA ===")
        
        # Select appropriate prompts based on task
        if FLAGS.task == "nl2sva_machine" or FLAGS.task == "nl2sva_opencore":
            prompts_module = prompts_nl2sva_machine
            print(f"Using nl2sva_machine prompts")
        else:  # nl2sva_human or others
            prompts_module = prompts_nl2sva_human
            print(f"Using nl2sva_human prompts")
        
        # Build RAG prompt with examples
        rag_prompt = ""
        if FLAGS.num_icl == 1:
            if hasattr(prompts_module, 'SVAGEN_HUMAN_ICL_EXAMPLE_1'):
                rag_prompt += prompts_module.SVAGEN_HUMAN_ICL_EXAMPLE_1
            elif hasattr(prompts_module, 'SVAGEN_MACHINE_ICL_EXAMPLE_1'):
                rag_prompt += prompts_module.SVAGEN_MACHINE_ICL_EXAMPLE_1
        elif FLAGS.num_icl == 3:
            if hasattr(prompts_module, 'SVAGEN_HUMAN_ICL_EXAMPLE_3'):
                rag_prompt += prompts_module.SVAGEN_HUMAN_ICL_EXAMPLE_3
            elif hasattr(prompts_module, 'SVAGEN_MACHINE_ICL_EXAMPLE_3'):
                rag_prompt += prompts_module.SVAGEN_MACHINE_ICL_EXAMPLE_3
        
        # Add testbench
        if rag_prompt:
            rag_prompt += "\n\n"
        rag_prompt += prompts_module.SVAGEN_TB_PREAMBLE
        rag_prompt += "\n" + testbench
        
        # Add examples if provided
        if examples:
            rag_prompt += "\n\nAdditional context from similar documents:\n"
            for ex in examples:
                formatted_example = f'Question: Create a SVA assertion that checks: {ex["prompt"]}\nReference Answer: {ex["reference_sva"]}'
                rag_prompt += f"\n{formatted_example}\n"
        
        # Add the question
        rag_prompt += "\n"
        rag_prompt += prompts_module.SVAGEN_QUESTION_PREAMBLE
        rag_prompt += prompt + "\n"
        rag_prompt += prompts_module.SVAGEN_QUESTION_POSTAMBLE
        
        # Generate SVA using LLM
        try:
            response = llm_inference(
                system_prompt=prompts_module.SVAGEN_HEADER,
                user_prompt=rag_prompt
            )
            
            # Extract SVA from response
            generated_sva = self._extract_sva_from_response(response)
            
            return generated_sva
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    
    def _extract_sva_from_response(self, response: str) -> str:
        """Extract SVA code from LLM response"""
        # Look for code blocks
        sva_match = re.search(r'```(?:systemverilog|sv|verilog)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sva_match:
            return sva_match.group(1).strip()
        
        # Look for assertion patterns
        assert_match = re.search(r'(assert\s+property.*?;)', response, re.DOTALL | re.IGNORECASE)
        if assert_match:
            return assert_match.group(1).strip()
        
        # Return the whole response if no pattern found
        return response.strip()
    
    def rank_traces_hybrid(self, traces: List[Dict], initial_sva: str, 
                           llm_weight: float = None, operator_weight: float = None) -> List[Dict]:
        """
        Rank traces using hybrid scoring (LLM + operator alignment).
        
        Args:
            traces: List of trace dicts with 'llm_score' and other metadata
            initial_sva: Initial SVA to extract operators from
            llm_weight: Weight for LLM score (default from FLAGS)
            operator_weight: Weight for operator alignment (default from FLAGS)
            
        Returns:
            Sorted list of traces with composite scores
        """
        print(f"\n=== Hybrid Ranking (LLM + Operator Alignment) ===")
        
        # Get weights from FLAGS if not provided
        if llm_weight is None:
            llm_weight = getattr(FLAGS, 'llm_score_weight', 0.7)
        if operator_weight is None:
            operator_weight = getattr(FLAGS, 'operator_score_weight', 0.3)
        
        print(f"LLM weight: {llm_weight}, Operator weight: {operator_weight}")
        
        # Extract operators from initial SVA
        target_operators = self.extract_operators_from_sva(initial_sva)
        print(f"Target operators from initial SVA: {target_operators}")
        
        # Calculate operator alignment scores for each trace
        for trace in traces:
            # Check if operator score was already calculated (e.g., by _score_traces_by_operators_only)
            # If so, use it instead of recalculating
            if 'operator_alignment_score' not in trace:
                # Extract operators from the actual suggestion text (trace_text)
                # This is what the user will see, so it's more meaningful
                trace_text = trace.get('trace_text', '')
                
                # Extract operators ONLY from what the trace actually says (trace_text)
                # DO NOT use reference_sva - that's from a different problem!
                # The trace should be scored on what it TEACHES, not what the source SVA contained
                trace_operators = self.extract_operators_from_sva(trace_text)
                
                # If trace_text doesn't mention operators, the trace doesn't teach about operators
                # Don't fallback to reference_sva - that would reward irrelevant operator matches
                if not trace_operators:
                    # Trace doesn't mention specific operators, so it's not teaching operator-specific lessons
                    # Give a neutral/low score instead of checking unrelated reference SVA
                    trace_operators = []  # Keep empty to indicate no operator guidance
                
                operator_score = self.calculate_operator_alignment_score(target_operators, trace_operators)
                
                trace['operator_alignment_score'] = operator_score
                trace['trace_operators'] = trace_operators
            else:
                # Use pre-calculated operator score (e.g., from rules-only extraction)
                operator_score = trace['operator_alignment_score']
            
            # Calculate composite score
            llm_score = trace.get('llm_score', 0)
            composite_score = (llm_weight * llm_score) + (operator_weight * operator_score)
            trace['composite_score'] = composite_score
        
        # Sort by composite score
        sorted_traces = sorted(traces, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        print(f"\nTop 5 traces after hybrid ranking:")
        for i, trace in enumerate(sorted_traces[:5], 1):
            print(f"  {i}. trace_id={trace.get('trace_id', 'N/A')[:50]}")
            print(f"     LLM score: {trace.get('llm_score', 0):.3f}, "
                  f"Operator score: {trace.get('operator_alignment_score', 0):.3f}, "
                  f"Composite: {trace.get('composite_score', 0):.3f}")
        
        return sorted_traces
    
    def prune_by_operators(self, traces: List[Dict], initial_sva: str, 
                          threshold: float = None) -> List[Dict]:
        """
        Prune traces based on operator alignment with initial SVA.
        
        Args:
            traces: List of trace dicts
            initial_sva: Initial SVA to extract operators from
            threshold: Minimum operator alignment score (default from FLAGS)
            
        Returns:
            Filtered list of traces
        """
        print(f"\n=== Operator-Based Pruning ===")
        
        if threshold is None:
            threshold = getattr(FLAGS, 'operator_pruning_threshold', 0.0)
        
        print(f"Pruning threshold: {threshold}")
        
        # Extract operators from initial SVA
        target_operators = self.extract_operators_from_sva(initial_sva)
        print(f"Target operators: {target_operators}")
        
        pruned_traces = []
        for trace in traces:
            operator_score = trace.get('operator_alignment_score', 0)
            
            if operator_score > threshold:
                pruned_traces.append(trace)
        
        print(f"Kept {len(pruned_traces)}/{len(traces)} traces after pruning")
        
        return pruned_traces
    
    def _prepare_qtree_index(self):
        """Pre-compute lookup structures for Q-Trees (copied from QTreeRAGTester)"""
        self.qtree_entries = []
        self.qtree_prompt_texts = []
        self.qtree_task_ids = []
        self.qtree_prompt_embeddings = None
        self.qtree_sva_texts = []
        self.qtree_sva_embeddings = None

        for entry in self.qtree_data:
            if FLAGS.task == "nl2sva_opencore":
                # For opencore, use design_name as task_id (replace '/' for consistency)
                task_id = entry.get('design_name', '').replace('/', '_')
            else:
                task_id = entry.get('task_id')
            qtree = entry.get('qtree_data', {})
            if not task_id or not qtree:
                continue

            prompt = entry.get('prompt') or qtree.get('prompt') or ''
            design_name = entry.get('design_name', '')
            nodes = qtree.get('nodes', [])
            root_nodes = self._get_root_nodes(nodes)
            reference_sva = (
                entry.get('reference_sva')
                or entry.get('generated_sva')
                or entry.get('ref_solution')
                or qtree.get('reference_sva')
            )

            # Pre-compute parent chains for all nodes in this qtree
            if task_id not in self.parent_chains_index:
                self.parent_chains_index[task_id] = {}
            
            for node in nodes:
                node_id = node.get('id')
                if node_id:
                    parent_chain = self.extract_tree_path(nodes, node_id)
                    self.parent_chains_index[task_id][node_id] = parent_chain

            self.qtree_entries.append({
                'task_id': task_id,
                'design_name': design_name,
                'prompt': prompt,
                'qtree': qtree,
                'root_nodes': root_nodes,
                'reference_sva': reference_sva,
                'entry': entry,
            })
            self.qtree_prompt_texts.append(str(prompt))
            self.qtree_task_ids.append(task_id)
            self.qtree_sva_texts.append(reference_sva or '')
        
        print(f"Prepared Q-Tree index: {len(self.qtree_entries)} Q-Trees available in database")
    
    def _get_root_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return root-level nodes (parent is None or empty)"""
        roots = []
        for node in nodes or []:
            parent_id = node.get('parent_id')
            if parent_id in (None, '', 'root'):
                roots.append({
                    'id': node.get('id'),
                    'question': node.get('question', '').strip(),
                    'level': node.get('level')
                })
        return roots
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for comparison"""
        return (question or '').strip().lower()
    
    def _load_training_examples(self):
        """Load training examples from CSV dataset"""
        # Check if debug mode is enabled and training_cases is specified
        if FLAGS.debug and hasattr(FLAGS, 'training_cases') and FLAGS.training_cases:
            print(f"DEBUG MODE: Using custom training_cases: {FLAGS.training_cases}")
            training_IDs = FLAGS.training_cases
        elif FLAGS.task == "nl2sva_human":
            # Hardcoded training IDs for nl2sva_human task
            training_IDs = [73, 28, 65, 36, 16, 59, 42, 47, 32, 45, 60, 4, 34, 20, 37, 61, 69, 1, 46, 54, 43, 72, 48, 0, 56, 7, 51, 41, 67, 24, 17, 35, 29, 25, 57, 63, 40, 23, 75, 12, 9, 11, 13, 49, 53, 19, 70, 21, 26, 6, 33, 76, 10, 15, 68, 14, 64, 55, 44, 50, 22, 77, 58, 18]
        elif FLAGS.task == "nl2sva_machine":
            training_IDs = [7, 243, 8, 136, 227, 130, 190, 148, 58, 146, 120, 277, 84, 27, 226, 57, 77, 127, 22, 169, 137, 156, 110, 143, 20, 18, 295, 261, 252, 88, 119, 108, 239, 93, 282, 257, 238, 162, 45, 241, 271, 265, 80, 297, 142, 268, 118, 28, 10, 213, 1, 51, 212, 244, 71, 133, 197, 4, 170, 101, 231, 0, 34, 275, 187, 6, 181, 48, 249, 66, 147, 68, 19, 105, 175, 274, 106, 290, 294, 199, 184, 298, 115, 160, 109, 234, 154, 283, 17, 222, 31, 23, 54, 35, 167, 285, 40, 248, 50, 161, 200, 215, 288, 113, 9, 139, 122, 164, 168, 246, 38, 230, 26, 99, 11, 53, 195, 245, 264, 49, 258, 29, 240, 3, 116, 260, 279, 145, 186, 224, 55, 254, 286, 79, 291, 152, 12, 251, 30, 178, 287, 70, 269, 250, 42, 262, 211, 173, 14, 37, 65, 149, 126, 205, 131, 196, 223, 242, 180, 210, 216, 61, 255, 111, 166, 76, 25, 208, 15, 194, 102, 123, 266, 218, 138, 87, 174, 135, 86, 151, 176, 299, 182, 129, 237, 256, 281, 183, 267, 155, 292, 247, 193, 270, 159, 64, 198, 229, 121, 13, 85, 98, 83, 75, 202, 2, 92, 63, 82, 192, 141, 236, 206, 107, 209, 189, 177, 96, 72, 91, 284, 158, 117, 171, 104, 24, 134, 233, 41, 62, 272, 56, 259, 221, 179, 201, 89, 232, 235, 74]
        elif FLAGS.task == "nl2sva_opencore":
            training_IDs = [885, 748, 832, 349, 134, 770, 642, 462, 715, 278, 8, 989, 857, 763, 189, 478, 654, 357, 499, 6, 882, 517, 629, 577, 227, 631, 338, 644, 186, 247, 515, 347, 965, 284, 373, 761, 445, 634, 475, 693, 18, 529, 309, 163, 181, 806, 894, 140, 430, 967, 732, 387, 799, 891, 764, 290, 439, 377, 960, 956, 60, 500, 191, 362, 751, 875, 289, 456, 393, 671, 354, 547, 954, 572, 218, 847, 374, 15, 460, 680, 153, 248, 736, 145, 117, 403, 397, 560, 508, 417, 239, 241, 203, 75, 172, 869, 297, 591, 743, 226, 473, 80, 52, 135, 942, 724, 541, 414, 673, 408, 729, 809, 595, 897, 302, 497, 868, 176, 432, 431, 695, 603, 25, 138, 527, 107, 204, 267, 993, 726, 814, 943, 999, 183, 304, 319, 556, 70, 523, 130, 210, 716, 336, 190, 787, 17, 917, 995, 801, 888, 668, 434, 803, 848, 479, 179, 834, 575, 187, 502, 467, 559, 315, 938, 265, 395, 136, 194, 649, 114, 873, 58, 866, 910, 900, 392, 657, 881, 10, 29, 261, 643, 696, 292, 285, 879, 398, 909, 549, 514, 449, 613, 794, 268, 940, 587, 583, 755, 535, 565, 386, 588, 73, 984, 985, 628, 701, 733, 484, 56, 205, 851, 886, 586, 146, 627, 105, 72, 930, 64, 270, 959, 325, 250, 229, 406, 101, 415, 589, 625, 367, 420, 438, 4, 490, 740, 0, 539, 660, 552, 538, 708, 382, 711, 404, 126, 429, 507, 970, 828, 618, 608, 169, 928, 506, 855, 207, 975, 363, 919, 564, 597, 802, 624, 323, 825, 876, 314, 528, 728, 520, 840, 93, 159, 796, 842, 481, 918, 21, 451, 585, 350, 283, 320, 394, 962, 142, 573, 892, 476, 769, 607, 141, 544, 421, 551, 714, 947, 950, 368, 606, 41, 482, 137, 493, 774, 878, 682, 206, 119, 501, 971, 383, 739, 294, 719, 616, 352, 487, 279, 46, 416, 973, 762, 494, 286, 19, 983, 251, 78, 570, 812, 99, 511, 252, 418, 578, 883, 348, 933, 924, 816, 332, 372, 407, 295, 826, 264, 42, 364, 76, 858, 510, 271, 635, 345, 477, 22, 413, 244, 530, 219, 379, 678, 173, 626, 758, 95, 425, 652, 180, 600, 246, 582, 28, 853, 782, 401, 409, 686, 222, 259, 272, 601, 944, 446, 199, 690, 390, 435, 62, 509, 746, 7, 650, 835, 905, 43, 856, 936, 531, 996, 16, 805, 850, 156, 274, 321, 647, 322, 697, 83, 433, 620, 157, 611, 798, 160, 687, 333, 196, 448, 312, 313, 672, 896, 122, 328, 375, 703, 849, 11, 474, 238, 705, 40, 747, 109, 795, 845, 619, 30, 262, 602, 760, 702, 504, 968, 829, 522, 990, 489, 920, 646, 480, 707, 91, 561, 115, 518, 230, 14, 35, 234, 356, 232, 288, 483, 118, 330, 496, 661, 951, 388, 691, 366, 174, 931, 797, 741, 813, 880, 580, 977, 780, 718, 300, 90, 108, 492, 158, 937, 20, 426, 452, 47, 71, 89, 359, 440, 469, 992, 777, 228, 630, 592, 710, 128, 308, 895, 113, 516, 738, 598, 887, 139, 648, 468, 640, 74, 884, 562, 590, 615, 824, 217, 200, 457, 391, 389, 994, 941, 964, 327, 720, 13, 907, 3, 713, 162, 147, 953, 360, 632, 709, 756, 85, 188, 745, 34, 454, 689, 331, 361, 948, 88, 23, 961, 838, 830, 659, 305, 864, 81, 669, 9, 957, 237, 301, 571, 932, 38, 311, 921, 2, 277, 839, 987, 233, 195, 97, 727, 266, 836, 275, 612, 79, 422, 213, 148, 427, 808, 593, 621, 198, 536, 997, 870, 906, 461, 821, 436, 553, 37, 901, 750, 704, 68, 257, 125, 92, 656, 216, 143, 827, 96, 161, 254, 202, 48, 773, 485, 817, 50, 949, 455, 717, 39, 558, 488, 658, 675, 916, 155, 730, 694, 410, 106, 742, 399, 44, 214, 865, 737, 121, 862, 841, 935, 486, 116, 911, 12, 981, 341, 365, 581, 744, 766, 823, 666, 221, 636, 537, 317, 224, 609, 800, 51, 735, 123, 712, 889, 281, 771, 55, 168, 334, 599, 692, 59, 150, 260, 596, 505, 820, 525, 785, 447, 913, 723, 843, 867, 245, 258, 444, 667, 307, 103, 833, 63, 779, 411, 495, 371, 874, 555, 351, 698, 540, 346, 604, 706, 296, 731, 988, 793, 633, 734, 253, 623, 53, 775, 567, 638, 129, 396, 923, 946, 818, 458, 242, 681, 26, 861, 171, 197, 166, 579, 871, 792, 151, 405, 617, 5, 184, 127, 653, 664, 877, 164, 384, 837, 811, 922, 282, 568, 472, 412, 215, 854, 419, 641, 378, 355, 914, 193, 144, 182, 208, 786, 859, 316, 235, 343, 209, 969, 663, 955, 674, 49, 269, 466, 754, 82, 124, 545, 112, 819, 519, 443, 358, 749, 402, 722, 178, 789, 974, 465, 470, 149]

        
        if self.csv_df is None:
            print(f"No CSV dataframe available for loading training examples")
            return
        
        self.training_examples = []
        for csv_row_id in training_IDs:
            if csv_row_id >= len(self.csv_df):
                continue
            row = self.csv_df.iloc[csv_row_id]
            prompt = str(row['prompt'] if 'prompt' in row else '').strip()
            reference_sva = row.get('ref_solution', None)
            if reference_sva and isinstance(reference_sva, str) and reference_sva.strip():
                self.training_examples.append({
                    'csv_row_id': csv_row_id,
                    'prompt': prompt,
                    'reference_sva': reference_sva
                })
        print(f"Loaded {len(self.training_examples)} training examples")
    
    def collect_all_training_rules(self) -> List[Rule]:
        """Collect all rules from successful training Q-Tree entries"""
        all_rules = []
        successful_entries = 0
        
        # Get training cases filter (for debug mode)
        training_case_ids = None
        if FLAGS.debug and hasattr(FLAGS, 'training_cases') and FLAGS.training_cases:
            training_case_ids = set(FLAGS.training_cases)
            print(f"DEBUG MODE: Filtering Q-Tree rules to only use training_cases: {sorted(training_case_ids)}")
        
        # Create CSV row mapping if we have csv_df
        csv_row_mapping = {}
        only_test_ids = getattr(FLAGS, 'only_test_ids', [])
        
        if self.csv_df is not None:
            for idx, row in self.csv_df.iterrows():
                design_name = row['design_name'] if 'design_name' in row else ''
                prompt = str(row['prompt'] if 'prompt' in row else '').strip()
                key = (design_name, prompt)
                csv_row_mapping[key] = idx
        
        # Step 1: Collect successful training entries
        successful_training_entries = []
        
        for entry_idx, entry in enumerate(self.qtree_data):
            task_id = entry.get('task_id', '')
            design_name = entry.get('design_name', '')
            prompt = entry.get('prompt', '').strip()
            reference_sva = entry.get('ref_solution', '')
            
            # Find CSV row index for this entry
            key = (design_name, prompt)
            csv_row = csv_row_mapping.get(key, -1)
            
            # Check if this entry was successful (pec=1.0 or bleu=1.0)
            final_metrics = entry.get('final_metrics', {})
            pec = final_metrics.get('pec', 0)
            bleu = final_metrics.get('bleu', 0)
            
            if pec != 1.0 and bleu != 1.0:
                continue
            
            # Only use training entries (not test entries) for rules
            is_training = csv_row not in only_test_ids
            if not is_training:
                continue
                
            successful_entries += 1
            successful_training_entries.append(entry)
            
            # # Extract rules from qtree nodes
            # qtree_data = entry.get('qtree_data', {})
            # nodes = qtree_data.get('nodes', [])
            
            # for node in nodes:
            #     if node.get('rules_generated'):
            #         for rule_text in node['rules_generated']:
            #             parent_chain = self.extract_tree_path(nodes, node['id'])
            #             reference_sva = entry.get('generated_sva') or entry.get('reference_sva')
            #             if not reference_sva:
            #                 reference_sva = entry.get('ref_solution')

            #             rule = Rule(
            #                 id=f"{task_id}_{node['id']}_{len(all_rules)}",
            #                 text=rule_text,
            #                 node_id=node['id'],
            #                 node_question=node.get('question', ''),
            #                 reference_sva=reference_sva,
            #                 prompt=prompt,
            #                 parent_chain=parent_chain,
            #                 source_task_id=task_id,
            #                 source_design=design_name,
            #                 source_metrics=final_metrics,
            #             )
            #             all_rules.append(rule)
                        
            #             # Store rule in the index for efficient lookup
            #             if task_id not in self.rules_index:
            #                 self.rules_index[task_id] = {}
            #             if node['id'] not in self.rules_index[task_id]:
            #                 self.rules_index[task_id][node['id']] = []
            #             self.rules_index[task_id][node['id']].append(rule)
        
        # Step 2: Apply filtering/subsampling strategy
        # PRIORITY 1: If training_case_ids is specified, use ONLY those IDs (no subsampling)
        # PRIORITY 2: Otherwise, apply subsample ratio if specified
        
        if training_case_ids is not None:
            # Filter by specific training case IDs
            filtered_entries = []
            for entry in successful_training_entries:
                design_name = entry.get('design_name', '')
                prompt = entry.get('prompt', '').strip()
                key = (design_name, prompt)
                csv_row = csv_row_mapping.get(key, -1)
                
                if csv_row in training_case_ids:
                    filtered_entries.append(entry)
            
            total_qtrees = len(self.qtree_data)
            print(f"\n=== Q-Tree Filtering by Training IDs ===")
            print(f"Total Q-Trees in database: {total_qtrees}")
            print(f"Successful training Q-Trees: {len(successful_training_entries)}")
            print(f"Filtered to training_cases: {sorted(training_case_ids)}")
            print(f"Q-Trees after filtering: {len(filtered_entries)}")
            print(f"Reduction: {total_qtrees} → {len(filtered_entries)} Q-Trees")
            print("=" * 50)
            
            # BREAKPOINT: Inspect filtered Q-Trees
            # Variables to check:
            #   - training_case_ids: set of training case IDs to filter by
            #   - filtered_entries: list of Q-Tree entries after filtering
            #   - csv_row_mapping: mapping from (design_name, prompt) to CSV row ID
            # breakpoint()
            
            entries_to_process = filtered_entries
            
        else:
            # Apply subsampling ratio if specified
            subsample_ratio = getattr(FLAGS, 'qtree_subsample_ratio', None)
            
            if subsample_ratio is not None and 0 < subsample_ratio < 1.0:
                original_count = len(successful_training_entries)
                # Adjust ratio relative to original training set (0.8)
                adjusted_ratio = subsample_ratio / 0.8
                target_count = int(original_count * adjusted_ratio)
                target_count = max(1, target_count)  # At least keep 1
                
                # Use fixed seed for sampling
                seed = getattr(FLAGS, 'seed', 42)
                np.random.seed(seed)
                indices = np.random.choice(original_count, size=target_count, replace=False)
                indices = sorted(indices)  # Keep order
                
                sampled_entries = [successful_training_entries[i] for i in indices]
                
                # Calculate effective ratio
                original_train_ratio = getattr(FLAGS, 'original_train_ratio', 0.8)
                effective_ratio = subsample_ratio / original_train_ratio
                
                print(f"\n=== Q-Tree Subsampling by Ratio ===")
                print(f"Successful training Q-Trees: {original_count}")
                print(f"Sampled Q-Trees: {len(sampled_entries)}")
                print(f"Subsample ratio: {subsample_ratio}")
                print(f"Effective ratio: {effective_ratio:.3f} (relative to original dataset)")
                print(f"Random seed: {seed}")
                print("=" * 50)
                
                entries_to_process = sampled_entries
            else:
                print(f"\n=== Q-Tree Selection: Using All Training Data ===")
                print(f"Using all {len(successful_training_entries)} successful training Q-Trees")
                print("=" * 50)
                entries_to_process = successful_training_entries

        # Step 3: Extract rules from sampled entries
        for entry in entries_to_process:
            task_id = entry.get('task_id', '')
            design_name = entry.get('design_name', '')
            prompt = entry.get('prompt', '').strip()
            
            final_metrics = entry.get('final_metrics', {})
            
            # Extract rules from qtree nodes
            qtree_data = entry.get('qtree_data', {})
            nodes = qtree_data.get('nodes', [])
            
            for node in nodes:
                if node.get('rules_generated'):
                    for rule_text in node['rules_generated']:
                        parent_chain = self.extract_tree_path(nodes, node['id'])
                        reference_sva = entry.get('generated_sva') or entry.get('reference_sva')
                        if not reference_sva:
                            reference_sva = entry.get('ref_solution')

                        rule = Rule(
                            id=f"{task_id}_{node['id']}_{len(all_rules)}",
                            text=rule_text,
                            node_id=node['id'],
                            node_question=node.get('question', ''),
                            reference_sva=reference_sva,
                            prompt=prompt,
                            parent_chain=parent_chain,
                            source_task_id=task_id,
                            source_design=design_name,
                            source_metrics=final_metrics,
                        )
                        all_rules.append(rule)
                        
                        # Store rule in the index for efficient lookup
                        if task_id not in self.rules_index:
                            self.rules_index[task_id] = {}
                        if node['id'] not in self.rules_index[task_id]:
                            self.rules_index[task_id][node['id']] = []
                        self.rules_index[task_id][node['id']].append(rule)

        print(f"Collected {len(all_rules)} total rules from {len(entries_to_process)} processed training entries")
        return all_rules
    
    def extract_tree_path(self, nodes: List[Dict[str, Any]], target_node_id: str) -> List[Dict[str, Any]]:
        """Extract the path from root to target node in the Q-tree"""
        node_map = {node.get('id'): node for node in nodes if node.get('id')}
        
        # Build parent relationships
        parent_map = {}
        for node in nodes:
            for child_id in node.get('children', []):
                parent_map[child_id] = node.get('id')
        
        # Build path from target to root
        path = []
        current_id = target_node_id
        while current_id and current_id in node_map:
            path.append(node_map[current_id])
            current_id = parent_map.get(current_id)
        
        # Reverse to get root-to-target path
        return list(reversed(path))
    
    def find_similar_qtrees(self, target_prompt: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k Q-Trees most similar to the target prompt"""
        if not target_prompt:
            return []

        if top_k is None:
            top_k = getattr(FLAGS, 'qtree_similarity_top_k', 5)

        if not self.qtree_entries:
            return []

        embedding_model = self._get_embedding_model()
        if self.qtree_prompt_embeddings is None:
            self.qtree_prompt_embeddings = {}
            if self.qtree_prompt_texts:
                embeddings = embedding_model.encode(self.qtree_prompt_texts, convert_to_tensor=False, show_progress_bar=False)
                for task_id, embedding in zip(self.qtree_task_ids, embeddings):
                    self.qtree_prompt_embeddings[task_id] = embedding

        target_embedding = embedding_model.encode([target_prompt], convert_to_tensor=False, show_progress_bar=False)[0]
        candidates: List[Dict[str, Any]] = []

        for entry in self.qtree_entries:
            task_id = entry['task_id']
            embedding = self.qtree_prompt_embeddings.get(task_id)
            if embedding is None:
                continue
            score = cosine_similarity([target_embedding], [embedding])[0][0]
            candidates.append({
                'task_id': task_id,
                'design_name': entry.get('design_name', ''),
                'prompt': entry.get('prompt', ''),
                'qtree': entry.get('qtree', {}),
                'root_nodes': entry.get('root_nodes', []),
                'score': float(score),
                'entry': entry,
                'has_parent_chains': task_id in self.parent_chains_index,
            })

        candidates.sort(key=lambda item: item['score'], reverse=True)
        top_candidates = candidates[:top_k]

        print(f"Retrieved {len(top_candidates)} candidate Q-Trees from {len(candidates)} total (top_k={top_k})")
        for idx, candidate in enumerate(top_candidates, 1):
            print(f"  Candidate {idx}: task_id={candidate['task_id']} design={candidate['design_name']} score={candidate['score']:.3f}")

        return top_candidates
    
    def find_similar_prompts(self, target_prompt: str, target_design: str, top_k: int = 2) -> List[Dict]:
        """Find most similar examples from training data using embedding-based similarity"""
        if not self.training_examples:
            return []
        
        # Initialize embedding model if not already done
        if not hasattr(self, 'example_embeddings'):
            embedding_model = self._get_embedding_model()
            self.example_embeddings = {}
            prompts = []
            csv_ids = []
            for example in self.training_examples:
                prompts.append(example['prompt'])
                csv_ids.append(example['csv_row_id'])
            
            if prompts:
                embeddings = embedding_model.encode(prompts, convert_to_tensor=False, show_progress_bar=False)
                for csv_id, embedding in zip(csv_ids, embeddings):
                    self.example_embeddings[csv_id] = embedding
                print(f"Pre-computed embeddings for {len(self.example_embeddings)} training examples")
        
        # Encode the target prompt
        embedding_model = self._get_embedding_model()
        target_embedding = embedding_model.encode([target_prompt], convert_to_tensor=False, show_progress_bar=False)[0]
        
        similarities = []
        for example in self.training_examples:
            csv_id = example['csv_row_id']
            if csv_id in self.example_embeddings:
                example_embedding = self.example_embeddings[csv_id]
                cosine_sim = cosine_similarity([target_embedding], [example_embedding])[0][0]
                similarities.append((cosine_sim, csv_id, example))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top k examples
        results = []
        for score, csv_id, example in similarities[:top_k]:
            results.append({
                'prompt': example['prompt'],
                'reference_sva': example['reference_sva']
            })
            
        return results
    
    def group_qtrees_by_root(self, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group candidate Q-Trees by normalized root question"""
        grouped: Dict[str, Dict[str, Any]] = {}

        for candidate in candidates:
            for root in candidate.get('root_nodes', []) or [{'id': None, 'question': ''}]:
                normalized = self._normalize_question(root.get('question', '')) or f"__root_{candidate['task_id']}"
                group = grouped.setdefault(normalized, {
                    'root_question': root.get('question', ''),
                    'members': []
                })
                group['members'].append({
                    'task_id': candidate['task_id'],
                    'design_name': candidate['design_name'],
                    'score': candidate['score'],
                    'root_question': root.get('question', ''),
                    'root_node_id': root.get('id'),
                    'qtree': candidate['qtree'],
                })

        print(f"Grouped candidates into {len(grouped)} root-question clusters")
        return grouped
    
    def _collect_traces_using_parent_chains(self, nodes: List[Dict[str, Any]], start_node_id: Optional[str], task_id: str) -> List[List[Dict[str, Any]]]:
        """Collect traces using pre-computed parent chains for efficiency"""
        if not nodes or not start_node_id or task_id not in self.parent_chains_index:
            return []
        
        node_map = {node.get('id'): node for node in nodes if node.get('id')}
        if start_node_id not in node_map:
            return []
        
        traces: List[List[Dict[str, Any]]] = []
        
        # Find all leaf nodes that descend from start_node_id
        def find_leaf_descendants(node_id: str, visited: set) -> List[str]:
            if node_id in visited:
                return []
            visited.add(node_id)
            
            node = node_map.get(node_id)
            if not node:
                return []
            
            children = node.get('children') or []
            if not children:
                return [node_id]
            
            leaves = []
            for child_id in children:
                leaves.extend(find_leaf_descendants(child_id, visited))
            return leaves
        
        leaf_nodes = find_leaf_descendants(start_node_id, set())
        
        # For each leaf node, use its pre-computed parent chain
        for leaf_id in leaf_nodes:
            if leaf_id in self.parent_chains_index[task_id]:
                parent_chain = self.parent_chains_index[task_id][leaf_id]
                # Filter to only include nodes from start_node_id to leaf
                filtered_path = []
                found_start = False
                for node in parent_chain:
                    if node.get('id') == start_node_id:
                        found_start = True
                    if found_start:
                        filtered_path.append(node)
                
                if filtered_path:
                    traces.append(filtered_path)
        
        if traces:
            print(f"  Generated {len(traces)} traces from node {start_node_id} (using pre-computed chains)")
        return traces
    
    def _format_trace(self, path: List[Dict[str, Any]], path_rules: List[Rule] = None, 
                      include_answers: bool = False) -> str:
        """
        Format a trace path as a readable string with normalized operators
        
        Args:
            path: List of nodes in the trace path
            path_rules: Associated Rule objects
            include_answers: Whether to include node answers (default: False, only questions)
        """
        lines = []
        
        # Get configuration for showing answers and normalization
        if include_answers is None:
            include_answers = getattr(FLAGS, 'show_qtree_answers_in_suggestions', False)
        
        normalize_in_display = getattr(FLAGS, 'normalize_operators_in_display', True)
        
        # Format hierarchical reasoning path
        for depth, node in enumerate(path):
            level = node.get('level', 'unknown')
            question = node.get('question', '')
            
            # Normalize operators in question if enabled
            if normalize_in_display and question:
                question = self.normalize_operators_in_text(question)
            
            # Add visual hierarchy based on level
            indent = ""
            if level == 'exploratory':
                indent = ""
                marker = "[E]"
            elif level == 'specific_analysis':
                indent = "  "
                marker = "[S]"
            elif level == 'rule_generation':
                indent = "    "
                marker = "[R]"
            else:
                marker = "[?]"
            
            lines.append(f"{indent}{marker} Q{depth}: {question}")
            
            # Only show answers if explicitly requested
            if include_answers:
                answer = node.get('answer', '')
                if answer:
                    # Normalize operators in answer if enabled
                    if normalize_in_display:
                        answer = self.normalize_operators_in_text(answer)
                    
                    lines.append(f"{indent}  >> Answer:")
                    for line in answer.split('\n'):
                        if line.strip():
                            lines.append(f"{indent}     {line}")
            
            # Format rules - ONLY for rule_generation level ([R])
            # [E] and [S] nodes don't generate rules, only [R] nodes do
            if level == 'rule_generation':
                rules = node.get('rules_generated') or []
                for i, rule in enumerate(rules, 1):
                    # Normalize operators in rules if enabled
                    rule_text = rule
                    if normalize_in_display and rule_text:
                        rule_text = self.normalize_operators_in_text(rule_text)
                    lines.append(f"{indent}  => Rule {i}: {rule_text}")
        
        # Add Rule metadata if available (compact format)
        if path_rules:
            lines.append("\n[Metadata]")
            unique_designs = set(rule.source_design for rule in path_rules)
            unique_tasks = set(rule.source_task_id for rule in path_rules)
            avg_pec = sum(rule.source_metrics.get('pec', 0) for rule in path_rules) / len(path_rules) if path_rules else 0
            lines.append(f"  {len(path_rules)} rules from {len(unique_tasks)} tasks, avg_PEC={avg_pec:.3f}")
        
        return '\n'.join(lines)
    
    def collect_branch_traces(self, group: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect root-to-leaf traces for all members in a grouped root cluster"""
        traces: List[Dict[str, Any]] = []
        members = group.get('members', [])

        for member_idx, member in enumerate(members, 1):
            qtree = member.get('qtree', {})
            nodes = qtree.get('nodes', [])
            root_node_id = member.get('root_node_id')
            task_id = member['task_id']
            
            # Use pre-computed parent chains if available
            if task_id in self.parent_chains_index:
                paths = self._collect_traces_using_parent_chains(nodes, root_node_id, task_id)
            else:
                paths = []

            for path_idx, path in enumerate(paths, 1):
                safe_design = member['design_name'].replace('/', '_').replace(' ', '_').replace(':', '_')
                
                if path:
                    leaf_node = path[-1]
                    leaf_node_id = leaf_node.get('id', f'path_{path_idx}')
                    trace_id = f"{safe_design}::{member['task_id']}::{leaf_node_id}"
                else:
                    trace_id = f"{safe_design}::{member['task_id']}::empty_path_{path_idx}"
                
                # Collect rules associated with nodes in this path
                path_rules = []
                if task_id in self.rules_index:
                    for node in path:
                        node_id = node.get('id')
                        if node_id and node_id in self.rules_index[task_id]:
                            path_rules.extend(self.rules_index[task_id][node_id])
                
                trace_text = self._format_trace(path, path_rules)
                traces.append({
                    'trace_id': trace_id,
                    'task_id': member['task_id'],
                    'design_name': member['design_name'],
                    'root_question': member.get('root_question', group.get('root_question', '')),
                    'score': member['score'],
                    'path': path,
                    'trace_text': trace_text,
                    'rules': path_rules,
                    'num_rules': len(path_rules),
                })

        print(f"Collected {len(traces)} traces for root question '{group.get('root_question', '')}'")
        return traces
    
    def rank_traces_with_llm(self, target_prompt: str, traces: List[Dict[str, Any]], testbench: str = '', max_traces: int = None) -> Dict[str, Any]:
        """Rank traces using LLM and return structured response"""
        if not traces:
            return {'raw_response': '', 'parsed': None}

        if max_traces is not None and len(traces) > max_traces:
            truncated_traces = traces[:max_traces]
            print(f"WARNING: Truncating from {len(traces)} to {max_traces} traces for LLM ranking")
        else:
            truncated_traces = traces
            print(f"Batch ranking all {len(truncated_traces)} traces")
        
        # Check if we should instruct LLM to skip misaligned traces
        instruct_skip_misaligned = getattr(FLAGS, 'instruct_skip_misaligned_traces', False)
            
        lines = [
            "You are an expert verification engineer evaluating Q-Tree reasoning traces for SystemVerilog assertion generation.",
            "",
            "Your task is to rank these traces based on their potential effectiveness for the new test case.",
            "",
            "Ranking criteria (in order of importance):",
            "1. **Historical Effectiveness**: If available, prioritize traces that have successfully fixed similar issues in the past",
            "2. **Relevance**: How well the trace's reasoning applies to the target prompt and testbench context",
            "3. **Rule Quality**: Concreteness, specificity, and actionability of the generated rules",
            "4. **Analysis Depth**: Quality of the exploratory → specific → rule generation reasoning chain",
            "",
            "Output: JSON array with objects containing:",
            "{\"trace_id\": string, \"rank\": integer (1-based), \"score\": float (0.000-1.000, 3 decimal places), \"rationale\": string}",
            "",
            "Ensure scores are meaningful and well-distributed (avoid clustering all scores near 0.5)."
        ]

        lines.append(f"\nTarget prompt: {target_prompt}")
        if testbench:
            lines.append(f"Testbench: {testbench}")

        lines.append("\nCandidate traces:")
        for idx, trace in enumerate(truncated_traces, 1):
            snippet = trace['trace_text']
            
            # Add rule performance info
            rule_info = ""
            if 'rules' in trace and trace['rules']:
                avg_pec = sum(r.source_metrics.get('pec', 0) for r in trace['rules']) / len(trace['rules'])
                avg_bleu = sum(r.source_metrics.get('bleu', 0) for r in trace['rules']) / len(trace['rules'])
                rule_info = f", rules_count={trace.get('num_rules', 0)}, avg_pec={avg_pec:.2f}, avg_bleu={avg_bleu:.2f}"
            
            lines.append(
                f"Trace {idx} (trace_id={trace['trace_id']}, task_id={trace['task_id']}, design={trace['design_name']}, initial_similarity={trace['score']:.4f}{rule_info}):\n{snippet}"
            )

        user_prompt = '\n\n'.join(lines)

        try:
            response = llm_inference(
                system_prompt="You evaluate strategies for SystemVerilog assertion fixes.",
                user_prompt=user_prompt
            )
            print(f"response: {response}")
            
        except Exception as exc:
            print(f"LLM ranking failed: {exc}")
            return {'raw_response': str(exc), 'parsed': None}

        parsed = None
        try:
            # Extract JSON from markdown code block if present
            response_to_parse = response.strip()
            
            # Check if response is wrapped in markdown code block
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_to_parse, re.DOTALL)
            if json_match:
                # Extract the JSON content from inside the code block
                response_to_parse = json_match.group(1).strip()
                print(f"Extracted JSON from markdown code block")
            
            # Now parse the cleaned JSON
            parsed = json.loads(response_to_parse)
            print(f"Successfully parsed {len(parsed) if isinstance(parsed, list) else 'N/A'} ranking items")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"WARNING: Failed to parse LLM ranking response as JSON: {e}")
            print(f"Response preview: {response[:200]}...")
            print(f"Attempting line-by-line fallback parsing...")
            
            # Fallback: Try to parse line by line
            parsed = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to extract trace_id, rank, score, and rationale from various formats
                # Format 1: {"trace_id": "xxx", "rank": N, "score": 0.xxx, "rationale": "..."}
                # Format 2: Trace N (trace_id=xxx): score 0.xxx - rationale
                # Format 3: N. trace_id: score - rationale
                # Format 4: rank N, trace_id, score 0.xxx, rationale
                
                item = {}
                
                # Try to find trace_id
                trace_id_match = re.search(r'trace_id[=:\s]+([a-zA-Z0-9_-]+)', line, re.IGNORECASE)
                if trace_id_match:
                    item['trace_id'] = trace_id_match.group(1)
                
                # Try to find rank
                rank_match = re.search(r'rank[=:\s]+(\d+)', line, re.IGNORECASE)
                if rank_match:
                    item['rank'] = int(rank_match.group(1))
                
                # Try to find score (float between 0 and 1)
                score_match = re.search(r'score[=:\s]+(0?\.\d+|1\.0+|0|1)', line, re.IGNORECASE)
                if score_match:
                    item['score'] = float(score_match.group(1))
                
                # Try to find rationale (text after "rationale:", "-", or ":")
                rationale_match = re.search(r'(?:rationale[:\s]+|[-–])\s*(.+)$', line, re.IGNORECASE)
                if rationale_match:
                    item['rationale'] = rationale_match.group(1).strip()
                
                # Only add if we found at least trace_id
                if 'trace_id' in item:
                    parsed.append(item)
            
            if parsed:
                print(f"Successfully parsed {len(parsed)} items using line-by-line fallback")
            else:
                print(f"Line-by-line fallback also failed to extract any valid items")
                parsed = None

        return {
            'raw_response': response,
            'parsed': parsed
        }
    
    def pickup_qtrees_for_test_case(self, test_id: int, test_row: Dict[str, Any], top_n: int = 3, initial_sva: str = None, skip_ranking: bool = True) -> Dict[str, Any]:
        """Main method to pick Q-Trees for a test case and return top suggestions"""
        design_name = test_row.get('design_name', '')
        prompt = str(test_row.get('prompt', '')).strip()
        testbench = test_row.get('testbench') or test_row.get('tb') or ''

        print(f"=" * 80)
        print(f"Evaluating Q-Trees for test_id={test_id} design='{design_name}'")
        print(f"Prompt: {prompt}")

        # Get retrieval mode from FLAGS
        # retrieval_mode = getattr(FLAGS, 'qtree_retrieval_mode', 'prompt')
        
        if FLAGS.qtree_ranking_mode == 'prompt':
            candidates = self.find_similar_qtrees(prompt)
        elif FLAGS.qtree_ranking_mode == 'sva':
            # For SVA mode, generate initial SVA first
            print(f"\nGenerating initial SVA for SVA-based retrieval...")
            examples = self.find_similar_prompts(prompt, design_name)
            if initial_sva:
                # Find similar Q-Trees based on SVA (not implemented yet, fallback to prompt)
                print(f"SVA-based Q-Tree matching not fully implemented, using prompt-based matching")
                candidates = self.find_similar_qtrees(prompt)
            else:
                candidates = self.find_similar_qtrees(prompt)
        else:
            candidates = self.find_similar_qtrees(prompt)
        
        # breakpoint()
            
        grouped = self.group_qtrees_by_root(candidates)

        # Collect ALL traces from all groups first (before LLM ranking)
        all_collected_traces = []
        group_results = []
        
        print(f"\n" + "=" * 60)
        print(f"Collecting traces from all groups...")
        print(f"=" * 60)

        # breakpoint()
        for group_key, group_data in grouped.items():
            # breakpoint()
            traces = self.collect_branch_traces(group_data)
            total_rules = sum(len(trace.get('rules', [])) for trace in traces)
            
            group_summary = {
                'group_key': group_key,
                'root_question': group_data.get('root_question', ''),
                'num_members': len(group_data.get('members', [])),
                'num_traces': len(traces),
                'num_rules': total_rules,
                'traces': traces
            }
            group_results.append(group_summary)
            
            # Add all traces to the global list
            all_collected_traces.extend(traces)
            
            print(f"Group '{group_summary['root_question'][:50]}...': {len(traces)} traces, {total_rules} rules")
        
        # breakpoint()
        print(f"\nTotal collected: {len(all_collected_traces)} traces from {len(group_results)} groups")
        
        # Now rank ALL traces together in ONE LLM call
        print(f"\n" + "=" * 60)
        print(f"Ranking ALL {len(all_collected_traces)} traces in single LLM call")
        print(f"=" * 60)
        
        if skip_ranking:
            all_ranked_traces = all_collected_traces
            return {
                'test_id': test_id,
                'design_name': design_name,
                'prompt': prompt,
                'testbench': testbench,
                'candidates': candidates,
                'groups': group_results,
                'total_rules_used': sum(g.get('num_rules', 0) for g in group_results),
                'top_suggestions': [],
                'all_ranked_traces': all_ranked_traces,
                'top_rule_strings': [],
                'rule_strings_only': [],
            }

        else:
            ranking = self.rank_traces_with_llm(prompt, all_collected_traces, testbench=testbench)
        
            parsed_rankings = ranking.get('parsed')
            all_ranked_traces = []
            
            if isinstance(parsed_rankings, list):
                parsed_rankings = sorted(parsed_rankings, key=lambda item: item.get('rank', 9999))
                
                print(f"\nLLM ranked {len(parsed_rankings)} traces:")
                for idx, item in enumerate(parsed_rankings[:10], 1):  # Show top 10
                    print(f"  {idx}. trace_id={item.get('trace_id', 'N/A')[:60]} score={item.get('score', 0):.3f}")
                
                # Build ranked traces with LLM scores
                trace_lookup = {trace['trace_id']: trace for trace in all_collected_traces}
                for ranked_item in parsed_rankings:
                    trace_id = ranked_item.get('trace_id')
                    if trace_id in trace_lookup:
                        trace_with_score = trace_lookup[trace_id].copy()
                        trace_with_score['llm_score'] = ranked_item.get('score', 0)
                        trace_with_score['llm_rank'] = ranked_item.get('rank', 9999)
                        trace_with_score['llm_rationale'] = ranked_item.get('rationale', '')
                        all_ranked_traces.append(trace_with_score)
            else:
                print(f"WARNING: LLM ranking not available or not in JSON format.")
                # Fallback: use all traces without LLM scores
                all_ranked_traces = all_collected_traces

            # Sort all traces by LLM score
            all_ranked_traces.sort(key=lambda x: x.get('llm_score', 0), reverse=True)
            
            # Extract top N suggestions globally
            top_n_suggestions = []
            top_n_rule_strings = []
            print(f"\n" + "=" * 60)
            print(f"GLOBAL TOP {top_n} SUGGESTIONS:")
            print(f"=" * 60)
            
            for idx, trace in enumerate(all_ranked_traces[:top_n], 1):
                print(f"\nGlobal Rank {idx}:")
                print(f"  Trace ID: {trace['trace_id']}")
                print(f"  Design: {trace['design_name']}")
                print(f"  LLM Score: {trace.get('llm_score', 0):.3f}")
                
                # Extract suggestion text from the trace
                suggestion_lines = []
                suggestion_lines.append(f"=== Suggestion {idx} (from {trace['design_name']}, score: {trace.get('llm_score', 0):.3f}) ===")
                suggestion_lines.append(trace['trace_text'])
                
                # Add rules from this trace
                if trace.get('rules'):
                    suggestion_lines.append("\nAssociated Rules:")
                    trace_rule_strings = []
                    for rule in trace['rules'][:top_n]:
                        rule_text = rule.text
                        suggestion_lines.append(f"- {rule_text}")
                        trace_rule_strings.append(rule_text)
                    
                    # Add this trace's rules to the global list
                    for rule_text in trace_rule_strings:
                        top_n_rule_strings.append({
                            'rule_text': rule_text,
                            'source_trace_id': trace['trace_id'],
                            'source_design': trace['design_name'],
                            'source_task_id': trace['task_id'],
                            'llm_score': trace['llm_score'],
                        })
                
                suggestion_text = '\n'.join(suggestion_lines)
                top_n_suggestions.append(suggestion_text)

            rule_strings_only = [r['rule_text'] for r in top_n_rule_strings]
            
            total_traces = sum(group['num_traces'] for group in group_results)
            total_rules_used = sum(group.get('num_rules', 0) for group in group_results)
            print(f"\n" + "=" * 60)
            print(f"Summary: {len(candidates)} candidate Q-Trees → {len(group_results)} groups → {total_traces} traces → {total_rules_used} rules leveraged")
            print(f"Global ranking: Selected top {len(top_n_suggestions)} suggestions from {len(all_ranked_traces)} total ranked traces")

            return {
                'test_id': test_id,
                'design_name': design_name,
                'prompt': prompt,
                'testbench': testbench,
                'candidates': candidates,
                'groups': group_results,
                'total_rules_used': total_rules_used,
                'top_suggestions': top_n_suggestions,
                'all_ranked_traces': all_ranked_traces,
                'top_rule_strings': top_n_rule_strings,
                'rule_strings_only': rule_strings_only,
            }
    
    def _generalize_signals(self, text: str, testbench: str = None) -> str:
        """
        Replace specific signal names and operator values with generic placeholders to make rules more reusable.
        
        Args:
            text: Text containing specific signal names and operators
            testbench: Optional testbench to extract signal patterns from
            
        Returns:
            Text with generalized signal references and normalized operators
        """
        if not text:
            return text
        
        # First, normalize operators (##1 -> ##N, [*3] -> [*N], etc.)
        generalized = self.normalize_operators_in_text(text)
        
        # Then, generalize signal name patterns
        # Pattern: word_word, tb_word, word_d1, word_vld, etc.
        signal_patterns = [
            (r'\b[a-z_]+_req\b', '<request_signal>'),
            (r'\b[a-z_]+_ack\b', '<acknowledge_signal>'),
            (r'\b[a-z_]+_vld\b', '<valid_signal>'),
            (r'\b[a-z_]+_rdy\b', '<ready_signal>'),
            (r'\b[a-z_]+_en\b', '<enable_signal>'),
            (r'\b[a-z_]+_rst\b', '<reset_signal>'),
            (r'\btb_[a-z_]+\b', '<testbench_signal>'),
            (r'\b[a-z_]+_d\d+\b', '<delayed_signal>'),
            (r'\b[a-z_]+_value\b', '<value_signal>'),
            (r'\b[a-z_]+_cnt\b', '<counter_signal>'),
            (r'\b[a-z_]+_count\b', '<counter_signal>'),
            (r'\bclk\b', '<clock>'),
            (r'\breset\b', '<reset>'),
            (r'sig_(\w+)', '<signal>'),
        ]
        
        for pattern, placeholder in signal_patterns:
            generalized = re.sub(pattern, placeholder, generalized, flags=re.IGNORECASE)
        
        return generalized
    
    def generate_rules_from_best_traces(self, best_traces: List[Dict],
                                       target_prompt: str, initial_sva: str,
                                       testbench: str = None, design_name: str = None,
                                       top_k: int = 5, include_answers: bool = False) -> List[str]:
        """
        Generate rules from the best-ranked traces.
        
        Args:
            best_traces: Top-ranked trace dicts after ranking and pruning
            target_prompt: Target natural language prompt
            initial_sva: Initial SVA to reference
            testbench: Testbench code for context
            design_name: Design module name
            top_k: Number of rules to generate
            include_answers: Whether to include node answers
            
        Returns:
            List of generated rule strings
        """
        # print(f"\n=== Generating Rules from {len(best_traces)} Best Traces ===")
        
        # Check if signal generalization is enabled
        generalize_signals = getattr(FLAGS, 'generalize_signals_in_rules', True)
        
        # Collect content from best traces
        trace_contents = []
        for trace in best_traces:
            path = trace.get('path', [])
            score = trace.get('composite_score', trace.get('llm_score', 0))
            
            content = {
                'trace_id': trace['trace_id'],
                'score': score,
                'questions': [],
                'rules': []
            }
            
            # Extract questions and existing rules from each node
            # breakpoint()
            for node in path:
                print(node)
                question = node.get('question', '')
                if question:
                    # Generalize signals in questions if enabled
                    if generalize_signals:
                        question = self._generalize_signals(question, testbench)
                    
                    answer = node.get('answer', '') if include_answers else None
                    if answer and generalize_signals:
                        answer = self._generalize_signals(answer, testbench)
                    
                    content['questions'].append({
                        'level': node.get('level', 'unknown'),
                        'question': question,
                        'answer': answer
                    })
                
                # Extract existing rules if any
                existing_rules = node.get('rules_generated', [])
                if existing_rules:
                    if generalize_signals:
                        generalized_rules = [self._generalize_signals(rule, testbench) for rule in existing_rules]
                        content['rules'].extend(generalized_rules)
                    else:
                        content['rules'].extend(existing_rules)
            
            trace_contents.append(content)
        
        # Build prompt for rule generation based on best traces
        rule_gen_prompt = f"""
You are an expert in SystemVerilog Assertions (SVA). Your task is to evaluate whether an automatically generated SVA correctly captures a natural language requirement.

Please analyze the Initial SVA by considering:
- The semantic meanings of the operators used
- Whether the timing relationships match the natural language description
- Insights from similar cases that have been successfully resolved
"""
        
        # Add design context if available
        # if design_name:
        #     rule_gen_prompt += f"Design module: {design_name}\n"
        
        # Add testbench context if available
        if testbench:
            # Show key signals and structure from testbench
            tb_preview = testbench
            rule_gen_prompt += f"\nTestbench context:\n{tb_preview}\n"
        
        rule_gen_prompt += f"\nNatural language requirement:\n{target_prompt}\n"
        rule_gen_prompt += f"\nInitial SVA generated:\n{initial_sva}\n"
        
        # Extract operators from initial SVA and add their explanations
        if initial_sva:
            initial_operators = self.extract_operators_from_sva(initial_sva)
            if initial_operators:
                operator_explanations = self.get_operator_explanations(initial_operators)
                
                if operator_explanations:
                    rule_gen_prompt += "\nSVA Operators in Initial SVA and their meanings:\n"
                    for op, explanation in operator_explanations.items():
                        rule_gen_prompt += f"  • {op}: {explanation}\n"
        
        # Extract operators from the rules in the traces and add their explanations
        trace_operators_set = set()
        for content in trace_contents:
            # Extract operators from rules in this trace
            for rule_text in content.get('rules', []):
                rule_operators = self.extract_operators_from_sva(rule_text)
                trace_operators_set.update(rule_operators)
        
        if trace_operators_set:
            trace_operator_explanations = self.get_operator_explanations(list(trace_operators_set))
            if trace_operator_explanations:
                rule_gen_prompt += "\nSVA Operators in Extracted Rules and their meanings:\n"
                for op, explanation in trace_operator_explanations.items():
                    rule_gen_prompt += f"  • {op}: {explanation}\n"
        
        rule_gen_prompt += "\n"
        
        rule_gen_prompt += f"\nExample reasoning traces from similar successful cases:\n"
        rule_gen_prompt += f"These traces show how similar problems were analyzed and resolved. Use them as reference points for your own analysis.\n"

        # breakpoint()
        
        for idx, content in enumerate(trace_contents, 1):
            rule_gen_prompt += f"\n--- Trace {idx} (score={content['score']:.3f}) ---"
            
            # Add questions from this trace
            for q_idx, q_data in enumerate(content['questions'], 1):  # Top 5 questions per trace
                level = q_data['level']
                question = q_data['question']
                rule_gen_prompt += f"\n[{level[0].upper()}] Q{q_idx}: {question}"
                
                if include_answers and q_data['answer']:
                    rule_gen_prompt += f"\n    Answer: {q_data['answer']}"
            
            # Show existing rules from this trace as reference
            if content['rules']:
                rule_gen_prompt += f"\n  Example solutions from this trace (consider adapting these insights):"
                for r_idx, rule in enumerate(content['rules'], 1):
                    rule_gen_prompt += f"\n    {r_idx}. {rule}..."
        
        rule_gen_prompt += f"""

Guidance from reasoning traces:
The traces above show common issues and solutions from similar cases. Use them as examples to help you reason about the current problem.

Your task:
1. **Consider the operator semantics**: Review the operator meanings provided above. Do the operators in the Initial SVA correctly capture the timing and logical relationships described in the natural language requirement?

2. **Evaluate operator suitability**: 
   - Does the overlapping implication (`|->`) vs non-overlapping implication (`|=>`) match the intended timing?
   - Do delay operators like `##[N:M]` accurately represent when the consequent should be evaluated?
   - Are there any mismatches between what the NL says and what the operators actually do?

3. **Check completeness**: Does the Initial SVA handle all aspects mentioned in the requirement (signals, timing, edge cases)?

4. **Reference the traces**: The reasoning traces provide helpful patterns and insights. Consider their analysis, but adapt the suggestions to fit this specific requirement.

"""
        
        # Check if we should instruct LLM to skip misaligned traces
        instruct_skip_misaligned = getattr(FLAGS, 'instruct_skip_misaligned_in_rules', False)
        
        if instruct_skip_misaligned:
            rule_gen_prompt += """
If the Initial SVA already correctly captures the target requirement (proper operators, timing, signals, and logic), you may output "None" for all rules - no changes are needed.
"""
        
        rule_gen_prompt += f"""
# Each rule should:
# 1. Be specific and actionable
# 2. Learn from that trace's successful pattern
# 3. Reference specific signals from the testbench when applicable
# 4. Focus on operators, timing, signals, or structure from that trace
# 5. Be directly applicable to the target requirement and initial SVA
"""
# Output format (exactly {top_k} {"rule" if top_k == 1 else "rules"}, one per trace, one line each):
        
        # Add format examples based on top_k
        if instruct_skip_misaligned:
            if top_k == 1:
                rule_gen_prompt += "- Rule 1: [Based on Trace 1] ... OR\n"
                rule_gen_prompt += "- None (if Initial SVA already captures the NL requirement correctly)\n"
            else:
                rule_gen_prompt += "- Rule 1: [Based on Trace 1] ... OR None (if Initial SVA already correct)\n"
                rule_gen_prompt += "...\n"
                rule_gen_prompt += f"- Rule {top_k}: [Based on Trace {top_k}] ... OR None\n"
                rule_gen_prompt += "\nNote: If Initial SVA is already correct, output 'None' for ALL rules.\n"
        else:
            if top_k == 1:
                rule_gen_prompt += "- Rule 1: [Based on Trace 1] ...\n"
            else:
                rule_gen_prompt += "- Rule 1: [Based on Trace 1] ...\n"
                rule_gen_prompt += f"- Rule {top_k}: [Based on Trace {top_k}] ...\n"
        
        # Add critical formatting instructions
        rule_gen_prompt += """
CRITICAL FORMATTING RULES:
- Do NOT include code examples (no "e.g. assert property...")
- Do NOT include "for example", "for instance", "e.g." followed by code
- Keep each rule concise and focused on WHAT to change, not HOW to implement it
- Stop each rule after stating the principle, do not show implementation details
"""
        
        # BREAKPOINT 1: Check prompt after building format
        # breakpoint()  # Uncomment to inspect rule_gen_prompt before adding task-specific hints
        
        if FLAGS.task == "nl2sva_human" or FLAGS.task == "nl2sva_machine":
            rule_gen_prompt += (
                f"\nExamples of useful suggestions:\n"
                f"- If the natural language query contains some signal names, just use the signals instead of adding more you imagine. For example, if the current natural language asks you to use a latency threshold, just extract one signal corresponding to that signal instead of overthinking and using additional signals.\n"
                f"- Do NOT use the implication operator '|->' if possible, i.e. only use '|->' if simpler alternatives are not enough. In such cases, directly write combinatorial logic using '&&', '!==', etc.\n"
                f"- If the natural language already contains some signal names, for example, if the natural language query contains the following sentence: 'Use the signals 'tb_gnt', 'last_gnt', 'hold', and 'tb_req'.' You may ignore/miss some variables such as hold, i.e. you don't have to use every signal in the query.\n"
                f"- Do NOT add the comments to the assertions.\n"
            )

        # BREAKPOINT 2: Check complete prompt before LLM call
        print(f"\n=== Prompt for LLM (length: {len(rule_gen_prompt)} chars) ===")
        print(rule_gen_prompt)
        # breakpoint()  # Uncomment to inspect final rule_gen_prompt before LLM call

        try:
            response = llm_inference(
                system_prompt="You are an SVA expert who generates actionable rules from successful reasoning traces.",
                user_prompt=rule_gen_prompt
            )
            
            # BREAKPOINT: Check LLM response
            print(f"\n=== LLM Response (length: {len(response)} chars) ===")
            print(response)
            # breakpoint()  # Check raw LLM response
            
            # Parse rules from response with robust extraction
            rules = []
            none_count = 0
            
            # Strategy 1: Look for explicit "Rules:" section first
            lines = response.split('\n')
            rules_section_started = False
            rules_section_lines = []
            
            for line in lines:
                stripped = line.strip()
                
                # Detect start of rules section
                if re.match(r'^\*\*Rules:\*\*|^Rules:|^---\s*$', stripped, flags=re.IGNORECASE):
                    rules_section_started = True
                    continue
                
                # If we're in the rules section, collect lines
                if rules_section_started:
                    # Stop if we hit another major section
                    if re.match(r'^\*\*[A-Z].*:\*\*|^[A-Z][a-z]+ [A-Z][a-z]+:', stripped):
                        break
                    rules_section_lines.append(line)
            
            # If we found a rules section, only parse from there
            if rules_section_lines:
                lines_to_parse = rules_section_lines
                print(f"  Found explicit Rules section with {len(rules_section_lines)} lines")
            else:
                # Otherwise parse the entire response
                lines_to_parse = lines
                print(f"  No explicit Rules section found, parsing entire response")
            
            # Parse each line looking for rule patterns
            for line in lines_to_parse:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                rule_text = None
                is_rule_line = False
                
                # STRICT FORMAT MATCHING - only accept lines that explicitly look like rules
                
                # Format 1: "- Rule N: ..." (most common from our prompt)
                match = re.match(r'^-\s*Rule\s+\d+:\s*(.+)$', line, flags=re.IGNORECASE)
                if match:
                    rule_text = match.group(1)
                    is_rule_line = True
                
                # Format 2: "Rule N: ..." (without leading dash)
                elif re.match(r'^Rule\s+\d+:\s*', line, flags=re.IGNORECASE):
                    rule_text = re.sub(r'^Rule\s+\d+:\s*', '', line, flags=re.IGNORECASE)
                    is_rule_line = True
                
                # Format 3: "**Rule N:**" (markdown bold)
                elif re.match(r'^\*\*Rule\s+\d+:\*\*\s*', line, flags=re.IGNORECASE):
                    rule_text = re.sub(r'^\*\*Rule\s+\d+:\*\*\s*', '', line, flags=re.IGNORECASE)
                    is_rule_line = True
                
                # Format 4: "N. Rule text" or "N) Rule text" (numbered list)
                # BUT only if it doesn't look like analysis (avoid "1. Question:", "2. Timing:", etc.)
                elif re.match(r'^\d+[\.\)]\s+', line):
                    # Skip if it looks like a section header or analysis point
                    if not re.match(r'^\d+[\.\)]\s+(Question|Timing|Trigger|Consequent|Signal|Check|Analysis|Operator|Compare|Evaluate|Consider):', line, flags=re.IGNORECASE):
                        rule_text = re.sub(r'^\d+[\.\)]\s+', '', line)
                        is_rule_line = True
                
                # If we found a potential rule, clean it up
                if rule_text:
                    # Check if this is a "None" response (misaligned trace)
                    if re.match(r'^None\s*$', rule_text, flags=re.IGNORECASE):
                        none_count += 1
                        print(f"  Found None response (rule #{none_count})")
                        continue
                    
                    # Clean up common prefixes that LLM might add
                    # Remove "Rule X: " prefix
                    rule_text = re.sub(r'^Rule\s+\d+:\s*', '', rule_text, flags=re.IGNORECASE)
                    
                    # Remove "[Based on Trace X]" prefix
                    rule_text = re.sub(r'^\[Based on Trace \d+\]\s*', '', rule_text, flags=re.IGNORECASE)
                    
                    # Remove "**Rule X:**" markdown prefix
                    rule_text = re.sub(r'^\*\*Rule\s+\d+:\*\*\s*', '', rule_text, flags=re.IGNORECASE)
                    
                    rule_text = rule_text.strip()
                    if rule_text and len(rule_text) > 5:  # Only add non-empty, meaningful rules
                        rules.append(rule_text)
            
            if none_count > 0:
                if none_count == top_k:
                    print(f"✅ LLM evaluated that Initial SVA already correctly captures the NL requirement!")
                    print(f"   All {none_count} rules returned 'None' - no additional suggestions needed.")
                else:
                    print(f"Filtered out {none_count}/{top_k} rules (returned 'None')")
            
            print(f"Parsed and cleaned {len(rules)} valid rules from LLM response")
            for i, rule in enumerate(rules[:top_k], 1):
                print(f"  Cleaned rule {i}: {rule[:100]}...")
            
            # If all rules are None, return empty list (Initial SVA is already good)
            if len(rules) == 0 and none_count > 0:
                print(f"\n🎯 Result: Initial SVA is sufficient. No improvement suggestions needed. Returning empty list []")
                return []
            
            return rules[:top_k]
            
        except Exception as e:
            print(f"Rule generation from traces failed: {e}")
            return []
    
#     def generate_rules_from_qtrees(self, similar_qtrees: List[Dict], 
#                                    target_prompt: str, top_k: int = 5,
#                                    include_answers: bool = False) -> List[str]:
#         """
#         Generate rules from similar Q-Trees instead of retrieving existing ones.
        
#         Args:
#             similar_qtrees: List of similar Q-Tree entries
#             target_prompt: Target natural language prompt
#             top_k: Number of rules to generate
#             include_answers: Whether to include node answers (default: False, only questions)
            
#         Returns:
#             List of generated rule strings
#         """
#         print(f"\n=== Rule Generation Mode ===")
#         print(f"Generating rules from {len(similar_qtrees)} similar Q-Trees")
#         print(f"Include answers: {include_answers}")
        
#         # Collect insights from similar Q-Trees
#         qtree_insights = []
#         for qtree_entry in similar_qtrees[:5]:  # Use top 5
#             qtree = qtree_entry.get('qtree_data', {})
#             nodes = qtree.get('nodes', [])
            
#             # Extract questions (and optionally answers) from exploratory nodes
#             for node in nodes:
#                 if node.get('level') == 'exploratory':
#                     question = node.get('question', '')
#                     if question:
#                         insight = {
#                             'question': question,
#                             'task_id': qtree_entry.get('task_id', 'unknown')
#                         }
                        
#                         # Only include answer if requested
#                         if include_answers:
#                             answer = node.get('answer', '')
#                             if answer:
#                                 insight['answer'] = answer
                        
#                         qtree_insights.append(insight)
        
#         print(f"Collected {len(qtree_insights)} insights from Q-Tree exploratory nodes")
        
#         # Build prompt for rule generation
#         rule_gen_prompt = f"""
# You are an expert in SystemVerilog Assertion (SVA) generation. Based on analysis from similar cases, 
# generate {top_k} specific, actionable rules to help write correct SVA assertions.

# Target requirement: {target_prompt}

# """
        
#         # Format insights differently based on whether we have answers
#         if include_answers:
#             rule_gen_prompt += "Key questions and insights from similar successful cases:\n"
#             for i, insight in enumerate(qtree_insights[:10], 1):  # Top 10 insights
#                 rule_gen_prompt += f"\n{i}. Question: {insight['question']}"
#                 if 'answer' in insight:
#                     rule_gen_prompt += f"\n   Insight: {insight['answer'][:200]}..."
#         else:
#             rule_gen_prompt += "Key questions considered in similar successful cases:\n"
#             for i, insight in enumerate(qtree_insights[:10], 1):  # Top 10 questions
#                 rule_gen_prompt += f"\n{i}. {insight['question']}"
        
#         rule_gen_prompt += f"""

# Based on these questions and the target requirement, generate exactly {top_k} rules that:
# 1. Are specific and actionable
# 2. Help avoid common mistakes
# 3. Are applicable to the target requirement
# 4. Focus on operators, timing, signals, or structure

# Output format:
# - Rule 1
# - Rule 2
# - ...
# """
        
#         # BREAKPOINT: Check prompt before LLM call
#         print(f"\n=== Prompt for LLM (length: {len(rule_gen_prompt)} chars) ===")
#         print(rule_gen_prompt)
#         # breakpoint()  # Check rule_gen_prompt content
        
#         try:
#             response = llm_inference(
#                 system_prompt="You are an SVA expert who generates actionable rules.",
#                 user_prompt=rule_gen_prompt
#             )
            
#             # BREAKPOINT: Check LLM response
#             print(f"\n=== LLM Response (length: {len(response)} chars) ===")
#             print(response)
#             # breakpoint()  # Check raw LLM response before parsing
            
#             # Parse rules from response
#             rules = []
#             for line in response.split('\n'):
#                 line = line.strip()
#                 if line.startswith('- ') and len(line) > 5:
#                     rules.append(line[2:])  # Remove "- " prefix
            
#             print(f"Generated {len(rules)} rules")
#             for i, rule in enumerate(rules, 1):
#                 print(f"  {i}. {rule[:80]}...")
            
#             return rules[:top_k]
            
#         except Exception as e:
#             print(f"Rule generation failed: {e}")
#             return []



def integrate_enhanced_retrieval(query: str, qtree_data: List[Dict], 
                                 testbench: str = None, 
                                 csv_df=None, top_k: int = 5):
    """
    Main integration point for enhanced Q-Tree retrieval.
    
    Uses FLAGS configuration:
    - FLAGS.qtree_ranking_mode: 'prompt' or 'sva' (how to find similar Q-Trees)
    - FLAGS.rule_source: 'retrieve' or 'generate' (whether to retrieve or generate rules)
    
    Args:
        query: Query string (may contain prompt, assertion, etc.)
        qtree_data: List of Q-Tree entries
        testbench: Optional testbench code
        csv_df: Optional training examples DataFrame
        top_k: Number of suggestions to return
        
    Returns:
        Dict with 'suggestions' (List[str]) and 'initial_sva' (str) keys
        OR List[str] for backward compatibility (when return_dict=False)
    """
    # Get configuration from FLAGS
    ranking_mode = getattr(FLAGS, 'qtree_ranking_mode', 'prompt')  # 'prompt' or 'sva'
    rule_source = getattr(FLAGS, 'rule_source', 'retrieve')  # 'retrieve' or 'generate'
    
    print(f"\n{'='*60}")
    print(f"Enhanced Q-Tree Retrieval Configuration:")
    print(f"  Ranking Mode: {ranking_mode}")
    print(f"  Rule Source: {rule_source}")
    print(f"{'='*60}")
    
    # BREAKPOINT 1: Check inputs and configuration
    # breakpoint()  # Uncomment to debug input parameters
    
    # Initialize enhanced retriever with full functionality
    retriever = EnhancedQTreeRetrieval(qtree_data, csv_df)
    
    # Parse query to extract prompt
    query_dict = {}
    if "<---->" in query:
        parts = query.split("<---->")
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                query_dict[key.strip()] = value.strip()
    else:
        query_dict['Question'] = query
    
    prompt = query_dict.get('Question', query)
    design_name = query_dict.get('Design', 'unknown')
    suggestions = []
    
    # Step 1: ALWAYS generate initial SVA first
    # This is needed for:
    # 1. Hybrid ranking (LLM score + operator alignment)
    # 2. Operator-based pruning
    # 3. SVA-based Q-Tree matching (if ranking_mode='sva')
    print(f"\n=== Step 1: Generate Initial SVA ===")
    examples = []
    if retriever.training_examples:
        examples = retriever.find_similar_prompts(prompt, design_name, top_k=getattr(FLAGS, 'Examples_top_k', 2))
        print(f"Found {len(examples)} similar training examples")
    
    initial_sva = retriever.generate_initial_sva(prompt, testbench or '', examples)
    
    # BREAKPOINT 2: Check initial SVA generation
    # breakpoint()  # Uncomment to inspect initial_sva and extracted operators
    
    # Extract operator explanations for the initial SVA
    operator_explanations = {}
    if not initial_sva:
        print(f"WARNING: Failed to generate initial SVA")
        print(f"Continuing without hybrid ranking and operator pruning...")
    else:
        print(f"Generated initial SVA: {initial_sva}")
        target_operators = retriever.extract_operators_from_sva(initial_sva)
        print(f"Extracted operators from initial SVA: {target_operators}")
        # Get explanations for operators in initial SVA
        operator_explanations = retriever.get_operator_explanations(target_operators)
        # print(f"Operator explanations: {list(operator_explanations.keys())}")
    
    # Step 2: Retrieve and rank traces (same for both modes)
    print(f"\n=== Step 2: Retrieve and Rank Traces ===")
    print(f"Using {ranking_mode}-based Q-Tree matching")
    
    # Prepare test_row format
    test_row = {
        'prompt': prompt,
        'testbench': testbench or '',
        'design_name': design_name
    }
    
    # Get Q-Tree suggestions with LLM ranking
    suggestions_dict = retriever.pickup_qtrees_for_test_case(0, test_row, top_n=top_k, initial_sva=initial_sva, skip_ranking=True)
    traces = suggestions_dict.get('all_ranked_traces', [])
    
    if not traces:
        return {'suggestions': [], 'initial_sva': initial_sva, 'operator_explanations': operator_explanations}
    
    # Step 3: Apply hybrid ranking if initial SVA was generated
    if initial_sva:
        retriever._score_traces_by_operators_only(traces, initial_sva)
    
    # Step 4: Branch Cutting (pruning)
    pruned = [t for t in traces if t.get('operator_alignment_score', 0.0) > 0.0]
    # breakpoint()
    print(f"Cut {len(traces)-len(pruned)} traces, kept {len(pruned)} for ranking.")

    if not pruned:
        print("All traces have been pruned. Returning the initial SVA directly.")
        return {'suggestions': [], 'initial_sva': initial_sva, 'operator_explanations': operator_explanations}

    # Step 5: LLM ranking (and hybrid ranking)
    ranking_result = retriever.rank_traces_with_llm(prompt, pruned, testbench=testbench)
    
    # Merge LLM scores back into the full trace objects
    parsed_rankings = ranking_result.get('parsed', [])
    if parsed_rankings:
        trace_lookup = {trace['trace_id']: trace for trace in pruned}
        for ranked_item in parsed_rankings:
            trace_id = ranked_item.get('trace_id')
            if trace_id in trace_lookup:
                # Add LLM scoring info to the original trace object
                trace_lookup[trace_id]['llm_score'] = ranked_item.get('score', 0)
                trace_lookup[trace_id]['llm_rank'] = ranked_item.get('rank', 9999)
                trace_lookup[trace_id]['llm_rationale'] = ranked_item.get('rationale', '')
    
    # Prune traces with LLM score = 0 (before hybrid ranking)
    before_llm_prune = len(pruned)
    pruned = [t for t in pruned if t.get('llm_score', 0.0) > 0.0]
    print(f"Pruned {before_llm_prune - len(pruned)} traces with LLM score = 0, kept {len(pruned)} for hybrid ranking.")
    
    if not pruned:
        print("All traces have been pruned by LLM scoring. Returning the initial SVA directly.")
        return {'suggestions': [], 'initial_sva': initial_sva, 'operator_explanations': operator_explanations}
    
    # Now call hybrid ranking with full trace objects that have both operator scores and LLM scores
    ranked_traces = retriever.rank_traces_hybrid(pruned, initial_sva)
    
    # Step 5: Generate or Extract rules based on mode
    print(f"\n=== Step 5: Extract/Generate Rules ===")
    
    if rule_source == 'generate':
        # Generate new rules from the BEST traces
        # Number of rules = FLAGS.Suggestions_top_k
        num_suggestions = getattr(FLAGS, 'Suggestions_top_k', 3)
        print(f"RULE GENERATION MODE: Generating {num_suggestions} rules from top {num_suggestions} traces")
        print(f"Each trace will generate ONE specific rule (1:1 mapping)")
        
        # Use exactly Suggestions_top_k traces
        best_traces = ranked_traces[:num_suggestions]
        
        print(f"\nUsing top {len(best_traces)} traces (one rule per trace):")
        for idx, trace in enumerate(best_traces, 1):
            score = trace.get('composite_score', trace.get('llm_score', 0))
            print(f"  Trace {idx}: {trace['trace_id'][:60]} (score={score:.3f})")
        
        # BREAKPOINT 6 (Generate): Check best traces before generation
        # breakpoint()  # Inspect best_traces selected for rule generation
        
        # Generate rules from these best traces (one rule per trace)
        include_answers = getattr(FLAGS, 'include_qtree_answers', False)
        rules = retriever.generate_rules_from_best_traces(
            best_traces=best_traces,
            target_prompt=prompt,
            initial_sva=initial_sva,
            testbench=testbench,
            design_name=design_name,
            top_k=num_suggestions,  # Generate num_suggestions rules
            include_answers=include_answers
        )
        
        # BREAKPOINT 7 (Generate): Check generated rules
        print(f"\n=== Generated {len(rules)} rules (should be {num_suggestions}) ===")
        for idx, rule in enumerate(rules, 1):
            print(f"  Rule {idx} (from Trace {idx}): {rule}...")
        # breakpoint()  # Check generated rules
        
        # Format as suggestions
        suggestions = []
        for idx, rule in enumerate(rules, 1):
            if not rule.strip().startswith('-'):
                suggestions.append(f"- {rule}")
            else:
                suggestions.append(rule)
        
        print(f"\n=== Final Suggestions ===")

        if len(suggestions) > num_suggestions:
            print(f"  ❌ ERROR: Returning {len(suggestions)} suggestions but should only return {num_suggestions}!")
            print(f"  Truncating to {num_suggestions}...")
            suggestions = suggestions[:num_suggestions]
        
        # breakpoint()  # Final check before return
        
        return {'suggestions': suggestions, 'initial_sva': initial_sva, 'operator_explanations': operator_explanations}
    
    else:  # rule_source == 'retrieve'
        # Extract existing rules/traces from best traces
        # Number of suggestions = FLAGS.Suggestions_top_k
        num_suggestions = getattr(FLAGS, 'Suggestions_top_k', 3)
        print(f"RULE RETRIEVAL MODE: Extracting from top {num_suggestions} traces")
        
        # Use the already ranked traces
        extract_rules_only = getattr(FLAGS, 'extract_rules_only', True)
        
        suggestions = []
        for idx, trace in enumerate(ranked_traces[:num_suggestions], 1):
            if extract_rules_only:
                # Extract only rules from the trace
                rules = trace.get('rules', [])
                if rules:
                    rule_lines = []
                    for rule in rules:
                        rule_text = rule.text if hasattr(rule, 'text') else str(rule)
                        # Clean up rule text
                        rule_text = rule_text.strip()
                        if rule_text.startswith('- '):
                            rule_text = rule_text[2:]
                        rule_lines.append(f"- {rule_text}")
                    
                    suggestion_text = '\n'.join(rule_lines)
                    suggestions.append(suggestion_text)
                    print(f"  Suggestion {idx}: {len(rules)} rules extracted")
                else:
                    print(f"  Suggestion {idx}: No rules found, skipping")
            else:
                # Use full trace text
                trace_text = trace.get('trace_text', '')
                if trace_text:
                    score = trace.get('composite_score', trace.get('llm_score', 0))
                    suggestions.append(f"=== Suggestion {idx} (score: {score:.3f}) ===\n{trace_text}")
                    print(f"  Suggestion {idx}: Full trace included")
    
    # BREAKPOINT 6: Check final suggestions before return
    print(f"\n=== Preview of Final Suggestions ===")
    for i, sugg in enumerate(suggestions, 1):
        lines = sugg.split('\n')
        print(f"\nSuggestion {i}:")
        for line in lines[:10]:  # Show first 10 lines
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... ({len(lines) - 10} more lines)")
    
    print(f"\nReturning {len(suggestions)} final suggestions")
    return {'suggestions': suggestions, 'initial_sva': initial_sva, 'operator_explanations': operator_explanations}