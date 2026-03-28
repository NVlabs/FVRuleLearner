"""
Q-Tree based inference system for SVA generation.

This module implements a decision tree based approach that uses saved Q-Trees
to reproduce the thinking process for fixing assertions.
"""

import json
import os
import pickle
import csv
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils_agent import initiate_chat_with_retry

from config import FLAGS
from saver import saver  # Import the singleton instance

# Create a wrapper function for print that uses Saver.log_info
def print(*args, **kwargs):
    # Convert all arguments to string and join them
    message = ' '.join(str(arg) for arg in args)
    saver.log_info(message)

@dataclass
class DecisionNode:
    """Node in the decision tree constructed from Q-Trees"""
    node_id: str
    question_pattern: str  # Pattern that this node checks
    condition: str        # Condition to evaluate
    action: str          # Action to take if condition is met
    confidence: float    # Success rate of this path
    children: List['DecisionNode'] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)  # Example Q-Tree nodes


@dataclass
class QTreePattern:
    """Common pattern extracted from Q-Trees"""
    pattern_type: str  # 'operator', 'timing', 'signal', 'structure'
    question_template: str
    common_fixes: List[str]
    success_rate: float
    example_cases: List[str]  # task_ids


class QTreeInferenceEngine:
    """Main inference engine using Q-Trees"""
    
    def __init__(self, qtree_logdir: str, csv_path: str = None):
        """
        Initialize the inference engine.
        
        Args:
            qtree_logdir: Directory containing saved Q-Trees
            csv_path: Optional path to CSV file containing prompts
        """
        self.qtree_logdir = qtree_logdir
        self.qtree_database = {}
        self.pattern_database = {}
        self.decision_tree = None
        self.vectorizer = None
        self.csv_prompts = {}  # Store prompts from CSV file
        
        # Load CSV prompts if path provided
        if csv_path and os.path.exists(csv_path):
            self.load_csv_prompts(csv_path)
        
        # Always load qtrees without checking
        self.load_qtrees()
        self.extract_patterns()
        if FLAGS.qtree_inference_mode == "single_decision_tree":
            self.build_decision_tree()
    
    def load_csv_prompts(self, csv_path: str):
        """
        Load prompts from CSV file (e.g., nl2sva_human.csv).
        
        Args:
            csv_path: Path to CSV file containing task_id and prompt columns
        """
        try:
            print(f"\n=== Loading prompts from CSV: {csv_path} ===")
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                loaded_count = 0
                
                for row in reader:
                    task_id = row.get('task_id', '').strip()
                    prompt = row.get('prompt', '').strip()
                    design_name = row.get('design_name', '').strip()
                    
                    if task_id and prompt:
                        self.csv_prompts[task_id] = {
                            'design_name': design_name,
                            'prompt': prompt
                        }
                        loaded_count += 1
            
            print(f"Loaded {loaded_count} prompts from CSV")
            if loaded_count > 0:
                print(f"Sample task IDs: {list(self.csv_prompts.keys())[:5]}")
            
        except Exception as e:
            print(f"Warning: Failed to load CSV prompts: {e}")
            print("Will continue with prompts from QTree files only")
    
    def load_qtrees(self):
        """Load saved Q-Trees from disk and filter based on success criteria"""
        qtree_data = saver.load_qtree_corrections(self.qtree_logdir)
        
        # Store original data for prompt lookup
        self._original_qtree_data = qtree_data if qtree_data else []
        
        if qtree_data:
            # Check if we should filter by success
            filter_on_load = getattr(FLAGS, 'filter_functionality', True)
            total_qtrees = 0
            loaded_qtrees = 0
            
            # Organize by task_id
            for entry in qtree_data:
                task_id = entry.get('task_id')
                qtree = entry.get('qtree_data', {})
                total_qtrees += 1
                
                if task_id and qtree:
                    # Get metrics from either final_metrics or qtree.metrics
                    final_metrics = entry.get('final_metrics', {})
                    qtree_metrics = qtree.get('metrics', {})
                    
                    # Use final_metrics if available, otherwise use qtree metrics
                    pec = final_metrics.get('pec', qtree_metrics.get('pec', 0))
                    bleu = final_metrics.get('bleu', qtree_metrics.get('bleu', 0))
                    
                    # Store metrics in qtree for later use
                    qtree['metrics'] = {'pec': pec, 'bleu': bleu}
                    
                    # Apply filter if enabled
                    if filter_on_load:
                        # Only load if pec=1.0 OR bleu=1.0
                        if (abs(pec - 1.0) < 1e-6 or abs(bleu - 1.0) < 1e-6):
                            self.qtree_database[task_id] = qtree
                            loaded_qtrees += 1
                    else:
                        # Load all QTrees
                        self.qtree_database[task_id] = qtree
                        loaded_qtrees += 1
            
            # Debug output
            print(f"\n=== Q-Tree Loading Debug ===")
            print(f"Total Q-Trees in file: {total_qtrees}")
            print(f"Filter enabled: {filter_on_load}")
            
            if filter_on_load:
                print(f"Loaded {loaded_qtrees} successful Q-Trees (pec=1.0 or bleu=1.0) out of {total_qtrees} total from {self.qtree_logdir}")
            else:
                print(f"Loaded all {loaded_qtrees} Q-Trees from {self.qtree_logdir}")
            print("=" * 50)
            # breakpoint()
        else:
            print(f"No Q-Trees found in {self.qtree_logdir}")
    
    def extract_patterns(self):
        """Extract common patterns from Q-Trees"""
        pattern_groups = defaultdict(list)
        
        for task_id, qtree in self.qtree_database.items():
            nodes = qtree.get('nodes', [])
            
            for node in nodes:
                if node['level'] == 'exploratory':
                    # Categorize by question type
                    question = node['question'].lower()
                    
                    if any(kw in question for kw in ['operator', '&&', '||', '!', 'logical']):
                        pattern_type = 'operator'
                    elif any(kw in question for kw in ['timing', 'delay', '##', 'eventually', 'temporal']):
                        pattern_type = 'timing'
                    elif any(kw in question for kw in ['signal', 'missing', 'extra']):
                        pattern_type = 'signal'
                    elif any(kw in question for kw in ['structure', 'antecedent', 'consequent']):
                        pattern_type = 'structure'
                    else:
                        pattern_type = 'other'
                    
                    pattern_groups[pattern_type].append({
                        'task_id': task_id,
                        'question': node['question'],
                        'answer': node.get('answer', ''),
                        'rules': self._get_rules_for_node(node, nodes),
                        'metrics': qtree.get('metrics', {})
                    })
        
        # Create pattern objects
        for pattern_type, examples in pattern_groups.items():
            # Find common fixes
            all_rules = []
            successful_cases = []
            
            for example in examples:
                all_rules.extend(example['rules'])
                if example['metrics'].get('pec', 0) == 1.0:
                    successful_cases.append(example['task_id'])
            
            # Count rule frequency
            rule_counts = defaultdict(int)
            for rule in all_rules:
                rule_counts[rule] += 1
            
            # Get most common rules
            common_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            pattern = QTreePattern(
                pattern_type=pattern_type,
                question_template=self._extract_question_template(
                    [ex['question'] for ex in examples]
                ),
                common_fixes=[rule for rule, _ in common_rules],
                success_rate=len(successful_cases) / len(examples) if examples else 0,
                example_cases=successful_cases[:5]
            )
            
            self.pattern_database[pattern_type] = pattern
        
        print(f"Extracted {len(self.pattern_database)} pattern types")
    
    def _get_rules_for_node(self, node, all_nodes):
        """Get rules generated from this node's subtree"""
        rules = []
        
        def collect_rules(node_id):
            for n in all_nodes:
                if n['id'] == node_id:
                    rules.extend(n.get('rules_generated', []))
                    for child_id in n.get('children', []):
                        collect_rules(child_id)
        
        collect_rules(node['id'])
        return rules
    
    def _extract_question_template(self, questions):
        """Extract a common template from similar questions"""
        if not questions:
            return ""
        
        # Simple approach: find common words
        word_counts = defaultdict(int)
        for q in questions:
            words = q.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] += 1
        
        # Get most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return f"Check for {', '.join([w for w, _ in common_words])}"
    
    def build_decision_tree(self):
        """Build decision tree from patterns"""
        print("\n=== Building Decision Tree ===")
        
        # Create root node
        self.decision_tree = DecisionNode(
            node_id="root",
            question_pattern="What type of issue needs to be fixed?",
            condition="always",
            action="categorize_issue",
            confidence=1.0
        )
        print(f"Created root node: {self.decision_tree.node_id}")
        print(f"  Question: {self.decision_tree.question_pattern}")
        print(f"  Action: {self.decision_tree.action}")
        
        # Add pattern-based branches
        branch_count = 0
        for pattern_type, pattern in self.pattern_database.items():
            print(f"\nProcessing pattern: {pattern_type}")
            print(f"  Template: {pattern.question_template}")
            print(f"  Success rate: {pattern.success_rate}")
            print(f"  Number of fixes: {len(pattern.common_fixes) if pattern.common_fixes else 0}")
            
            if pattern.common_fixes:  # Only add if we have fixes
                pattern_node = DecisionNode(
                    node_id=f"pattern_{pattern_type}",
                    question_pattern=pattern.question_template,
                    condition=f"issue_type == '{pattern_type}'",
                    action="check_specific_pattern",
                    confidence=pattern.success_rate
                )
                print(f"  Created pattern node: {pattern_node.node_id}")
                
                # Add fix nodes
                for i, fix in enumerate(pattern.common_fixes):
                    fix_node = DecisionNode(
                        node_id=f"fix_{pattern_type}_{i}",
                        question_pattern=f"Apply fix: {fix[:50]}...",
                        condition=self._extract_condition_from_fix(fix),
                        action=fix,
                        confidence=pattern.success_rate,
                        examples=pattern.example_cases
                    )
                    pattern_node.children.append(fix_node)
                    print(f"    Added fix node {i}: {fix_node.node_id}")
                    print(f"      Fix preview: {fix[:80]}...")
                    print(f"      Condition: {fix_node.condition}")
                
                self.decision_tree.children.append(pattern_node)
                branch_count += 1
                print(f"  Added pattern branch to tree (total fixes: {len(pattern_node.children)})")
            else:
                print(f"  Skipped pattern (no fixes available)")
        
        print(f"\n=== Decision Tree Built Successfully ===")
        print(f"Total main branches: {len(self.decision_tree.children)}")
        print(f"Tree structure:")
        self._print_tree_structure(self.decision_tree, indent=0)
        # breakpoint()
        print("=" * 50)
    
    def _print_tree_structure(self, node, indent=0):
        """Helper method to print tree structure"""
        prefix = "  " * indent
        print(f"{prefix}├─ {node.node_id} (confidence: {node.confidence})")
        print(f"{prefix}│  Question: {node.question_pattern[:60]}...")
        print(f"{prefix}│  Condition: {node.condition}")
        if node.examples and indent > 0:  # Don't print examples for root
            print(f"{prefix}│  Examples: {len(node.examples)} cases")
        
        for i, child in enumerate(node.children):
            if i == len(node.children) - 1:
                print(f"{prefix}│")
            self._print_tree_structure(child, indent + 1)
    
    def _extract_condition_from_fix(self, fix):
        """Extract condition from fix rule"""
        fix_lower = fix.lower()
        
        if 'replace' in fix_lower:
            # Extract what to replace
            if 'eventually' in fix_lower:
                return "'eventually' in assertion"
            elif '&&' in fix_lower:
                return "'&&' in antecedent"
            elif '||' in fix_lower:
                return "'||' in assertion"
        elif 'add' in fix_lower:
            if 'strong' in fix_lower:
                return "'strong' not in assertion"
        
        return "check_assertion_structure"
    
    def find_similar_qtrees(self, query_prompt: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Find most similar Q-Trees based on prompt similarity.
        By default, only considers successful QTrees (pec = 1.0 OR bleu = 1.0).
        
        Returns:
            List of (task_id, similarity_score) tuples
        """
        if top_k is None:
            top_k = getattr(FLAGS, 'qtree_similarity_top_k', 5)
        
        # Create corpus of prompts from already-filtered QTrees
        task_ids = []
        prompts = []
        qtree_entries = []  # Store full entries for debugging
        
        # All QTrees in database are already filtered based on FLAGS.filter_functionality
        # breakpoint()
        for task_id, qtree in self.qtree_database.items():
            task_ids.append(task_id)
            
            # Priority 1: Check CSV prompts (most accurate source)
            prompt = None
            prompt_source = None
            
            if task_id in self.csv_prompts:
                prompt = self.csv_prompts[task_id]['prompt']
                prompt_source = 'csv'
            
            # Priority 2: Check prompt in qtree data
            if not prompt:
                prompt = qtree.get('prompt', None)
                if prompt:
                    prompt_source = 'qtree'
            
            # Priority 3: Check parent entry
            if not prompt:
                # Look for prompt in the original loaded data
                for entry in getattr(self, '_original_qtree_data', []):
                    if entry.get('task_id') == task_id:
                        prompt = entry.get('prompt', None)
                        if prompt:
                            prompt_source = 'parent'
                        break
            
            # No fallback - let it fail if no prompt found
            if not prompt:
                print(f"ERROR: No prompt found for task_id: {task_id}")
                print(f"  - Not in CSV file")
                print(f"  - Not in QTree data")
                print(f"  - Not in parent entry")
                # Skip this QTree entirely
                continue
            
            prompts.append(prompt)
            qtree_entries.append({
                'task_id': task_id,
                'prompt': prompt,
                'prompt_source': prompt_source
            })
        
        if not task_ids:
            print(f"Warning: No QTrees available in database for similarity matching")
            return []
        
        print(f"\n=== QTree Similarity Search ===")
        print(f"Query: {query_prompt}")
        print(f"Searching among {len(task_ids)} QTrees for similarity")
        
        # Count prompt sources
        source_counts = {}
        for entry in qtree_entries:
            source = entry['prompt_source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\nPrompt sources breakdown:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count} prompts")
        
        # Check if we have any QTrees left after filtering
        if not qtree_entries:
            print(f"\nERROR: No QTrees with valid prompts found!")
            print(f"Total QTrees in database: {len(self.qtree_database)}")
            print(f"QTrees skipped due to missing prompts: {len(self.qtree_database) - len(qtree_entries)}")
            return []
        
        # Add query
        prompts.append(query_prompt)
        
        # Vectorize
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 3),
                max_features=100
            )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(prompts)
            
            # Calculate similarities
            query_vector = tfidf_matrix[-1]
            similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
            
            # Get top-k
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            similar_qtrees = []
            print(f"\n=== Top {top_k} Similar QTrees ===")
            for rank, idx in enumerate(top_indices, 1):
                task_id = task_ids[idx]
                similarity = similarities[idx]
                entry = qtree_entries[idx]
                
                similar_qtrees.append((task_id, similarity))
                
                print(f"\nRank {rank}:")
                print(f"  Task ID: {task_id}")
                print(f"  Similarity: {similarity:.4f}")
                print(f"  Prompt source: {entry['prompt_source']}")
                # if entry['has_real_prompt']:
                print(f"  Prompt preview: {entry['prompt'][:100]}...")
            
            print("\n=== Candidate QTrees Selected ===")
            print(f"Candidates: {[t[0] for t in similar_qtrees]}")
            
            # Add breakpoint here to inspect candidates
            # breakpoint()
            
            return similar_qtrees
            
        except Exception as e:
            print(f"ERROR: Failed to find similar Q-Trees: {e}")
            print(f"Stack trace:")
            import traceback
            traceback.print_exc()
            # Don't fall back - let it fail
            raise Exception(f"QTree similarity search failed: {e}")
    
    def traverse_decision_tree(self, assertion: str, similar_qtrees: List[str]) -> List[Dict]:
        """
        Traverse decision tree to find applicable fixes.
        
        Returns:
            List of applicable fixes with confidence scores
        """
        fixes = []
        
        def traverse_node(node: DecisionNode, path: List[str]):
            # Evaluate condition
            if self._evaluate_condition(node.condition, assertion, similar_qtrees):
                path.append(node.node_id)
                
                # If leaf node with action
                if node.action and node.action not in ['categorize_issue', 'check_specific_pattern']:
                    fixes.append({
                        'action': node.action,
                        'confidence': node.confidence,
                        'path': path.copy(),
                        'examples': node.examples
                    })
                
                # Traverse children
                for child in node.children:
                    traverse_node(child, path.copy())
        
        traverse_node(self.decision_tree, [])
        
        # Sort by confidence
        fixes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return fixes
    
    def _evaluate_condition(self, condition: str, assertion: str, similar_qtrees: List[str]) -> bool:
        """Evaluate if condition is met"""
        if condition == "always":
            return True
        
        # Simple pattern matching
        if " in " in condition:
            pattern, target = condition.split(" in ")
            pattern = pattern.strip("'\"")
            return pattern in assertion
        
        # Issue type conditions
        if "issue_type == " in condition:
            # Analyze assertion to determine issue type
            assertion_lower = assertion.lower()
            
            if 'operator' in condition and any(op in assertion_lower for op in ['&&', '||', '!', 'and', 'or']):
                return True
            elif 'timing' in condition and any(t in assertion_lower for t in ['##', 'eventually', '->', '=>']):
                return True
            elif 'signal' in condition:
                # This would need more sophisticated analysis
                return True
            elif 'structure' in condition:
                return '|->' in assertion or '|=>' in assertion
        
        return False
    
    def apply_fixes(self, assertion: str, fixes: List[Dict]) -> str:
        """
        Apply selected fixes to assertion.
        
        Args:
            assertion: Original assertion
            fixes: List of fixes to apply
            
        Returns:
            Fixed assertion
        """
        fixed_assertion = assertion
        confidence_threshold = getattr(FLAGS, 'qtree_decision_confidence_threshold', 0.7)
        
        for fix in fixes:
            if fix['confidence'] < confidence_threshold:
                continue
            
            action = fix['action']
            
            # Parse and apply fix
            if 'Replace' in action or 'replace' in action:
                fixed_assertion = self._apply_replacement(fixed_assertion, action)
            elif 'Add' in action or 'add' in action:
                fixed_assertion = self._apply_addition(fixed_assertion, action)
            elif 'Remove' in action or 'remove' in action:
                fixed_assertion = self._apply_removal(fixed_assertion, action)
        
        return fixed_assertion
    
    
    def fix_assertion_with_qtree(self, prompt: str, assertion: str = None, 
    reference: str = None, testbench: str = None) -> Tuple[str, Dict]:
        """
        Main method to fix assertion using Q-Tree inference.
        Can work with just Natural Language (prompt) when no assertion is provided.
        
        Args:
            prompt: Natural language description
            assertion: Generated assertion to fix (optional)
            reference: Reference assertion (optional, for evaluation)
            
        Returns:
            Tuple of (fixed_assertion, inference_info)
        """
        # DEBUG
        print("\n=== Q-TREE FIX_ASSERTION_WITH_QTREE ===")
        print(f"Received prompt: {prompt}")
        if assertion:
            print(f"Received assertion to fix:\n{assertion}")
        else:
            print("No assertion provided - will retrieve based on Natural Language only")
        print(f"Reference provided: {'Yes' if reference else 'No'}")
        print("=" * 50)
        
        inference_info = {
            'similar_qtrees': [],
            'applied_fixes': [],
            'decision_path': [],
            'no_assertion_mode': assertion is None
        }
        
        # Find similar Q-Trees based on natural language
        similar_qtrees = self.find_similar_qtrees(prompt)
        inference_info['similar_qtrees'] = similar_qtrees
        
        # Extract task_ids
        similar_task_ids = [task_id for task_id, _ in similar_qtrees]
        # Get mode early to check conditions
        mode = FLAGS.qtree_inference_mode
        
        # Apply fixes based on mode (already defined above)
        if mode == 'single_decision_tree':
            # Traverse decision tree
            applicable_fixes = self.traverse_decision_tree(assertion, similar_task_ids)
            inference_info['applied_fixes'] = applicable_fixes
            fixed_assertion = self.apply_fixes(assertion, applicable_fixes)
        elif mode == 'best_of_multiple_decision_trees':
            # Build customized QTree from multiple similar QTrees
            customized_qtree = self.build_customized_qtree(similar_task_ids)
            print(f"customized_qtree: {customized_qtree}")
            breakpoint()
            # Perform layer-by-layer inference
            suggestions = self.layer_by_layer_inference(
                customized_qtree, prompt, assertion, testbench
            )
            
            breakpoint()
            # Apply the selected suggestions
            fixed_assertion = self.apply_qtree_suggestions(assertion, suggestions)
            inference_info['customized_qtree'] = customized_qtree
            inference_info['layer_suggestions'] = suggestions

        
        return fixed_assertion, inference_info
    
#     def generate_assertion_from_qtrees(self, prompt: str, similar_task_ids: List[str]) -> str:
#         """
#         Generate an initial assertion based on similar QTrees when no assertion is provided.
        
#         Args:
#             prompt: Natural language description
#             similar_task_ids: List of similar task IDs
            
#         Returns:
#             Generated assertion string
#         """
#         print(f"\nGenerating assertion from {len(similar_task_ids)} similar QTrees")
        
#         # QTree doesn't need a sophisticated starting assertion
#         # The layer-by-layer inference will apply suggestions from similar cases
#         print("  Creating minimal template assertion...")
#         print("  The real work happens in layer-by-layer inference:")
#         print("  - Exploratory questions identify issues (YES/NO)")
#         print("  - For each YES, ALL suggestions from that branch are applied")
        
#         # Minimal template - will be completely transformed by QTree suggestions
#         template_assertion = """asrt: assert property (@(posedge clk) disable iff (tb_reset)
#     (1'b1)  // Will be replaced by QTree suggestions
# );"""
        
#         return template_assertion
    
    def generate_with_qtree_prompt(self, prompt: str, assertion: str, 
    similar_task_ids: List[str]) -> str:
        """
        Generate assertion using Q-Tree thinking process in prompt.
        """
        # Build prompt with Q-Tree thinking
        qtree_prompt = f"""
        Natural Language: {prompt}
        Current Assertion: {assertion}
        
        Based on similar successful cases, follow this thinking process:
        """
        
        # Add thinking from similar Q-Trees
        for task_id in similar_task_ids[:3]:  # Top 3
            if task_id in self.qtree_database:
                qtree = self.qtree_database[task_id]
                qtree_prompt += f"\n\nFrom case {task_id}:\n"
                
                # Add exploratory questions
                for node in qtree.get('nodes', []):
                    if node['level'] == 'exploratory':
                        qtree_prompt += f"- Question: {node['question']}\n"
                        if node.get('answer'):
                            qtree_prompt += f"  Key insight: {node['answer'][:200]}...\n"
        
        qtree_prompt += "\nApply these insights to fix the assertion."
        
        # In practice, this would call the LLM
        # For now, return the original with basic fixes
        return assertion
    
    def evaluate_fix(self, original: str, fixed: str, reference: str = None) -> Dict:
        """Evaluate the quality of the fix"""
        metrics = {
            'changed': original != fixed,
            'fix_count': 0
        }
        
        # Count specific fixes applied
        if 'strong' in fixed and 'strong' not in original:
            metrics['fix_count'] += 1
        if 'eventually' not in fixed and 'eventually' in original:
            metrics['fix_count'] += 1
        
        return metrics
    
    def build_customized_qtree(self, similar_task_ids: List[str], assertion: str = None) -> Dict:
        """
        Build a customized QTree by merging insights from multiple similar QTrees.
        
        Args:
            similar_task_ids: List of similar task IDs
            assertion: Current assertion to fix (optional)
            
        Returns:
            Customized QTree structure
        """
        print(f"\n=== Building Customized QTree from {len(similar_task_ids)} similar QTrees ===")
        print(f"Similar task IDs: {similar_task_ids}...")
        
        # Collect all nodes from similar QTrees
        all_nodes = defaultdict(list)  # Group by level
        
        for task_id in similar_task_ids[:getattr(FLAGS, 'qtree_similarity_top_k', 5)]:
            if task_id in self.qtree_database:
                qtree = self.qtree_database[task_id]
                for node in qtree.get('nodes', []):
                    level = node.get('level', 'unknown')
                    all_nodes[level].append({
                        'task_id': task_id,
                        'node': node,
                        'metrics': qtree.get('metrics', {})
                    })
        
        # Build customized tree structure
        customized_qtree = {
            'nodes': [],
            'levels': ['exploratory', 'refinement', 'final_suggestion'],
            'node_id_counter': 0
        }
        
        # Merge exploratory questions
        exploratory_questions = self._merge_exploratory_questions(
            all_nodes.get('exploratory', []), assertion
        )
        
        # Create root node
        root_node = {
            'id': self._get_next_node_id(customized_qtree),
            'level': 'root',
            'question': 'What aspects of the assertion need to be checked?',
            'children': []
        }
        customized_qtree['nodes'].append(root_node)
        
        # Add exploratory nodes
        for question_data in exploratory_questions:
            exp_node = {
                'id': self._get_next_node_id(customized_qtree),
                'level': 'exploratory',
                'question': question_data['question'],
                'relevance_score': question_data['relevance'],
                'source_tasks': question_data['sources'],
                'children': []
            }
            root_node['children'].append(exp_node['id'])
            customized_qtree['nodes'].append(exp_node)
            
            # Add refinement nodes for this exploratory question
            refinements = self._get_relevant_refinements(
                question_data, all_nodes.get('refinement', [])
            )
            
            for ref_data in refinements:
                ref_node = {
                    'id': self._get_next_node_id(customized_qtree),
                    'level': 'refinement',
                    'question': ref_data['question'],
                    'parent_exploratory': exp_node['id'],
                    'children': []
                }
                exp_node['children'].append(ref_node['id'])
                customized_qtree['nodes'].append(ref_node)
                
                # Add suggestion nodes
                suggestions = self._get_relevant_suggestions(
                    ref_data, all_nodes.get('final_suggestion', [])
                )
                
                for sug_data in suggestions:
                    sug_node = {
                        'id': self._get_next_node_id(customized_qtree),
                        'level': 'final_suggestion',
                        'suggestion': sug_data['suggestion'],
                        'confidence': sug_data['confidence'],
                        'parent_refinement': ref_node['id']
                    }
                    ref_node['children'].append(sug_node['id'])
                    customized_qtree['nodes'].append(sug_node)
        
        print(f"Built customized QTree with {len(customized_qtree['nodes'])} nodes")
        return customized_qtree
    
    def _get_next_node_id(self, qtree: Dict) -> str:
        """Generate next node ID"""
        node_id = f"node_{qtree['node_id_counter']}"
        qtree['node_id_counter'] += 1
        return node_id
    
    def _merge_exploratory_questions(self, exploratory_nodes: List[Dict], 
                                   assertion: str) -> List[Dict]:
        """
        Merge and rank exploratory questions from multiple QTrees.
        """
        question_groups = defaultdict(list)
        
        # Group similar questions
        for node_data in exploratory_nodes:
            question = node_data['node']['question']
            # Simple grouping by key concepts
            key = self._extract_question_key(question)
            question_groups[key].append(node_data)
        
        # Rank and select top questions
        ranked_questions = []
        for key, group in question_groups.items():
            # Calculate relevance based on frequency (how often this question appears)
            # All included QTrees have either pec=1.0 or bleu=1.0
            relevance = len(group) / len(exploratory_nodes) if exploratory_nodes else 0
            
            # Select representative question (prefer one with both pec=1.0 and bleu=1.0 if available)
            representative = group[0]
            for g in group:
                pec = g['metrics'].get('pec', 0)
                bleu = g['metrics'].get('bleu', 0)
                if abs(pec - 1.0) < 1e-6 and abs(bleu - 1.0) < 1e-6:
                    representative = g
                    break
            
            ranked_questions.append({
                'question': representative['node']['question'],
                'relevance': relevance,  # Frequency-based
                'sources': [g['task_id'] for g in group],
                'original_nodes': group
            })
        
        # Sort by relevance (frequency) and return top questions
        ranked_questions.sort(key=lambda x: x['relevance'], reverse=True)
        return ranked_questions[:5]  # Top 5 exploratory questions
    
    def _extract_all_suggestions_from_qtree(self, customized_qtree: Dict) -> List[Dict]:
        """
        Extract all suggestions from customized QTree for RAG retrieval.
        No assertion analysis needed - just collect all available suggestions.
        
        Args:
            customized_qtree: The customized QTree structure
            
        Returns:
            List of all suggestions with metadata
        """
        print("\n=== Extracting All Suggestions from Customized QTree ===")
        all_suggestions = []
        
        # Iterate through all nodes in the customized QTree
        for node in customized_qtree['nodes']:
            if node['level'] == 'final_suggestion' and node.get('suggestion'):
                # Find the path to this suggestion
                parent_ref_id = node.get('parent_refinement')
                parent_exp_id = None
                
                # Find parent refinement node
                for ref_node in customized_qtree['nodes']:
                    if ref_node['id'] == parent_ref_id:
                        parent_exp_id = ref_node.get('parent_exploratory')
                        break
                
                # Build reasoning path
                exploratory_q = ""
                refinement_q = ""
                
                for exp_node in customized_qtree['nodes']:
                    if exp_node['id'] == parent_exp_id:
                        exploratory_q = exp_node.get('question', '')
                        break
                
                for ref_node in customized_qtree['nodes']:
                    if ref_node['id'] == parent_ref_id:
                        refinement_q = ref_node.get('question', '')
                        break
                
                all_suggestions.append({
                    'suggestion': node['suggestion'],
                    'confidence': node.get('confidence', 0.5),
                    'reasoning_path': {
                        'exploratory': exploratory_q,
                        'refinement': refinement_q,
                        'source_tasks': node.get('source_tasks', [])
                    }
                })
        
        print(f"Extracted {len(all_suggestions)} suggestions from customized QTree")
        return all_suggestions
    
    def _extract_question_key(self, question: str) -> str:
        """Extract key concepts from question for grouping"""
        question_lower = question.lower()
        
        # Define key concept patterns
        concepts = []
        if any(kw in question_lower for kw in ['operator', 'logical', '&&', '||']):
            concepts.append('operator')
        if any(kw in question_lower for kw in ['timing', 'delay', 'eventually', 'temporal']):
            concepts.append('timing')
        if any(kw in question_lower for kw in ['signal', 'missing', 'extra']):
            concepts.append('signal')
        if any(kw in question_lower for kw in ['structure', 'antecedent', 'consequent']):
            concepts.append('structure')
        
        return '_'.join(concepts) if concepts else 'general'
    
    def _get_relevant_refinements(self, exploratory_data: Dict, 
                                refinement_nodes: List[Dict]) -> List[Dict]:
        """Get relevant refinement questions for an exploratory question"""
        relevant_refinements = []
        
        # Find refinements from the same source tasks
        for ref_node_data in refinement_nodes:
            if ref_node_data['task_id'] in exploratory_data['sources']:
                relevant_refinements.append({
                    'question': ref_node_data['node']['question'],
                    'source_task': ref_node_data['task_id']
                })
        
        return relevant_refinements[:3]  # Top 3 refinements
    
    def _get_relevant_suggestions(self, refinement_data: Dict,
                                suggestion_nodes: List[Dict]) -> List[Dict]:
        """Get relevant suggestions for a refinement question"""
        relevant_suggestions = []
        
        # Find suggestions from the same source task
        for sug_node_data in suggestion_nodes:
            if sug_node_data['task_id'] == refinement_data.get('source_task'):
                node = sug_node_data['node']
                suggestion_text = node.get('suggestion', '')
                if not suggestion_text and node.get('rules_generated'):
                    # Use rules as suggestions
                    suggestion_text = '; '.join(node['rules_generated'])
                
                if suggestion_text:
                    relevant_suggestions.append({
                        'suggestion': suggestion_text,
                        'confidence': sug_node_data['metrics'].get('pec', 0.5)
                    })
        
        return relevant_suggestions[:2]  # Top 2 suggestions per refinement
    
    def layer_by_layer_inference(self, customized_qtree: Dict, 
                               prompt: str, assertion: str, testbench: str) -> List[Dict]:
        """
        Perform layer-by-layer inference using a single LLM call for efficiency.
        
        This method collects all questions from the customized QTree (exploratory,
        refinement, and suggestion levels) and analyzes them in one comprehensive
        LLM call. The LLM determines which paths are relevant and which suggestions
        should be applied.
        
        Args:
            customized_qtree: The customized QTree structure
            prompt: Natural language description
            assertion: Current assertion to fix
            
        Returns:
            List of selected suggestions with reasoning paths
        """
        print("\n=== Layer-by-Layer Inference ===")
        
        selected_suggestions = []
        
        # Get root node
        root_node = next(n for n in customized_qtree['nodes'] if n['level'] == 'root')
        
        # Collect all questions in a structured way
        question_tree = []
        
        # Get exploratory nodes
        exploratory_nodes = [
            n for n in customized_qtree['nodes'] 
            if n['level'] == 'exploratory' and n['id'] in root_node['children']
        ]
        
        for exp_node in exploratory_nodes:
            exp_entry = {
                'id': exp_node['id'],
                'question': exp_node['question'],
                'level': 'exploratory',
                'refinements': []
            }
            
            # Get refinement nodes for this exploratory question
            refinement_nodes = [
                n for n in customized_qtree['nodes']
                if n['level'] == 'refinement' and n['id'] in exp_node['children']
            ]
            
            for ref_node in refinement_nodes:
                ref_entry = {
                    'id': ref_node['id'],
                    'question': ref_node['question'],
                    'level': 'refinement',
                    'suggestions': []
                }
                
                # Get suggestion nodes
                suggestion_nodes = [
                    n for n in customized_qtree['nodes']
                    if n['level'] == 'final_suggestion' and n['id'] in ref_node['children']
                ]
                
                for sug_node in suggestion_nodes:
                    ref_entry['suggestions'].append({
                        'id': sug_node['id'],
                        'text': sug_node['suggestion'],
                        'confidence': sug_node['confidence']
                    })
                
                exp_entry['refinements'].append(ref_entry)
            
            question_tree.append(exp_entry)
        
        # Make one LLM call to analyze all questions
        print(f"question_tree: {question_tree}")
        breakpoint()
        batch_answers = self._batch_llm_analysis(question_tree, prompt, assertion, testbench)
        
        # Process batch answers to extract suggestions
        for exp_data in batch_answers:
            if exp_data.get('is_relevant', False):
                # For each YES answer, include all suggestions from all refinements
                for ref_data in exp_data.get('refinements', []):
                    for sug_data in ref_data.get('suggestions', []):
                        selected_suggestions.append({
                            'suggestion': sug_data['text'],
                            'confidence': sug_data['confidence'],
                            'reasoning_path': {
                                'exploratory': exp_data['question'],
                                'exploratory_answer': f"{exp_data['answer']} - {exp_data.get('reason', '')}",
                                'refinement': ref_data['question'],
                                'refinement_answer': 'Applied because exploratory question identified an issue'
                            }
                        })
        
        print(f"\nSelected {len(selected_suggestions)} suggestions through layer-by-layer inference")
        return selected_suggestions
    
    def _batch_llm_analysis(self, question_tree: List[Dict], prompt: str, assertion: str, testbench: str) -> List[Dict]:
        """
        Analyze exploratory questions in a single LLM call for efficiency.
        """
        # Build simplified prompt focusing only on exploratory questions
        batch_prompt = f"""
        You are analyzing a SystemVerilog assertion to identify potential issues.
        
        Natural Language Requirement: {prompt}
        
        Current Testbench:
        {testbench}

        Current Assertion:
        {assertion}
        
        Please answer the following exploratory questions with YES or NO.
        Answer YES if the question identifies an issue that needs fixing.
        Answer NO if the assertion is correct in that aspect.
        
        Questions:
        """
        
        # List all exploratory questions
        for i, exp_data in enumerate(question_tree, 1):
            batch_prompt += f"\n{i}. {exp_data['question']}"
        
        batch_prompt += """
        
        OUTPUT FORMAT:
        Provide a JSON array with your answers:
        [
          {
            "question_number": 1,
            "answer": "YES" or "NO",
            "brief_reason": "One sentence explanation (optional)"
          },
          ...
        ]
        
        Be strict - only answer YES if there's a real issue that needs fixing.
        """
        
        try:
            # Make the LLM call
            response = initiate_chat_with_retry(
                system_prompt="You are an expert in SystemVerilog assertions. Provide responses in valid JSON format.",
                user_prompt=batch_prompt
            )

            print(f"response: {response}")
            breakpoint()
            
            # Parse JSON response
            import json
            try:
                # Extract JSON from response (handle potential markdown formatting)
                response_text = response.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                batch_results = json.loads(response_text.strip())
                
                # Map the simple YES/NO results back to our question tree structure
                mapped_results = []
                for answer_data in batch_results:
                    q_num = answer_data.get('question_number', 0) - 1  # Convert to 0-based index
                    if 0 <= q_num < len(question_tree):
                        exp_data = question_tree[q_num]
                        
                        # Create result structure based on YES/NO answer
                        is_relevant = answer_data.get('answer', '').upper() == 'YES'
                        
                        result = {
                            'id': exp_data['id'],
                            'question': exp_data['question'],
                            'answer': answer_data.get('answer', 'NO'),
                            'reason': answer_data.get('brief_reason', ''),
                            'is_relevant': is_relevant,
                            'refinements': []
                        }
                        
                        # If YES, include all refinements and suggestions
                        if is_relevant:
                            for ref_data in exp_data['refinements']:
                                ref_result = {
                                    'id': ref_data['id'],
                                    'question': ref_data['question'],
                                    'is_relevant': True,  # Assume relevant if parent is YES
                                    'suggestions': []
                                }
                                
                                # Include all suggestions for this refinement
                                for sug in ref_data['suggestions']:
                                    ref_result['suggestions'].append({
                                        'id': sug['id'],
                                        'text': sug['text'],
                                        'confidence': sug['confidence'],
                                        'is_applicable': True  # Apply all suggestions if parent is YES
                                    })
                                
                                result['refinements'].append(ref_result)
                        
                        mapped_results.append(result)
                
                return mapped_results
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print("Falling back to sequential processing")
                # Return empty list to trigger fallback
                return []
                
        except Exception as e:
            print(f"Error in batch LLM analysis: {e}")
            # Return empty list to trigger fallback
            return []
    
    
    def apply_qtree_suggestions(self, assertion: str, suggestions: List[Dict]) -> str:
        """
        Apply the selected suggestions to fix the assertion.
        
        Args:
            assertion: Original assertion
            suggestions: List of suggestions with reasoning
            
        Returns:
            Fixed assertion
        """
        print(f"\n=== Applying {len(suggestions)} QTree Suggestions ===")
        
        fixed_assertion = assertion
        
        # Sort suggestions by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply suggestions
        for i, suggestion in enumerate(suggestions):
            print(f"\nApplying suggestion {i+1}: {suggestion['suggestion'][:100]}...")
            print(f"Confidence: {suggestion['confidence']:.2f}")
            
            # Parse and apply the suggestion
            suggestion_text = suggestion['suggestion']
            
            # Try different application strategies
            if 'replace' in suggestion_text.lower():
                fixed_assertion = self._apply_replacement_suggestion(
                    fixed_assertion, suggestion_text
                )
            elif 'add' in suggestion_text.lower():
                fixed_assertion = self._apply_addition_suggestion(
                    fixed_assertion, suggestion_text
                )
            elif 'remove' in suggestion_text.lower():
                fixed_assertion = self._apply_removal_suggestion(
                    fixed_assertion, suggestion_text
                )
            else:
                # Try to apply as a direct transformation
                fixed_assertion = self._apply_direct_suggestion(
                    fixed_assertion, suggestion_text
                )
        
        return fixed_assertion
    
    def _apply_replacement_suggestion(self, assertion: str, suggestion: str) -> str:
        """Apply a replacement suggestion"""
        # Parse replacement pattern
        if "eventually" in suggestion.lower() and "strong" in suggestion.lower():
            if "eventually" in assertion and "strong" not in assertion:
                return assertion.replace("eventually", "strong(eventually)")
        
        # Add more replacement patterns as needed
        return assertion
    
    def _apply_addition_suggestion(self, assertion: str, suggestion: str) -> str:
        """Apply an addition suggestion"""
        # Implement addition logic
        return assertion
    
    def _apply_removal_suggestion(self, assertion: str, suggestion: str) -> str:
        """Apply a removal suggestion"""
        # Implement removal logic
        return assertion
    
    def _apply_direct_suggestion(self, assertion: str, suggestion: str) -> str:
        """Apply a direct transformation suggestion"""
        # For now, return the original assertion
        # In a full implementation, this would parse and apply the suggestion
        return assertion


# Utility functions for integration

def create_inference_engine(csv_path: str = None) -> QTreeInferenceEngine:
    """
    Create and initialize inference engine.
    
    Args:
        csv_path: Optional path to CSV file containing prompts
    
    Returns:
        Initialized QTreeInferenceEngine
    """
    # Use FLAGS.dataset_path if csv_path not provided
    if csv_path is None and hasattr(FLAGS, 'dataset_path'):
        csv_path = FLAGS.dataset_path
        print(f"Using FLAGS.dataset_path: {csv_path}")
    
    # Always use FLAGS.load_suggestions_path
    return QTreeInferenceEngine(FLAGS.load_suggestions_path, csv_path)


def test_qtree_inference():
    """Test the Q-Tree inference system"""
    # Create engine - will use FLAGS.dataset_path automatically
    engine = create_inference_engine()
    
    # Test case 1: With assertion provided
    print("\n=== TEST 1: With Assertion Provided ===")
    test_prompt = "When the FIFO is not empty, data must eventually be popped"
    test_assertion = """
    asrt_response_pending: assert property (@(posedge clk) disable iff (tb_reset)
        (!fifo_empty && !rd_pop) |-> eventually (rd_pop)
    );
    """
    
    # Fix assertion
    fixed, info = engine.fix_assertion_with_qtree(
        test_prompt, test_assertion
    )
    
    print(f"Original: {test_assertion}")
    print(f"Fixed: {fixed}")
    print(f"Mode: {info.get('no_assertion_mode', False)}")
    
    # Test case 2: Without assertion (Natural Language only)
    print("\n\n=== TEST 2: Natural Language Only (No Assertion) ===")
    test_prompt2 = "Check that valid signal is always followed by ready within 5 cycles"
    
    # Generate and fix assertion from Natural Language only
    fixed2, info2 = engine.fix_assertion_with_qtree(
        test_prompt2, assertion=None
    )
    
    print(f"Generated and Fixed: {fixed2}")
    print(f"Mode: {info2.get('no_assertion_mode', False)}")
    print(f"Generated from scratch: {info2.get('generated_from_scratch', False)}")


if __name__ == "__main__":
    test_qtree_inference()
