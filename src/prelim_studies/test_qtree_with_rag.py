#!/usr/bin/env python3
"""
Test QTree suggestions using RAG approach.
Uses the regular RAG prompt structure with use_RAG = True and RAG_content = ['Examples', 'Suggestions']
"""

import json
import os
import sys
import re
import tempfile
import shutil
import csv
from typing import List, Dict, Any, Tuple, Optional
# from dataclasses import dataclass, field  # Not available in Python 3.6
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np  # For embedding arrays
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLAGS
from FVEval.fv_eval import utils as utils2
from FVEval.fv_eval.fv_tool_execution import check_single_assertion_error
from FVEval.fv_eval import prompts_nl2sva_human
from utils import get_user, get_host, save_pickle
from utils_agent import llm_inference
from saver import Saver, saver
import pandas as pd
from rag_agent import RAGAgent
import pickle

# Import evaluate_response and related dependencies
from reproduce_bleu_bug import evaluate_response
from FVEval.fv_eval.data import InputData

# Override print with saver logging
print = saver.log_info

class Rule:
    """Represents a rule generated from the qtree"""
    def __init__(self, id, text, node_id, node_question, parent_chain=None, 
                 source_task_id=None, source_design=None, source_metrics=None):
        self.id = id
        self.text = text
        self.node_id = node_id
        self.node_question = node_question
        self.parent_chain = parent_chain or []
        self.source_task_id = source_task_id  # Which training task this rule came from
        self.source_design = source_design  # Which design it was for
        self.source_metrics = source_metrics or {}  # The metrics it achieved


class RAGTestResult:
    """Result of testing suggestions using RAG approach"""
    def __init__(self, test_id, task_id, design_name, prompt, rules_tested, 
                 new_metrics, fixed, improvements, sva, test_type="unknown", 
                 reference_sva=None, trace_id=None):
        self.test_id = test_id
        self.task_id = task_id
        self.design_name = design_name
        self.prompt = prompt
        self.rules_tested = rules_tested
        self.new_metrics = new_metrics
        self.fixed = fixed
        self.improvements = improvements
        self.sva = sva
        self.test_type = test_type
        self.reference_sva = reference_sva  # Reference/ground truth SVA
        self.trace_id = trace_id  # Unique identifier following qtree_choice pattern

class QTreeRAGTester:
    def __init__(self, qtree_file: str = None, 
                 test_individual_rules: bool = True,
                 test_all_rules_together: bool = True,
                 test_rules_by_node: bool = True,
                 test_top_level_rules: bool = True,
                 check_functionality: bool = True):
        """Initialize the RAG-based tester
        
        Args:
            qtree_file: Path to the qtree JSON file
            test_individual_rules: Whether to test each rule one at a time
            test_all_rules_together: Whether to test all rules together
            test_rules_by_node: Whether to test rules grouped by node
            test_top_level_rules: Whether to test only top-level rules
            check_functionality: Whether to check functionality using JasperGold (requires JasperGold installation)
        """
        
        # Test configuration
        self.test_individual_rules = test_individual_rules
        self.test_all_rules_together = test_all_rules_together
        self.test_rules_by_node = test_rules_by_node
        self.test_top_level_rules = test_top_level_rules
        self.check_functionality = check_functionality
        
        # Use FLAGS configuration
        self.only_test_ids = FLAGS.only_test_ids if hasattr(FLAGS, 'only_test_ids') else [78]
        
        # Determine qtree file path
        if qtree_file:
            self.qtree_file = qtree_file
        else:
            # Use the specific qtree file mentioned in the conversation
            self.qtree_file = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-22T09-03-15.426631_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtrees.json"
                
        print(f"Using qtree file: {self.qtree_file}")
        print(f"Testing IDs: {self.only_test_ids}")
        print(f"RAG Configuration: use_RAG={FLAGS.use_RAG if hasattr(FLAGS, 'use_RAG') else False}, RAG_content={FLAGS.RAG_content if hasattr(FLAGS, 'RAG_content') else []}")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"qtree_rag_test_{get_user()}_{get_host()}_")
        
        # Results storage
        self.test_results = []
        self.successful_fixes = []
        
        # Load qtree data
        with open(self.qtree_file, 'r') as f:
            self.qtree_data = json.load(f)
            
        # Initialize RAG agent for examples if configured
        self.example_rag_agent = None
        self.training_ids_for_rules = []     # Track which training IDs are used for rules
        
        # Initialize training examples from specified IDs
        self.training_examples = []
        if FLAGS.use_RAG and 'Examples' in FLAGS.RAG_content:
            self._load_training_examples()

        # breakpoint()
            
        # Load CSV dataset to get proper row mappings
        print(f"Loading CSV dataset from: {FLAGS.dataset_path}")
        self.csv_df = pd.read_csv(FLAGS.dataset_path)
        
        # Create mapping from design_name/prompt to CSV row index
        self.csv_row_mapping = {}
        for idx, row in self.csv_df.iterrows():
            # Store multiple keys for robust matching
            design_name = row['design_name'] if 'design_name' in row else ''
            prompt = str(row['prompt'] if 'prompt' in row else '').strip()
            key = (design_name, prompt)
            self.csv_row_mapping[key] = idx
            
        print(f"Loaded {len(self.csv_df)} rows from CSV")
        
        # Log which CSV rows are available in qtrees.json
        available_csv_rows = set()
        for entry in self.qtree_data:
            design_name = entry.get('design_name', '')
            prompt = entry.get('prompt', '').strip()
            key = (design_name, prompt)
            csv_row = self.csv_row_mapping.get(key, -1)
            if csv_row >= 0:
                available_csv_rows.add(csv_row)
                
        print(f"Available CSV rows in qtrees.json: {sorted(available_csv_rows)}")
        print(f"Requested CSV rows: {self.only_test_ids}")
        
        # Check if requested rows are available
        missing_rows = set(self.only_test_ids) - available_csv_rows
        if missing_rows:
            print(f"WARNING: Requested CSV rows not found in qtrees.json: {sorted(missing_rows)}")
    
    def _load_training_examples(self):
        """Load training examples from specified training IDs"""
        training_IDs = [73, 28, 65, 36, 16, 59, 42, 47, 32, 45, 60, 4, 34, 20, 37, 61, 69, 1, 46, 54, 43, 72, 48, 0, 56, 7, 51, 41, 67, 24, 17, 35, 29, 25, 57, 63, 40, 23, 75, 12, 9, 11, 13, 49, 53, 19, 70, 21, 26, 6, 33, 76, 10, 15, 68, 14, 64, 55, 44, 50, 22, 77, 58, 18]

        # Extract all NL (natural language prompt) and reference SVA from the dataset for the specified training IDs
        self.nl_and_sva_pairs = []
        # Note: self.csv_df is loaded after this method is called in __init__
        df = pd.read_csv(FLAGS.dataset_path)
        for csv_row_id in training_IDs:
            if csv_row_id >= len(df):
                continue
            row = df.iloc[csv_row_id]
            # breakpoint()
            prompt = str(row['prompt'] if 'prompt' in row else '').strip()
            # Try to get the reference SVA from the CSV or from qtree_data
            reference_sva = row.get('ref_solution', None)
            if reference_sva and isinstance(reference_sva, str) and reference_sva.strip():
                self. training_examples.append({
                    'csv_row_id': csv_row_id,
                    'prompt': prompt,
                    'reference_sva': reference_sva
                })
        print(f"Loading training examples from {len(training_IDs)} specified training IDs...")
    
    def find_similar_examples(self, target_prompt: str, target_design: str, top_k: int = 2) -> List[Dict]:
        """Find most similar examples from training data using embedding-based similarity"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Initialize embedding model if not already done
        if not hasattr(self, 'embedding_model'):
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Initialized SentenceTransformer for similarity calculation")
            
            # Pre-compute embeddings for all training examples
            self.example_embeddings = {}
            prompts = []
            csv_ids = []
            for example in self.training_examples:
                prompts.append(example['prompt'])
                csv_ids.append(example['csv_row_id'])
            
            if prompts:
                # Batch encode all prompts
                embeddings = self.embedding_model.encode(prompts, convert_to_tensor=False, show_progress_bar=False)
                for csv_id, embedding in zip(csv_ids, embeddings):
                    self.example_embeddings[csv_id] = embedding
                print(f"Pre-computed embeddings for {len(self.example_embeddings)} training examples")
        
        # Encode the target prompt
        target_embedding = self.embedding_model.encode([target_prompt], convert_to_tensor=False, show_progress_bar=False)[0]
        
        similarities = []
        
        for example in self.training_examples:
            csv_id = example['csv_row_id']
            if csv_id in self.example_embeddings:
                # Calculate cosine similarity
                example_embedding = self.example_embeddings[csv_id]
                cosine_sim = cosine_similarity([target_embedding], [example_embedding])[0][0]
                
                similarities.append((cosine_sim, csv_id, example))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Print similarity rankings
        # print(f"\n  Similarity Rankings for: '{target_prompt[:80]}...'")
        # print(f"  Target Design: {target_design}")
        # print(f"  {'Rank':<6} {'Score':<8} {'CSV ID':<8} {'Task ID':<20} {'Design':<30} {'Prompt':<60}")
        # print("  " + "="*140)
        # Show top 20 rankings (or all if less than 20)
        # num_rankings_to_show = 40  # Can be made configurable via FLAGS if needed
        # num_to_show = min(num_rankings_to_show, len(similarities))
        # for rank, (score, csv_id, example) in enumerate(similarities[:num_to_show], 1):
        #     # Get design name and task ID from CSV data
        #     if hasattr(self, 'csv_df') and csv_id < len(self.csv_df):
        #         row = self.csv_df.iloc[csv_id]
        #         design_name = str(row.get('design_name', 'Unknown'))[:28] + '...' if len(str(row.get('design_name', 'Unknown'))) > 28 else str(row.get('design_name', 'Unknown'))
        #         task_id = str(row.get('task_id', f'row_{csv_id}'))[:18] + '...' if len(str(row.get('task_id', f'row_{csv_id}'))) > 18 else str(row.get('task_id', f'row_{csv_id}'))
        #     else:
        #         design_name = 'N/A'
        #         task_id = f'row_{csv_id}'
        #     prompt_preview = example['prompt'][:58] + '...' if len(example['prompt']) > 58 else example['prompt']
        #     print(f"  {rank:<6} {score:<8.4f} {csv_id:<8} {task_id:<20} {design_name:<30} {prompt_preview}")
        # if len(similarities) > num_to_show:
        #     print(f"  ... and {len(similarities) - num_to_show} more training examples")
        # print(f"\n  Selected top {top_k} most similar examples (scores: {', '.join(f'{s[0]:.4f}' for s in similarities[:top_k])})")
        
        # breakpoint()

        # Return top k examples
        results = []
        for score, csv_id, example in similarities[:top_k]:
            # breakpoint()
            results.append({
                'prompt': example['prompt'],
                'reference_sva': example['reference_sva']
            })
            
        return results
        
    def collect_all_training_rules(self) -> List[Rule]:
        """Collect all rules from SUCCESSFUL training qtree entries (pec=1.0 or bleu=1.0) - separate from examples"""
        all_rules = []
        successful_entries = 0
        unique_training_ids = set()
        
        for entry_idx, entry in enumerate(self.qtree_data):
            task_id = entry.get('task_id', '')
            design_name = entry.get('design_name', '')
            prompt = entry.get('prompt', '').strip()
            
            # Find CSV row index for this entry
            key = (design_name, prompt)
            csv_row = self.csv_row_mapping.get(key, -1)
            
            # Check if this entry was successful (pec=1.0 or bleu=1.0)
            final_metrics = entry.get('final_metrics', {})
            pec = final_metrics.get('pec', 0)
            bleu = final_metrics.get('bleu', 0)
            
            if pec != 1.0 and bleu != 1.0:
                # Skip non-successful entries
                continue
            
            # Only use training entries (not test entries) for rules
            is_training = csv_row not in self.only_test_ids
            if not is_training:
                continue
                
            successful_entries += 1
            unique_training_ids.add((task_id, csv_row))
            
            # Extract rules from qtree nodes
            qtree_data = entry.get('qtree_data', {})
            nodes = qtree_data.get('nodes', [])
            
            for node in nodes:
                if node.get('rules_generated'):
                    for rule_text in node['rules_generated']:
                        parent_chain = self.extract_tree_path(nodes, node['id'])
                        rule = Rule(
                            id=f"{task_id}_{node['id']}_{len(all_rules)}",
                            text=rule_text,
                            node_id=node['id'],
                            node_question=node.get('question', ''),
                            parent_chain=parent_chain,
                            source_task_id=task_id,
                            source_design=design_name,
                            source_metrics=final_metrics
                        )
                        all_rules.append(rule)
            
            # Also add top-level rules if any
            # top_level_rules = qtree_data.get('rules', [])
            # for idx, rule_text in enumerate(top_level_rules):
            #     rule = Rule(
            #         id=f"{task_id}_top_{idx}",
            #         text=rule_text,
            #         node_id='top_level',
            #         node_question='Top-level rule',
            #         parent_chain=['Top-level'],
            #         source_task_id=task_id,
            #         source_design=design_name,
            #         source_metrics=final_metrics
            #     )
            #     all_rules.append(rule)
        
        # Track which training IDs are used for rules
        self.training_ids_for_rules = list(unique_training_ids)
        
        print(f"Collected {len(all_rules)} total rules from {successful_entries} successful TRAINING entries")
        print(f"Rules collected from {len(unique_training_ids)} unique training task IDs")
        
        # breakpoint()

        # Log summary of what's used for what
        if FLAGS.debug:
            print("\n" + "="*60)
            print("TRAINING DATA USAGE SUMMARY:")
            print("="*60)
            print(f"Training IDs used for RULES: {len(self.training_ids_for_rules)}")
        
        return all_rules
        
    def extract_tree_path(self, nodes: List[Dict], node_id: str) -> List[str]:
        """Extract the full tree path for a node"""
        node_map = {n['id']: n for n in nodes}
        path = []
        current_id = node_id
        
        while current_id and current_id in node_map:
            node = node_map[current_id]
            path.append(f"{node['id']}: {node['question']}")  # Full question text
            current_id = node.get('parent_id')
            
        return list(reversed(path))
    
    def build_rag_prompt(self, original_prompt: str, testbench: str, suggestions: List[str], examples: List[str] = None) -> str:
        """Build prompt using RAG style with suggestions (and optionally examples)"""
        
        prompt = ""
        if FLAGS.num_icl == 1:
            prompt += prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_1
        elif FLAGS.num_icl == 3:
            prompt += prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_3
        
        # Add testbench preamble and testbench - this must come right after ICL examples
        if prompt:  # If we had ICL examples, add separator
            prompt += "\n\n"
        prompt += prompts_nl2sva_human.SVAGEN_TB_PREAMBLE
        prompt += "\n" + testbench
        
        # Debug: check if testbench was added
        if FLAGS.debug and not testbench:
            print("[DEBUG] WARNING: Testbench is empty!")
    
        # Add examples if provided (from RAG retrieval)
        if examples and FLAGS.RAG_content and 'Examples' in FLAGS.RAG_content:
            prompt += "\n\nAdditional context from similar documents:\n"
            prompt += "\n\n".join(f"{example}" for example in examples)
        
        # Add suggestions as additional context before the question
        if suggestions and FLAGS.RAG_content and 'Suggestions' in FLAGS.RAG_content:
            prompt += "\n\nAdditional suggestions that may help improve your assertion:\n"
            for suggestion in suggestions:
                prompt += f"{suggestion}\n"
        
        # Add the question using proper format
        prompt += "\n"  # Add separator before question
        prompt += prompts_nl2sva_human.SVAGEN_QUESTION_PREAMBLE
        prompt += original_prompt + "\n"
        prompt += prompts_nl2sva_human.SVAGEN_QUESTION_POSTAMBLE
        
        # Save prompt for analysis (you can also copy these to saver's directory if needed)
        prompt_file = os.path.join(self.temp_dir, f"rag_prompt_{len(self.test_results)}.txt")
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        # print(f"\nFull prompt saved to: {prompt_file}")

        # breakpoint()
        
        return prompt
    
    def extract_sva_from_response(self, response: str) -> str:
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
    
    def test_with_rag_for_test_case(self, test_id: int, test_row: Dict, rule_group: List[Rule], test_type: str = "training_rules") -> Optional[RAGTestResult]:
        """Test suggestions on a specific test case using RAG approach"""
        # Extract test case information
        design_name = test_row['design_name'] if 'design_name' in test_row else ''
        prompt = str(test_row['prompt'] if 'prompt' in test_row else '').strip()
        testbench = test_row['testbench'] if 'testbench' in test_row else (test_row['tb'] if 'tb' in test_row else '')
        # Try to get reference SVA from test row (ground truth)
        reference_sva = test_row.get('ref_solution', None)
        # Get the actual task_id from the row
        task_id = test_row.get('task_id', None)

        # print(f"reference_sva: {reference_sva}")

        # breakpoint()
        
        # Retrieve most similar examples
        examples = []
        Examples_top_k = FLAGS.Examples_top_k
        
        # Find similar examples from training data
        similar_examples = self.find_similar_examples(prompt, design_name, top_k=Examples_top_k)
        
        # Format examples like in hardware_general_agent.py
        for ex in similar_examples:
            formatted_example = f'Question: Create a SVA assertion that checks: {ex["prompt"]}\nReference Answer: {ex["reference_sva"]}'
            examples.append(formatted_example)
            
        print(f"  Retrieved {len(examples)} most similar examples from {len(self.training_examples)} training examples")
        
        # Build RAG prompt with the suggestions (extract text from Rule objects) and examples
        suggestion_texts = [rule.text for rule in rule_group]
        
        # Always show a quick summary of rule texts being used
        print(f"\n  Using {len(rule_group)} rules as suggestions:")
        for i, rule in enumerate(rule_group[:5], 1):  # Show first 5
            print(f"    {i}. {rule.text[:100]}{'...' if len(rule.text) > 100 else ''}")
        if len(rule_group) > 5:
            print(f"    ... and {len(rule_group) - 5} more rules")
        
        rag_prompt = self.build_rag_prompt(prompt, testbench, suggestion_texts, examples=examples)
        
        # Add breakpoint before LLM call
        # if FLAGS.debug:
        #     print("\n[BREAKPOINT] About to call LLM with RAG prompt")
        #     breakpoint()
            
        # Generate new assertion using LLM with RAG prompt
        try:
            response = llm_inference(
                system_prompt=prompts_nl2sva_human.SVAGEN_HEADER,
                user_prompt=rag_prompt
            )
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
            
        # Extract SVA from response
        # generated_sva = self.extract_sva_from_response(response)
        # Add breakpoint after SVA extraction
        # if FLAGS.debug:
        #     print(f"\n[BREAKPOINT] Extracted SVA: {generated_sva}")
            # breakpoint()
            
        # Test the generated assertion
        new_metrics = self.test_assertion(response, testbench, prompt, design_name, 
                                         task_id, reference_sva=reference_sva)
        
        # Check if fixed (all functionality metrics = 1.0)
        fixed = (
            new_metrics.get('pec', 0) == 1.0
        )
        
        # Track improvements
        improvements = {}
        # Since this is a test case, we might not have original metrics
        # Just report the achieved metrics
        for metric in ['syntax', 'pec', 'relax_pec']:
            val = new_metrics.get(metric, 0)
            if val > 0:
                improvements[metric] = f"achieved {val}"
        
        # breakpoint()
        
        # Generate trace_id following the pattern from test_qtree_choice.py
        # trace_id should identify the SOURCE of the rule, not the test case
        # Use source design_name/task_id/rule_id for uniqueness
        
        if len(rule_group) == 1:
            # Single rule: use the source information
            rule = rule_group[0]
            safe_source_design = rule.source_design.replace('/', '_').replace(' ', '_').replace(':', '_')
            trace_id = f"{safe_source_design}::{rule.source_task_id}::{rule.node_id}"
        else:
            # Multiple rules: check if they're all from the same source
            unique_sources = set((r.source_design, r.source_task_id) for r in rule_group)
            unique_nodes = set(r.node_id for r in rule_group)
            
            if len(unique_sources) == 1:
                # All rules from same source
                source_design, source_task_id = unique_sources.pop()
                safe_source_design = source_design.replace('/', '_').replace(' ', '_').replace(':', '_')
                if len(unique_nodes) == 1:
                    # Same node too
                    trace_id = f"{safe_source_design}::{source_task_id}::{rule_group[0].node_id}"
                else:
                    # Different nodes from same source
                    trace_id = f"{safe_source_design}::{source_task_id}::multi_nodes_{len(rule_group)}_rules"
            else:
                # Rules from multiple sources - create composite identifier
                trace_id = f"multi_source::{test_type}_{len(rule_group)}_rules_from_{len(unique_sources)}_sources"
                
        result = RAGTestResult(
            test_id=test_id,
            task_id=task_id,
            design_name=design_name,
            prompt=prompt,
            # rules_tested=[rule.text for rule in rule_group],
            rules_tested=rule_group,
            new_metrics=new_metrics,
            fixed=fixed,
            improvements=improvements,
            test_type=test_type,
            sva = self.extract_sva_from_response(response),
            reference_sva=reference_sva,
            trace_id=trace_id
        )
        
        # Log result - more detailed with BLEU score
        if fixed:
            print(f"  ✓ FIXED! (pec={new_metrics.get('pec', 0)}, syntax={new_metrics.get('syntax', 0)}, bleu={new_metrics.get('bleu', 0):.3f})")
            self.successful_fixes.append(result)
        else:
            print(f"  ✗ Failed (pec={new_metrics.get('pec', 0)}, bleu={new_metrics.get('bleu', 0):.3f})")
            
        return result
    
    def test_assertion(self, sva_code: str, testbench: str, prompt: str, design_name: str, 
                      task_id: str, reference_sva: str = None) -> Dict[str, Any]:
        """Test an assertion and return comprehensive metrics using evaluate_response"""
        
        # Create InputData object for evaluate_response
        row = InputData(
            task_id=task_id,
            prompt=prompt,
            testbench=testbench,
            design_name=design_name,
            ref_solution=reference_sva,  # Use reference SVA if available
        )
        
        # If no reference SVA provided, use the generated SVA as reference for self-comparison
        # This allows us to at least get syntax checking even without ground truth
        
        # First, evaluate without functionality checking to get BLEU score
        metrics = evaluate_response(
            response=sva_code,
            reference=reference_sva,
            row=row,
            check_functionality=False
        )
        print(f"response: {sva_code}")
        print(f"reference: {reference_sva}")
        print(f"metrics: {metrics}")
        # breakpoint()
        
        # If BLEU = 1.0, skip functionality checking as the assertion is identical to reference
        if metrics.get('bleu', 0) == 1.0:
            print("  BLEU = 1.0 (perfect match), skipping JasperGold functionality check")
            # Set functionality metrics to 1.0 since we have perfect match
            metrics['syntax'] = 1.0
            metrics['pec'] = 1.0
            metrics['relax_pec'] = 1.0
        else:
            # BLEU != 1.0, run functionality check if configured
            if self.check_functionality:

                metrics = evaluate_response(
                    response=sva_code,
                    reference=reference_sva,
                    row=row,
                    check_functionality=True
                )
        
        # breakpoint()
        return metrics

    def run(self):
        """Run RAG-based tests: test all training rules on test cases"""
        all_results = []
        
        # Load test dataset
        df = pd.read_csv(FLAGS.dataset_path)
        
        # Debug: Check CSV columns and sample data
        if FLAGS.debug:
            print(f"\n[DEBUG] CSV columns: {list(df.columns)}")
            if len(df) > 0 and self.only_test_ids[0] < len(df):
                print(f"[DEBUG] Row {self.only_test_ids} data:")
                for col in df.columns:
                    val = df.iloc[self.only_test_ids[0]][col]
                    if isinstance(val, str) and len(val) > 100:
                        print(f"  - {col}: {val[:100]}... (length: {len(val)})")
                    else:
                        print(f"  - {col}: {val}")
        
        # Show training examples loaded
        print(f"\n{'='*60}")
        print(f"TRAINING EXAMPLES DATABASE: {len(self.training_examples)} examples loaded")
        print(f"{'='*60}")
        
        # Collect ALL rules from training qtrees
        print("\nCollecting successful training rules...")
        all_training_rules = self.collect_all_training_rules()

        # breakpoint()
        
        # For each test case we want to test
        for test_id in self.only_test_ids:
            if test_id >= len(df):
                print(f"Test ID {test_id} out of range (CSV has {len(df)} rows)")
                continue
                
            # Get the test case from CSV
            test_row = df.iloc[test_id]
            
            # Test different combinations of training rules on this test case:
            
            # 1. Test each rule individually (one at a time)
            if self.test_individual_rules:
                print("\n" + "-"*60)
                print("TESTING EACH TRAINING RULE INDIVIDUALLY")
                print("-"*60)
                
                for i, rule in enumerate(all_training_rules, 1):
                    # Concise one-line output
                    print(f"\nRule {i}/{len(all_training_rules)}: {rule.text[:80]}... (from {rule.source_task_id})")
                    
                    result = self.test_with_rag_for_test_case(test_id, test_row, [rule], 
                                                                test_type="individual_rule")
                    if result:
                        all_results.append(result)
                        # Success message already logged in test_with_rag_for_test_case
            
            # 2. Test all rules together
            if self.test_all_rules_together:
                print("\n" + "-"*60)
                print("TESTING ALL TRAINING RULES TOGETHER")
                print("-"*60)
                
                if all_training_rules:
                    result = self.test_with_rag_for_test_case(test_id, test_row, all_training_rules,
                                                                test_type="all_rules")
                    if result:
                        all_results.append(result)
            
            # 3. Test rules by node
            if self.test_rules_by_node:
                print("\n" + "-"*60)
                print("TESTING RULES GROUPED BY NODE")
                print("-"*60)
                
                nodes_rules = defaultdict(list)
                for rule in all_training_rules:
                    if rule.node_id != "top_level":
                        nodes_rules[rule.node_id].append(rule)
                
                for node_id, node_rules in nodes_rules.items():
                    if len(node_rules) >= 2:  # Test nodes with multiple rules
                        print(f"\nTesting {len(node_rules)} rules from node {node_id}")
                        result = self.test_with_rag_for_test_case(test_id, test_row, node_rules,
                                                                        test_type=f"node_{node_id}")
                        if result:
                            all_results.append(result)
            
            # 4. Test top-level rules only
            if self.test_top_level_rules:
                print("\n" + "-"*60)
                print("TESTING TOP-LEVEL RULES ONLY")
                print("-"*60)
                
                top_rules = [r for r in all_training_rules if r.node_id == "top_level"]

                if top_rules:
                    print(f"\nTesting {len(top_rules)} top-level rules only")
                    result = self.test_with_rag_for_test_case(test_id, test_row, top_rules,
                                                                test_type="top_level_rules")
                    if result:
                        all_results.append(result)
            
        self.test_results = all_results
        return all_results
    
    def generate_report(self):
        """Generate a comprehensive report"""
        report = []
        report.append("=" * 80)
        report.append("QTREE RAG TESTING REPORT")
        report.append("=" * 80)
        report.append(f"\nConfiguration:")
        report.append(f"  Task: {FLAGS.task}")
        report.append(f"  Model: {FLAGS.llm_model}")
        report.append(f"  Test IDs: {self.only_test_ids}")
        report.append(f"  QTree file: {self.qtree_file}")
        report.append(f"  RAG Settings: use_RAG=True, RAG_content=['Examples', 'Suggestions']")
        report.append(f"  Check Functionality (JasperGold): {self.check_functionality}")
        
        # Add training data usage summary
        report.append(f"\nTraining Data Usage:")
        report.append(f"  Training examples loaded: {len(self.training_examples)} (from specified training IDs)")
        report.append(f"  Training entries used for RULES: {len(self.training_ids_for_rules)}")
        
        report.append(f"\nResults Summary:")
        report.append(f"  Total RAG tests: {len(self.test_results)}")
        report.append(f"  Successful fixes: {len(self.successful_fixes)}")
        
        # Detailed results for successful fixes
        if self.successful_fixes:
            report.append("\n" + "="*60)
            report.append("SUCCESSFUL RAG FIXES")
            report.append("="*60)
            
            for fix in self.successful_fixes:
                report.append(f"\nTest ID {fix.test_id} - {fix.task_id}")
                report.append(f"Trace ID: {fix.trace_id}")
                report.append(f"Design: {fix.design_name}")
                report.append(f"Prompt: {fix.prompt}")  # Full prompt
                report.append(f"\nNumber of suggestions used: {len(fix.rules_tested)}")
                report.append("Suggestions used:")
                for rule in fix.rules_tested[:5]:  # Show first 5
                    report.append(f"  - {rule.text}")  # Full rule text
                    report.append(f"    From: {rule.source_task_id} (pec={rule.source_metrics.get('pec', 0)})")
                    report.append(f"    Branch: {' → '.join(rule.parent_chain)}")  # Full branch path
                if len(fix.rules_tested) > 5:
                    report.append(f"  ... and {len(fix.rules_tested) - 5} more")
                report.append("\nMetrics achieved:")
                report.append(f"  BLEU: {fix.new_metrics.get('bleu', 0):.3f}")
                report.append(f"  ROUGE-L: {fix.new_metrics.get('rouge', 0):.3f}")
                report.append(f"  Exact Match: {fix.new_metrics.get('exact_match', 0)}")
                report.append(f"  Syntax: {fix.new_metrics.get('syntax', 0)}")
                report.append(f"  PEC: {fix.new_metrics.get('pec', 0)}")
                report.append(f"  Relaxed PEC: {fix.new_metrics.get('relax_pec', 0)}")
                
                report.append("\nMetric improvements:")
                for metric, improvement in fix.improvements.items():
                    report.append(f"  {metric}: {improvement}")
                    
                if fix.reference_sva:
                    report.append(f"\nReference SVA (ground truth):")
                    report.append(fix.reference_sva)
                report.append(f"\nGenerated SVA (with RAG):")
                report.append(fix.sva)
                    
        # Summary of which suggestion groups work best
        report.append("\n\n" + "="*80)
        report.append("EFFECTIVENESS SUMMARY")
        report.append("="*80)
        
        # Separate individual rule fixes
        individual_fixes = [r for r in self.test_results if r.fixed and len(r.rules_tested) == 1]
        if individual_fixes:
            report.append(f"\nINDIVIDUAL RULES THAT FIX THE ISSUE ({len(individual_fixes)} total):")
            for fix in individual_fixes:
                rule = fix.rules_tested[0]
                report.append(f"\n  ✓ Rule: {rule.text}")
                report.append(f"    From QTree: {rule.source_task_id} (design: {rule.source_design})")
                report.append(f"    Branch: {' → '.join(rule.parent_chain)}")
                report.append(f"    Node: {rule.node_id} - {rule.node_question}")  # Full node question
                report.append(f"    Source achieved: pec={rule.source_metrics.get('pec', 0)}, bleu={rule.source_metrics.get('bleu', 0)}")
                report.append(f"    Test metrics: {fix.new_metrics}")
        
        # Group by number of suggestions
        fixes_by_count = defaultdict(int)
        for result in self.test_results:
            if result.fixed:
                fixes_by_count[len(result.rules_tested)] += 1
                
        if fixes_by_count:
            report.append("\n\nFixes by number of suggestions used:")
            for count, num_fixes in sorted(fixes_by_count.items()):
                if count == 1:
                    report.append(f"  {count} suggestion (individual rules): {num_fixes} fixes")
                else:
                    report.append(f"  {count} suggestions: {num_fixes} fixes")
                
        # Group by node type
        fixes_by_node_type = defaultdict(int)
        for result in self.test_results:
            if result.fixed:
                if all(r.node_id == "top_level" for r in result.rules_tested):
                    fixes_by_node_type["Top-level rules only"] += 1
                elif all(r.node_id != "top_level" for r in result.rules_tested):
                    fixes_by_node_type["Node-specific rules"] += 1
                else:
                    fixes_by_node_type["Mixed rules"] += 1
                    
        if fixes_by_node_type:
            report.append("\nFixes by rule type:")
            for node_type, count in fixes_by_node_type.items():
                report.append(f"  {node_type}: {count} fixes")
        
        # Track which source QTrees and branches contributed to fixes
        source_contributions = defaultdict(lambda: {'successful_fixes': 0})
        branch_contributions = defaultdict(int)
        
        for result in self.test_results:
            if result.fixed:
                for rule in result.rules_tested:
                    # Use composite key for unique identification
                    source_key = (rule.source_task_id, rule.source_design)
                    source_contributions[source_key]['successful_fixes'] += 1
                    # Track branch effectiveness with design name
                    branch_key = f"{rule.source_task_id}_{rule.source_design}: {' → '.join(rule.parent_chain)}"
                    branch_contributions[branch_key] += 1
        
        if source_contributions:
            report.append(f"\n\nMost effective source QTrees:")
            sorted_sources = sorted(source_contributions.items(), 
                                  key=lambda x: x[1]['successful_fixes'], 
                                  reverse=True)
            for source_key, data in sorted_sources[:10]:  # Top 10
                if data['successful_fixes'] > 0:
                    task_id, design_name = source_key
                    report.append(f"  {task_id} ({design_name}): {data['successful_fixes']} successful uses")
        
        if branch_contributions:
            report.append(f"\n\nMost effective branches:")
            sorted_branches = sorted(branch_contributions.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
            for branch, count in sorted_branches[:10]:  # Top 10
                report.append(f"  {branch}: {count} successful uses")
                
        return "\n".join(report)
    
    def generate_csv_summary(self, output_file: str):
        """Generate a CSV summary of results"""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'trace_id', 'test_id', 'task_id', 'design_name', 'num_suggestions', 'rule_types',
                'new_syntax', 'new_pec', 'new_relax_pec', 
                'bleu', 'rouge', 'exact_match',
                'fixed', 'improvements', 'suggestions_preview', 'source_qtrees', 'branches'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.test_results:
                # Determine rule types
                rule_types = "Mixed"
                if all(r.node_id == "top_level" for r in result.rules_tested):
                    rule_types = "Top-level"
                elif all(r.node_id != "top_level" for r in result.rules_tested):
                    rule_types = "Node-specific"
                    
                # Create suggestions preview
                suggestions_preview = " | ".join(r.text for r in result.rules_tested[:3])
                if len(result.rules_tested) > 3:
                    suggestions_preview += f" (+{len(result.rules_tested) - 3} more)"
                
                # Get unique source QTrees with design names
                source_qtrees_set = set()
                for r in result.rules_tested:
                    # Create unique identifier with both task_id and design_name
                    source_id = f"{r.source_task_id}_{r.source_design}"
                    source_qtrees_set.add(source_id)
                source_qtrees = list(source_qtrees_set)
                source_qtrees_str = ", ".join(source_qtrees[:5])  # Show first 5
                if len(source_qtrees) > 5:
                    source_qtrees_str += f" (+{len(source_qtrees) - 5} more)"
                
                # Get unique branches
                unique_branches = []
                seen_branches = set()
                for r in result.rules_tested:
                    branch_str = ' → '.join(r.parent_chain[:2])
                    if branch_str not in seen_branches:
                        unique_branches.append(branch_str)
                        seen_branches.add(branch_str)
                branches_str = " | ".join(unique_branches[:3])
                if len(unique_branches) > 3:
                    branches_str += f" (+{len(unique_branches) - 3} more)"
                
                row = {
                    'trace_id': result.trace_id,
                    'test_id': result.test_id,
                    'task_id': result.task_id,
                    'design_name': result.design_name,
                    'num_suggestions': len(result.rules_tested),
                    'rule_types': rule_types,
                    'new_syntax': result.new_metrics.get('syntax', 0),
                    'new_pec': result.new_metrics.get('pec', 0),
                    'new_relax_pec': result.new_metrics.get('relax_pec', 0),
                    'bleu': result.new_metrics.get('bleu', 0),
                    'rouge': result.new_metrics.get('rouge', 0),
                    'exact_match': result.new_metrics.get('exact_match', 0),
                    'fixed': result.fixed,
                    'improvements': '; '.join(f"{k}: {v}" for k, v in result.improvements.items()),
                    'suggestions_preview': suggestions_preview,
                    'source_qtrees': source_qtrees_str,
                    'branches': branches_str
                }
                writer.writerow(row)
    
    def generate_plots(self, output_dir: str):
        """Generate visualization plots for the test results"""
        # Set style - try multiple options for compatibility
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                # Use default if seaborn styles not available
                pass
        
        # 1. Success Rate by Test Type
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        test_type_stats = defaultdict(lambda: {'total': 0, 'fixed': 0})
        
        for result in self.test_results:
            test_type_stats[result.test_type]['total'] += 1
            if result.fixed:
                test_type_stats[result.test_type]['fixed'] += 1
        
        test_types = list(test_type_stats.keys())
        success_rates = [test_type_stats[tt]['fixed'] / test_type_stats[tt]['total'] * 100 
                        if test_type_stats[tt]['total'] > 0 else 0 
                        for tt in test_types]
        totals = [test_type_stats[tt]['total'] for tt in test_types]
        
        bars = ax1.bar(test_types, success_rates)
        ax1.set_xlabel('Test Type')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate by Test Configuration')
        
        # Add count labels on bars
        for bar, total, rate in zip(bars, totals, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(rate)}%\n(n={total})',
                    ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'plot_success_rate_by_type.png'), dpi=150)
        plt.close(fig1)
        
        # 2. Distribution of Number of Rules in Successful Fixes
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        successful_rule_counts = [len(r.rules_tested) for r in self.test_results if r.fixed]
        
        if successful_rule_counts:
            ax2.hist(successful_rule_counts, bins=20, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Number of Rules Used')
            ax2.set_ylabel('Number of Successful Fixes')
            ax2.set_title('Distribution of Rule Count in Successful Fixes')
            ax2.axvline(np.mean(successful_rule_counts), color='red', linestyle='dashed', 
                       linewidth=2, label=f'Mean: {np.mean(successful_rule_counts):.1f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No successful fixes found', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Rule Count in Successful Fixes')
        
        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'plot_rule_count_distribution.png'), dpi=150)
        plt.close(fig2)
        
        # 3. Metrics Comparison (BLEU, PEC, Syntax)
        fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_to_plot = ['bleu', 'pec', 'syntax', 'relax_pec']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            # Group by test type
            metric_by_type = defaultdict(list)
            for result in self.test_results:
                metric_val = result.new_metrics.get(metric, 0)
                metric_by_type[result.test_type].append(metric_val)
            
            # Create box plot
            data_to_plot = []
            labels = []
            for test_type, values in metric_by_type.items():
                if values:
                    data_to_plot.append(values)
                    labels.append(f"{test_type}\n(n={len(values)})")
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} Score by Test Type')
                ax.set_ylim(0, 1.1)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric.upper()} Score by Test Type')
        
        plt.suptitle('Metrics Comparison Across Test Types', fontsize=16)
        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'plot_metrics_comparison.png'), dpi=150)
        plt.close(fig3)
        
        # 4. Rule Source Effectiveness
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        source_effectiveness = defaultdict(lambda: {'used': 0, 'successful': 0})
        
        for result in self.test_results:
            for rule in result.rules_tested:
                source_key = f"{rule.source_task_id}_{rule.source_design}"
                source_effectiveness[source_key]['used'] += 1
                if result.fixed:
                    source_effectiveness[source_key]['successful'] += 1
        
        # Sort by success count
        sorted_sources = sorted(source_effectiveness.items(), 
                               key=lambda x: x[1]['successful'], 
                               reverse=True)[:20]  # Top 20
        
        if sorted_sources:
            source_names = [s[0] for s in sorted_sources]
            success_counts = [s[1]['successful'] for s in sorted_sources]
            total_counts = [s[1]['used'] for s in sorted_sources]
            
            x = np.arange(len(source_names))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, total_counts, width, label='Total Uses', alpha=0.7)
            bars2 = ax4.bar(x + width/2, success_counts, width, label='Successful Uses', alpha=0.7)
            
            ax4.set_xlabel('Source (Task ID + Design)')
            ax4.set_ylabel('Count')
            ax4.set_title('Top 20 Most Effective Rule Sources')
            ax4.set_xticks(x)
            ax4.set_xticklabels(source_names, rotation=90, ha='right')
            ax4.legend()
            
            # Add effectiveness percentage
            for i, (total, success) in enumerate(zip(total_counts, success_counts)):
                if total > 0:
                    effectiveness = success / total * 100
                    ax4.text(i, max(total, success) + 0.5, f'{effectiveness:.0f}%', 
                            ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No rule usage data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Rule Source Effectiveness')
        
        plt.tight_layout()
        fig4.savefig(os.path.join(output_dir, 'plot_rule_source_effectiveness.png'), dpi=150)
        plt.close(fig4)
        
        # 5. Success vs Number of Rules Scatter Plot
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        rule_counts = []
        bleu_scores = []
        pec_scores = []
        is_fixed = []
        
        for result in self.test_results:
            rule_counts.append(len(result.rules_tested))
            bleu_scores.append(result.new_metrics.get('bleu', 0))
            pec_scores.append(result.new_metrics.get('pec', 0))
            is_fixed.append(result.fixed)
        
        if rule_counts:
            # Create scatter plot with different colors for fixed/not fixed
            colors = ['green' if f else 'red' for f in is_fixed]
            scatter = ax5.scatter(rule_counts, bleu_scores, c=colors, alpha=0.6, s=100)
            
            ax5.set_xlabel('Number of Rules Used')
            ax5.set_ylabel('BLEU Score')
            ax5.set_title('BLEU Score vs Number of Rules Used')
            ax5.set_ylim(-0.05, 1.05)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='g', markersize=10, label='Fixed'),
                             Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='r', markersize=10, label='Not Fixed')]
            ax5.legend(handles=legend_elements)
            
            # Add trend line
            if len(rule_counts) > 1:
                z = np.polyfit(rule_counts, bleu_scores, 1)
                p = np.poly1d(z)
                ax5.plot(sorted(rule_counts), p(sorted(rule_counts)), 
                        "b--", alpha=0.5, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
        else:
            ax5.text(0.5, 0.5, 'No test results available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('BLEU Score vs Number of Rules Used')
        
        plt.tight_layout()
        fig5.savefig(os.path.join(output_dir, 'plot_bleu_vs_rule_count.png'), dpi=150)
        plt.close(fig5)
        
        # 6. Metrics Correlation Heatmap
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        
        # Collect all metrics in a matrix
        metrics_data = []
        metrics_names = ['BLEU', 'PEC', 'Syntax', 'Relax PEC', 'Num Rules']
        
        for result in self.test_results:
            metrics_data.append([
                result.new_metrics.get('bleu', 0),
                result.new_metrics.get('pec', 0),
                result.new_metrics.get('syntax', 0),
                result.new_metrics.get('relax_pec', 0),
                len(result.rules_tested)
            ])
        
        if metrics_data:
            # Calculate correlation matrix
            metrics_array = np.array(metrics_data)
            correlation_matrix = np.corrcoef(metrics_array.T)
            
            # Create heatmap
            im = ax6.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax6.set_xticks(np.arange(len(metrics_names)))
            ax6.set_yticks(np.arange(len(metrics_names)))
            ax6.set_xticklabels(metrics_names)
            ax6.set_yticklabels(metrics_names)
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax6.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax6)
            cbar.set_label('Correlation Coefficient')
            
            # Add correlation values as text
            for i in range(len(metrics_names)):
                for j in range(len(metrics_names)):
                    text = ax6.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
            
            ax6.set_title('Correlation Matrix of Metrics')
        else:
            ax6.text(0.5, 0.5, 'No data for correlation analysis', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Correlation Matrix of Metrics')
        
        plt.tight_layout()
        fig6.savefig(os.path.join(output_dir, 'plot_metrics_correlation.png'), dpi=150)
        plt.close(fig6)
        
        # 7. Generate tree visualizations for successful fixes
        self.generate_tree_visualizations(output_dir)
        
        print(f"\nGenerated plots in: {output_dir}")
        print("  - plot_success_rate_by_type.png")
        print("  - plot_rule_count_distribution.png")
        print("  - plot_metrics_comparison.png")
        print("  - plot_rule_source_effectiveness.png")
        print("  - plot_bleu_vs_rule_count.png")
        print("  - plot_metrics_correlation.png")
        print("  - plot_successful_qtree_*.png (tree visualizations for each source)")
    
    def generate_tree_visualizations(self, output_dir: str):
        """Generate tree visualizations showing which nodes/branches led to successful fixes"""
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        try:
            import networkx as nx
            HAS_NETWORKX = True
        except ImportError:
            HAS_NETWORKX = False
            print("  Note: NetworkX not found. Generating simplified tree visualizations.")
        
        # Collect successful rules by source task AND design
        successful_rules_by_source = defaultdict(list)
        for result in self.test_results:
            if result.fixed:
                for rule in result.rules_tested:
                    # Use both task_id and design_name as key
                    source_key = (rule.source_task_id, rule.source_design)
                    successful_rules_by_source[source_key].append({
                        'rule': rule,
                        'test_id': result.test_id,
                        'metrics': result.new_metrics
                    })
        
        # Generate a tree visualization for each source that contributed to fixes
        for source_idx, (source_key, rule_data_list) in enumerate(successful_rules_by_source.items()):
            if not rule_data_list:
                continue
                
            source_task_id, source_design_name = source_key
            
            # Find the corresponding qtree data - match BOTH task_id and design_name
            qtree_entry = None
            for entry in self.qtree_data:
                if (entry.get('task_id') == source_task_id and 
                    entry.get('design_name') == source_design_name):
                    qtree_entry = entry
                    break
            
            if not qtree_entry or 'qtree_data' not in qtree_entry:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 12))
            
            if not HAS_NETWORKX:
                # Simplified visualization without NetworkX
                self._generate_simple_tree_visualization(ax, qtree_entry, rule_data_list, successful_rules_by_source)
                plt.tight_layout()
                # Include design name in filename for uniqueness
                safe_design = source_design_name.replace('/', '_').replace(' ', '_')[:30]
                filename = f"plot_successful_qtree_{source_idx}_{source_task_id}_{safe_design}.png"
                fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Generated tree visualization: {filename}")
                continue
            
            # Extract nodes and build graph
            nodes = qtree_entry['qtree_data'].get('nodes', [])
            if not nodes:
                continue
                
            # Create a directed graph
            G = nx.DiGraph()
            
            # Track which nodes contributed to successful fixes
            successful_node_ids = set()
            for data in rule_data_list:
                successful_node_ids.add(data['rule'].node_id)
            
            # Add nodes to graph
            node_positions = {}
            node_labels = {}
            node_colors = []
            
            # Build parent-child relationships
            for node in nodes:
                node_id = node['id']
                G.add_node(node_id)
                
                # Create label with node question (truncated for display)
                question = node.get('question', '')[:40]
                if len(node.get('question', '')) > 40:
                    question += '...'
                node_labels[node_id] = f"{node_id}\n{question}"
                
                # Color nodes based on success
                if node_id in successful_node_ids:
                    node_colors.append('#90EE90')  # Light green for successful nodes
                else:
                    node_colors.append('#FFFFE0')  # Light yellow for other nodes
                
                # Add edges
                if 'parent_id' in node and node['parent_id']:
                    G.add_edge(node['parent_id'], node_id)
            
            # Add top-level rules if any contributed to success
            if 'top_level' in successful_node_ids:
                G.add_node('root')
                node_labels['root'] = 'Root\n(Top-level rules)'
                node_colors.insert(0, '#90EE90')  # Green if top-level successful
                
                # Connect root to all parentless nodes
                for node in nodes:
                    if 'parent_id' not in node or not node['parent_id']:
                        G.add_edge('root', node['id'])
            
            # Use hierarchical layout
            try:
                # Try to use graphviz layout for better tree structure
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                # Fallback to spring layout if graphviz not available
                pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw the graph
            nx.draw(G, pos, ax=ax, 
                   node_color=node_colors,
                   node_size=3000,
                   with_labels=True,
                   labels=node_labels,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   linewidths=2)
            
            # Add title and legend
            design_name = qtree_entry.get('design_name', 'Unknown')
            prompt = qtree_entry.get('prompt', '')[:80]
            if len(qtree_entry.get('prompt', '')) > 80:
                prompt += '...'
            
            ax.set_title(f"QTree from {source_task_id} ({design_name})\n"
                        f"Prompt: {prompt}\n"
                        f"Nodes in green contributed to {len(rule_data_list)} successful fix(es)",
                        fontsize=12, pad=20)
            
            # Add legend
            green_patch = mpatches.Patch(color='#90EE90', label='Successful nodes (led to fixes)')
            yellow_patch = mpatches.Patch(color='#FFFFE0', label='Other nodes')
            ax.legend(handles=[green_patch, yellow_patch], loc='upper right')
            
            # Add success details as text box
            success_text = f"Success Details:\n"
            for data in rule_data_list[:3]:  # Show first 3
                rule = data['rule']
                success_text += f"\n• Fixed Test ID {data['test_id']}:"
                success_text += f"\n  Rule: {rule.text[:60]}..."
                success_text += f"\n  BLEU: {data['metrics'].get('bleu', 0):.3f}, PEC: {data['metrics'].get('pec', 0)}"
            
            if len(rule_data_list) > 3:
                success_text += f"\n\n... and {len(rule_data_list) - 3} more successful applications"
            
            # Add text box with success details
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.02, success_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', bbox=props)
            
            plt.tight_layout()
            
            # Save the figure with unique filename including design name
            safe_design = source_design_name.replace('/', '_').replace(' ', '_')[:30]
            filename = f"plot_successful_qtree_{source_idx}_{source_task_id}_{safe_design}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Generated tree visualization: {filename}")
        
        # Also generate a summary tree showing all successful paths across all sources
        self.generate_summary_tree_visualization(output_dir, successful_rules_by_source)
    
    def generate_summary_tree_visualization(self, output_dir: str, successful_rules_by_source):
        """Generate a summary visualization showing success patterns across all trees"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Count how many times each node pattern led to success
        node_pattern_success = defaultdict(int)
        branch_pattern_success = defaultdict(int)
        
        for source_key, rule_data_list in successful_rules_by_source.items():
            for data in rule_data_list:
                rule = data['rule']
                # Count node questions
                node_pattern = rule.node_question[:50] if rule.node_question else 'Unknown'
                node_pattern_success[node_pattern] += 1
                
                # Count branch patterns (simplified)
                if rule.parent_chain:
                    branch_pattern = ' → '.join([p.split(':')[0] for p in rule.parent_chain[-2:]])
                    branch_pattern_success[branch_pattern] += 1
        
        # Create bar chart of most successful node patterns
        top_patterns = sorted(node_pattern_success.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_patterns:
            patterns, counts = zip(*top_patterns)
            y_pos = np.arange(len(patterns))
            
            bars = ax.barh(y_pos, counts, color='skyblue', edgecolor='navy')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(patterns, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Number of Successful Fixes')
            ax.set_title('Most Successful Node Patterns Across All QTrees', fontsize=14, pad=20)
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                       str(count), ha='left', va='center')
            
            # Add summary stats
            total_fixes = sum(len(rules) for rules in successful_rules_by_source.values())
            unique_sources = len(successful_rules_by_source)
            
            summary_text = (f"Summary Statistics:\n"
                          f"• Total successful fixes: {total_fixes}\n"
                          f"• Unique source QTrees: {unique_sources}\n"
                          f"• Most common successful pattern: {patterns[0]}")
            
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
        else:
            ax.text(0.5, 0.5, 'No successful patterns found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'plot_successful_patterns_summary.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("  Generated summary visualization: plot_successful_patterns_summary.png")
    
    def _generate_simple_tree_visualization(self, ax, qtree_entry, rule_data_list, successful_rules_by_source):
        """Generate a simplified tree visualization without NetworkX"""
        # Clear axes
        ax.clear()
        ax.axis('off')
        
        # Get successful node IDs
        successful_node_ids = set()
        for data in rule_data_list:
            successful_node_ids.add(data['rule'].node_id)
        
        # Build tree structure text
        tree_text = []
        tree_text.append(f"QTree: {qtree_entry.get('task_id')} ({qtree_entry.get('design_name', 'Unknown')})")
        tree_text.append(f"Prompt: {qtree_entry.get('prompt', '')[:80]}...")
        tree_text.append(f"\nSuccessful Nodes (contributed to {len(rule_data_list)} fixes):")
        tree_text.append("="*60)
        
        # Show successful branches
        for data in rule_data_list:
            rule = data['rule']
            tree_text.append(f"\n✓ Node: {rule.node_id}")
            tree_text.append(f"  Question: {rule.node_question}")
            tree_text.append(f"  Rule: {rule.text[:80]}...")
            tree_text.append(f"  Branch Path:")
            for i, node in enumerate(rule.parent_chain):
                tree_text.append(f"    {'  ' * i}└─ {node}")
            tree_text.append(f"  Fixed Test ID: {data['test_id']}")
            tree_text.append(f"  Metrics: BLEU={data['metrics'].get('bleu', 0):.3f}, PEC={data['metrics'].get('pec', 0)}")
            tree_text.append("-"*40)
        
        # Create text display
        full_text = '\n'.join(tree_text)
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add title
        ax.set_title(f"Successful Branches from QTree {qtree_entry.get('task_id')}", 
                    fontsize=14, pad=20)
    
    def cleanup(self, keep_temp_files=False):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            if keep_temp_files:
                print(f"\nTemporary files kept in: {self.temp_dir}")
            else:
                shutil.rmtree(self.temp_dir)
                print(f"\nTemporary directory cleaned up: {self.temp_dir}")


def main(test_individual_rules=True, 
         test_all_rules_together=True,
         test_rules_by_node=True,
         test_top_level_rules=True):
    """Main function
    
    Args:
        test_individual_rules: Whether to test each rule one at a time
        test_all_rules_together: Whether to test all rules together  
        test_rules_by_node: Whether to test rules grouped by node
        test_top_level_rules: Whether to test only top-level rules
        check_functionality: Whether to check functionality using JasperGold
    """
    
    # Override with FLAGS if they exist
    test_individual_rules = getattr(FLAGS, 'test_individual_rules', test_individual_rules)
    test_all_rules_together = getattr(FLAGS, 'test_all_rules_together', test_all_rules_together)
    test_rules_by_node = getattr(FLAGS, 'test_rules_by_node', test_rules_by_node)
    test_top_level_rules = getattr(FLAGS, 'test_top_level_rules', test_top_level_rules)
    
    print(f"\n{'='*60}")
    print(f"QTREE RAG TESTING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Test IDs: {FLAGS.only_test_ids}")
    print(f"Task: {FLAGS.task}")
    print(f"Model: {FLAGS.llm_model if hasattr(FLAGS, 'llm_model') else 'default'}")
    print(f"Test configurations:")
    print(f"  - Test individual rules: {test_individual_rules}")
    print(f"  - Test all rules together: {test_all_rules_together}")
    print(f"  - Test rules by node: {test_rules_by_node}")
    print(f"  - Test top-level rules: {test_top_level_rules}")
    print(f"{'='*60}\n")
    
    # Initialize local_saver for organized logging
    local_saver = Saver()
    print(f"Using log directory: {local_saver.get_log_dir()}")
    
    # Create tester with configuration
    tester = QTreeRAGTester(
        test_individual_rules=test_individual_rules,
        test_all_rules_together=test_all_rules_together,
        test_rules_by_node=test_rules_by_node,
        test_top_level_rules=test_top_level_rules
    )
    
    try:
        # Run tests
        results = tester.run()
        
        # Generate report
        report = tester.generate_report()
        
        # Save outputs using saver
        # Save text report
        report_file = os.path.join(local_saver.get_log_dir(), "qtree_rag_test_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
            
        # Save CSV summary
        csv_file = "qtree_rag_test_summary.csv"
        tester.generate_csv_summary(os.path.join(local_saver.get_log_dir(), csv_file))
        
        # Generate visualization plots
        tester.generate_plots(local_saver.get_log_dir())
        
        # Save test results as pickle for later analysis
        save_pickle(results, os.path.join(local_saver.get_log_dir(), "test_results.pkl"))
        
        # Save successful fixes separately
        successful_fixes = [r for r in results if r.fixed]
        if successful_fixes:
            save_pickle(successful_fixes, os.path.join(local_saver.get_log_dir(), "successful_fixes.pkl"))
            
        # Save summary statistics
        stats = {
            "total_tests": len(results),
            "successful_fixes": len(successful_fixes),
            "test_ids": FLAGS.only_test_ids,
            "configurations": {
                "test_individual_rules": test_individual_rules,
                "test_all_rules_together": test_all_rules_together,
                "test_rules_by_node": test_rules_by_node,
                "test_top_level_rules": test_top_level_rules
            }
        }
        # Save as JSON
        import json
        with open(os.path.join(local_saver.get_log_dir(), "test_summary.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nRAG testing complete!")
        print(f"All results saved to: {local_saver.get_log_dir()}")
        print(f"Report: qtree_rag_test_report.txt")
        print(f"CSV summary: {csv_file}")
        print(f"Test results: test_results.pkl")
        if successful_fixes:
            print(f"Successful fixes: successful_fixes.pkl")
        print(f"\nVisualization plots:")
        print(f"  - plot_success_rate_by_type.png - Success rates for different test configurations")
        print(f"  - plot_rule_count_distribution.png - Distribution of rule counts in successful fixes")
        print(f"  - plot_metrics_comparison.png - Box plots comparing metrics across test types")
        print(f"  - plot_rule_source_effectiveness.png - Top 20 most effective rule sources")
        print(f"  - plot_bleu_vs_rule_count.png - Scatter plot of BLEU vs number of rules")
        print(f"  - plot_metrics_correlation.png - Correlation heatmap between different metrics")
        print(f"  - plot_successful_qtree_*.png - Tree visualizations showing successful nodes/branches")
        print(f"  - plot_successful_patterns_summary.png - Summary of most successful node patterns")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Training Examples Database: {len(tester.training_examples)} examples loaded")
        print(f"Rules Database Size: {len(tester.training_ids_for_rules)} unique training entries")
        print(f"Total tests run: {len(results)}")
        print(f"Successful fixes: {len(tester.successful_fixes)}")
        
        if tester.successful_fixes:
            print("\nSuccessful RAG combinations:")
            for fix in tester.successful_fixes[:3]:  # Show first 3
                print(f"  Test ID {fix.test_id}: {len(fix.rules_tested)} suggestions used")
                print(f"    New metrics: {fix.new_metrics}")
                print(f"    Improvements: {fix.improvements}")
        else:
            print("\nNo successful fixes found!")
            print("Consider:")
            print("  - Checking if the QTree file contains data for the test IDs")
            print("  - Verifying the suggestions are relevant")
            print("  - Running with FLAGS.debug = True to inspect prompts")
            print(f"  - Current test IDs: {FLAGS.only_test_ids}")
                
    except Exception as e:
        print(f"Error during RAG testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Keep temp files if debug mode is on
        tester.cleanup(keep_temp_files=FLAGS.debug)
        # Close saver
        local_saver.close()


if __name__ == "__main__":
    # Direct configuration - change these values as needed
    # Setting to True means that test type will run
    
    # Option 1: Test ONLY individual rules (recommended for finding specific fixes)
    test_individual_rules = True      # Test each rule one at a time
    test_all_rules_together = False   # Test all rules together
    test_rules_by_node = False        # Test rules grouped by node
    test_top_level_rules = False      # Test only top-level rules
    
    # Option 2: Test all combinations (uncomment to use)
    # test_individual_rules = False
    # test_all_rules_together = True
    # test_rules_by_node = True
    # test_top_level_rules = True
    
    # Option 3: Test everything (uncomment to use - this takes the longest)
    # test_individual_rules = True
    # test_all_rules_together = True
    # test_rules_by_node = True
    # test_top_level_rules = True
    
    # Run with the configured settings
    main(
        test_individual_rules=test_individual_rules,
        test_all_rules_together=test_all_rules_together,
        test_rules_by_node=test_rules_by_node,
        test_top_level_rules=test_top_level_rules
    )
