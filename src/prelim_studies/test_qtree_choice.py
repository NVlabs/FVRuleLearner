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
    def __init__(self, id, text, node_id, node_question,reference_sva,
     prompt, parent_chain=None, source_task_id=None, source_design=None, source_metrics=None):
        self.id = id
        self.text = text
        self.node_id = node_id
        self.node_question = node_question
        self.reference_sva = reference_sva
        self.prompt = prompt
        self.parent_chain = parent_chain or []
        self.source_task_id = source_task_id  # Which training task this rule came from
        self.source_design = source_design  # Which design it was for
        self.source_metrics = source_metrics or {}  # The metrics it achieved
        

class QTreeRAGTester:
    def __init__(self, qtree_file: str = None, 
                 check_functionality: bool = True,
                 effectiveness_csv_path: str = None):
        """Initialize the RAG-based tester"""
        
        # Use FLAGS configuration
        self.only_test_ids = FLAGS.only_test_ids
        
        # Determine qtree file path
        if qtree_file is None:
            qtree_file = os.path.join(FLAGS.load_suggestions_path, "qtrees.json")
        self.qtree_file = qtree_file
        
        # Store effectiveness CSV path
        if effectiveness_csv_path is None:
            effectiveness_csv_path = FLAGS.suggestions_effectiveness_csv_path
        self.effectiveness_csv_path = effectiveness_csv_path

        print(f"Using qtree file: {self.qtree_file}")
        print(f"Testing IDs: {self.only_test_ids}")
        print(f"RAG Configuration: use_RAG={FLAGS.use_RAG if hasattr(FLAGS, 'use_RAG') else False}, RAG_content={FLAGS.RAG_content if hasattr(FLAGS, 'RAG_content') else []}")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"qtree_rag_test_{get_user()}_{get_host()}_")
        
        # Results storage
        self.test_results = []
        self.successful_fixes = []
        self.qtree_ranking_results = []
        self.suggestion_effectiveness = {}  # Store effectiveness results
        
        # Load qtree data
        with open(self.qtree_file, 'r') as f:
            self.qtree_data = json.load(f)
            
        # Initialize RAG agent for examples if configured
        self.example_rag_agent = None
        self.training_ids_for_rules = []     # Track which training IDs are used for rules

        # Storage for qtree lookup utilities
        self.qtree_entries: List[Dict[str, Any]] = []
        self.qtree_prompt_texts: List[str] = []
        self.qtree_task_ids: List[str] = []
        self.qtree_prompt_embeddings: Optional[Dict[str, Any]] = None
        self.qtree_sva_texts: List[str] = []
        self.qtree_sva_embeddings: Optional[Dict[str, Any]] = None
        # Index to store Rules by task_id -> node_id -> Rule
        self.rules_index: Dict[str, Dict[str, Rule]] = {}
        # Index to store pre-computed parent chains by task_id -> node_id -> parent_chain
        self.parent_chains_index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._prepare_qtree_index()
        
        # Initialize training examples from specified IDs
        self.training_examples = []
        if FLAGS.use_RAG and 'Examples' in FLAGS.RAG_content:
            self._load_training_examples()
            
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
        
        # Collect all training rules to populate rules_index
        print("Collecting training rules from successful qtree entries...")
        self.all_training_rules = self.collect_all_training_rules()
        print(f"Collected {len(self.all_training_rules)} rules total")
        
        # Check if requested rows are available
        missing_rows = set(self.only_test_ids) - available_csv_rows
        if missing_rows:
            print(f"WARNING: Requested CSV rows not found in qtrees.json: {sorted(missing_rows)}")
            
        # Load effectiveness data if provided
        # breakpoint()
        if self.effectiveness_csv_path:
            self._load_effectiveness_data()

    def _load_effectiveness_data(self):
        """Load effectiveness data from CSV if provided"""
        try:
            print(f"\nLoading effectiveness data from: {self.effectiveness_csv_path}")
            effectiveness_df = pd.read_csv(self.effectiveness_csv_path)
            
            # Create mapping of trace_id to effectiveness data
            self.effectiveness_mapping = {}
            total_entries = 0
            fixed_entries = 0
            
            for idx, row in effectiveness_df.iterrows():
                trace_id = row.get('trace_id', '')
                if trace_id:
                    total_entries += 1
                    fixed = row.get('fixed', False)
                    if isinstance(fixed, str):
                        fixed = fixed.lower() == 'true'
                    if fixed:
                        fixed_entries += 1
                        
                    self.effectiveness_mapping[trace_id] = {
                        'fixed': fixed,
                        'test_id': row.get('test_id'),
                        'task_id': row.get('task_id'),
                        'design_name': row.get('design_name'),
                        'improvements': row.get('improvements', ''),
                        'suggestions_preview': row.get('suggestions_preview', ''),
                        'new_syntax': row.get('new_syntax', 0),
                        'new_pec': row.get('new_pec', 0),
                        'new_relax_pec': row.get('new_relax_pec', 0),
                        'bleu': row.get('bleu', 0),
                        'rouge': row.get('rouge', 0),
                        'exact_match': row.get('exact_match', 0)
                    }
            
            print(f"Loaded {total_entries} effectiveness entries")
            print(f"Found {fixed_entries} successful fixes ({fixed_entries/total_entries*100:.1f}%)")
            
            # breakpoint()  
            # Analyze fix rates by design
            design_stats = defaultdict(lambda: {'total': 0, 'fixed': 0})
            for trace_id, data in self.effectiveness_mapping.items():
                design = data['design_name']
                design_stats[design]['total'] += 1
                if data['fixed']:
                    design_stats[design]['fixed'] += 1
            
            print("\nFix rates by design:")
            for design, stats in sorted(design_stats.items()):
                fix_rate = stats['fixed'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"  {design}: {stats['fixed']}/{stats['total']} ({fix_rate:.1f}%)")
                
        except Exception as e:
            print(f"WARNING: Failed to load effectiveness data: {e}")
            self.effectiveness_mapping = {}

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

    def collect_all_training_rules(self) -> List[Rule]:
        """Collect all rules from SUCCESSFUL training qtree entries (pec=1.0 or bleu=1.0) - separate from examples"""
        all_rules = []
        successful_entries = 0
        unique_training_ids = set()
        
        for entry_idx, entry in enumerate(self.qtree_data):
            task_id = entry.get('task_id', '')
            design_name = entry.get('design_name', '')
            prompt = entry.get('prompt', '').strip()
            reference_sva = entry.get('ref_solution', '')
            
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
                        print(f"Adding rule: {rule.id}")
                        # breakpoint()

                        all_rules.append(rule)
                        
                        # Store rule in the index for efficient lookup
                        if task_id not in self.rules_index:
                            self.rules_index[task_id] = {}
                        if node['id'] not in self.rules_index[task_id]:
                            self.rules_index[task_id][node['id']] = []
                        self.rules_index[task_id][node['id']].append(rule)
        
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

    def find_similar_prompts(self, target_prompt: str, target_design: str, top_k: int = 2) -> List[Dict]:
        # Here target_prompt is the raw prompt from the csv file
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

        # Return top k examples
        results = []
        for score, csv_id, example in similarities[:top_k]:
            # breakpoint()
            results.append({
                'prompt': example['prompt'],
                'reference_sva': example['reference_sva']
            })
            
        return results

    def find_similar_svas(self, target_sva: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find most similar Q-Trees based on reference SVA content."""
        if not target_sva or not target_sva.strip():
            return []

        if top_k is None:
            top_k = getattr(FLAGS, 'qtree_similarity_top_k', 5)

        embedding_model = self._get_embedding_model()
        from sklearn.metrics.pairwise import cosine_similarity

        if self.qtree_sva_embeddings is None:
            self.qtree_sva_embeddings = {}
            texts_to_encode = []
            ids_to_encode = []
            for task_id, sva_text in zip(self.qtree_task_ids, self.qtree_sva_texts):
                if sva_text and sva_text.strip():
                    ids_to_encode.append(task_id)
                    texts_to_encode.append(sva_text)

            if texts_to_encode:
                embeddings = embedding_model.encode(texts_to_encode, convert_to_tensor=False, show_progress_bar=False)
                for task_id, embedding in zip(ids_to_encode, embeddings):
                    self.qtree_sva_embeddings[task_id] = embedding
                print(f"Pre-computed embeddings for {len(self.qtree_sva_embeddings)} Q-Tree reference SVAs")

        if not self.qtree_sva_embeddings:
            print("No Q-Tree SVAs available for similarity search")
            return []

        target_embedding = embedding_model.encode([target_sva], convert_to_tensor=False, show_progress_bar=False)[0]

        candidates: List[Dict[str, Any]] = []
        for entry in self.qtree_entries:
            task_id = entry['task_id']
            embedding = self.qtree_sva_embeddings.get(task_id)
            reference_sva = entry.get('reference_sva')

            if embedding is None or not reference_sva:
                continue

            score = cosine_similarity([target_embedding], [embedding])[0][0]

            candidates.append({
                'task_id': task_id,
                'design_name': entry.get('design_name', ''),
                'prompt': entry.get('prompt', ''),
                'reference_sva': reference_sva,
                'qtree': entry.get('qtree', {}),
                'root_nodes': entry.get('root_nodes', []),
                'score': float(score),
                'entry': entry,
            })

        candidates.sort(key=lambda item: item['score'], reverse=True)
        top_candidates = candidates[:top_k]

        print(f"Retrieved {len(top_candidates)} candidate Q-Trees via SVA similarity (top_k={top_k})")
        for idx, candidate in enumerate(top_candidates, 1):
            print(f"  SVA Candidate {idx}: task_id={candidate['task_id']} design={candidate['design_name']} score={candidate['score']:.3f}")

        return top_candidates

    def _prepare_qtree_index(self):
        """Pre-compute lookup structures for Q-Trees."""
        self.qtree_entries = []
        self.qtree_prompt_texts = []
        self.qtree_task_ids = []
        self.qtree_prompt_embeddings = None
        self.qtree_sva_texts = []
        self.qtree_sva_embeddings = None

        for entry in self.qtree_data:
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

    def _get_embedding_model(self):
        """Return cached sentence transformer model."""
        if not hasattr(self, '_embedding_model'):
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Initialized SentenceTransformer for similarity calculation")
        return self._embedding_model

    def _get_root_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return root-level nodes (parent is None or empty)."""
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
        return (question or '').strip().lower()

    def group_training_rules_by_source(self, rules: List[Rule]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Group training rules by (design_name, task_id) source."""
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rule in rules:
            key = (rule.source_design or 'unknown', rule.source_task_id or 'unknown')
            group = grouped.setdefault(key, {
                'design_name': rule.source_design,
                'task_id': rule.source_task_id,
                'rules': [],
                'reference_sva': rule.reference_sva,
            })

            if rule.reference_sva and not group.get('reference_sva'):
                group['reference_sva'] = rule.reference_sva

            group['rules'].append(rule)

        print(f"Grouped {len(rules)} rules into {len(grouped)} design/task clusters")
        return grouped

    def rank_rule_groups_by_similarity(self, target_prompt: str, grouped_rules: Dict[Tuple[str, str], Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rank grouped rule sets by cosine similarity of reference SVAs to target prompt."""
        if not grouped_rules:
            return []

        if top_k is None:
            top_k = getattr(FLAGS, 'Suggestions_top_k', 2)

        embedding_model = self._get_embedding_model()
        from sklearn.metrics.pairwise import cosine_similarity

        prompt_embedding = embedding_model.encode([target_prompt], convert_to_tensor=False, show_progress_bar=False)[0]

        ranked = []
        for key, group in grouped_rules.items():
            reference_sva = group.get('reference_sva')
            if not reference_sva:
                continue

            group_embedding = embedding_model.encode([reference_sva], convert_to_tensor=False, show_progress_bar=False)[0]
            similarity = cosine_similarity([prompt_embedding], [group_embedding])[0][0]

            ranked.append({
                'design_name': group.get('design_name'),
                'task_id': group.get('task_id'),
                'rules': group.get('rules', []),
                'reference_sva': reference_sva,
                'similarity': float(similarity)
            })

        ranked.sort(key=lambda item: item['similarity'], reverse=True)
        return ranked[:top_k]

    def _collect_traces_from_node(self, nodes: List[Dict[str, Any]], start_node_id: Optional[str]) -> List[List[Dict[str, Any]]]:
        if not nodes or not start_node_id:
            return []

        node_map = {node.get('id'): node for node in nodes if node.get('id')}
        if start_node_id not in node_map:
            return []

        traces: List[List[Dict[str, Any]]] = []

        def dfs(current_id: str, path: List[Dict[str, Any]], visited: set):
            if current_id in visited:
                return
            node = node_map.get(current_id)
            if not node:
                return

            next_path = path + [node]
            next_visited = visited | {current_id}
            children = node.get('children') or []

            if not children:
                # This is a leaf node - we have a complete trace
                traces.append(next_path)
                return

            for child_id in children:
                if child_id not in next_visited:
                    dfs(child_id, next_path, next_visited)

        dfs(start_node_id, [], set())
        if traces:
            print(f"  Generated {len(traces)} traces from node {start_node_id}")
        return traces

    def _collect_traces_using_parent_chains(self, nodes: List[Dict[str, Any]], start_node_id: Optional[str], task_id: str) -> List[List[Dict[str, Any]]]:
        """Collect traces using pre-computed parent chains for efficiency."""
        if not nodes or not start_node_id or task_id not in self.parent_chains_index:
            return []
        
        node_map = {node.get('id'): node for node in nodes if node.get('id')}
        if start_node_id not in node_map:
            return []
        
        traces: List[List[Dict[str, Any]]] = []
        
        # Find all leaf nodes that descend from start_node_id
        start_node = node_map.get(start_node_id)
        if not start_node:
            return []
        
        # Get all descendant leaf nodes
        def find_leaf_descendants(node_id: str, visited: set) -> List[str]:
            if node_id in visited:
                return []
            visited.add(node_id)
            
            node = node_map.get(node_id)
            if not node:
                return []
            
            children = node.get('children') or []
            if not children:
                # This is a leaf node
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

    def _trim_text(self, text: str, max_chars: int = 480) -> str:
        if not text:
            return ''
        text = str(text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + '...'
    
    def check_suggestion_effectiveness(self, selected_traces: List[Dict[str, Any]], test_id: int) -> List[Dict[str, Any]]:
        """Check if selected suggestions appear in effectiveness data and whether they led to fixes"""
        results = []
        
        print("\n" + "="*60)
        print("SUGGESTION EFFECTIVENESS CHECK:")
        print("="*60)
        
        # BREAKPOINT 3: Check inputs and mapping
        print(f"[DEBUG] check_suggestion_effectiveness called with {len(selected_traces)} traces")
        print(f"[DEBUG] effectiveness_mapping keys sample (first 5): {list(self.effectiveness_mapping.keys())[:5]}")
        # breakpoint()
        
        for idx, trace in enumerate(selected_traces, 1):
            trace_id = trace.get('trace_id', '')
            effectiveness_data = self.effectiveness_mapping.get(trace_id, None)
            
            # BREAKPOINT 4: Check each trace lookup
            print(f"[DEBUG] Looking up trace_id: '{trace_id}'")
            print(f"[DEBUG] Found in mapping: {effectiveness_data is not None}")
            # breakpoint()
            
            result = {
                'rank': idx,
                'trace_id': trace_id,
                'llm_score': trace.get('llm_score', 0),
                'found_in_csv': effectiveness_data is not None,
                'fixed': False,
                'metrics': {},
                'improvements': ''
            }
            
            if effectiveness_data:
                result['fixed'] = effectiveness_data['fixed']
                result['metrics'] = {
                    'new_syntax': effectiveness_data['new_syntax'],
                    'new_pec': effectiveness_data['new_pec'],
                    'new_relax_pec': effectiveness_data['new_relax_pec'],
                    'bleu': effectiveness_data['bleu'],
                    'rouge': effectiveness_data['rouge'],
                    'exact_match': effectiveness_data['exact_match']
                }
                result['improvements'] = effectiveness_data['improvements']
                result['test_id_match'] = effectiveness_data.get('test_id') == test_id
                
                print(f"\nRank {idx}: {trace_id}")
                print(f"  Found in CSV: YES")
                print(f"  Fixed: {'YES' if result['fixed'] else 'NO'}")
                print(f"  Test ID Match: {'YES' if result['test_id_match'] else 'NO'}")
                print(f"  Metrics: syntax={result['metrics']['new_syntax']}, pec={result['metrics']['new_pec']}, bleu={result['metrics']['bleu']:.3f}")
                if result['improvements']:
                    print(f"  Improvements: {result['improvements']}")
            else:
                print(f"\nRank {idx}: {trace_id}")
                print(f"  Found in CSV: NO")
                print(f"  (No effectiveness data available)")
            
            results.append(result)
        
        # Summary statistics
        found_count = sum(1 for r in results if r['found_in_csv'])
        fixed_count = sum(1 for r in results if r['fixed'])
        
        print("\n" + "-"*60)
        print("EFFECTIVENESS SUMMARY:")
        print(f"  Total suggestions: {len(selected_traces)}")
        print(f"  Found in CSV: {found_count}/{len(selected_traces)} ({found_count/len(selected_traces)*100:.0f}%)")
        if found_count > 0:
            print(f"  Fixed issues: {fixed_count}/{found_count} ({fixed_count/found_count*100:.0f}%)")
        
        # Store in instance variable for later analysis
        if not hasattr(self, 'all_effectiveness_results'):
            self.all_effectiveness_results = []
        self.all_effectiveness_results.append({
            'test_id': test_id,
            'results': results,
            'summary': {
                'total': len(selected_traces),
                'found': found_count,
                'fixed': fixed_count
            }
        })
        
        return results

    def extract_tree_path(self, nodes: List[Dict[str, Any]], target_node_id: str) -> List[Dict[str, Any]]:
        """Extract the path from root to target node in the Q-tree."""
        # Build a map of node_id to node
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

    def _format_trace(self, path: List[Dict[str, Any]], path_rules: List[Rule] = None) -> str:
        lines = []
        
            # Format hierarchical reasoning path
        for depth, node in enumerate(path):
            level = node.get('level', 'unknown')
            question = node.get('question', '')  # No trimming
            
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
            
            lines.append(f"{indent}{marker} Step {depth}: {question}")
            
            # Show full answer without trimming
            answer = node.get('answer', '')
            if answer:
                lines.append(f"{indent}  >> Answer:")
                # Format answer with proper indentation for readability
                for line in answer.split('\n'):
                    if line.strip():
                        lines.append(f"{indent}     {line}")
            
            # Format rules without trimming
            rules = node.get('rules_generated') or []
            for i, rule in enumerate(rules, 1):
                lines.append(f"{indent}  => Rule {i}: {rule}")
        
        # Add Rule metadata if available
        if path_rules:
            lines.append("\n[Rule Metadata]")
            unique_designs = set(rule.source_design for rule in path_rules)
            unique_tasks = set(rule.source_task_id for rule in path_rules)
            lines.append(f"  Rules from {len(unique_tasks)} tasks across {len(unique_designs)} designs")
            
            # Show performance metrics of rules
            avg_pec = sum(rule.source_metrics.get('pec', 0) for rule in path_rules) / len(path_rules) if path_rules else 0
            lines.append(f"  Average PEC score: {avg_pec:.3f}")
            
            # Sample rule details
            for i, rule in enumerate(path_rules[:2]):  # Show first 2 rules
                lines.append(f"  Rule {i+1}: from {rule.source_task_id} ({rule.source_design})")
                if rule.source_metrics:
                    lines.append(f"    Metrics: PEC={rule.source_metrics.get('pec', 0)}, BLEU={rule.source_metrics.get('bleu', 0):.3f}")
            
            if len(path_rules) > 2:
                lines.append(f"  ... and {len(path_rules) - 2} more rules")
        
        return '\n'.join(lines)

    def find_similar_qtrees(self, target_prompt: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k Q-Trees most similar to the target prompt."""
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

        from sklearn.metrics.pairwise import cosine_similarity

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

        print(f"Retrieved {len(top_candidates)} candidate Q-Trees (top_k={top_k})")
        for idx, candidate in enumerate(top_candidates, 1):
            print(f"  Candidate {idx}: task_id={candidate['task_id']} design={candidate['design_name']} score={candidate['score']:.3f}")

        return top_candidates

    def group_qtrees_by_root(self, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group candidate Q-Trees by normalized root question."""
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

    def collect_branch_traces(self, group: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect root-to-leaf traces for all members in a grouped root cluster."""
        traces: List[Dict[str, Any]] = []
        members = group.get('members', [])

        for member_idx, member in enumerate(members, 1):
            qtree = member.get('qtree', {})
            nodes = qtree.get('nodes', [])
            root_node_id = member.get('root_node_id')
            task_id = member['task_id']
            
            # First try to use pre-computed parent chains if available
            if task_id in self.parent_chains_index:
                paths = self._collect_traces_using_parent_chains(nodes, root_node_id, task_id)
            else:
                # Fallback to computing paths dynamically
                paths = self._collect_traces_from_node(nodes, root_node_id)

            for path_idx, path in enumerate(paths, 1):
                # Include design_name for uniqueness since task_id alone is not unique
                # Sanitize design_name to avoid issues with special characters
                safe_design = member['design_name'].replace('/', '_').replace(' ', '_').replace(':', '_')
                
                # Use the leaf node's ID (last node in path) for the trace_id
                if path:
                    leaf_node = path[-1]  # Get the last node in the path (leaf node)
                    leaf_node_id = leaf_node.get('id', f'path_{path_idx}')
                    trace_id = f"{safe_design}::{member['task_id']}::{leaf_node_id}"
                else:
                    # Fallback if path is empty
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
                    'rules': path_rules,  # Include Rule objects for this trace
                    'num_rules': len(path_rules),
                })

        print(f"Collected {len(traces)} traces for root question '{group.get('root_question', '')}'")
        return traces

    def rank_traces_with_llm(self, target_prompt: str, traces: List[Dict[str, Any]], testbench: str = '', max_traces: int = None) -> Dict[str, Any]:
        """Rank traces using LLM and return structured response.
        
        This uses batch processing to rank multiple traces in a single LLM call for efficiency.
        """
        if not traces:
            return {'raw_response': '', 'parsed': None}

        # Batch processing: Take all traces unless max_traces is specified
        if max_traces is not None and len(traces) > max_traces:
            truncated_traces = traces[:max_traces]
            print(f"WARNING: Truncating from {len(traces)} to {max_traces} traces for LLM ranking")
        else:
            truncated_traces = traces
            print(f"Batch ranking all {len(truncated_traces)} traces")
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

        lines.append(f"\nTarget prompt: {target_prompt}")  # Full prompt
        if testbench:
            lines.append(f"Testbench: {testbench}")  # Full testbench

        lines.append("\nCandidate traces:")
        for idx, trace in enumerate(truncated_traces, 1):
            snippet = trace['trace_text']  # Full trace text, no trimming
            
            # Analyze reasoning path depth and quality
            path = trace.get('path', [])
            reasoning_depth = {
                'exploratory': sum(1 for n in path if n.get('level') == 'exploratory'),
                'specific': sum(1 for n in path if n.get('level') == 'specific_analysis'),
                'rules': sum(1 for n in path if n.get('level') == 'rule_generation'),
                'total_rules_generated': sum(len(n.get('rules_generated', [])) for n in path)
            }
            
            # Add rule performance info to help LLM rank better
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

            # breakpoint()

        except Exception as exc:
            print(f"LLM ranking failed: {exc}")
            return {'raw_response': str(exc), 'parsed': None}

        parsed = None
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        return {
            'raw_response': response,
            'parsed': parsed
        }

    def pickup_qtrees_for_test_case(self, test_id: int, test_row: Dict[str, Any], top_n: int = FLAGS.top_n) -> Dict[str, Any]:
        design_name = test_row.get('design_name', '')
        prompt = str(test_row.get('prompt', '')).strip()
        testbench = test_row.get('testbench') or test_row.get('tb') or ''

        print("=" * 80)
        print(f"Evaluating Q-Trees for test_id={test_id} design='{design_name}'")
        print(f"Prompt: {prompt}")  # Full prompt

        candidates = self.find_similar_qtrees(prompt)
        grouped = self.group_qtrees_by_root(candidates)

        group_results = []
        all_ranked_traces = []  # Collect all traces with scores for global ranking
        
        for group_key, group_data in grouped.items():
            traces = self.collect_branch_traces(group_data)
            ranking = self.rank_traces_with_llm(prompt, traces, testbench=testbench)

            parsed_rankings = ranking.get('parsed')
            if isinstance(parsed_rankings, list):
                parsed_rankings = sorted(parsed_rankings, key=lambda item: item.get('rank', 9999))

            # Count total rules used in this group
            total_rules = sum(len(trace.get('rules', [])) for trace in traces)
            
            group_summary = {
                'group_key': group_key,
                'root_question': group_data.get('root_question', ''),
                'num_members': len(group_data.get('members', [])),
                'num_traces': len(traces),
                'num_rules': total_rules,
                'ranking': ranking,
                'traces': traces
            }
            group_results.append(group_summary)

            print("-" * 60)
            print(f"Root question group: {group_summary['root_question']}")  # Full question
            print(f"Members: {group_summary['num_members']} | Traces: {group_summary['num_traces']} | Rules: {total_rules}")

            if parsed_rankings:
                top_items = parsed_rankings[:top_n]
                print("Top-ranked traces (from LLM):")
                for item in top_items:
                    print(f"  rank={item.get('rank')} trace_id={item.get('trace_id')} score={item.get('score', 0):.3f}")
                    print(f"    rationale: {item.get('rationale', '')}")  # Full rationale
                    
                # Collect all ranked traces with their scores for global ranking
                trace_lookup = {trace['trace_id']: trace for trace in traces}
                for ranked_item in parsed_rankings:
                    trace_id = ranked_item.get('trace_id')
                    if trace_id in trace_lookup:
                        trace_with_score = trace_lookup[trace_id].copy()
                        trace_with_score['llm_score'] = ranked_item.get('score', 0)
                        trace_with_score['llm_rank'] = ranked_item.get('rank', 9999)
                        trace_with_score['llm_rationale'] = ranked_item.get('rationale', '')
                        all_ranked_traces.append(trace_with_score)
                
                # BREAKPOINT 2: Check traces being added from each group
                # Shows how many traces from this group and their scores
                # print(f"[DEBUG] Added {len([r for r in parsed_rankings if r.get('trace_id') in trace_lookup])} traces from group '{group_key[:50]}...'")
                # breakpoint()
            else:
                print("LLM ranking not available or not in JSON format.")

        # Global ranking: Sort all traces by LLM score
        all_ranked_traces.sort(key=lambda x: x.get('llm_score', 0), reverse=True)
        
        # BREAKPOINT 1: Check all_ranked_traces after global sorting
        # This shows all traces from all groups sorted by LLM score
        # breakpoint()
        
        # Extract top 3 suggestions globally
        top_n_suggestions = []
        top_n_rule_strings = []  # Collect just the rule strings separately
        print("\n" + "=" * 60)
        print("GLOBAL TOP 3 SUGGESTIONS:")
        print("=" * 60)
        
        for idx, trace in enumerate(all_ranked_traces[:top_n], 1):
            print(f"\nGlobal Rank {idx}:")
            print(f"  Trace ID: {trace['trace_id']}")
            print(f"  Design: {trace['design_name']}")
            print(f"  Task ID: {trace['task_id']}")
            print(f"  Root Question: {trace['root_question']}")
            print(f"  LLM Score: {trace.get('llm_score', 0):.3f}")
            print(f"  LLM Rationale: {trace.get('llm_rationale', '')}")
            
            # Extract suggestion text from the trace
            suggestion_lines = []
            
            # Add the full reasoning path
            suggestion_lines.append(f"=== Suggestion {idx} (from {trace['design_name']}, score: {trace.get('llm_score', 0):.3f}) ===")
            suggestion_lines.append(trace['trace_text'])
            
            # Add any rules from this trace
            if trace.get('rules'):
                suggestion_lines.append("\nAssociated Rules:")
                trace_rule_strings = []  # Collect rules for this trace
                for rule in trace['rules'][:top_n]:
                    rule_text = rule.text
                    suggestion_lines.append(f"- {rule_text}")
                    trace_rule_strings.append(rule_text)
                
                # Add this trace's rules to the global list with metadata
                for rule_text in trace_rule_strings:
                    top_n_rule_strings.append({
                        'rule_text': rule_text,
                        'source_trace_id': trace['trace_id'],
                        'source_design': trace['design_name'],
                        'source_task_id': trace['task_id'],
                        'llm_score': trace.get('llm_score', 0)
                    })
            
            suggestion_text = '\n'.join(suggestion_lines)
            top_n_suggestions.append(suggestion_text)
            
            # BREAKPOINT 3: Check each top suggestion being created
            # Shows the actual suggestion text that will be used in RAG
            # print(f"[DEBUG] Created suggestion {idx} with {len(suggestion_lines)} lines, {len(trace.get('rules', []))} rules")
            # breakpoint()
            
            # Also print a preview
            print(f"\n  Suggestion Preview:")
            suggestion_lines_split = suggestion_text.split('\n')
            preview_lines = suggestion_lines_split[:10]  # Show first 10 lines
            for line in preview_lines:
                print(f"    {line}")
            if len(suggestion_lines_split) > 10:
                print(f"    ... ({len(suggestion_lines_split) - 10} more lines)")

        # Print extracted rule strings
        if top_n_rule_strings:
            print("\n" + "-" * 60)
            print("EXTRACTED RULE STRINGS:")
            print("-" * 60)
            for i, rule_info in enumerate(top_n_rule_strings, 1):
                print(f"{i}. {rule_info['rule_text']}")
                print(f"   (from {rule_info['source_design']}, score: {rule_info['llm_score']:.3f})")
            print("-" * 60)
        
        # Summary of Rule usage
        total_traces = sum(group['num_traces'] for group in group_results)
        total_rules_used = sum(group.get('num_rules', 0) for group in group_results)
        print("\n" + "=" * 60)
        print(f"Summary: {len(candidates)} candidate Q-Trees → {len(group_results)} groups → {total_traces} traces → {total_rules_used} rules leveraged")
        
        # Show trace distribution
        trace_distribution = [group['num_traces'] for group in group_results]
        if trace_distribution:
            print(f"Trace distribution per group: min={min(trace_distribution)}, max={max(trace_distribution)}, avg={sum(trace_distribution)/len(trace_distribution):.1f}")
        
        if self.all_training_rules:
            print(f"Rule utilization: {total_rules_used}/{len(self.all_training_rules)} ({total_rules_used/len(self.all_training_rules)*100:.1f}%)")
        
        print(f"\nGlobal ranking: Selected top {len(top_n_suggestions)} suggestions from {len(all_ranked_traces)} total ranked traces")
        print(f"Extracted {len(top_n_rule_strings)} rule strings from top suggestions")
        
        # Also create a simple list of just the rule text strings for easy access
        rule_strings_only = [r['rule_text'] for r in top_n_rule_strings]
        
        # Check effectiveness of selected suggestions if data is available
        effectiveness_results = []
        if hasattr(self, 'effectiveness_mapping') and self.effectiveness_mapping:
            effectiveness_results = self.check_suggestion_effectiveness(all_ranked_traces[:top_n], test_id)
        
        # breakpoint()

        return {
            'test_id': test_id,
            'design_name': design_name,
            'prompt': prompt,
            'testbench': testbench,
            'candidates': candidates,
            'groups': group_results,
            'total_rules_used': total_rules_used,
            'top_suggestions': top_n_suggestions,  # NEW: Return the global top 3 suggestions
            'all_ranked_traces': all_ranked_traces,  # NEW: Return all ranked traces for analysis
            'top_rule_strings': top_n_rule_strings,  # NEW: Rule strings with metadata
            'rule_strings_only': rule_strings_only,    # NEW: Just the rule text strings
            'effectiveness_results': effectiveness_results  # NEW: Effectiveness check results
        }

    def build_rag_prompt(self, original_prompt: str, testbench: str, suggestions: List[str] = None, examples: List[str] = None) -> str:
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

    def initial_sva_generation(self, test_id: int, test_row: Dict) -> str:
        """Generate an initial SVA for a test case"""
        # Extract test case information
        design_name = test_row['design_name'] if 'design_name' in test_row else ''
        prompt = str(test_row['prompt'] if 'prompt' in test_row else '').strip()
        testbench = test_row['testbench'] if 'testbench' in test_row else (test_row['tb'] if 'tb' in test_row else '')
        # Try to get reference SVA from test row (ground truth)
        
        # Retrieve most similar examples
        Examples_top_k = FLAGS.Examples_top_k
        examples = []
        
        # Find similar examples from training data
        similar_examples = self.find_similar_prompts(prompt, design_name, top_k=Examples_top_k)

        # Format examples like in hardware_general_agent.py
        for ex in similar_examples:
            formatted_example = f'Question: Create a SVA assertion that checks: {ex["prompt"]}\nReference Answer: {ex["reference_sva"]}'
            examples.append(formatted_example)
            
        # Compose the RAG prompt
        rag_prompt = self.build_rag_prompt(prompt, testbench, suggestions=None, examples=examples)

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
        generated_sva = self.extract_sva_from_response(response)
        
        return generated_sva

    def second_sva_generation(self, test_id: int, test_row: Dict, suggestions: Dict[str, Any]) -> str:
        """Generate an second SVA for a test case"""
        # Extract test case information
        design_name = test_row['design_name'] if 'design_name' in test_row else ''
        prompt = str(test_row['prompt'] if 'prompt' in test_row else '').strip()
        testbench = test_row['testbench'] if 'testbench' in test_row else (test_row['tb'] if 'tb' in test_row else '')
        # Try to get reference SVA from test row (ground truth)
        
        # Retrieve most similar examples
        Examples_top_k = FLAGS.Examples_top_k
        examples = []
        
        # Find similar examples from training data
        similar_examples = self.find_similar_prompts(prompt, design_name, top_k=Examples_top_k)

        # Format examples like in hardware_general_agent.py
        for ex in similar_examples:
            formatted_example = f'Question: Create a SVA assertion that checks: {ex["prompt"]}\nReference Answer: {ex["reference_sva"]}'
            examples.append(formatted_example)
        
        # Extract the top suggestions from the Dict
        top_suggestions = suggestions.get('top_suggestions', [])
        rule_strings_only = suggestions.get('rule_strings_only', [])
        
        # BREAKPOINT 4: Check suggestions being used in second generation
        # Shows the actual suggestions that will be added to the prompt
        # print(f"[DEBUG] Using {len(top_suggestions)} suggestions for second generation")
        # print(f"[DEBUG] Found {len(rule_strings_only)} rule strings")
        # for i, rule in enumerate(rule_strings_only):
        #     print(f"[DEBUG] Rule {i+1}: {rule[:100]}...")
        # breakpoint()
        
        # You can choose to use either:
        # Option 1: Use full suggestions with reasoning paths (current)
        # rag_prompt = self.build_rag_prompt(prompt, testbench, suggestions=top_suggestions, examples=examples)
        
        # Option 2: Use only the rule strings as suggestions
        # Format rule strings as suggestions
        if rule_strings_only:
            formatted_rule_suggestions = [f"{rule}" for i, rule in enumerate(rule_strings_only)]
        else:
            formatted_rule_suggestions = []
            
        # Compose the RAG prompt using formatted rule suggestions
        rag_prompt = self.build_rag_prompt(prompt, testbench, suggestions=formatted_rule_suggestions, examples=examples)

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
        generated_sva = self.extract_sva_from_response(response)
        
        return generated_sva

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
            metrics = evaluate_response(
                response=sva_code,
                reference=reference_sva,
                row=row,
                check_functionality=True
            )
        
        # breakpoint()
        return metrics

    def generate_effectiveness_report(self):
        """Generate a comprehensive report on suggestion effectiveness"""
        if not hasattr(self, 'all_effectiveness_results') or not self.all_effectiveness_results:
            print("\nNo effectiveness results to report.")
            return
        
        print("\n" + "="*80)
        print("FINAL EFFECTIVENESS REPORT:")
        print("="*80)
        
        total_tests = len(self.all_effectiveness_results)
        total_suggestions = sum(r['summary']['total'] for r in self.all_effectiveness_results)
        total_found = sum(r['summary']['found'] for r in self.all_effectiveness_results)
        total_fixed = sum(r['summary']['fixed'] for r in self.all_effectiveness_results)
        
        print(f"\nOverall Statistics:")
        print(f"  Tests analyzed: {total_tests}")
        print(f"  Total suggestions: {total_suggestions}")
        print(f"  Suggestions found in CSV: {total_found}/{total_suggestions} ({total_found/total_suggestions*100:.1f}%)")
        if total_found > 0:
            print(f"  Suggestions that fixed issues: {total_fixed}/{total_found} ({total_fixed/total_found*100:.1f}%)")
        
        # Per-test breakdown
        print(f"\nPer-Test Breakdown:")
        for test_result in self.all_effectiveness_results:
            test_id = test_result['test_id']
            summary = test_result['summary']
            print(f"\n  Test ID {test_id}:")
            print(f"    Suggestions: {summary['total']}")
            print(f"    Found in CSV: {summary['found']}")
            print(f"    Fixed: {summary['fixed']}")
            
            # Show which suggestions fixed issues
            for result in test_result['results']:
                if result['fixed']:
                    print(f"    ✓ {result['trace_id']} (rank {result['rank']}, score {result['llm_score']:.3f})")
        
        # Analysis by rank
        print(f"\nEffectiveness by Rank:")
        rank_stats = defaultdict(lambda: {'total': 0, 'found': 0, 'fixed': 0})
        for test_result in self.all_effectiveness_results:
            for result in test_result['results']:
                rank = result['rank']
                rank_stats[rank]['total'] += 1
                if result['found_in_csv']:
                    rank_stats[rank]['found'] += 1
                    if result['fixed']:
                        rank_stats[rank]['fixed'] += 1
        
        for rank in sorted(rank_stats.keys()):
            stats = rank_stats[rank]
            found_rate = stats['found'] / stats['total'] * 100 if stats['total'] > 0 else 0
            fix_rate = stats['fixed'] / stats['found'] * 100 if stats['found'] > 0 else 0
            print(f"  Rank {rank}: {stats['fixed']}/{stats['found']} fixed ({fix_rate:.1f}%) out of {stats['total']} suggestions")
        
        print("\n" + "="*80)

    def run(self):
        results = []
        df = self.csv_df

        target_ids = self.only_test_ids if self.only_test_ids else [0]
        # rules = self.collect_all_training_rules()

        for test_id in target_ids:
            if test_id >= len(df):
                print(f"Test ID {test_id} out of range (CSV has {len(df)} rows)")
                continue

            test_row = df.iloc[test_id]
            suggestions = self.pickup_qtrees_for_test_case(test_id, test_row)
            second_sva = self.second_sva_generation(test_id, test_row, suggestions)

            # breakpoint()
            # print(f"second_sva: {second_sva}")

            result = self.test_assertion(second_sva, test_row['testbench'], test_row['prompt'], test_row['design_name'], 
                                         test_row['task_id'], reference_sva=test_row['ref_solution'])

            print(f"result: {result}")
            # breakpoint()

            results.append(result)

        # Generate effectiveness report if data was loaded
        if hasattr(self, 'effectiveness_mapping') and self.effectiveness_mapping:
            self.generate_effectiveness_report()

        return results

def main():
    """Main function"""
    tester = QTreeRAGTester()
    tester.run()


if __name__ == "__main__":
    main()
    