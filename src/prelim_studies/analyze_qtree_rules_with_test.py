#!/usr/bin/env python3
"""
Analyze qtree rules and test which ones can fix specific test cases.
Uses FLAGS configuration for only_test_ids.
"""

import json
import os
import sys
import re
import tempfile
import shutil
import csv
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FLAGS
from FVEval.fv_eval import utils as utils2
from FVEval.fv_eval.fv_tool_execution import check_single_assertion_error
from utils import setup_logger, get_user, get_host


@dataclass
class Rule:
    """Represents a rule generated from the qtree"""
    id: str
    text: str
    node_id: str
    node_question: str
    parent_chain: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of testing a specific suggestion"""
    test_id: int
    task_id: str
    design_name: str
    prompt: str
    rule: Rule
    original_sva: str
    modified_sva: str
    original_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    fixed: bool
    improvements: Dict[str, str]


@dataclass
class GroupTestResult:
    """Result of testing a group of suggestions together"""
    test_id: int
    task_id: str
    design_name: str
    prompt: str
    rule_group: List[Rule]
    original_sva: str
    modified_sva: str
    original_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    fixed: bool
    improvements: Dict[str, str]


class QTreeRuleTester:
    def __init__(self, qtree_file: str = None):
        """Initialize the tester using FLAGS configuration"""
        self.logger = setup_logger("QTreeRuleTester")
        
        # Use FLAGS configuration
        self.only_test_ids = FLAGS.only_test_ids if hasattr(FLAGS, 'only_test_ids') else [1]
        
        # Determine qtree file path
        if qtree_file:
            self.qtree_file = qtree_file
        else:
            # Use default path based on FLAGS
            base_dir = os.path.join(FLAGS.proj_dir, "Documents/fv/hardware-agent-marco/src/logs")
            # Find the most recent qtree file
            qtree_files = list(Path(base_dir).glob("*/qtrees.json"))
            if qtree_files:
                self.qtree_file = str(max(qtree_files, key=os.path.getmtime))
            else:
                raise FileNotFoundError("No qtree.json files found in logs directory")
                
        self.logger.info(f"Using qtree file: {self.qtree_file}")
        self.logger.info(f"Testing IDs: {self.only_test_ids}")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"qtree_test_{get_user()}_{get_host()}_")
        
        # Results storage
        self.test_results = []
        self.group_test_results = []
        self.successful_fixes = []
        
        # Load qtree data
        with open(self.qtree_file, 'r') as f:
            self.qtree_data = json.load(f)
            
    def extract_tree_path(self, nodes: List[Dict], node_id: str) -> List[str]:
        """Extract the full tree path for a node"""
        node_map = {n['id']: n for n in nodes}
        path = []
        current_id = node_id
        
        while current_id and current_id in node_map:
            node = node_map[current_id]
            path.append(f"{node['id']}: {node['question'][:50]}...")
            current_id = node.get('parent_id')
            
        return list(reversed(path))
    
    def apply_rule_to_sva(self, sva: str, rule_text: str) -> Optional[str]:
        """Apply a single rule to modify an SVA assertion"""
        # Extract clean SVA code from markdown if present
        sva_match = re.search(r'```(?:systemverilog)?\s*(.*?)\s*```', sva, re.DOTALL)
        if sva_match:
            sva_code = sva_match.group(1).strip()
        else:
            sva_code = sva.strip()
            
        modified_sva = sva_code
        made_change = False
        
        # Apply different rule patterns based on the rule text
        
        # 1. Replace reduction OR with logical OR
        if "Replace the reduction OR operator (`|`) with the logical OR operator (`||`)" in rule_text:
            # Pattern: !signal |-> becomes !signal ||->
            new_sva = re.sub(r'(\![a-zA-Z_]\w*)\s*\|(?!->)', r'\1 ||', modified_sva)
            if new_sva != modified_sva:
                modified_sva = new_sva
                made_change = True
                
        # 2. Add parentheses for clarity
        elif "Add parentheses around the antecedent and consequent" in rule_text:
            # Find implication patterns
            match = re.search(r'(.*?)\s*(\|->|\|=>)\s*(.*?)(?=\);|$)', modified_sva, re.DOTALL)
            if match:
                antecedent = match.group(1).strip()
                operator = match.group(2)
                consequent = match.group(3).strip()
                
                # Add parentheses if not already present
                if not (antecedent.startswith('(') and antecedent.endswith(')')):
                    antecedent = f"({antecedent})"
                    made_change = True
                if not (consequent.startswith('(') and consequent.endswith(')')):
                    consequent = f"({consequent})"
                    made_change = True
                    
                if made_change:
                    modified_sva = modified_sva.replace(match.group(0), f"{antecedent} {operator} {consequent}")
                    
        # 3. Remove unnecessary operators
        elif "Remove unnecessary operators" in rule_text and "bitwise `|`" in rule_text:
            new_sva = re.sub(r'\(\s*([!a-zA-Z_]\w*)\s*\|\s*(?=\|->)', r'\1 ', modified_sva)
            if new_sva != modified_sva:
                modified_sva = new_sva
                made_change = True
                
        # 4. Change implication type
        elif "non-overlapping implication (`|=>`)" in rule_text:
            if "|->" in modified_sva:
                modified_sva = modified_sva.replace("|->", "|=>")
                made_change = True
        elif "overlapping implication (`|->`)" in rule_text:
            if "|=>" in modified_sva:
                modified_sva = modified_sva.replace("|=>", "|->")
                made_change = True
                
        # 5. Timing adjustments
        elif "##[0:$]" in rule_text:
            if "##[1:$]" in modified_sva:
                modified_sva = modified_sva.replace("##[1:$]", "##[0:$]")
                made_change = True
        elif "##[1:$]" in rule_text:
            if "##[0:$]" in modified_sva:
                modified_sva = modified_sva.replace("##[0:$]", "##[1:$]")
                made_change = True
                
        # 6. Strong/weak sequence operators
        elif "add strong" in rule_text.lower() or "strong(" in rule_text:
            if "strong(" not in modified_sva and "##[" in modified_sva:
                # Add strong to timing sequences
                modified_sva = re.sub(r'(\|(?:->|=>)\s*)(##\[[^\]]+\]\s*\w+)', r'\1strong(\2)', modified_sva)
                if "strong(" in modified_sva:
                    made_change = True
        elif "remove strong" in rule_text.lower():
            if "strong(" in modified_sva:
                modified_sva = re.sub(r'strong\s*\((.*?)\)', r'\1', modified_sva)
                made_change = True
                
        # 7. Fix double parentheses
        elif "double parentheses" in rule_text.lower():
            modified_sva = re.sub(r'\(\(([^)]+)\)\)', r'(\1)', modified_sva)
            if modified_sva != sva_code:
                made_change = True
        
        if made_change:
            # Wrap back in markdown if original was wrapped
            if sva_match:
                return f"```systemverilog\n{modified_sva}\n```"
            return modified_sva
            
        return None
    
    def apply_rules_to_sva(self, sva: str, rules: List[str]) -> str:
        """Apply multiple rules to an SVA assertion in sequence"""
        current_sva = sva
        for rule in rules:
            modified = self.apply_rule_to_sva(current_sva, rule)
            if modified:
                current_sva = modified
        return current_sva
    
    def test_assertion(self, entry: Dict, modified_sva: str) -> Dict[str, Any]:
        """Test a modified assertion and return metrics"""
        # Extract clean SVA code
        sva_match = re.search(r'```(?:systemverilog)?\s*(.*?)\s*```', modified_sva, re.DOTALL)
        if sva_match:
            clean_sva = sva_match.group(1).strip()
        else:
            clean_sva = modified_sva.strip()
            
        # Get necessary data from entry
        testbench = entry.get('testbench', '')
        prompt = entry.get('prompt', '')
        design_name = entry.get('design_name', '')
        task_id = entry.get('task_id', '')
        
        # Create a dummy row object for compatibility
        class DummyRow:
            def __init__(self, testbench, prompt):
                self.testbench = testbench
                self.prompt = prompt
                
        row = DummyRow(testbench, prompt)
        
        # Package the testbench with the modified assertion
        packaged_tb = utils2.package_testbench(row, clean_sva, FLAGS.task)
        
        # Test the assertion
        try:
            is_valid, jasper_output, metrics = check_single_assertion_error(
                prompt, packaged_tb, FLAGS.task, task_id,
                "test", self.temp_dir, FLAGS.tcl_check_path,
                "qtree_test", 'signal', FLAGS.task, True,
                True, True, design_name
            )
            return metrics if metrics else {}
        except Exception as e:
            self.logger.error(f"Error checking assertion: {e}")
            return {}
    
    def test_individual_rules(self, entry: Dict) -> List[TestResult]:
        """Test each rule individually on the entry"""
        results = []
        task_id = entry.get('task_id', '')
        
        # Extract test ID
        try:
            test_id = int(task_id.split('_')[-1]) if '_' in task_id else -1
        except:
            test_id = -1
            
        # Skip if not in our test list
        if test_id not in self.only_test_ids:
            return results
            
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Testing individual rules for {task_id} (test_id: {test_id})")
        self.logger.info(f"{'='*60}")
        
        original_sva = entry.get('generated_sva', '')
        original_metrics = entry.get('final_metrics', entry.get('previous_metrics', {}))
        design_name = entry.get('design_name', '')
        prompt = entry.get('prompt', '')
        
        # Extract all rules from the qtree
        qtree = entry.get('qtree_data', {})
        nodes = qtree.get('nodes', [])
        
        all_rules = []
        
        # Get rules from individual nodes
        for node in nodes:
            tree_path = self.extract_tree_path(nodes, node['id'])
            for idx, rule_text in enumerate(node.get('rules_generated', [])):
                if isinstance(rule_text, str) and rule_text.strip():
                    rule = Rule(
                        id=f"{node['id']}_rule_{idx}",
                        text=rule_text,
                        node_id=node['id'],
                        node_question=node['question'],
                        parent_chain=tree_path
                    )
                    all_rules.append(rule)
        
        # Get aggregated top-level rules
        for idx, rule_text in enumerate(qtree.get('rules', [])):
            if isinstance(rule_text, str) and rule_text.strip():
                rule = Rule(
                    id=f"top_level_rule_{idx}",
                    text=rule_text,
                    node_id="top_level",
                    node_question="Aggregated rules from all nodes",
                    parent_chain=["Top level aggregated rules"]
                )
                all_rules.append(rule)
        
        self.logger.info(f"Found {len(all_rules)} rules to test")
        
        # Test each rule individually
        for rule in all_rules:
            self.logger.info(f"\nTesting rule: {rule.text[:60]}...")
            
            modified_sva = self.apply_rule_to_sva(original_sva, rule.text)
            if not modified_sva or modified_sva == original_sva:
                self.logger.info("  → No changes made by this rule")
                continue
                
            # Test the modified assertion
            new_metrics = self.test_assertion(entry, modified_sva)
            
            if not new_metrics:
                self.logger.warning("  → Failed to get metrics")
                continue
            
            # Check if fixed (all key metrics are 1.0)
            key_metrics = ['syntax', 'pec', 'relax_pec']
            fixed = all(new_metrics.get(m, 0) == 1.0 for m in key_metrics)
            
            # Calculate improvements
            improvements = {}
            for metric in key_metrics:
                old_val = original_metrics.get(metric, 0)
                new_val = new_metrics.get(metric, 0)
                if new_val > old_val:
                    improvements[metric] = f"{old_val} → {new_val}"
                    
            result = TestResult(
                test_id=test_id,
                task_id=task_id,
                design_name=design_name,
                prompt=prompt,
                rule=rule,
                original_sva=original_sva,
                modified_sva=modified_sva,
                original_metrics=original_metrics,
                new_metrics=new_metrics,
                fixed=fixed,
                improvements=improvements
            )
            
            results.append(result)
            
            # Log result
            if fixed:
                self.logger.info(f"  ✓ FIXED! All metrics are now 1.0")
                self.successful_fixes.append(result)
            elif improvements:
                self.logger.info(f"  → Improved: {improvements}")
            else:
                self.logger.info(f"  → No improvement")
                
        return results
    
    def test_rule_combinations(self, entry: Dict) -> List[GroupTestResult]:
        """Test combinations of rules together"""
        results = []
        task_id = entry.get('task_id', '')
        
        # Extract test ID
        try:
            test_id = int(task_id.split('_')[-1]) if '_' in task_id else -1
        except:
            test_id = -1
            
        # Skip if not in our test list
        if test_id not in self.only_test_ids:
            return results
            
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Testing rule combinations for {task_id}")
        self.logger.info(f"{'='*60}")
        
        original_sva = entry.get('generated_sva', '')
        original_metrics = entry.get('final_metrics', entry.get('previous_metrics', {}))
        design_name = entry.get('design_name', '')
        prompt = entry.get('prompt', '')
        
        qtree = entry.get('qtree_data', {})
        nodes = qtree.get('nodes', [])
        
        # Test 1: All rules from each node together
        for node in nodes:
            node_rules = []
            for idx, rule_text in enumerate(node.get('rules_generated', [])):
                if isinstance(rule_text, str) and rule_text.strip():
                    rule = Rule(
                        id=f"{node['id']}_rule_{idx}",
                        text=rule_text,
                        node_id=node['id'],
                        node_question=node['question'],
                        parent_chain=self.extract_tree_path(nodes, node['id'])
                    )
                    node_rules.append(rule)
                    
            if len(node_rules) > 1:
                self.logger.info(f"\nTesting {len(node_rules)} rules from node {node['id']} together")
                
                # Apply all rules from this node
                modified_sva = self.apply_rules_to_sva(
                    original_sva, 
                    [r.text for r in node_rules]
                )
                
                if modified_sva != original_sva:
                    new_metrics = self.test_assertion(entry, modified_sva)
                    
                    if new_metrics:
                        # Check if fixed
                        key_metrics = ['syntax', 'pec', 'relax_pec']
                        fixed = all(new_metrics.get(m, 0) == 1.0 for m in key_metrics)
                        
                        # Calculate improvements
                        improvements = {}
                        for metric in key_metrics:
                            old_val = original_metrics.get(metric, 0)
                            new_val = new_metrics.get(metric, 0)
                            if new_val > old_val:
                                improvements[metric] = f"{old_val} → {new_val}"
                                
                        result = GroupTestResult(
                            test_id=test_id,
                            task_id=task_id,
                            design_name=design_name,
                            prompt=prompt,
                            rule_group=node_rules,
                            original_sva=original_sva,
                            modified_sva=modified_sva,
                            original_metrics=original_metrics,
                            new_metrics=new_metrics,
                            fixed=fixed,
                            improvements=improvements
                        )
                        
                        results.append(result)
                        
                        if fixed:
                            self.logger.info(f"  ✓ FIXED by rule group from node {node['id']}")
                        elif improvements:
                            self.logger.info(f"  → Improved: {improvements}")
        
        # Test 2: All top-level aggregated rules together
        top_rules = []
        for idx, rule_text in enumerate(qtree.get('rules', [])):
            if isinstance(rule_text, str) and rule_text.strip():
                rule = Rule(
                    id=f"top_level_rule_{idx}",
                    text=rule_text,
                    node_id="top_level",
                    node_question="Aggregated rules",
                    parent_chain=["Top level"]
                )
                top_rules.append(rule)
                
        if top_rules:
            self.logger.info(f"\nTesting all {len(top_rules)} top-level rules together")
            
            modified_sva = self.apply_rules_to_sva(
                original_sva,
                [r.text for r in top_rules]
            )
            
            if modified_sva != original_sva:
                new_metrics = self.test_assertion(entry, modified_sva)
                
                if new_metrics:
                    # Check if fixed
                    key_metrics = ['syntax', 'pec', 'relax_pec']
                    fixed = all(new_metrics.get(m, 0) == 1.0 for m in key_metrics)
                    
                    # Calculate improvements
                    improvements = {}
                    for metric in key_metrics:
                        old_val = original_metrics.get(metric, 0)
                        new_val = new_metrics.get(metric, 0)
                        if new_val > old_val:
                            improvements[metric] = f"{old_val} → {new_val}"
                            
                    result = GroupTestResult(
                        test_id=test_id,
                        task_id=task_id,
                        design_name=design_name,
                        prompt=prompt,
                        rule_group=top_rules,
                        original_sva=original_sva,
                        modified_sva=modified_sva,
                        original_metrics=original_metrics,
                        new_metrics=new_metrics,
                        fixed=fixed,
                        improvements=improvements
                    )
                    
                    results.append(result)
                    
                    if fixed:
                        self.logger.info("  ✓ FIXED by all top-level rules")
                    elif improvements:
                        self.logger.info(f"  → Improved: {improvements}")
                        
        return results
    
    def run(self):
        """Run all tests"""
        self.logger.info(f"\nStarting QTree rule testing for test IDs: {self.only_test_ids}")
        
        all_individual_results = []
        all_group_results = []
        
        for entry in self.qtree_data:
            task_id = entry.get('task_id', '')
            
            # Check if this entry should be tested
            try:
                test_id = int(task_id.split('_')[-1]) if '_' in task_id else -1
            except:
                continue
                
            if test_id not in self.only_test_ids:
                continue
            
            # Test individual rules
            individual_results = self.test_individual_rules(entry)
            all_individual_results.extend(individual_results)
            
            # Test rule combinations
            group_results = self.test_rule_combinations(entry)
            all_group_results.extend(group_results)
            
        self.test_results = all_individual_results
        self.group_test_results = all_group_results
        
        return all_individual_results, all_group_results
    
    def generate_report(self):
        """Generate a comprehensive report"""
        report = []
        report.append("=" * 80)
        report.append("QTREE RULE TESTING REPORT")
        report.append("=" * 80)
        report.append(f"\nConfiguration:")
        report.append(f"  Task: {FLAGS.task}")
        report.append(f"  Model: {FLAGS.llm_model}")
        report.append(f"  Test IDs: {self.only_test_ids}")
        report.append(f"  QTree file: {self.qtree_file}")
        
        report.append(f"\nResults Summary:")
        report.append(f"  Total individual rule tests: {len(self.test_results)}")
        report.append(f"  Total group rule tests: {len(self.group_test_results)}")
        
        # Individual rule successes
        individual_fixes = [r for r in self.test_results if r.fixed]
        report.append(f"  Successful individual rule fixes: {len(individual_fixes)}")
        
        # Group rule successes
        group_fixes = [r for r in self.group_test_results if r.fixed]
        report.append(f"  Successful group rule fixes: {len(group_fixes)}")
        
        # Detailed results for successful fixes
        if individual_fixes:
            report.append("\n" + "="*60)
            report.append("SUCCESSFUL INDIVIDUAL RULE FIXES")
            report.append("="*60)
            
            for fix in individual_fixes:
                report.append(f"\nTest ID {fix.test_id} - {fix.task_id}")
                report.append(f"Design: {fix.design_name}")
                report.append(f"Prompt: {fix.prompt[:80]}...")
                report.append(f"\nRule that fixed the issue:")
                report.append(f"  ID: {fix.rule.id}")
                report.append(f"  Text: {fix.rule.text}")
                report.append(f"  From node: {fix.rule.node_id}")
                report.append(f"  Question: {fix.rule.node_question}")
                report.append("\nTree path:")
                for p in fix.rule.parent_chain:
                    report.append(f"  → {p}")
                report.append("\nMetric improvements:")
                for metric, improvement in fix.improvements.items():
                    report.append(f"  {metric}: {improvement}")
                report.append(f"\nOriginal SVA:")
                report.append(fix.original_sva)
                report.append(f"\nFixed SVA:")
                report.append(fix.modified_sva)
                    
        if group_fixes:
            report.append("\n\n" + "="*60)
            report.append("SUCCESSFUL GROUP RULE FIXES")
            report.append("="*60)
            
            for fix in group_fixes:
                report.append(f"\nTest ID {fix.test_id} - {fix.task_id}")
                report.append(f"Design: {fix.design_name}")
                report.append(f"Number of rules in group: {len(fix.rule_group)}")
                report.append("\nRules in group:")
                for rule in fix.rule_group:
                    report.append(f"  - {rule.text}")
                report.append("\nMetric improvements:")
                for metric, improvement in fix.improvements.items():
                    report.append(f"  {metric}: {improvement}")
                    
        # Summary of which rules are most effective
        report.append("\n\n" + "="*80)
        report.append("RULE EFFECTIVENESS SUMMARY")
        report.append("="*80)
        
        if individual_fixes:
            # Count fixes by rule pattern
            rule_patterns = defaultdict(int)
            for fix in individual_fixes:
                if "reduction OR" in fix.rule.text:
                    rule_patterns["Replace reduction OR with logical OR"] += 1
                elif "parentheses" in fix.rule.text:
                    rule_patterns["Add parentheses for clarity"] += 1
                elif "unnecessary operators" in fix.rule.text:
                    rule_patterns["Remove unnecessary operators"] += 1
                elif "overlapping implication" in fix.rule.text:
                    rule_patterns["Change implication type"] += 1
                elif "##[" in fix.rule.text:
                    rule_patterns["Adjust timing delays"] += 1
                elif "strong" in fix.rule.text.lower():
                    rule_patterns["Modify sequence strength"] += 1
                else:
                    rule_patterns["Other"] += 1
                    
            report.append("\nMost effective rule types:")
            for pattern, count in sorted(rule_patterns.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {pattern}: {count} fixes")
                
        return "\n".join(report)
    
    def generate_csv_summary(self, output_file: str):
        """Generate a CSV summary of results"""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'test_id', 'task_id', 'design_name', 'rule_type', 'rule_id', 'rule_text',
                'original_syntax', 'original_pec', 'original_relax_pec',
                'new_syntax', 'new_pec', 'new_relax_pec',
                'fixed', 'improvements'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write individual rule results
            for result in self.test_results:
                row = {
                    'test_id': result.test_id,
                    'task_id': result.task_id,
                    'design_name': result.design_name,
                    'rule_type': 'individual',
                    'rule_id': result.rule.id,
                    'rule_text': result.rule.text,
                    'original_syntax': result.original_metrics.get('syntax', 0),
                    'original_pec': result.original_metrics.get('pec', 0),
                    'original_relax_pec': result.original_metrics.get('relax_pec', 0),
                    'new_syntax': result.new_metrics.get('syntax', 0),
                    'new_pec': result.new_metrics.get('pec', 0),
                    'new_relax_pec': result.new_metrics.get('relax_pec', 0),
                    'fixed': result.fixed,
                    'improvements': '; '.join(f"{k}: {v}" for k, v in result.improvements.items())
                }
                writer.writerow(row)
                
            # Write group rule results
            for result in self.group_test_results:
                row = {
                    'test_id': result.test_id,
                    'task_id': result.task_id,
                    'design_name': result.design_name,
                    'rule_type': 'group',
                    'rule_id': f"group_{len(result.rule_group)}_rules",
                    'rule_text': ' | '.join(r.text[:50] + '...' for r in result.rule_group),
                    'original_syntax': result.original_metrics.get('syntax', 0),
                    'original_pec': result.original_metrics.get('pec', 0),
                    'original_relax_pec': result.original_metrics.get('relax_pec', 0),
                    'new_syntax': result.new_metrics.get('syntax', 0),
                    'new_pec': result.new_metrics.get('pec', 0),
                    'new_relax_pec': result.new_metrics.get('relax_pec', 0),
                    'fixed': result.fixed,
                    'improvements': '; '.join(f"{k}: {v}" for k, v in result.improvements.items())
                }
                writer.writerow(row)
    
    def cleanup(self):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main():
    """Main function"""
    logger = setup_logger("main")
    
    # Optional: Override FLAGS.only_test_ids if needed
    # FLAGS.only_test_ids = [1]  # Start with test_id 1
    
    logger.info(f"Starting QTree rule analysis with test IDs: {FLAGS.only_test_ids}")
    
    # Create tester (will use FLAGS configuration)
    tester = QTreeRuleTester()
    
    try:
        # Run tests
        individual_results, group_results = tester.run()
        
        # Generate report
        report = tester.generate_report()
        
        # Save outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_dir = os.path.join(FLAGS.proj_dir, "Documents/fv/hardware-agent-marco/src")
        
        # Save text report
        report_file = os.path.join(base_output_dir, f"qtree_rule_test_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
            
        # Save CSV summary
        csv_file = os.path.join(base_output_dir, f"qtree_rule_test_summary_{timestamp}.csv")
        tester.generate_csv_summary(csv_file)
        
        logger.info(f"\nAnalysis complete!")
        logger.info(f"Report saved to: {report_file}")
        logger.info(f"CSV summary saved to: {csv_file}")
        
        # Print summary
        individual_fixes = [r for r in individual_results if r.fixed]
        group_fixes = [r for r in group_results if r.fixed]
        
        print(f"\nSummary:")
        print(f"  Individual rule fixes: {len(individual_fixes)}/{len(individual_results)}")
        print(f"  Group rule fixes: {len(group_fixes)}/{len(group_results)}")
        
        if individual_fixes:
            print("\nSuccessful individual rules:")
            for fix in individual_fixes[:5]:  # Show first 5
                print(f"  Test ID {fix.test_id}: {fix.rule.text[:80]}...")
                
        if group_fixes:
            print("\nSuccessful rule groups:")
            for fix in group_fixes[:3]:  # Show first 3
                print(f"  Test ID {fix.test_id}: {len(fix.rule_group)} rules combined")
                
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
