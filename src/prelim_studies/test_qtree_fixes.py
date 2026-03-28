#!/usr/bin/env python3
"""
Test which QTree suggestions can fix specific test cases.
This script applies rules/suggestions and tests if they fix SVA issues.
"""

import json
import os
import sys
import re
import tempfile
import shutil
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FLAGS
from FVEval.fv_eval import utils as utils2
from FVEval.fv_eval.fv_tool_execution import check_single_assertion_error
from utils import setup_logger


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
    rule_group: List[Rule]
    original_sva: str
    modified_sva: str
    original_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    fixed: bool
    improvements: Dict[str, str]


class QTreeFixTester:
    def __init__(self, qtree_file: str, only_test_ids: List[int] = None):
        """Initialize the tester"""
        self.qtree_file = qtree_file
        self.only_test_ids = only_test_ids or [1]  # Default to test_id 1
        self.logger = setup_logger("QTreeFixTester")
        self.temp_dir = tempfile.mkdtemp(prefix="qtree_fix_test_")
        self.test_results = []
        self.group_test_results = []
        
        # Load qtree data
        with open(qtree_file, 'r') as f:
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
        
        # Apply different rule patterns
        if "Replace the reduction OR operator (`|`) with the logical OR operator (`||`)" in rule_text:
            # Look for patterns like !signal |-> and replace with !signal ||->
            new_sva = re.sub(r'(\![a-zA-Z_]\w*)\s*\|(?!->)', r'\1 ||', modified_sva)
            if new_sva != modified_sva:
                modified_sva = new_sva
                made_change = True
                
        elif "Add parentheses around the antecedent and consequent" in rule_text:
            # Pattern: antecedent |-> consequent or antecedent |=> consequent
            match = re.search(r'(.*?)\s*(\|->|\|=>)\s*(.*?)(?=\);|$)', modified_sva, re.DOTALL)
            if match:
                antecedent = match.group(1).strip()
                operator = match.group(2)
                consequent = match.group(3).strip()
                # Only add parentheses if not already present
                if not (antecedent.startswith('(') and antecedent.endswith(')')):
                    antecedent = f"({antecedent})"
                    made_change = True
                if not (consequent.startswith('(') and consequent.endswith(')')):
                    consequent = f"({consequent})"
                    made_change = True
                if made_change:
                    modified_sva = modified_sva.replace(match.group(0), f"{antecedent} {operator} {consequent}")
                    
        elif "Remove unnecessary operators" in rule_text and "bitwise `|`" in rule_text:
            # Remove redundant | operators
            new_sva = re.sub(r'\(\s*([!a-zA-Z_]\w*)\s*\|\s*(?=\|->)', r'\1 ', modified_sva)
            if new_sva != modified_sva:
                modified_sva = new_sva
                made_change = True
                
        elif "non-overlapping implication (`|=>`)" in rule_text:
            if "|->" in modified_sva:
                modified_sva = modified_sva.replace("|->", "|=>")
                made_change = True
                
        elif "overlapping implication (`|->`)" in rule_text:
            if "|=>" in modified_sva:
                modified_sva = modified_sva.replace("|=>", "|->")
                made_change = True
                
        elif "##[0:$]" in rule_text:
            if "##[1:$]" in modified_sva:
                modified_sva = modified_sva.replace("##[1:$]", "##[0:$]")
                made_change = True
                
        elif "##[1:$]" in rule_text:
            if "##[0:$]" in modified_sva:
                modified_sva = modified_sva.replace("##[0:$]", "##[1:$]")
                made_change = True
                
        elif "strong(" in rule_text:
            if "strong(" not in modified_sva and "##[" in modified_sva:
                # Add strong to timing sequences
                modified_sva = re.sub(r'(\|->\s*)(##\[[^\]]+\]\s*\w+)', r'\1strong(\2)', modified_sva)
                made_change = True
                
        elif "remove strong" in rule_text.lower():
            if "strong(" in modified_sva:
                modified_sva = re.sub(r'strong\s*\((.*?)\)', r'\1', modified_sva)
                made_change = True
        
        if made_change:
            # Wrap back in markdown if original was wrapped
            if sva_match:
                return f"```systemverilog\n{modified_sva}\n```"
            return modified_sva
            
        return None
    
    def apply_rules_to_sva(self, sva: str, rules: List[str]) -> str:
        """Apply multiple rules to an SVA assertion"""
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
            
        # Get testbench and other info
        testbench = entry.get('testbench', '')
        prompt = entry.get('prompt', '')
        design_name = entry.get('design_name', '')
        task_id = entry.get('task_id', '')
        
        # Create a dummy row object
        class DummyRow:
            def __init__(self, testbench, prompt):
                self.testbench = testbench
                self.prompt = prompt
                
        row = DummyRow(testbench, prompt)
        
        # Package the testbench
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
        """Test each rule individually"""
        results = []
        task_id = entry.get('task_id', '')
        
        # Extract test ID
        try:
            test_id = int(task_id.split('_')[-1]) if '_' in task_id else -1
        except:
            test_id = -1
            
        if self.only_test_ids and test_id not in self.only_test_ids:
            return results
            
        self.logger.info(f"\nTesting individual rules for {task_id} (test_id: {test_id})")
        
        original_sva = entry.get('generated_sva', '')
        original_metrics = entry.get('final_metrics', entry.get('previous_metrics', {}))
        design_name = entry.get('design_name', '')
        
        # Extract all rules
        qtree = entry.get('qtree_data', {})
        nodes = qtree.get('nodes', [])
        
        all_rules = []
        
        # Get rules from nodes
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
        
        # Get top-level rules
        for idx, rule_text in enumerate(qtree.get('rules', [])):
            if isinstance(rule_text, str) and rule_text.strip():
                rule = Rule(
                    id=f"top_level_rule_{idx}",
                    text=rule_text,
                    node_id="top_level",
                    node_question="Aggregated rules",
                    parent_chain=["Top level"]
                )
                all_rules.append(rule)
        
        # Test each rule
        for rule in all_rules:
            self.logger.info(f"Testing rule: {rule.text[:60]}...")
            
            modified_sva = self.apply_rule_to_sva(original_sva, rule.text)
            if not modified_sva or modified_sva == original_sva:
                continue
                
            new_metrics = self.test_assertion(entry, modified_sva)
            
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
                    
            result = TestResult(
                test_id=test_id,
                task_id=task_id,
                design_name=design_name,
                rule=rule,
                original_sva=original_sva,
                modified_sva=modified_sva,
                original_metrics=original_metrics,
                new_metrics=new_metrics,
                fixed=fixed,
                improvements=improvements
            )
            
            results.append(result)
            
            if fixed:
                self.logger.info(f"✓ FIXED by rule: {rule.text[:60]}...")
                
        return results
    
    def test_rule_combinations(self, entry: Dict) -> List[GroupTestResult]:
        """Test combinations of rules"""
        results = []
        task_id = entry.get('task_id', '')
        
        # Extract test ID
        try:
            test_id = int(task_id.split('_')[-1]) if '_' in task_id else -1
        except:
            test_id = -1
            
        if self.only_test_ids and test_id not in self.only_test_ids:
            return results
            
        self.logger.info(f"\nTesting rule combinations for {task_id}")
        
        original_sva = entry.get('generated_sva', '')
        original_metrics = entry.get('final_metrics', entry.get('previous_metrics', {}))
        design_name = entry.get('design_name', '')
        
        # Get all rules (same as above)
        qtree = entry.get('qtree_data', {})
        all_rules = self.extract_all_rules(qtree)
        
        # Test different combinations
        # 1. All rules from same node
        nodes = qtree.get('nodes', [])
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
                self.logger.info(f"Testing {len(node_rules)} rules from node {node['id']}")
                modified_sva = self.apply_rules_to_sva(
                    original_sva, 
                    [r.text for r in node_rules]
                )
                
                if modified_sva != original_sva:
                    new_metrics = self.test_assertion(entry, modified_sva)
                    
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
                        self.logger.info(f"✓ FIXED by rule group from node {node['id']}")
        
        # 2. All top-level rules
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
            self.logger.info(f"Testing all {len(top_rules)} top-level rules together")
            modified_sva = self.apply_rules_to_sva(
                original_sva,
                [r.text for r in top_rules]
            )
            
            if modified_sva != original_sva:
                new_metrics = self.test_assertion(entry, modified_sva)
                
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
                    self.logger.info("✓ FIXED by all top-level rules")
                    
        return results
    
    def extract_all_rules(self, qtree: Dict) -> List[Rule]:
        """Extract all rules from a qtree"""
        all_rules = []
        nodes = qtree.get('nodes', [])
        
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
                    
        for idx, rule_text in enumerate(qtree.get('rules', [])):
            if isinstance(rule_text, str) and rule_text.strip():
                rule = Rule(
                    id=f"top_level_rule_{idx}",
                    text=rule_text,
                    node_id="top_level",
                    node_question="Aggregated rules",
                    parent_chain=["Top level"]
                )
                all_rules.append(rule)
                
        return all_rules
    
    def run(self):
        """Run all tests"""
        all_individual_results = []
        all_group_results = []
        
        for entry in self.qtree_data:
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
        report.append("QTREE FIX TESTING REPORT")
        report.append("=" * 80)
        report.append(f"\nTest IDs analyzed: {self.only_test_ids}")
        report.append(f"Total individual rule tests: {len(self.test_results)}")
        report.append(f"Total group rule tests: {len(self.group_test_results)}")
        
        # Individual rule successes
        individual_fixes = [r for r in self.test_results if r.fixed]
        report.append(f"\nSuccessful individual rule fixes: {len(individual_fixes)}")
        
        if individual_fixes:
            report.append("\n" + "="*60)
            report.append("INDIVIDUAL RULE FIXES")
            report.append("="*60)
            
            for fix in individual_fixes:
                report.append(f"\nTest ID {fix.test_id} - {fix.task_id}")
                report.append(f"Design: {fix.design_name}")
                report.append(f"Rule: {fix.rule.text}")
                report.append(f"From node: {fix.rule.node_id} - {fix.rule.node_question[:50]}...")
                report.append("Improvements:")
                for metric, improvement in fix.improvements.items():
                    report.append(f"  {metric}: {improvement}")
                    
        # Group rule successes
        group_fixes = [r for r in self.group_test_results if r.fixed]
        report.append(f"\n\nSuccessful group rule fixes: {len(group_fixes)}")
        
        if group_fixes:
            report.append("\n" + "="*60)
            report.append("GROUP RULE FIXES")
            report.append("="*60)
            
            for fix in group_fixes:
                report.append(f"\nTest ID {fix.test_id} - {fix.task_id}")
                report.append(f"Design: {fix.design_name}")
                report.append(f"Number of rules in group: {len(fix.rule_group)}")
                report.append("Rules in group:")
                for rule in fix.rule_group:
                    report.append(f"  - {rule.text[:80]}...")
                report.append("Improvements:")
                for metric, improvement in fix.improvements.items():
                    report.append(f"  {metric}: {improvement}")
                    
        return "\n".join(report)
    
    def cleanup(self):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main():
    """Main function"""
    # Configuration - change these as needed
    only_test_ids = [1]  # Start with test_id 1 as requested
    qtree_file = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-22T09-03-15.426631_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtrees.json"
    
    print(f"Testing QTree fixes for test IDs: {only_test_ids}")
    
    tester = QTreeFixTester(qtree_file, only_test_ids)
    
    try:
        individual_results, group_results = tester.run()
        
        # Generate report
        report = tester.generate_report()
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/qtree_fix_test_report_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write(report)
            
        print(f"\nTest complete!")
        print(f"Report saved to: {output_file}")
        
        # Print summary
        individual_fixes = [r for r in individual_results if r.fixed]
        group_fixes = [r for r in group_results if r.fixed]
        
        print(f"\nSummary:")
        print(f"  Individual rule fixes: {len(individual_fixes)}/{len(individual_results)}")
        print(f"  Group rule fixes: {len(group_fixes)}/{len(group_results)}")
        
        if individual_fixes:
            print("\nSuccessful individual rules:")
            for fix in individual_fixes[:5]:  # Show first 5
                print(f"  - {fix.rule.text[:80]}...")
                
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
