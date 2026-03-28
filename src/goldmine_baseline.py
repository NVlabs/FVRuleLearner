import re
import subprocess
import os
import shutil
from typing import List, Tuple, Dict
import multiprocessing
import random

from config import FLAGS
from saver import saver
from FVEval.fv_eval import utils as utils2
from FVEval.fv_eval.fv_tool_execution import (
    launch_jg,
    launch_jg_with_queue,
    calculate_coverage_metric,
    calculate_error_metric,
    check_single_assertion_error,
    check_assertions_error_parallel,
    check_assertions_report_parallel,
    analyze_coverage,
    analyze_uncovered_lines,
    format_uncovered_items,
    
)
from pprint import pformat
import time
from design_preprocessing import preprocess_design

print = saver.log_info

def parse_goldmine_assertions(testbench_path: str) -> List[str]:
    """
    Parse assertions from a GOLDMINE testbench file.
    
    Args:
        testbench_path (str): Path to the testbench file
        
    Returns:
        List[str]: List of individual assertions
    """
    with open(testbench_path, 'r') as f:
        content = f.read()
    
    # Split content into individual assertions
    assertions = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('assert property'):
            assertions.append(line)
    
    return assertions

def print_copyable_metrics(results: Dict):
    """
    Print metric values tab-separated with 4 decimal places (0-1 scale).
    """
    metrics = results["coverage_metrics"]
    
    # Extract and format values
    values = [
        str(results["total_assertions"]),
        str(results["passing_count"]),
        f"{metrics.get('coverage_coi_statement', 0)/100:.4f}",
        f"{metrics.get('coverage_coi_branch', 0)/100:.4f}", 
        f"{metrics.get('coverage_coi_functional', 0)/100:.4f}",
        f"{metrics.get('coverage_coi_toggle', 0)/100:.4f}",
        f"{metrics.get('coverage_coi_expression', 0)/100:.4f}"
    ]
    
    print("\t".join(values))

def check_goldmine_assertions(
    design_rtl: str,
    goldmine_testbench_path: str,
    exp_id: str,
    task_id: str,
    temp_dir: str,
    tcl_check_path: str, 
    tcl_report_path: str,
    iteration: int,
    signal: str,
    task: str,
    cleanup_temp_files: bool,
    row
) -> Dict:
    """
    Verify GOLDMINE assertions and calculate coverage.
    """
    # Parse assertions from GOLDMINE testbench
    assertions = parse_goldmine_assertions(goldmine_testbench_path)
    if not assertions:
        return {
            "passing_assertions": [],
            "coverage_metrics": {},
            "total_assertions": 0,
            "passing_count": 0
        }
    
    # Step 1: Check each assertion individually
    passing_assertions = []
    
    for assertion in assertions:
        # Package the assertion into testbench
        packaged_tb = utils2.package_testbench(row, assertion, task)
        
        # Verify single assertion
        result = check_assertions_error_parallel(
            design_rtl=design_rtl,
            packaged_tb_texts=[packaged_tb],
            exp_id=exp_id,
            task_id=task_id,
            temp_dir=temp_dir,
            tcl_check_path=tcl_check_path,
            num_processes=1, 
            iteration=iteration,
            signal=signal,
            task=task,
            cleanup_temp_files=cleanup_temp_files
        )
        
        if result[0][0]: # If assertion passed
            passing_assertions.append(assertion)
            
    # Step 2: Get coverage metrics for all passing assertions combined
    if passing_assertions:
        # Combine all passing assertions
        combined_assertions = "\n".join(passing_assertions)
        
        # Package combined assertions
        packaged_combined_tb = utils2.package_testbench(row, combined_assertions, task)
        
        # Get coverage metrics
        combined_results = check_assertions_report_parallel(
            design_rtl=design_rtl,
            packaged_tb_texts=[packaged_combined_tb],
            exp_id=exp_id,
            task_id=task_id,
            temp_dir=temp_dir,
            tcl_check_path=tcl_report_path,
            num_processes=1,
            iteration=iteration,
            signal=signal,
            task=task,
            cleanup_temp_files=cleanup_temp_files,
            whole_batch=True
        )
        
        if combined_results and len(combined_results) > 0:
            _, _, coverage_metrics = combined_results[0]
        else:
            coverage_metrics = {}
    else:
        coverage_metrics = {}
        
    results = {
        "passing_assertions": passing_assertions,
        "coverage_metrics": coverage_metrics, 
        "total_assertions": len(assertions),
        "passing_count": len(passing_assertions)
    }
    
    printable_results = print_copyable_metrics(results)
    print(f"All metrics: {printable_results}")
    print(f"total assertions:{total_assertions}")
    print(f"passing count:{passing_count}")
    
    return results


