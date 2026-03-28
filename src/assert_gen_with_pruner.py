import re
import subprocess
import os
import shutil
from typing import List, Tuple, Dict, Optional
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
    format_assertion_collection,
    remove_comments,
    eth_rxethmac,
    ge_1000baseX_rx,
)
from FVEval.fv_eval import (
    prompts_design2sva)
from utils_agent import initiate_chat_with_retry
from pprint import pformat
import time
from design_preprocessing import preprocess_design
from assertion_checker import AssertionTracker

print = saver.log_info

# Cleanup function (optional use)
def cleanup_temp_dirs(temp_dir, exp_id, task_id, iteration, num_items):
    for i in range(num_items):
        tmp_jg_proj_dir = os.path.join(temp_dir, "jg", f"{exp_id}_{i}_{task_id}_{iteration}")
        if os.path.isdir(tmp_jg_proj_dir):
            shutil.rmtree(tmp_jg_proj_dir)

def calculate_average_coverage(coverage_dict: Dict[str, float]) -> float:
    # List of COI models and stimuli models
    coi_models = FLAGS.evaluate_models
    
    # Sum the coverage percentages for the COI models
    total_coi_coverage = sum(coverage_dict.get(f'coverage_coi_{model}', 0.0) for model in coi_models)

    # Calculate the total number of models (coi + stimuli)
    total_models = len(coi_models)
    
    # Calculate the average
    return total_coi_coverage / total_models if total_models > 0 else 0.0


def generate_assumption(agents, user_prompt: str, row, target_coverage: float) -> str:
    if not FLAGS.only_assertion:
        assumption_response = initiate_chat_with_retry(
            agents["user"],
            agents["Assumption_Generation"],
            message=row.message_assumption
        )
        assumption_generation = utils2.parse_code_response(assumption_response)
    else:
        assumption_generation = row.assumption
    return assumption_generation

def generate_assertions(assumption, agents, user_prompt: str, row, target_coverage: float, max_iterations: int, signals_set, tracker: Optional[AssertionTracker] = None) -> Tuple[str, Dict[str, float]]:
    all_assertions = []  # NEW: Track all assertions regardless of validity
    all_valid_assertions = []  # Track only valid assertions
    unique_assertions = set()
    current_coverage = {}
    iteration = 0
    temp_dir = os.path.join(saver.logdir, "tmp_sva")
    os.makedirs(temp_dir, exist_ok=True)
    if assumption:
        combined_assertions = assumption
    start_time = time.time()
    Jasper_coverage_report = ""

    # To store metrics (time, coverage, number of correct assertions) for each iteration
    all_metrics = []
    # last_three_coverages = []
    exit_loops = False

    test_text = ""
    initial_tb = utils2.package_testbench(row, test_text, FLAGS.task)
    tb_modul_str = utils2.extract_module_name(initial_tb)
    
    # Get initial uncovered baseline
    initial_output = analyze_coverage(
        row.prompt, initial_tb, FLAGS.task, row.task_id,
        temp_dir, FLAGS.tcl_report_path, FLAGS.cleanup_temp_files,
        tb_modul_str, iteration=0
    )

    if tracker:
        tracker.set_row(row)
        initial_uncovered = analyze_uncovered_lines(
            jasper_out_str=initial_output,
            design_code=row.prompt
        )

        tracker.uncovered_holes_baseline = initial_uncovered
        # Save iteration 0 baseline using the dedicated function
        tracker.save_iteration_baseline(0, initial_uncovered)
        # design_meta = preprocess_design(agents, row.prompt)
        # design_meta = tracker.generate_line_block_summaries(agents, row.prompt)

    gen_assertions_count = 0
    # tracker.prev_iteration_uncovered = tracker.uncovered_holes_baseline if tracker else None

    while iteration < max_iterations:
        elapsed_time = time.time() - start_time
        if elapsed_time >= FLAGS.MAX_TIME:
            print(f"Stopping due to elapsed time exceeding {FLAGS.MAX_TIME / 3600:.2f} hours ({elapsed_time / 3600:.2f} hours elapsed).")
            print(f"All metrics: {' '.join(str(item) for sublist in all_metrics for item in sublist)}")
            exit_loops = True
            break

        metric_iteration = []

        if assumption:
            user_prompt_with_assumption = user_prompt.replace('assumption_para', combined_assertions)
        else:
            user_prompt_with_assumption = user_prompt

        if tracker and iteration > 0:
            # Get assertions and reasonings - uses current probabilities
            selected_assertions, reasoning_response = tracker.select_assertions_with_reasoning(
                iteration, agents, row.prompt, FLAGS.max_assertions
            )
            if len(selected_assertions) > 0:
                user_prompt_with_assumption += "\n\nPrevious useful assertions and their coverage impact:\n"
                user_prompt_with_assumption += reasoning_response

        if FLAGS.all_signals_in_one_shot:
            randomized_signals = random.sample(signals_set, len(signals_set))
            signals_str = ', '.join(randomized_signals)
            signals_size = len(signals_set)
            assertion_size = signals_size * FLAGS.num_assertions
            if iteration == 0:
                # Lily: DEBUG!!! Change it!!!
                signal_rule = f"Generate a total of {assertion_size} different assertions, with {FLAGS.num_assertions} assertions for each signal in the set '{signals_str}'."
            else:
                signal_rule = ""
        else:
            signal_rule = f"Only generate assertions related to signal '{next(iter(signals_set))}'."  # Will be updated in loop

        # Determine signals to process based on flags
        if FLAGS.with_individual_signal and not FLAGS.all_signals_in_one_shot:
            signals_to_process = signals_set
        else:
            signals_to_process = [', '.join(signals_set)] if signals_set else [None]

        assertion_candidates = ''

        for signal in signals_to_process:
            # signal_rule = f"Only generate assertions related to signal(s) '{signal}'." if signal else ''
            current_prompt = user_prompt_with_assumption.replace("signal_rule_PH", signal_rule)
            if FLAGS.all_signals_in_one_shot:
                current_prompt = current_prompt.replace('signal_size_PH', str(assertion_size))

            if FLAGS.coverage_report and iteration != 0:
                current_prompt += Jasper_coverage_report
                coverage_instruction = (
                    "\n\nPlease generate additional assertions different from the previous assertions to improve coverage, especially focusing on the lines not covered."
                )
                if FLAGS.use_inter_param:
                    coverage_instruction += (
                        " Each feature's code should be self-contained. Ensure that any intermediate signals (e.g., state, in_A, in_B, etc.) are declared and assigned within the feature's code block."
                    )
                else:
                    coverage_instruction += " Please do not generate intermediate parameters."
                if FLAGS.all_signals_in_one_shot:
                    coverage_instruction += f" Generate a total of {assertion_size} different assertions, with {FLAGS.num_assertions} assertions for each signal in the set '{signals_str}'."
                current_prompt += coverage_instruction

            # Generate assertions using the language model
            for temperature in FLAGS.temperature_list:
                agents["Coding"].llm_config["temperature"] = temperature
                lm_response = initiate_chat_with_retry(
                    agents["user"],
                    agents["Coding"],
                    message=current_prompt
                )
                assertion_blocks = utils2.parse_code_response_multi(lm_response)

                # Filter out duplicate assertions after removing comments
                pruned_assertions = []
                for assertion in assertion_blocks:
                    assertion_no_comments = remove_comments(assertion)
                    if assertion_no_comments not in unique_assertions:
                        unique_assertions.add(assertion_no_comments)
                        pruned_assertions.append(assertion)

                if FLAGS.with_pruner:
                    assertion_blocks_formatted = "\n\n".join(
                    [f'```systemverilog\n{assertion}\n```' for assertion in pruned_assertions]
                    )
                    assertion_candidates += assertion_blocks_formatted
                # SVA pruning agent
                    if iteration != 0:
                        user_prompt_pruning = user_prompt
                        user_prompt_pruning += Jasper_coverage_report
                        user_prompt_pruning += "Below is a list of SystemVerilog Assertions (SVAs) generated for the design. These assertions are intended to cover various aspects of the design's behavior, including signal interactions, state transitions, and key outputs. The assertions aim to maximize coverage, especially focusing on the Cone of Influence (COI) of critical signals and modules. Your task is to review the generated assertions and apply pruning, ensuring that the remaining assertions provide the highest value for comprehensive formal verification." + assertion_candidates
                        user_prompt_pruning += f"""
        Criteria for LLM Pruning of SVAs:
        1. Ensure No Issues: Retain only assertions without syntax errors or counterexample (CEX) issues.
        2. Remove Redundancy: Eliminate assertions that cover the same signal or behavior in different forms.
        3. Focus on COI: Keep assertions within the Cone of Influence (COI) of critical signals or modules.
        4. Optimize Signal Interactions: Retain assertions that test complex or critical signal interactions.
        5. Address Coverage Gaps: Preserve assertions that fill gaps, especially for corner cases or less-exercised paths.
        6. Minimize for Efficiency: Prune to the fewest high-impact assertions for faster JasperGold verification.
        7. Address Uncovered Lines: Retain or refine assertions that target lines or branches identified as uncovered in the coverage report, ensuring they are effectively exercised.

        Please give me {int(FLAGS.assertion_top_k_ratios*assertion_size)} assertions after pruning, wrapped in the following format. Ensure that there is no intermediate signals.
        ```systemverilog
        <pruned_assertion_0>
        ```
        ```systemverilog
        <pruned_assertion_1>
        ```
        """
                        lm_response = initiate_chat_with_retry(
                            agents["user"],
                            agents["Prune_Assertions"],
                            message=user_prompt_pruning
                        )

            gen_assertions_count += len(pruned_assertions)
            design_rtl = row.prompt
            packaged_tb_texts = [
                utils2.package_testbench(row, assertion, FLAGS.task) for assertion in pruned_assertions
            ]

            # Validate assertions in parallel
            if not FLAGS.post_coverage_calc:
                if tracker:
                    validity_results = check_assertions_report_parallel(
                    design_rtl, packaged_tb_texts, FLAGS.task, row.task_id,
                    temp_dir, FLAGS.tcl_report_path, FLAGS.nparallel, iteration, None, FLAGS.task, FLAGS.cleanup_temp_files, whole_batch = False)
                else:
                    validity_results = check_assertions_error_parallel(
                    design_rtl, packaged_tb_texts, FLAGS.task, row.task_id,
                    temp_dir, FLAGS.tcl_check_path, FLAGS.nparallel, iteration, None, FLAGS.task, FLAGS.cleanup_temp_files)
            else:
                validity_results = check_assertions_error_parallel(
                design_rtl, packaged_tb_texts, FLAGS.task, row.task_id,
                temp_dir, FLAGS.tcl_check_path, FLAGS.nparallel, iteration, None, FLAGS.task, FLAGS.cleanup_temp_files)

            if tracker:
                # Process each assertion individually
                for assertion, (is_valid, jasper_output, metrics) in zip(pruned_assertions, validity_results):
                    if is_valid:
                        all_valid_assertions.append(assertion)  # Directly append to all_valid_assertions
                        assertion_uncovered = metrics['coverage_details']

                        if not FLAGS.post_coverage_calc:
                            # Get uncovered holes for this assertion

                            # Calculate newly covered holes by comparing with previous iteration's coverage
                            newly_covered = {}
                            prev_holes = set(tracker.uncovered_holes_baseline.keys())
                            curr_holes = set(assertion_uncovered.keys())
                            covered_holes = prev_holes - curr_holes
                            
                            for hole in covered_holes:
                                newly_covered[hole] = tracker.uncovered_holes_baseline[hole]

                            # Save assertion results
                            tracker.save_rag_results(
                                assertion=assertion,
                                metrics=metrics,
                                newly_covered_holes=newly_covered
                            )

                        else:
                            # Save valid assertion with initial metrics from syntax/CEX check
                            tracker.save_rag_results(
                                assertion=assertion,
                                metrics={
                                    "metrics": metrics,
                                    "coverage_details": {}  # Empty initially, will be updated when selected
                                },
                                newly_covered_holes={}  # Empty initially, will be updated when selected
                            )

                # Add deduplication after appending
                all_valid_assertions = list(dict.fromkeys(all_valid_assertions))
                combined_assertions = (assumption + "\n\n" + "\n".join(all_valid_assertions) 
                    if FLAGS.with_assumptions else "\n".join(all_valid_assertions))
                
                # Analyze coverage
                packaged_combined_tb = utils2.package_testbench(row, combined_assertions, FLAGS.task)

                combined_output = analyze_coverage(
                    row.prompt, packaged_combined_tb, FLAGS.task, row.task_id,
                    temp_dir, FLAGS.tcl_report_path, FLAGS.cleanup_temp_files,
                    tb_modul_str, iteration=iteration
                )

                # print(f"combined_output: {combined_output}")
                
                # Update previous iteration's uncovered holes for next iteration
                tracker.uncovered_holes_baseline = analyze_uncovered_lines(
                    jasper_out_str=combined_output,
                    design_code=row.prompt
                )

                tracker.save_iteration_baseline(iteration + 1, initial_uncovered)

            else:
                valid_assertions = [assertion for assertion, (is_valid, _, _) in zip(pruned_assertions, validity_results) 
                                if is_valid]
                all_valid_assertions.extend(valid_assertions)
                all_valid_assertions = list(dict.fromkeys(all_valid_assertions))

                # Combine assertions
                combined_assertions = (assumption + "\n\n" + "\n".join(all_valid_assertions) 
                                    if FLAGS.with_assumptions else "\n".join(all_valid_assertions))

                # Analyze coverage
                packaged_combined_tb = utils2.package_testbench(row, combined_assertions, FLAGS.task)

                # Analyze coverage
                combined_output = analyze_coverage(
                    row.prompt, packaged_combined_tb, FLAGS.task, row.task_id,
                    temp_dir, FLAGS.tcl_report_path, FLAGS.cleanup_temp_files, tb_modul_str,
                    iteration=iteration
                )

            # if tracker:
            #     # Update baseline for the uncovered holes after each iteration
            #     current_uncovered = analyze_uncovered_lines(jasper_out_str=combined_output, design_code=row.prompt)
            #     tracker.uncovered_holes_baseline = current_uncovered  # Update baseline cumulatively

            # **Calculate coverage metrics**
            
            coverage_metrics = calculate_coverage_metric(combined_output)
            current_coverage = {
                f'coverage_{ctype}_{model}': coverage_metrics.get(f'coverage_{ctype}_{model}', 0.0)
                for ctype in ['stimuli', 'coi']
                for model in ['statement', 'branch', 'functional', 'toggle', 'expression']
            }

            coverage_results = {
            'metrics': coverage_metrics,
            'coverage_details': tracker.uncovered_holes_baseline}

            if FLAGS.unified_sample:
                pass
            else:
                if tracker:
                    tracker.update_probabilities_after_coverage(iteration, coverage_results, FLAGS.max_iterations)

            # Calculate average coverage
            avg_coverage = calculate_average_coverage(current_coverage)
            correct_assertions_count = len(all_valid_assertions)
            # last_three_coverages.append(avg_coverage)
            # if len(last_three_coverages) > 3:
            #     last_three_coverages.pop(0)

            print(f"Iteration {iteration + 1}: Generated {correct_assertions_count} valid assertions. Average COI coverage: {avg_coverage:.2f}%")
            print(f"Coverage breakdown: {current_coverage}")

            # Check for convergence
            # if len(last_three_coverages) == 3 and all(
            #     abs(last_three_coverages[i] - last_three_coverages[i - 1]) < 0.001
            #     for i in range(1, 3)
            # ):
            #     print(f"Coverage has converged at {avg_coverage:.2f}% for three consecutive iterations. Stopping...")
            #     exit_loops = True
            #     break

            # Record metrics
            end_time = time.time()
            saver.save_stats(f'time_{iteration}', end_time - start_time)
            metric_iteration.extend([end_time - start_time, gen_assertions_count, correct_assertions_count])
            for key, value in current_coverage.items():
                saver.save_stats(f'iteration_{iteration}_{key}', value)
                metric_iteration.append(value)

            # Update coverage report
            if FLAGS.coverage_report:
                Jasper_coverage_report = "\n\nCOVERAGE REPORT\n" + "\n".join(
                    f"{key.replace('_', '|')}|{value:.2f}%" for key, value in current_coverage.items()
                ) + "\n"
                if avg_coverage != 0.0:
                    uncovered_lines_report = analyze_uncovered_lines(
                        jasper_out_str=combined_output,
                        design_code=row.prompt
                    )

                    Jasper_coverage_report += format_uncovered_items(uncovered_lines_report)

            # Check if the target coverage has been reached
            # Lily: DEBUG
            if avg_coverage >= target_coverage:
                exit_loops = True
                break

        if exit_loops:
            all_metrics.append(metric_iteration)
            break
        iteration += 1
        all_metrics.append(metric_iteration)

    print(f"All metrics: {' '.join(str(item) for sublist in all_metrics for item in sublist)}")
    return combined_assertions, current_coverage


def run_assertion_generation_w_pruner(agents, user_prompt, row):
    if "design2sva" in FLAGS.task:
        # if row.task_id == "eth_rxethmac":
        #     row.prompt = eth_rxethmac
        # elif row.task_id == "ge_1000baseX_rx":
        #     row.prompt = ge_1000baseX_rx
        signals_set = utils2.extract_signals_from_module(row.prompt, row.task_id)

        # Initialize the tracker
        if FLAGS.use_ICRL:
            tracker = AssertionTracker(save_dir=os.path.join(saver.logdir, "assertion_tracking"), base_probability = FLAGS.base_probability, iteration_weight = FLAGS.iteration_weight, min_probability = FLAGS.min_probability, delta = FLAGS.delta)
        else:
            tracker = None

        # Iterative optimization process
        if FLAGS.with_assumptions:
            assumption = generate_assumption(agents, user_prompt, row, FLAGS.target_coverage)
        else:
            assumption = None

        final_assertions, final_coverage = generate_assertions(
            assumption=assumption, 
            agents=agents,
            user_prompt=user_prompt, 
            row=row,
            target_coverage=FLAGS.target_coverage,
            max_iterations=FLAGS.max_iterations,
            signals_set=signals_set,
            tracker=tracker  # Pass the initialized tracker
        )

        # final_assertions, final_coverage = generate_assertions(
        #     assumption, agents, user_prompt, row, FLAGS.target_coverage, FLAGS.max_iterations, signals_set
        # )
        avg_coverage = calculate_average_coverage(final_coverage)
        print(f"Final average coverage: {avg_coverage:.2f}%")
        print(f"Final coverage breakdown: {final_coverage}")
        # Combine assumption and assertions
        testbench_generation = final_assertions
        return testbench_generation