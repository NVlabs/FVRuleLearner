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
    analyze_coverage,
    analyze_uncovered_lines,
    format_uncovered_items,
    eth_rxethmac,
    ge_1000baseX_rx,
    AutoSVA2_rules,
)
from FVEval.fv_eval import (
    prompts_design2sva)
from utils_agent import initiate_chat_with_retry
from pprint import pformat
import time
from design_preprocessing import preprocess_design

print = saver.log_info

# Cleanup function (optional use)
def cleanup_temp_dirs(temp_dir, exp_id, task_id, iteration, num_items):
    for i in range(num_items):
        tmp_jg_proj_dir = os.path.join(temp_dir, "jg", f"{exp_id}_{i}_{task_id}_{iteration}")
        if FLAGS.cleanup_temp_files:
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

def generate_assertions(assumption, agents, user_prompt: str, row, target_coverage: float, max_iterations: int, signals_set) -> Tuple[str, Dict[str, float]]:
    all_valid_assertions = []
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

    # Track last three coverage values to check for convergence
    # last_three_coverages = []
    exit_loops = False  # Flag to control loop exit

    gen_assertions_count = 0
    while iteration < max_iterations:
        elapsed_time = time.time() - start_time
        if elapsed_time >= FLAGS.MAX_TIME:
            print(f"Stopping due to elapsed time exceeding 4 hours ({elapsed_time / 3600:.2f} hours).")
            # Print all collected metrics before breaking
            print(f"All metrics: {' '.join(str(item) for sublist in all_metrics for item in sublist)}")
            exit_loops = True
            break

        metric_iteration = []
        # Update the prompt with the combined assertions and coverage report if available
        if assumption:
            user_prompt_with_assumption_default = user_prompt.replace('assumption_para', combined_assertions)
        else:
            user_prompt_with_assumption_default = user_prompt

            # Check if signals_set is empty or not
        if FLAGS.with_individual_signal:
            # Set up based on whether we're doing one-shot or individual processing
            if FLAGS.all_signals_in_one_shot:
                randomized_signals = random.sample(signals_set, len(signals_set))
                signals_str = ', '.join(randomized_signals)
                signals_size = len(signals_set)
                assertion_size = signals_size * FLAGS.num_assertions
                signal_rule = f"Generate a total of {assertion_size} different assertions, with {FLAGS.num_assertions} assertions for each signal in the set '{signals_str}'."
            else:
                signal_rule = f"Only generate assertions related to signal '{next(iter(signals_set))}'."  # Will be updated in loop
                
            if assumption:
                user_prompt_with_assumption_default = user_prompt.replace('assumption_para', combined_assertions)
            else:
                user_prompt_with_assumption_default = user_prompt

            # Prepare the base prompt
            user_prompt_with_assumption = user_prompt_with_assumption_default.replace('signal_rule_PH', signal_rule)
            if FLAGS.all_signals_in_one_shot:
                user_prompt_with_assumption = user_prompt_with_assumption.replace('signal_size_PH', str(assertion_size))
            
            if FLAGS.baseline_AutoSVA2:
                user_prompt_with_assumption += AutoSVA2_rules

            if FLAGS.coverage_report and iteration != 0:
                user_prompt_with_assumption += Jasper_coverage_report
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
                user_prompt_with_assumption += coverage_instruction

            # Process signals either all at once or individually
            if FLAGS.all_signals_in_one_shot:
                signals_to_process = [signals_set]  # Process all signals in one iteration
            else:
                signals_to_process = [[signal] for signal in signals_set]  # Process each signal individually

            for current_signals in signals_to_process:
                if not FLAGS.all_signals_in_one_shot:
                    current_signal = next(iter(current_signals))  # Get the single signal
                    # Update prompt for individual signal
                    user_prompt_with_assumption = user_prompt_with_assumption_default.replace(
                        "with_hyphen_PH", 
                        f"- The signal '{current_signal}' should be included in the assertions."
                    )
                    user_prompt_with_assumption = user_prompt_with_assumption.replace(
                        "signal_rule_PH", 
                        f"Only generate assertions related to signal '{current_signal}'."
                    )

                # Get assertions from the language model
                lm_response = initiate_chat_with_retry(
                    agents["user"],
                    agents["Coding"],
                    message=user_prompt_with_assumption
                )
                assertion_blocks = utils2.parse_code_response_multi(lm_response)
                gen_assertions_count += len(assertion_blocks)

                # LAAG-RV: Add reflection and refinement module
                if FLAGS.baseline_LAAGRV:
                    refined_assertions = []
                    for assertion in assertion_blocks:
                        refinement_count = 0
                        current_assertion = assertion
                        
                        while refinement_count < FLAGS.laagrv_max_cex_refinements:
                            packaged_tb = utils2.package_testbench(row, current_assertion, FLAGS.task)
                            
                            # Verify assertion
                            is_valid, jasper_output, metrics = check_single_assertion_error(
                                row.prompt, packaged_tb, FLAGS.task, row.task_id,
                                str(len(refined_assertions)), temp_dir, FLAGS.tcl_check_path,
                                str(iteration), 'signal', FLAGS.task, FLAGS.cleanup_temp_files,
                                whole_batch=False
                            )
                            
                            if is_valid:
                                refined_assertions.append(current_assertion)
                                break
                            
                            # Get error feedback
                            error_lines = [line for line in re.findall(r'ERROR.*', jasper_output) 
                                            if 'dummy' not in line]
                            error_string = '\n'.join(error_lines)
                            
                            if refinement_count >= FLAGS.laagrv_max_cex_refinements - 1:
                                print(f"LAAG-RV: Assertion abandoned after {FLAGS.laagrv_max_cex_refinements} attempts")
                                break
                            
                            # Generate error-specific reflection prompt
                            reflection_prompt = f"""Analyze the following assertion and its verification error:
                            
                            RTL Design:
                            {row.prompt}

                            Assertion:
                            {current_assertion}
                            
                            Verification Error:
                            {error_lines}
                            
                            Please fix the issues while maintaining the original verification intent.

                            Provide only the corrected assertion."""

                            refinement_response = initiate_chat_with_retry(
                                agents["user"],
                                agents["Coding"],
                                message=reflection_prompt
                            )
                            
                            current_assertion = utils2.parse_code_response(refinement_response)
                            refinement_count += 1
                    
                    # Use refined assertions
                    assertion_blocks = refined_assertions

                # Prepend the assumption to each assertion block
                if FLAGS.with_assumptions:
                    assertion_blocks_with_assumptions = [assumption + "\n\n" + assertion for assertion in assertion_blocks]
                else:
                    assertion_blocks_with_assumptions = [assertion for assertion in assertion_blocks]

                # Package the testbench for each assertion block
                design_rtl = row.prompt
                packaged_tb_texts = [utils2.package_testbench(row, assertion, FLAGS.task) for assertion in assertion_blocks_with_assumptions]

                # Validate assertions in parallel
                validity_results = check_assertions_error_parallel(
                    design_rtl, packaged_tb_texts, FLAGS.task, row.task_id, temp_dir, 
                    FLAGS.tcl_check_path, FLAGS.nparallel, iteration, 
                    'signal' if FLAGS.all_signals_in_one_shot else current_signal, 
                    FLAGS.task, FLAGS.cleanup_temp_files
                )

                # Filter assertions based on validity
                valid_assertions = [
                    assertion for assertion, result in zip(assertion_blocks, validity_results) if result[0]
                ]
                all_valid_assertions.extend(valid_assertions)

                # Deduplicate while preserving order
                all_valid_assertions = list(dict.fromkeys(all_valid_assertions))

                # Combine valid assertions
                if FLAGS.with_assumptions:
                    combined_assertions = assumption + "\n\n" + "\n".join(all_valid_assertions)
                else:
                    combined_assertions = "\n".join(all_valid_assertions)

                signal_desc = signals_str if FLAGS.all_signals_in_one_shot else current_signal
                print(f"Combined assertions after signal(s) '{signal_desc}':\n{combined_assertions}")

                # Package the combined testbench
                packaged_combined_tb = utils2.package_testbench(row, combined_assertions, FLAGS.task)

                tb_modul_str = utils2.extract_module_name(packaged_combined_tb)
                # Analyze coverage
                jasper_output = analyze_coverage(
                    row.prompt, packaged_combined_tb, FLAGS.task, row.task_id,
                    temp_dir, FLAGS.tcl_report_path, FLAGS.cleanup_temp_files, tb_modul_str, iteration=iteration
                )

                coverage_metrics = calculate_coverage_metric(jasper_out_str=jasper_output)
                
                # Initialize coverage metrics dictionary
                current_coverage = {}
                coverage_types = ['stimuli', 'coi']
                coverage_models = ['statement', 'branch', 'functional', 'toggle', 'expression']

                # Extract coverage metrics
                for ctype in coverage_types:
                    for model in coverage_models:
                        key = f'coverage_{ctype}_{model}'
                        current_coverage[key] = coverage_metrics.get(key, 0.0)

                # Calculate and track coverage metrics
                avg_coverage = calculate_average_coverage(current_coverage)
                correct_assertions_count = len(all_valid_assertions)

                # last_three_coverages.append(avg_coverage)
                # if len(last_three_coverages) > 3:
                #     last_three_coverages.pop(0)

                print(f"Iteration {iteration + 1} after signal(s) '{signal_desc}': Generated {len(valid_assertions)} valid assertions. Average COI coverage: {avg_coverage:.2f}%")
                print(f"Coverage breakdown after signal(s) '{signal_desc}': {current_coverage}")
                # print(f"Last three coverage values: {[f'{x:.2f}%' for x in last_three_coverages]}")

                # Record metrics
                end_time = time.time()
                saver.save_stats(f'time_{iteration}', end_time - start_time)
                metric_iteration.extend([end_time - start_time, gen_assertions_count, correct_assertions_count])

                # Save coverage metrics
                for key, value in current_coverage.items():
                    saver.save_stats(f'iteration_{iteration}_{key}', value)
                    metric_iteration.append(value)

                # Check for coverage convergence
                # if len(last_three_coverages) == 3 and all(
                #     abs(last_three_coverages[i] - last_three_coverages[i-1]) < 0.001 
                #     for i in range(1, 3)
                # ):
                #     print(f"Coverage has converged at {avg_coverage:.2f}% for three consecutive iterations. Stopping...")
                #     print(f"All metrics: {' '.join(str(item) for sublist in all_metrics for item in sublist)}")
                #     exit_loops = True
                #     break

                # Update coverage report if needed
                if FLAGS.coverage_report:
                    Jasper_coverage_report = "\n\nCOVERAGE REPORT\n" + "\n".join(
                        f"{key.replace('_', '|')}|{value:.2f}%" for key, value in current_coverage.items()
                    ) + "\n"
                    if avg_coverage != 0.0:
                        uncovered_lines_report = analyze_uncovered_lines(
                            jasper_out_str=jasper_output,
                            design_code=row.prompt
                        )

                        Jasper_coverage_report += format_uncovered_items(uncovered_lines_report)

                    # Check if target coverage reached
                    if avg_coverage >= target_coverage:
                        print(f"All metrics: {' '.join(str(item) for sublist in all_metrics for item in sublist)}")
                        exit_loops = True
                        break

        if exit_loops:
            all_metrics.append(metric_iteration)
            break
        iteration += 1
        print(f"iteration:{iteration}")

        all_metrics.append(metric_iteration)

    print(f"All metrics: {' '.join(str(item) for sublist in all_metrics for item in sublist)}")
    return combined_assertions, current_coverage

# def generate_assertions_AutoSVA(assumption, agents, user_prompt: str, row, target_coverage: float, signals_set) -> Tuple[str, Dict[str, float]]:
#     all_valid_assertions = []
#     all_metrics = []
#     metric_iteration = []
#     start_time = time.time()
#     temp_dir = os.path.join(saver.logdir, "tmp_sva")
#     os.makedirs(temp_dir, exist_ok=True)

#     # Preprocess the design prompt
#     row.prompt = preprocess_design(agents=agents, design_code=row.prompt)
    
#     # Initialize the combined assertions with the assumption if provided
#     combined_assertions = assumption if assumption else ""
    
#     # Construct the prompt with assumptions and signals if applicable
#     user_prompt_with_assumption = user_prompt.replace('assumption_para', combined_assertions)

#     if signals_set:
#         signals_str = ', '.join(signals_set)
#         user_prompt_with_assumption = user_prompt_with_assumption.replace(
#             'with_hyphen_PH', f"- The assertions should use the interface parameters: {signals_str}."
#         )
#         user_prompt_with_assumption = user_prompt_with_assumption.replace(
#             'signal_rule_PH', f"Only generate assertions related to the signals: {signals_str}."
#         )
    
#     signals_str = ', '.join(signals_set)
#     signals_size = len(signals_set)
#     assertion_size = signals_size * FLAGS.num_assertions

#     user_prompt_with_assumption = user_prompt_with_assumption.replace('signal_size_PH', str(assertion_size))

#     user_prompt_with_assumption += AutoSVA2_rules
#     user_prompt_with_assumption += f"Generate a total of {assertion_size} different  assertions, with {FLAGS.num_assertions} assertions for each signal in the set '{signals_str}'."

#     # Generate assertions using the language model
#     lm_response = initiate_chat_with_retry(agents["user"], agents["Coding"], message=user_prompt_with_assumption)
#     assertion_blocks = utils2.parse_code_response_multi(lm_response)
#     gen_assertions_count = len(assertion_blocks)
    
#     # Prepend the assumption to each assertion block if required
#     if FLAGS.with_assumptions:
#         assertion_blocks_with_assumptions = [assumption + "\n\n" + assertion for assertion in assertion_blocks]
#     else:
#         assertion_blocks_with_assumptions = assertion_blocks

#     # Package the testbench for each assertion block
#     design_rtl = row.prompt
#     packaged_tb_texts = [utils2.package_testbench(row, assertion, FLAGS.task) for assertion in assertion_blocks_with_assumptions]
    
#     # Validate assertions in parallel
#     validity_results = check_assertions_error_parallel(
#         design_rtl, packaged_tb_texts, FLAGS.task, row.task_id, temp_dir,
#         FLAGS.tcl_check_path, FLAGS.nparallel, 0, 'signal', FLAGS.task, FLAGS.cleanup_temp_files
#     )
    
#     # Filter out valid assertions
#     valid_assertions = [assertion for assertion, (is_valid, _) in zip(assertion_blocks, validity_results) if is_valid]
#     all_valid_assertions.extend(valid_assertions)

#     # Deduplicate while preserving order
#     all_valid_assertions = list(dict.fromkeys(all_valid_assertions))
#     correct_assertions_count = len(valid_assertions)
    
#     # Combine all valid assertions
#     combined_assertions = "\n".join(all_valid_assertions) if not FLAGS.with_assumptions else assumption + "\n\n" + "\n".join(all_valid_assertions)
    
#     # Package the combined testbench
#     packaged_combined_tb = utils2.package_testbench(row, combined_assertions, FLAGS.task)
    
#     # Analyze coverage
#     tb_modul_str = utils2.extract_module_name(packaged_combined_tb)
#     iteration = 0
#     jasper_output = analyze_coverage(
#         row.prompt, packaged_combined_tb, FLAGS.task, row.task_id,
#         temp_dir, FLAGS.tcl_report_path, FLAGS.cleanup_temp_files, tb_modul_str, iteration=iteration
#     )

#     # Calculate coverage metrics
#     coverage_metrics = calculate_coverage_metric(jasper_out_str=jasper_output)
#     current_coverage = {f'coverage_{ctype}_{model}': coverage_metrics.get(f'coverage_{ctype}_{model}', 0.0)
#                         for ctype in ['stimuli', 'coi'] for model in ['statement', 'branch', 'functional', 'toggle', 'expression']}
    
#     avg_coverage = calculate_average_coverage(current_coverage)
    
#     print(f"Generated {len(valid_assertions)} valid assertions with average COI coverage: {avg_coverage:.2f}%")
#     print(f"Coverage breakdown: {current_coverage}")
#     end_time = time.time()
#     saver.save_stats(f'time_{iteration}', end_time - start_time)
#     metric_iteration.append(end_time - start_time)
#     metric_iteration.append(gen_assertions_count)
#     metric_iteration.append(correct_assertions_count)

#     # Save coverage metrics for each type and model
#     for key, value in current_coverage.items():
#         saver.save_stats(f'iteration_{iteration}_{key}', value)
#         metric_iteration.append(value)

#     all_metrics.append(metric_iteration)
#     print(f"All metrics: {' '.join(str(item) for sublist in all_metrics for item in sublist)}")

#     return combined_assertions, current_coverage


def run_assertion_generation(agents, user_prompt, row):
    if FLAGS.baseline_GOLDMINE:
        from goldmine_baseline import check_goldmine_assertions
        goldmine_results = check_goldmine_assertions(
            design_rtl=row.prompt,
            goldmine_testbench_path=FLAGS.goldmine_testbench_path,
            exp_id=FLAGS.task,
            task_id=row.task_id,
            temp_dir = os.path.join(saver.logdir, "tmp_sva"),
            tcl_check_path=FLAGS.tcl_check_path,
            tcl_report_path=FLAGS.tcl_report_path,
            iteration=0,
            signal='signal',
            task=FLAGS.task,
            cleanup_temp_files=FLAGS.cleanup_temp_files,
            row=row
        )
        
        # If using GOLDMINE baseline, return its results instead of generating new assertions
        return "\n".join(goldmine_results["passing_assertions"])

    if "design2sva" in FLAGS.task:
        signals_set = utils2.extract_signals_from_module(row.prompt, row.task_id)
        if FLAGS.iterative_optimization:
            # Iterative optimization process
            if FLAGS.with_assumptions:
                assumption = generate_assumption(agents, user_prompt, row, FLAGS.target_coverage)
            else:
                assumption = None
            final_assertions, final_coverage = generate_assertions(
                assumption, agents, user_prompt, row, FLAGS.target_coverage, FLAGS.max_iterations, signals_set
            )
            avg_coverage = calculate_average_coverage(final_coverage)
            print(f"Final average coverage: {avg_coverage:.2f}%")
            print(f"Final coverage breakdown: {final_coverage}")
            # Combine assumption and assertions
            testbench_generation = final_assertions
            return testbench_generation

        # elif FLAGS.baseline_AutoSVA2:
        #     if FLAGS.with_assumptions:
        #         assumption = generate_assumption(agents, user_prompt, row, FLAGS.target_coverage)
        #     else:
        #         assumption = None
        #     final_assertions, final_coverage = generate_assertions_AutoSVA(
        #         assumption, agents, user_prompt, row, FLAGS.target_coverage, signals_set
        #     )
        #     avg_coverage = calculate_average_coverage(final_coverage)
        #     print(f"Final average coverage: {avg_coverage:.2f}%")
        #     print(f"Final coverage breakdown: {final_coverage}")
        #     # Combine assumption and assertions
        #     testbench_generation = final_assertions
        #     return testbench_generation

        else:
            # One-time inference
            if not FLAGS.only_assertion:
                # Generate assumption using the Assumption_Generation agent
                assumption_response = initiate_chat_with_retry(
                    agents["user"],
                    agents["Assumption_Generation"],
                    message=row.message_assumption
                )
                assumption_generation = utils2.parse_code_response(assumption_response)
            else:
                # Use the provided assumption
                assumption_generation = row.assumption

            # Replace placeholder in user prompt with the generated assumption
            message_w_assumption = user_prompt.replace('assumption_para', assumption_generation)

            # Generate assertions using the Coding agent
            assertion_response = initiate_chat_with_retry(
                agents["user"],
                agents["Coding"],
                message=message_w_assumption
            )
            assertion_generation_list = utils2.parse_code_response_multi(assertion_response)

            # Combine assumption and assertions
            assertion_generation = "\n\n".join(assertion_generation_list)
            testbench_generation = assumption_generation + "\n\n" + assertion_generation
            return testbench_generation