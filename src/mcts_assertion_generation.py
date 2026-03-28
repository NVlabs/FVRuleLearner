import os
import random
import math
from typing import List, Dict, Tuple
from config import FLAGS
from saver import saver
from FVEval.fv_eval import utils as utils2
from utils_agent import initiate_chat_with_retry
from FVEval.fv_eval.fv_tool_execution import (
    launch_jg,
    calculate_coverage_metric,
    check_assertions_error_parallel,
    analyze_coverage,
    analyze_uncovered_lines,
)

print = saver.log_info

class Node:
    def __init__(self, state, parent=None, agents=None, row=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.agents = agents
        self.untried_actions = None
        self.row = row
        self.temp_dir = os.path.join(saver.logdir, "tmp_sva")

    def get_untried_actions(self, user_prompt):
        if self.untried_actions is None:
            self.untried_actions = self.generate_actions_with_llm(user_prompt)
        return self.untried_actions

    def generate_actions_with_llm(self, user_prompt):
        coverage_report = self.state.get('coverage_report', '')
        prompt = user_prompt
        prompt += f"""
Uncovered lines or areas to focus on:
{coverage_report}

Given the current state of assertions and signals, suggest 5 possible actions for generating new assertions.
Each action should be a brief description or template for an assertion.

Current state:
Signals: {self.state['signals']}
Existing assertions: {self.state['assertions']}

Please provide 5 diverse actions for generating new assertions that target the uncovered areas.
"""
        response = initiate_chat_with_retry(self.agents["user"], self.agents["Coding"], message=prompt)
        actions = utils2.parse_list_response(response)
        return actions[:5]

    def is_fully_expanded(self, user_prompt):
        return len(self.get_untried_actions(user_prompt)) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.value / c.visits) + c_param * ((math.log(self.visits) / c.visits) ** 0.5)
            for c in self.children.values()
        ]
        return list(self.children.values())[choices_weights.index(max(choices_weights))]

    def rollout(self, user_prompt):
        current_state = self.state.copy()
        rollout_depth = 0
        max_rollout_depth = FLAGS.max_rollout_depth
        while not self.is_terminal(current_state) and rollout_depth < max_rollout_depth:
            possible_moves = self.generate_actions_with_llm(user_prompt)
            if not possible_moves:
                break
            action = random.choice(possible_moves)
            current_state = self.move(current_state, action)
            rollout_depth += 1
        return self.get_reward(current_state)

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

    def is_terminal(self, state):
        return state.get("coverage", 0) >= FLAGS.target_coverage

    def move(self, state, action):
        new_state = state.copy()
        new_assertion = self.generate_assertion(action)
        is_valid, error_type = self.validate_assertion(new_assertion)
        if is_valid:
            new_state["assertions"].append(new_assertion)
            if len(new_state["assertions"]) % FLAGS.coverage_check_interval == 0:
                new_state["coverage"], new_state["coverage_report"] = self.calculate_coverage(new_state["assertions"])
        else:
            if error_type == 'syntax_error':
                print(f"Syntax error in assertion: {new_assertion}")
            elif error_type == 'cex_error':
                print(f"CEX error in assertion: {new_assertion}")
        return new_state

    def generate_assertion(self, action):
        prompt = f"""
Given the action:
{action}

Generate a SystemVerilog assertion using the following signals:
{self.state['signals']}

Ensure that the assertion targets the uncovered areas if possible.

Provide only the assertion code, without any explanation.
"""
        lm_response = initiate_chat_with_retry(
            self.agents["user"],
            self.agents["Coding"],
            message=prompt
        )
        return utils2.parse_code_response(lm_response)

    def validate_assertion(self, assertion):
        design_rtl = self.row.prompt
        packaged_tb_text = utils2.package_testbench(self.row, assertion, FLAGS.task)
        validity_results = check_assertions_error_parallel(
            design_rtl, [packaged_tb_text], FLAGS.task, self.row.task_id,
            self.temp_dir, FLAGS.tcl_check_path, FLAGS.nparallel, 0, FLAGS.task
        )
        is_valid, error_type = validity_results[0]
        return is_valid, error_type

    def calculate_coverage(self, assertions):
        if assertions:
            combined_assertions = "\n".join(assertions)
            packaged_combined_tb = utils2.package_testbench(self.row, combined_assertions, FLAGS.task)

            jasper_output = analyze_coverage(
                self.row.prompt, packaged_combined_tb, FLAGS.task, self.row.task_id,
                self.temp_dir, FLAGS.tcl_report_path
            )

            coverage_metrics = calculate_coverage_metric(jasper_out_str=jasper_output)
            avg_coverage = sum(coverage_metrics.values()) / len(coverage_metrics)

            design_code = self.row.prompt
            coverage_report = analyze_uncovered_lines(jasper_out_str=jasper_output, design_code=design_code)

            return avg_coverage, coverage_report
        return 0, ""

    def get_reward(self, state):
        return state.get("coverage", 0)

def uct_search(root, max_iterations, user_prompt):
    for _ in range(max_iterations):
        node = root
        while not node.is_terminal(node.state) and node.is_fully_expanded(user_prompt):
            node = node.best_child()

        if not node.is_fully_expanded(user_prompt):
            action = random.choice(node.get_untried_actions(user_prompt))
            node.untried_actions.remove(action)
            new_state = node.move(node.state, action)
            child_node = Node(new_state, parent=node, agents=node.agents, row=node.row)
            node.children[action] = child_node
            node = child_node

        reward = node.rollout(user_prompt)
        node.backpropagate(reward)

    return root.best_child(c_param=0.)

def run_mcts_assertion_generation(agents, user_prompt: str, row):
    signals = utils2.extract_signals_from_module(row.prompt)
    root_state = {"assertions": [], "coverage": 0, "coverage_report": "", "signals": signals}
    root = Node(root_state, agents=agents, row=row)

    best_node = uct_search(root, FLAGS.max_iterations, user_prompt)

    final_assertions = []
    current_node = best_node
    while current_node.parent is not None:
        action = [action for action, child in current_node.parent.children.items() if child == current_node][0]
        assertion = current_node.state["assertions"][-1]
        final_assertions.insert(0, assertion)
        current_node = current_node.parent

    temp_dir = os.path.join(saver.logdir, "tmp_sva")
    coverage, valid_assertions, coverage_report = evaluate_assertions(agents, row, final_assertions, temp_dir)

    print(f"Final coverage: {coverage}")
    print(f"Valid assertions:\n{valid_assertions}")
    print(f"Coverage Report:\n{coverage_report}")

    return "\n".join(valid_assertions)

def evaluate_assertions(agents, row, assertions, temp_dir):
    design_rtl = row.prompt
    packaged_tb_texts = [utils2.package_testbench(row, assertion, FLAGS.task) for assertion in assertions]

    validity_results = check_assertions_error_parallel(
        design_rtl, packaged_tb_texts, FLAGS.task, row.task_id,
        temp_dir, FLAGS.tcl_check_path, FLAGS.nparallel, 0, FLAGS.task
    )
    valid_assertions = []
    for assertion, (is_valid, error_type) in zip(assertions, validity_results):
        if is_valid:
            valid_assertions.append(assertion)
        else:
            print(f"Invalid assertion: {assertion}")
            print(f"Error type: {error_type}")

    if valid_assertions:
        combined_assertions = "\n".join(valid_assertions)
        packaged_combined_tb = utils2.package_testbench(row, combined_assertions, FLAGS.task)

        jasper_output = analyze_coverage(
            row.prompt, packaged_combined_tb, FLAGS.task, row.task_id,
            temp_dir, FLAGS.tcl_report_path
        )

        coverage_metrics = calculate_coverage_metric(jasper_out_str=jasper_output)
        avg_coverage = sum(coverage_metrics.values()) / len(coverage_metrics)

        design_code = row.prompt
        coverage_report = analyze_uncovered_lines(jasper_out_str=jasper_output, design_code=design_code)

        return avg_coverage, valid_assertions, coverage_report

    return 0, [], ""

# Helper functions needed for analyze_uncovered_lines
def extract_and_parse_items(coverage_text):
    items = []
    lines = coverage_text.split('\n')
    current_item = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith("ITEM:"):
            if current_item:
                items.append(current_item)
            current_item = {}
        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            current_item[key] = value
    
    if current_item:
        items.append(current_item)
    
    return items

def extract_source_code_snippets(source_locations, design_code):
    snippets = {}
    design_lines = design_code.split('\n')
    
    for location in source_locations:
        if ':' in location:
            file_name, line_number = location.rsplit(':', 1)
            try:
                line_number = int(line_number)
                # Extract 3 lines before and after the specified line
                start = max(0, line_number - 4)
                end = min(len(design_lines), line_number + 3)
                relevant_lines = design_lines[start:end]
                snippet = '\n'.join(f"{i+start+1}: {line}" for i, line in enumerate(relevant_lines))
                snippets[location] = snippet
            except ValueError:
                snippets[location] = f"Error: Unable to parse line number from {location}"
        else:
            snippets[location] = f"Error: Invalid source location format {location}"
    
    return snippets

def analyze_uncovered_lines(jasper_out_str, design_code):
    START_MARKER = "### COVERAGE_REPORT_START ###"
    END_MARKER = "### COVERAGE_REPORT_END ###"

    start_idx = jasper_out_str.find(START_MARKER)
    end_idx = jasper_out_str.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        return ""

    coverage_text = jasper_out_str[start_idx + len(START_MARKER):end_idx].strip()

    undetectable_text = ""
    unprocessed_text = ""

    undetectable_start = coverage_text.find("puts $undetectable_coverage")
    unprocessed_start = coverage_text.find("puts $unprocessed_coverage")

    if undetectable_start != -1:
        undetectable_end = coverage_text.find("set unprocessed_coverage")
        if undetectable_end == -1:
            undetectable_end = len(coverage_text)
        undetectable_text = coverage_text[undetectable_start:undetectable_end].strip()

    if unprocessed_start != -1:
        unprocessed_text = coverage_text[unprocessed_start:].strip()

    if not undetectable_text and not unprocessed_text:
        return ""

    undetectable_items = extract_and_parse_items(undetectable_text)
    unprocessed_items = extract_and_parse_items(unprocessed_text)

    if not undetectable_items and not unprocessed_items:
        return ""

    source_locations = []
    for item in undetectable_items + unprocessed_items:
        if "source_location" in item:
            source_locations.append(item["source_location"])

    extracted_snippets = extract_source_code_snippets(source_locations, design_code)

    elements_to_keep = ['expression', 'formal_status', 'stimuli_status', 'checker_status', 'cover_item_type', 'bound', 'description']

    def format_item(item):
        result = ""
        for key in elements_to_keep:
            value = item.get(key, '')
            if value:
                if isinstance(value, dict):
                    value = str(value)
                if key == 'description':
                    description = value.strip('"\'').strip()
                    result += f"{key}: {description}\n"
                else:
                    result += f"{key}: {value.strip()}\n"
        return result

    coverage_report_str = ""

    if undetectable_items:
        coverage_report_str += "Undetectable Coverage Items:\n"
        for i, item in enumerate(undetectable_items, 1):
            if item:
                formal_status = item.get('formal_status', '')
                if 'Unreachable' in formal_status or 'Undetectable' in formal_status:
                    coverage_report_str += f"\nItem {i}:\n"
                    coverage_report_str += "----------------------------------------\n"
                    coverage_report_str += format_item(item)
                    if "source_location" in item and item["source_location"] in extracted_snippets:
                        coverage_report_str += f"Source Code:{extracted_snippets[item['source_location']]}\n"

    if unprocessed_items:
        coverage_report_str += "\nUnprocessed Coverage Items:\n"
        for i, item in enumerate(unprocessed_items, 1):
            if item:
                formal_status = item.get('formal_status', '')
                if 'Unreachable' in formal_status or 'Undetectable' in formal_status:
                    coverage_report_str += f"\nItem {i}:\n"
                    coverage_report_str += "----------------------------------------\n"
                    coverage_report_str += format_item(item)
                    if "source_location" in item and item["source_location"] in extracted_snippets:
                        coverage_report_str += f"Source Code:{extracted_snippets[item['source_location']]}\n"

    return coverage_report_str.strip()