import json
import networkx as nx
from typing import List, Any, Optional, Dict, Tuple
from config import FLAGS
from saver import saver
from utils_agent import initiate_chat_with_retry
import os
import shutil
import time
import multiprocessing
from pprint import pformat
from design_preprocessing import preprocess_design
from FVEval.fv_eval import utils as utils2
from FVEval.fv_eval.fv_tool_execution import (
    launch_jg,
    launch_jg_with_queue,
    calculate_coverage_metric,
    calculate_error_metric,
    check_assertions_error_parallel,
    read_file,
)
from FVEval.fv_eval import prompts_design2sva

# Override the built-in print function with saver.log_info for logging
print = saver.log_info

golden_testbench_path = '/home/scratch.liwan_mobile/training/arbiter/plain_arbiter_checker.sv'
vip_path = ["/home/scratch.liwan_mobile/training/vips/nv_assert_vip_arbiter.sv"]

def traverse_feature_graph(
    feature_graph_json_str: str,
    design_module: str,
    golden_testbench: str,
    vip_contents: Optional[Dict[str, Optional[str]]],
    agents: Dict[str, Any],
    row: Any
) -> List[str]:
    """
    Traverse the feature graph from the root node, generate assertions at leaf nodes,
    and evaluate the assertions.
    
    :param feature_graph_json_str: JSON string representing the feature graph.
    :param design_module: The design module content.
    :param golden_testbench: The golden testbench content.
    :param vip_contents: Dictionary mapping VIP file paths to their contents or None.
    :param agents: Dictionary of agent instances for communication.
    :param row: Row data containing necessary information for testbench packaging.
    :return: List of valid assertions.
    """
    feature_graph = json.loads(feature_graph_json_str)
    G = nx.DiGraph()
    
    for node in feature_graph['nodes']:
        G.add_node(node['id'], **node)
    
    for edge in feature_graph['edges']:
        G.add_edge(edge['from'], edge['to'], reason=edge.get('reason', ''))

    root = [node for node in G.nodes() if G.in_degree(node) == 0][0]
    return traverse_node(G, root, design_module, golden_testbench, vip_contents, agents, row)

def traverse_node(
    G: nx.DiGraph,
    node_id: str,
    design_module: str,
    golden_testbench: str,
    vip_contents: str,
    agents: Dict[str, Any],
    row: Any
) -> List[str]:
    node = G.nodes[node_id]
    node_type = node.get('type')
    
    if node_type == 'description':
        valid_assertions = []
        for successor in G.successors(node_id):
            valid_assertions.extend(traverse_node(G, successor, design_module, golden_testbench, vip_contents, agents, row))
        return valid_assertions
    
    elif node_type == 'feature':
        valid_assertions = []
        for successor in G.successors(node_id):
            valid_assertions.extend(traverse_node(G, successor, design_module, golden_testbench, vip_contents, agents, row))
        return valid_assertions
    
    elif node_type == 'assertion':
        assertion_text = node.get('assertion_text', '')
        if assertion_text:
            return validate_assertion(assertion_text, design_module, agents, row)
    
    return []

def validate_assertion(
    assertion_text: str,
    design_module: str,
    agents: Dict[str, Any],
    row: Any
) -> List[str]:
    """
    Validate a single assertion.

    :param assertion_text: The SystemVerilog assertion to validate.
    :param design_module: The design module content.
    :param agents: Dictionary of agent instances for communication.
    :param row: Row data containing necessary information for testbench packaging.
    :return: List containing the validated assertion if it's valid, otherwise an empty list.
    """
    # Package the testbench for the assertion
    packaged_tb = package_testbench(row, assertion_text, FLAGS.task)

    # Validate the assertion
    temp_dir = os.path.join(saver.logdir, "tmp_sva")
    os.makedirs(temp_dir, exist_ok=True)
    
    validity_results = check_assertions_error_parallel(
        design_module, [packaged_tb], FLAGS.task, row.task_id,
        temp_dir, FLAGS.tcl_check_path, FLAGS.nparallel, 0, FLAGS.task
    )

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Return the assertion if it's valid
    if validity_results[0][0]:
        return [assertion_text]
    else:
        return []

# def generate_and_validate_assertions(
#     feature_description: str,
#     suggestion: str,
#     design_module: str,
#     agents: Dict[str, Any],
#     row: Any
# ) -> List[str]:
#     """
#     Generate assertions based on the feature description and suggestion,
#     validate them, and collect only the passing assertions.
#     """
#     assertion_prompt = f"""
# Please analyze the following design module{' along with its associated VIP files' if vip_contents else ''} and the golden testbench. Generate a detailed 3-level feature tree in JSON format that captures the design features and assertions. The generated graph should align with the golden model to ensure consistency and correctness. Each node should be a key, and suggestions connecting specific assertions should be represented as edges in the JSON. Ensure that the leaf nodes are the assertions. The structure should be suitable for easy extraction and conversion into a graph for further use.

# Level 0 (Design Description): Provide an overview of the design, including its purpose, scope, and high-level behavior. This level sets the context for understanding the features and assertions.
# Level 1 (Design Features): Identify and list the main features of the design, ensuring they align with the functionalities specified in the golden testbench. There can be multiple features at this level, each representing a high-level functionality of the design.
# Level 2 (Specific Assertions): For each design feature at Level 1, provide specific assertions with natural language descriptions that correspond to the behaviors verified in the golden testbench. These assertions should be the leaf nodes of the tree.
# Suggestions (Edges): Include any suggestions (rules/guidelines) connecting specific assertions as edges, ensuring they reflect the relationships defined in the golden testbench.

# Design Module: 
# {design_module}

# Golden Testbench: 
# {golden_testbench}

# VIP Files:
# {vip_section}

# Output format:
# 1. Present the feature tree in JSON format.
# 2. Include 'edges' as a separate JSON key at the top level, which captures the relationships (suggestions) between specific assertions.
# 3. Ensure the output graph aligns with the golden model and is formatted for easy use in further processing.
# 4. The tree structure should be hierarchical and indented accordingly to reflect the design's features and assertions.
# 5. Ensure that the leaf nodes are the assertions.

# Example Output:
# json
# {{
# "nodes": [
#     {{
#     "id": "DesignDescription",
#     "type": "description",
#     "description": "An overview of the design's purpose and behavior."
#     }},
#     {{
#     "id": "Feature1",
#     "type": "feature",
#     "description": "This feature allows toggling the light on and off."
#     }},
#     {{
#     "id": "Feature2",
#     "type": "feature",
#     "description": "This feature adjusts the brightness of the light."
#     }},
#     {{
#     "id": "Assertion1",
#     "type": "assertion",
#     "description": "The light turns on when the switch is in the 'on' position."
#     }},
#     {{
#     "id": "Assertion2",
#     "type": "assertion",
#     "description": "The brightness increases when the up button is pressed."
#     }}
# ],
# "edges": [
#     {{
#     "from": "DesignDescription",
#     "to": "Feature1",
#     "reason": "Feature of toggling light derived from design purpose."
#     }},
#     {{
#     "from": "DesignDescription",
#     "to": "Feature2",
#     "reason": "Feature of adjusting brightness derived from design purpose."
#     }},
#     {{
#     "from": "Feature1",
#     "to": "Assertion1",
#     "reason": "Assertion verifying light toggling functionality."
#     }},
#     {{
#     "from": "Feature2",
#     "to": "Assertion2",
#     "reason": "Assertion verifying brightness adjustment functionality."
#     }}
# ]
# }}
# ```
# """

    # assertion_response = initiate_chat_with_retry(
    #     agents["user"],
    #     agents["Coding"],
    #     message=assertion_prompt
    # )

    # assertion_blocks = utils2.parse_code_response_multi(assertion_response)

    # # Validate the assertions
    # return validate_assertions(assertion_blocks, design_module, agents, row)

def validate_assertions(
    assertion_blocks: List[str],
    design_module: str,
    agents: Dict[str, Any],
    row: Any
) -> List[str]:
    """
    Validate assertions by checking syntax and counterexamples (CEX).
    Only keep the passing assertions.
    """
    temp_dir = os.path.join(saver.logdir, "tmp_sva")
    os.makedirs(temp_dir, exist_ok=True)
    iteration = 0  # Assuming single iteration for this context

    # Package the testbench for each assertion block
    packaged_tb_texts = [package_testbench(row, assertion, FLAGS.task) for assertion in assertion_blocks]

    # Validate assertions in parallel
    validity_results = check_assertions_error_parallel(
        design_module, packaged_tb_texts, FLAGS.task, row.task_id,
        temp_dir, FLAGS.tcl_check_path, FLAGS.nparallel, iteration, FLAGS.task
    )

    # Collect valid assertions
    valid_assertions = [assertion for assertion, (is_valid, _) in zip(assertion_blocks, validity_results) if is_valid]

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    return valid_assertions

def train_feature_graph(agents: Dict[str, Any], message: Dict[str, Any], row: Any) -> Dict[str, Any]:
    """
    High-level function to train the feature graph for the 'train' task in design2sva.

    :param agents: Dictionary of agent instances for communication.
    :param message: Dictionary containing necessary information for training.
    :param row: Row data containing additional information.
    :return: Dictionary containing the trained feature graph and related information.
    """
    # Extract necessary information from the message
    design_module = row.prompt
    # Read golden testbench and VIP file
    golden_testbench = read_file(golden_testbench_path)

    # Read all VIP files
    vip_contents = ""
    for path in vip_path:
        vip_contents += read_file(path) + "\n\n"  # Add two newlines between file contents for clarity

    # Preprocess the design module
    if FLAGS.add_comment:
        design_module = preprocess_design(agents=agents, design_code=design_module)

    iteration = 0
    valid_assertions = []
    coverage = 0

    while coverage < FLAGS.target_coverage and iteration < FLAGS.max_iterations:
        # Generate feature graph using an agent
        feature_graph_prompt = f"""
    Please analyze the following design module and the golden testbench. Generate a detailed 3-level feature tree in JSON format that captures the design features and assertions. The generated graph should align with the golden model to ensure consistency and correctness. Each node should be a key, and suggestions connecting specific assertions should be represented as edges in the JSON. Ensure that the leaf nodes are the assertions. The structure should be suitable for easy extraction and conversion into a graph for further use.

    Level 0 (Design Description): Provide an overview of the design, including its purpose, scope, and high-level behavior. This level sets the context for understanding the features and assertions.
    Level 1 (Design Features): Identify and list the main features of the design, ensuring they align with the functionalities specified in the golden testbench. There can be multiple features at this level, each representing a high-level functionality of the design.
    Level 2 (Specific Assertions): For each design feature at Level 1, provide specific assertions with natural language descriptions that correspond to the behaviors verified in the golden testbench. These assertions should be the leaf nodes of the tree.
    Suggestions (Edges): Include any suggestions (rules/guidelines) connecting specific assertions as edges, ensuring they reflect the relationships defined in the golden testbench.

    Design Module: 
    {design_module}

    Golden Testbench: 
    {golden_testbench}

    VIP Files:
    {vip_contents}

    Output format:
    1. Present the feature tree in JSON format.
    2. Include 'edges' as a separate JSON key at the top level, which captures the relationships (suggestions) between specific assertions.
    3. Ensure the output graph aligns with the golden model and is formatted for easy use in further processing.
    4. The tree structure should be hierarchical and indented accordingly to reflect the design's features and assertions.
    5. Ensure that the leaf nodes are the assertions.

    Example Output:
    ```json
    {{
    "nodes": [
        {{
            "id": "DesignDescription",
            "type": "description",
            "description": "An overview of the design's purpose and behavior."
        }},
        {{
            "id": "Feature1",
            "type": "feature",
            "description": "This feature allows toggling the light on and off."
        }},
        {{
            "id": "Feature2",
            "type": "feature",
            "description": "This feature adjusts the brightness of the light."
        }},
        {{
            "id": "Assertion1",
            "type": "assertion",
            "description": "The light turns on when the switch is in the 'on' position.",
            "assertion_text": "assert property (@(posedge clk) switch_on |-> ##[1:5] light_on);"
        }},
        {{
            "id": "Assertion2",
            "type": "assertion",
            "description": "The brightness increases when the up button is pressed.",
            "assertion_text": "assert property (@(posedge clk) up_button_pressed |-> brightness > $past(brightness));"
        }}
    ],
    "edges": [
        {{
            "from": "DesignDescription",
            "to": "Feature1",
            "reason": "Feature of toggling light derived from design purpose."
        }},
        {{
            "from": "DesignDescription",
            "to": "Feature2",
            "reason": "Feature of adjusting brightness derived from design purpose."
        }},
        {{
            "from": "Feature1",
            "to": "Assertion1",
            "reason": "Assertion verifying light toggling functionality."
        }},
        {{
            "from": "Feature2",
            "to": "Assertion2",
            "reason": "Assertion verifying brightness adjustment functionality."
        }}
    ]
    }}
    ```
    """

        feature_graph_response = initiate_chat_with_retry(
            agents["user"],
            agents["Coding"],
            message=feature_graph_prompt
        )

        feature_graph_json_str = utils2.parse_code_response_json(feature_graph_response)
        
        # Traverse the feature graph and generate/validate assertions
        new_valid_assertions = traverse_feature_graph(
            feature_graph_json_str,
            design_module,
            golden_testbench,
            vip_contents,
            agents,
            row
        )

        valid_assertions.extend(new_valid_assertions)

        # Calculate coverage
        coverage = calculate_coverage_metric(valid_assertions, design_module)

        iteration += 1

    return {
        'trained_feature_graph': feature_graph_json_str,
        'iterations': iteration,
        'final_assertions': valid_assertions,
        'coverage': coverage
    }
