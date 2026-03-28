from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import math
import random
import hashlib
import os
import re
from config import FLAGS
from FVEval.fv_eval import utils as utils2
from utils_agent import initiate_chat_with_retry
from FVEval.fv_eval.fv_tool_execution import format_newly_covered_items, format_assertion_collection
from FVEval.fv_eval.fv_tool_execution import (
    calculate_coverage_metric,
    calculate_error_metric,
    check_single_assertion_error,
    check_assertions_error_parallel,
    check_assertions_report_parallel,
    analyze_coverage,
    analyze_uncovered_lines,
    format_newly_covered_items,
    format_uncovered_items,
    format_assertion_collection,
    remove_comments,
    eth_rxethmac,
    ge_1000baseX_rx,
)

class AssertionTracker:
    def __init__(self, save_dir: str, base_probability: float, iteration_weight: float, min_probability: float, delta: float):
        # Existing initialization
        self.save_dir = Path(save_dir)
        self.coverage_baseline = None
        self.iterations = {}
        self.line_descriptions = {}
        self.block_descriptions = {}
        self.assertion_groups = {}
        self.group_probabilities = {}
        self.base_probability = base_probability
        self.iteration_weight = iteration_weight
        self.min_probability = min_probability  # 10% of base_probability
        self.delta = delta  # Coverage-based probability adjustment factor
        
        # Coverage tracking
        self.uncovered_holes_baseline = None
        # self.prev_iteration_uncovered = None  # Stores previous iteration's combined coverage holes
        self.assertion_coverage_impact = {}
        self.coverage_calculated = {}  # Key: assertion_hash, Value: bool
        
        # Enhanced assertion tracking
        self.assertion_rag_info = {}  # Key: assertion_hash, Value: dict with assertion info
        self.iteration_data = {}  # Key: iteration, Value: dict with iteration data
        
        # Add storage for iteration baselines
        self.iteration_baselines = {}  # Key: iteration number, Value: uncovered holes for that iteration
        self.uncovered_holes_baseline = None  # Keep this for initial baseline
        self.design_row = None  # Add storage for row object

    def set_row(self, row):
        """Set the row object for use in testbench generation"""
        self.design_row = row

    def generate_design_meta(self, agents: Dict, design_code: str) -> None:
        """Generate initial design metadata using LLM"""
        specs_prompt = """
        Analyze this RTL design and identify:
        1. Important code blocks (modules, always blocks, FSMs, etc.) with their line ranges
        2. Critical individual lines (state changes, key assignments, etc.)

        Provide output in this JSON format:
        ```json
        {
            "blocks": [
                {
                    "location": "10-20",
                    "description": "Main FSM controller block implementing protocol handshake. Controls ready/valid signaling and handles backpressure."
                }
            ],
            "lines": [
                {
                    "location": "15",
                    "description": "Critical state transition from IDLE to BUSY based on valid signal assertion."
                }
            ]
        }
        ```

        Design Code:
        ```verilog
        {code}
        ```
        """

        response = initiate_chat_with_retry(
            agents["user"],
            agents["Analysis"],
            message=specs_prompt.format(code=design_code)
        )
        
        try:
            specs_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if specs_match:
                specs = json.loads(specs_match.group(1))
                self.design_meta = specs
                
                # Initialize mapping dictionaries for quick lookups
                self._init_design_mappings(design_code)
        except Exception as e:
            print(f"Error generating design metadata: {e}")
            self.design_meta = {"blocks": [], "lines": []}
        
        self._init_design_mappings(design_code)

    def _init_design_mappings(self, design_code: str) -> None:
        """Initialize mappings for quick line/block lookups"""
        self.line_to_block = {}  # Maps line numbers to containing blocks
        self.line_meta = {}      # Maps line numbers to line metadata
        code_lines = design_code.splitlines()
        
        # Process blocks
        for block in self.design_meta["blocks"]:
            start, end = map(int, block["location"].split("-"))
            block["code"] = "\n".join(code_lines[start-1:end])
            for line_num in range(start, end + 1):
                if line_num not in self.line_to_block:
                    self.line_to_block[line_num] = []
                self.line_to_block[line_num].append(block)
        
        # Process lines
        for line in self.design_meta["lines"]:
            line_num = int(line["location"])
            self.line_meta[line_num] = line
            line["code"] = code_lines[line_num-1].strip()

    def _extract_location(self, item: Dict) -> Optional[Tuple[int, int, int, int]]:
        """Extract line range from uncovered item in (row,col)-(row,col) format"""
        try:
            if "source_location" in item:
                # Format: (174,49)-(174,72)
                pattern = r'\((\d+),(\d+)\)-\(\d+,(\d+)\)'
                match = re.match(pattern, item["source_location"])
                if match:
                    row = int(match.group(1))
                    start_col = int(match.group(2))
                    end_col = int(match.group(3))
                    return (row, start_col, row, end_col)
        except Exception as e:
            print(f"Error extracting location: {e}")
        return None

    def match_uncovered_to_design(self, uncovered_data: Dict) -> Dict:
        """Match uncovered items with design metadata and line/block specifications"""
        matched = {}
        
        for item_hash, item in uncovered_data.items():
            location = self._extract_location(item)
            if not location:
                continue
                
            row = location[0]  # Row number from (row,col)-(row,col)
            
            matched[item_hash] = {
                "item": item,
                "location": location,
                "blocks": [],
                "lines": [],
                "description": "",
                "code": ""
            }
            
            # Map to blocks containing this line
            for block in self.design_meta["blocks"]:
                start_line, end_line = map(int, block["location"].split("-"))
                if start_line <= row <= end_line:
                    matched[item_hash]["blocks"].append(block)
                    
            # Map to specific line if it's marked as critical
            for line in self.design_meta["lines"]:
                if int(line["location"]) == row:
                    matched[item_hash]["lines"].append(line)
                    
            # Add combined description
            matched[item_hash]["description"] = self._generate_combined_description(
                matched[item_hash]["blocks"],
                matched[item_hash]["lines"]
            )
            
            # Add code snippet
            if self.design_meta.get("code"):
                code_lines = self.design_meta["code"].splitlines()
                matched[item_hash]["code"] = code_lines[row-1]
        
        return matched

    def update_iteration_coverage(self, iteration: int, uncovered_holes: Dict):
        """
        Update the tracking of uncovered holes for each iteration.
        
        Args:
            iteration: Current iteration number
            uncovered_holes: Dict of currently uncovered holes
        """
        self.iteration_uncovered[iteration] = uncovered_holes
        
    def get_coverage_impact(self, assertion_hash: str) -> Dict:
        """
        Get the coverage impact details for a specific assertion.
        
        Args:
            assertion_hash: Hash of the assertion to analyze
            
        Returns:
            Dict containing coverage impact metrics
        """
        if assertion_hash in self.assertion_rag_info:
            info = self.assertion_rag_info[assertion_hash]
            return {
                "newly_covered_holes": len(info.get("newly_covered_holes", {})),
                "coverage_metrics": info.get("metrics", {}),
                "coverage_details": info.get("coverage_details", {})
            }
        return {}

    def get_iteration_coverage_progress(self, iteration: int) -> Dict:
        """
        Get coverage progress for a specific iteration.
        
        Args:
            iteration: Iteration number to analyze
            
        Returns:
            Dict containing coverage progress metrics
        """
        if iteration in self.iteration_uncovered:
            current_holes = set(self.iteration_uncovered[iteration].keys())
            baseline_holes = set(self.uncovered_holes_baseline.keys())
            
            return {
                "total_covered": len(baseline_holes - current_holes),
                "remaining_uncovered": len(current_holes),
                "coverage_percentage": (1 - len(current_holes) / len(baseline_holes)) * 100
                if baseline_holes else 0
            }
        return {}

    def retrieve_rag_info(self, iteration: int) -> Dict:
        """
        Retrieve RAG info for the current iteration.
        """
        iteration_rag_info = {
            assertion_hash: info
            for assertion_hash, info in self.assertion_rag_info.items()
            if info.get("iteration") == iteration
        }
        self.iteration_data[iteration] = iteration_rag_info
        return iteration_rag_info

    def select_assertions_for_prompt(self, iteration: int) -> List[Dict]:
        """
        Select assertions based on their probabilities for the next prompt.
        """
        selected = []
        for assertion_hash, prob in self.group_probabilities.items():
            if prob > 0:
                info = self.assertion_rag_info.get(assertion_hash, {})
                selected.append({
                    "type": "assertion",
                    "assertion": info.get("assertion", ""),
                    "coverage": info.get("metrics", {}).get("coverage_summary", "No coverage data"),
                    "newly_covered_holes": info.get("newly_covered_holes", {})
                })
        return selected

    def save_iteration_baseline(self, iteration: int, uncovered_holes: Dict):
        """Save the baseline for a specific iteration"""
        self.iteration_baselines[iteration] = uncovered_holes.copy()

    def get_newly_covered_items(self, current_uncovered: Dict) -> Dict:
        """Calculate newly covered items by comparing with baseline"""
        baseline_set = set(self.uncovered_holes_baseline.keys())
        current_set = set(current_uncovered.keys())
        
        covered_items = baseline_set - current_set
        newly_covered = {}
        
        for item in covered_items:
            assertion_hash = self.find_covering_assertion(item)
            if assertion_hash:
                if assertion_hash not in newly_covered:
                    newly_covered[assertion_hash] = []
                newly_covered[assertion_hash].append(item)
                    
        return newly_covered

    def get_assertion_hash(self, assertion: str) -> str:
        """Generate unique hash for assertion"""
        return hashlib.md5(assertion.encode()).hexdigest()

    def get_assertion_by_hash(self, assertion_hash: str) -> Optional[str]:
        """Retrieve assertion by its hash"""
        for data in self.iteration_data.values():
            for assertion_info in data.values():
                if self.get_assertion_hash(assertion_info["assertion"]) == assertion_hash:
                    return assertion_info["assertion"]
        return None

    def find_covering_assertion(self, item: str) -> Optional[str]:
        """Find the assertion that covered a specific item"""
        for assertion_hash, info in self.assertion_rag_info.items():
            if item in info.get("newly_covered_holes", []):
                return assertion_hash
        return None

    def get_coverage_impact_summary(self, assertion_hash: str) -> Dict:
        """Get coverage impact summary for a specific assertion"""
        if assertion_hash in self.assertion_rag_info:
            info = self.assertion_rag_info[assertion_hash]
            return {
                "newly_covered_count": len(info.get("newly_covered_holes", {})),
                "coverage_metrics": info.get("metrics", {}),
                "coverage_details": info.get("coverage_details", {})
            }
        return {}

    def _generate_combined_description(self, blocks: List[Dict], lines: List[Dict]) -> str:
        """Generate combined description from block and line metadata"""
        descriptions = []
        
        if blocks:
            descriptions.append("Blocks:")
            for block in blocks:
                descriptions.append(f"- {block['description']}")
                
        if lines:
            descriptions.append("Lines:")
            for line in lines:
                descriptions.append(f"- {line['description']}")
                
        return "\n".join(descriptions)

    def update_coverage_tracking(self, 
                               iteration: int,
                               current_uncovered: Dict,
                               assertions: List[str]) -> None:
        """Update coverage tracking with design context"""
        # Calculate newly covered items
        newly_covered = self.get_newly_covered_items(current_uncovered)
        
        # Match with design metadata
        matched_coverage = self.match_uncovered_to_design(newly_covered)
        
        # Update assertion impact tracking
        for assertion in assertions:
            assertion_hash = self.get_assertion_hash(assertion)
            
            # Initialize if needed
            if assertion_hash not in self.assertion_coverage_impact:
                self.assertion_coverage_impact[assertion_hash] = {
                    "total_impact": {
                        "blocks_covered": set(),
                        "lines_covered": set()
                    },
                    "iterations": {}
                }
            
            impact = self.assertion_coverage_impact[assertion_hash]
            
            # Record impact for this iteration
            impact["iterations"][iteration] = matched_coverage
            
            # Update total impact
            for item in matched_coverage.values():
                for block in item["blocks"]:
                    impact["total_impact"]["blocks_covered"].add(
                        block["location"]
                    )
                for line in item["lines"]:
                    impact["total_impact"]["lines_covered"].add(
                        line["location"]
                    )

    def generate_line_block_summaries(self, agents, design_code: str) -> Dict:
        """
        Get simple block and line specifications using LLM.
        """
        specs_prompt = """
        Analyze this design code and identify:
        1. Important code blocks (modules, always blocks, FSMs, etc.) with their line ranges
        2. Critical individual lines (state changes, key assignments, etc.)

        Provide output in this JSON format:
        ```json
        {
            "blocks": [
                {
                    "location": "10-20",
                    "description": "Main FSM controller block implementing protocol handshake. Controls ready/valid signaling and handles backpressure. Key signals include ready, valid, and state transitions.",
                    "code": "actual_code_snippet"
                }
            ],
            "lines": [
                {
                    "location": "15",
                    "description": "Critical state transition from IDLE to BUSY based on valid signal assertion. Controls the start of new transaction processing.",
                    "code": "actual_line_code"
                }
            ]
        }
        ```

        Design Code:
        ```verilog
        code_PH
        ```
        """

        # Get specifications from LLM
        specs_response = initiate_chat_with_retry(
            agents["user"],
            agents["Design_Meta_Specification"],
            message=specs_prompt.replace('code_PH', design_code)
        )
            
        # Extract and parse JSON
        try:
            specs_match = re.search(r'```json\s*(.*?)\s*```', specs_response, re.DOTALL)
            if specs_match:
                specs = json.loads(specs_match.group(1))
                return {
                    "code": design_code,
                    "blocks": self.process_blocks(specs["blocks"], design_code),
                    "lines": self.process_lines(specs["lines"], design_code)
                }
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return {"code": design_code, "blocks": [], "lines": []}

    def process_blocks(self, blocks: List[Dict], code: str) -> List[Dict]:
        """Process block specifications and add code snippets."""
        processed = []
        code_lines = code.splitlines()
        
        for block in blocks:
            try:
                start, end = map(int, block["location"].split("-"))
                if 1 <= start <= end <= len(code_lines):
                    processed.append({
                        "location": block["location"],
                        "description": block["description"],
                        "code": "\n".join(code_lines[start-1:end])
                    })
            except (ValueError, KeyError) as e:
                print(f"Error processing block: {e}")
                
        return processed

    def process_lines(self,lines: List[Dict], code: str) -> List[Dict]:
        """Process line specifications and add code snippets."""
        processed = []
        code_lines = code.splitlines()
        
        for line in lines:
            try:
                line_num = int(line["location"])
                if 1 <= line_num <= len(code_lines):
                    processed.append({
                        "location": line["location"],
                        "description": line["description"],
                        "code": code_lines[line_num-1].strip()
                    })
            except (ValueError, KeyError) as e:
                print(f"Error processing line: {e}")
                
        return processed

    # def select_assertions_with_reasoning(self, iteration: int, agents: Dict, design_code: str, max_assertions: int = 4) -> Tuple[List[str], str]:
    #     """
    #     Select assertions based on their individual probabilities and generate reasoning
    #     including coverage impact information. Limits the number of selected assertions.
        
    #     Args:
    #         iteration: Current iteration number
    #         agents: Dictionary of available agents
    #         design_code: The RTL design code
    #         max_assertions: Maximum number of assertions to select (default: 4)
            
    #     Returns:
    #         Tuple containing:
    #             - List of selected assertions
    #             - Combined reasoning response
    #     """
    #     selected_assertions_info = []
    #     assertions_needing_reasoning = []
    #     assertions_with_reasoning = []
        
    #     # First pass: Select assertions and identify which need new reasoning
    #     for assertion_hash, prob in self.group_probabilities.items():
    #         if prob > 0:
    #             info = self.assertion_rag_info.get(assertion_hash, {})
    #             assertion = info.get("assertion", "")
    #             if assertion and random.random() < prob:
    #                 # Break if we've reached the maximum number of assertions
    #                 if len(selected_assertions_info) >= max_assertions:
    #                     break
                        
    #                 # Get coverage information
    #                 coverage_info = {
    #                     "newly_covered_holes": info.get("newly_covered_holes", {}),
    #                     "metrics": info.get("metrics", {}),
    #                     "coverage_details": info.get("coverage_details", {})
    #                 }
                    
    #                 existing_reasoning = self.get_reasoning_for_assertion(assertion_hash)
    #                 if existing_reasoning:
    #                     assertions_with_reasoning.append({
    #                         "index": len(selected_assertions_info) + 1,
    #                         "assertion": assertion,
    #                         "hash": assertion_hash,
    #                         "reasoning": existing_reasoning,
    #                         "coverage_info": coverage_info
    #                     })
    #                 else:
    #                     assertions_needing_reasoning.append({
    #                         "index": len(selected_assertions_info) + 1,
    #                         "assertion": assertion,
    #                         "hash": assertion_hash,
    #                         "coverage_info": coverage_info
    #                     })
    #                 selected_assertions_info.append({
    #                     "assertion": assertion,
    #                     "hash": assertion_hash,
    #                     "coverage_info": coverage_info
    #                 })
        
    #     if not selected_assertions_info:
    #         return [], ""
        
    #     # Generate reasoning for all new assertions in one batch if needed
    #     if assertions_needing_reasoning:
    #         # Format assertions and their coverage information
    #         assertions_text = format_assertion_collection(assertions_needing_reasoning)
            
    #         batch_prompt = f"""
    #         Given this RTL design code:
            
    #         ```verilog
    #         {design_code}
    #         ```
            
    #         Analyze each of the following assertions and explain:
    #         1. The verification intent and what aspects of the design it checks
    #         2. Why the assertion is effective based on its coverage impact
    #         3. How the newly covered lines relate to critical design functionality
            
    #         For each assertion, provide your reasoning in this format:
            
    #         Assertion <number>:
    #         ```systemverilog
    #         <assertion code>
    #         ```
            
    #         Reasoning:
    #         <your detailed analysis of both the assertion and its coverage impact>
            
    #         Here are the assertions to analyze:
            
    #         {assertions_text}
    #         """
            
    #         batch_response = initiate_chat_with_retry(
    #             agents["user"],
    #             agents["ICRL_Reasoning"],
    #             message=batch_prompt
    #         )
            
    #         # Parse individual reasonings from batch response
    #         new_reasonings = self.parse_reasoning_response(batch_response, len(assertions_needing_reasoning))
            
    #         # Store and append reasonings
    #         for assertion_info, reasoning in zip(assertions_needing_reasoning, new_reasonings):
    #             if reasoning is not None:
    #                 self.save_reasoning_for_assertion(assertion_info["hash"], reasoning)
    #                 assertions_with_reasoning.append({
    #                     "index": assertion_info["index"],
    #                     "assertion": assertion_info["assertion"],
    #                     "hash": assertion_info["hash"],
    #                     "reasoning": reasoning,
    #                     "coverage_info": assertion_info["coverage_info"]
    #                 })
        
    #     # Sort by index to maintain order
    #     assertions_with_reasoning.sort(key=lambda x: x["index"])
        
    #     # Generate final combined response
    #     reasoning_response = []
    #     for info in assertions_with_reasoning:
    #         if info.get("reasoning"):
    #             reasoning_response.append(
    #                 f"Assertion {info['index']}:\n\n"
    #                 f"```systemverilog\n{info['assertion']}\n```\n"
    #                 f"Reasoning:\n{info['reasoning']}"
    #             )
        
    #     # Store the selected assertions for this iteration
    #     self.iteration_data[iteration] = {
    #         info["hash"]: {
    #             "assertion": info["assertion"],
    #             "previous_prob": self.group_probabilities[info["hash"]]
    #         } for info in selected_assertions_info
    #     }
        
    #     # Return selected assertions and combined reasoning
    #     return (
    #         [info["assertion"] for info in selected_assertions_info],
    #         "\n\n".join(reasoning_response)
    #     )


    def select_assertions_with_reasoning(self, iteration: int, agents: Dict, design_code: str, max_assertions: int = 3) -> Tuple[List[str], str]:
        """
        Select assertions based on their individual probabilities and generate reasoning
        including coverage impact information. Limits the number of selected assertions.
        
        Args:
            iteration: Current iteration number
            agents: Dictionary of available agents
            design_code: The RTL design code
            max_assertions: Maximum number of assertions to select (default: 4)
            
        Returns:
            Tuple containing:
                - List of selected assertions
                - Combined reasoning response
        """
        selected_assertions_info = []
        assertions_needing_reasoning = []
        assertions_with_reasoning = []
        
        # Use previous iteration's baseline for comparison
        baseline_to_compare = self.iteration_baselines.get(iteration - 1, self.uncovered_holes_baseline)

        # Helper function to extract cid from hole info
        def get_hole_cid(hole_info):
            """Extract or generate cid from hole info"""
            if 'cid' in hole_info:
                return hole_info['cid']
            elif 'source_location' in hole_info:
                # Generate consistent cid from source location
                return hashlib.md5(hole_info['source_location'].encode()).hexdigest()
            return None

        # First pass: Select assertions based on probabilities
        for assertion_hash, prob in self.group_probabilities.items():
            if prob > 0:
                info = self.assertion_rag_info.get(assertion_hash, {})
                assertion = info.get("assertion", "")
                if assertion and random.random() < prob:
                    if len(selected_assertions_info) >= max_assertions:
                        break
                    
                    if FLAGS.post_coverage_calc and not self.coverage_calculated.get(assertion_hash, False):
                        # Get empty testbench and calculate coverage (existing code)
                        test_text = assertion
                        initial_tb = utils2.package_testbench(self.design_row, test_text, FLAGS.task)
                        tb_module_str = utils2.extract_module_name(initial_tb)
                        
                        coverage_output = analyze_coverage(
                            design_code, initial_tb, FLAGS.task, self.design_row.task_id,
                            os.path.join(self.save_dir, "tmp_sva"), FLAGS.tcl_report_path, 
                            FLAGS.cleanup_temp_files, tb_module_str, iteration=iteration
                        )
                        
                        coverage_metrics = calculate_coverage_metric(coverage_output)
                        uncovered_holes = analyze_uncovered_lines(
                            jasper_out_str=coverage_output,
                            design_code=design_code
                        )
                        
                        # Print for debugging
                        # if FLAGS.debug:
                        #     print("\nOriginal hole locations:")
                        #     for hole_info in uncovered_holes.values():
                        #         print(f"Original: {hole_info.get('source_location', 'No location')}")
                        
                        # Normalize hole locations and merge identical ones
                        # Convert holes dict to use cid as key
                        current_holes = {}
                        for hole_info in uncovered_holes.values():
                            cid = get_hole_cid(hole_info)
                            if cid:
                                current_holes[cid] = hole_info
                        
                        # Similarly convert baseline holes
                        baseline_holes = {}
                        if baseline_to_compare:
                            for hole_info in baseline_to_compare.values():
                                cid = get_hole_cid(hole_info)
                                if cid:
                                    baseline_holes[cid] = hole_info
                        
                        # Calculate newly covered holes using cids
                        newly_covered = {}
                        if baseline_holes:
                            baseline_cids = set(baseline_holes.keys())
                            current_cids = set(current_holes.keys())
                            covered_cids = baseline_cids - current_cids
                            for cid in covered_cids:
                                newly_covered[cid] = baseline_holes[cid]
                        
                        # Update assertion info
                        self.assertion_rag_info[assertion_hash].update({
                            "metrics": coverage_metrics,
                            "coverage_details": current_holes,
                            "newly_covered_holes": newly_covered
                        })
                        self.coverage_calculated[assertion_hash] = True
                        
                        # Update info reference
                        info = self.assertion_rag_info[assertion_hash]
                    
                    # Get coverage information
                    coverage_info = {
                        "newly_covered_holes": info.get("newly_covered_holes", {}),
                        "metrics": info.get("metrics", {}),
                        "coverage_details": info.get("coverage_details", {})
                    }
                    
                    # Rest of the selection logic remains the same...
                    existing_reasoning = self.get_reasoning_for_assertion(assertion_hash)
                    if existing_reasoning:
                        assertions_with_reasoning.append({
                            "index": len(selected_assertions_info) + 1,
                            "assertion": assertion,
                            "hash": assertion_hash,
                            "reasoning": existing_reasoning,
                            "coverage_info": coverage_info
                        })
                    else:
                        assertions_needing_reasoning.append({
                            "index": len(selected_assertions_info) + 1,
                            "assertion": assertion,
                            "hash": assertion_hash,
                            "coverage_info": coverage_info
                        })
                    selected_assertions_info.append({
                        "assertion": assertion,
                        "hash": assertion_hash,
                        "coverage_info": coverage_info
                    })
        
        if not selected_assertions_info:
            return [], ""
        
        # Generate reasoning for all new assertions in one batch if needed
        if assertions_needing_reasoning:
            # Format assertions and their coverage information
            assertions_text = format_assertion_collection(assertions_needing_reasoning)
            
            batch_prompt = f"""
            Given this RTL design code:
            
            ```verilog
            {design_code}
            ```
            
            Analyze each of the following assertions and explain:
            1. The verification intent and what aspects of the design it checks
            2. Why the assertion is effective based on its coverage impact
            3. How the newly covered lines relate to critical design functionality
            
            For each assertion, provide your reasoning in this format:
            
            Assertion <number>:
            ```systemverilog
            <assertion code>
            ```
            
            Reasoning:
            <your detailed analysis of both the assertion and its coverage impact>
            
            Here are the assertions to analyze:
            
            {assertions_text}
            """
            
            batch_response = initiate_chat_with_retry(
                agents["user"],
                agents["ICRL_Reasoning"],
                message=batch_prompt
            )
            
            # Parse individual reasonings from batch response
            new_reasonings = self.parse_reasoning_response(batch_response, len(assertions_needing_reasoning))
            
            # Store and append reasonings
            for assertion_info, reasoning in zip(assertions_needing_reasoning, new_reasonings):
                if reasoning is not None:
                    self.save_reasoning_for_assertion(assertion_info["hash"], reasoning)
                    assertions_with_reasoning.append({
                        "index": assertion_info["index"],
                        "assertion": assertion_info["assertion"],
                        "hash": assertion_info["hash"],
                        "reasoning": reasoning,
                        "coverage_info": assertion_info["coverage_info"]
                    })
        
        # Sort by index to maintain order
        assertions_with_reasoning.sort(key=lambda x: x["index"])
        
        # Generate final combined response
        reasoning_response = []
        for info in assertions_with_reasoning:
            if info.get("reasoning"):
                reasoning_response.append(
                    f"Assertion {info['index']}:\n\n"
                    f"```systemverilog\n{info['assertion']}\n```\n"
                    f"Reasoning:\n{info['reasoning']}"
                )
        
        # Store the selected assertions for this iteration
        self.iteration_data[iteration] = {
            info["hash"]: {
                "assertion": info["assertion"],
                "previous_prob": self.group_probabilities[info["hash"]]
            } for info in selected_assertions_info
        }
        
        return (
            [info["assertion"] for info in selected_assertions_info],
            "\n\n".join(reasoning_response)
        )


    def get_scaling_factor(self, iteration: int, max_iterations: int) -> float:
        """
        Calculate the exponential scaling factor for a given iteration.
        
        Args:
            iteration: Current iteration number
            max_iterations: Total number of iterations for scaling calculations.
            
        Returns:
            float: Scaling factor between 0 and 1
        """
        return math.exp(-self.iteration_weight * iteration / max_iterations)

    def decay_unused_probabilities(self, iteration: int):
        """
        Apply exponential decay to probabilities of assertions not used in the current iteration.
        
        Args:
            iteration: Current iteration number
        """
        scaling_factor = self.get_scaling_factor(iteration)
        current_assertions = set(self.iteration_data.get(iteration, {}).keys())
        
        for assertion_hash in self.group_probabilities:
            if assertion_hash not in current_assertions:
                current_prob = self.group_probabilities[assertion_hash]
                self.group_probabilities[assertion_hash] = max(
                    current_prob * scaling_factor,
                    self.min_probability
                )

    def update_probabilities_after_coverage(self, iteration: int, coverage_results: Dict, max_iterations: int):
        """
        First applies time-based exponential scaling to ALL assertions,
        then handles coverage-based probability adjustments separately.
        
        Args:
            iteration: Current iteration number
            coverage_results: Coverage analysis results from current iteration
        """
        
        # 1. Apply time-based exponential scaling to ALL assertions
        scaling_factor = self.get_scaling_factor(iteration, max_iterations)
        for assertion_hash in self.group_probabilities:
            current_prob = self.group_probabilities[assertion_hash]
            # Apply universal time scaling
            scaled_prob = max(current_prob * scaling_factor, self.min_probability)
            self.group_probabilities[assertion_hash] = scaled_prob
            
            # Record time scaling effect in RAG info
            if assertion_hash in self.assertion_rag_info:
                self.assertion_rag_info[assertion_hash].update({
                    'time_scaling': {
                        'iteration': iteration,
                        'before_scaling': current_prob,
                        'after_scaling': scaled_prob,
                        'scaling_factor': scaling_factor
                    }
                })
        
        # 2. Apply coverage-based adjustments only to assertions used in this iteration
        if iteration in self.iteration_data:
            used_assertions = self.iteration_data[iteration]
            
            # Get coverage metrics
            metrics = coverage_results.get('metrics', {})
            newly_covered = coverage_results.get('coverage_details', {})
            
            for assertion_hash, info in used_assertions.items():
                # Check coverage improvement
                metrics_improved = (
                    metrics.get('functionality', 0) == 1.0 and
                    any(metrics.get(f'coverage_coi_{model}', 0) > 0 
                        for model in ['statement', 'branch', 'functional', 'toggle', 'expression'])
                )
                coverage_improved = len(newly_covered) > 0
                
                # Get probability after time scaling
                current_prob = self.group_probabilities[assertion_hash]
                
                # Apply coverage-based adjustment
                if metrics_improved or coverage_improved:
                    # Increase probability
                    new_prob = min(current_prob + self.delta, 1.0)
                else:
                    # Decrease probability
                    new_prob = max(current_prob - self.delta, self.min_probability)
                
                self.group_probabilities[assertion_hash] = new_prob
                
                # Record coverage-based adjustment in RAG info
                if assertion_hash in self.assertion_rag_info:
                    self.assertion_rag_info[assertion_hash].update({
                        'coverage_adjustment': {
                            'iteration': iteration,
                            'before_adjustment': current_prob,
                            'after_adjustment': new_prob,
                            'coverage_improved': metrics_improved or coverage_improved,
                            'metrics_improved': metrics_improved,
                            'newly_covered_count': len(newly_covered)
                        }
                    })
        
        # Save the updated data
        self.save_data()


    def get_probability_history(self, assertion_hash: str) -> List[Dict]:
        """
        Get complete probability update history for an assertion.
        
        Args:
            assertion_hash: Hash of the assertion to analyze
            
        Returns:
            List[Dict]: History of probability updates including both time scaling
                    and coverage adjustments
        """
        history = []
        if assertion_hash in self.assertion_rag_info:
            info = self.assertion_rag_info[assertion_hash]
            for iteration in sorted(info.get('time_scaling', {}).keys()):
                update = {
                    'iteration': iteration,
                    'time_scaling': info['time_scaling'].get(iteration, {}),
                }
                if iteration in info.get('coverage_adjustment', {}):
                    update['coverage_adjustment'] = info['coverage_adjustment'][iteration]
                history.append(update)
        return history

    def get_assertion_probabilities(self) -> Dict[str, float]:
        """
        Get current probabilities for all assertions.
        
        Returns:
            Dict mapping assertion hashes to their current probabilities
        """
        return self.group_probabilities.copy()

    def get_assertion_coverage_history(self, assertion_hash: str) -> List[Dict]:
        """
        Get coverage impact history for a specific assertion.
        
        Args:
            assertion_hash: Hash of the assertion to analyze
            
        Returns:
            List of dictionaries containing coverage impact data across iterations
        """
        history = []
        
        if assertion_hash in self.assertion_rag_info:
            info = self.assertion_rag_info[assertion_hash]
            for iteration in sorted(info.get('iterations', {})):
                history.append({
                    'iteration': iteration,
                    'metrics': info['iterations'][iteration].get('metrics', {}),
                    'newly_covered': len(info['iterations'][iteration].get('newly_covered_holes', {})),
                    'probability': info['iterations'][iteration].get('probability', self.base_probability)
                })
                
        return history

    def save_rag_results(self, assertion: str, metrics: Dict, newly_covered_holes: Dict):
        """
        Save RAG results for an assertion without modifying probabilities.
        Probabilities will be updated later based on coverage impact.
        """
        assertion_hash = self.get_assertion_hash(assertion)
        
        # Initialize if not present
        if assertion_hash not in self.assertion_rag_info:
            self.assertion_rag_info[assertion_hash] = {
                "assertion": assertion,
                "metrics": {},
                "newly_covered_holes": {},
                "coverage_details": {}
            }
            # Initialize probability if not already set
            if assertion_hash not in self.group_probabilities:
                self.group_probabilities[assertion_hash] = self.base_probability
        
        # Update assertion info
        self.assertion_rag_info[assertion_hash].update({
            "metrics": metrics,
            "newly_covered_holes": newly_covered_holes,
            "coverage_details": metrics.get("coverage_details", {})
        })

    # Update probabilities based on coverage impact after running assertions
    def update_assertion_probability(assertion_hash, coverage_improved: bool):
        current_prob = self.group_probabilities.get(assertion_hash, self.base_probability)
        scaling_factor = math.exp(-self.iteration_weight * iteration)
        
        if coverage_improved:
            new_prob = min(current_prob + 0.1 * scaling_factor, 1.0)
        else:
            new_prob = max(current_prob - 0.1 * scaling_factor, 0.0)
        
        self.group_probabilities[assertion_hash] = new_prob
    
        # Return selected assertions and combined reasoning
        return (
            [info["assertion"] for info in selected_assertions_info],
            "\n\n".join(reasoning_response)
        )

    # def parse_reasoning_response(self, response: str, num_assertions: int) -> List[str]:
    #     # Parse the reasoning from the LLM response
    #     reasoning_paragraphs = []
    #     pattern = r"Assertion (\d+):\s*```systemverilog\s*(.*?)\s*```\s*Reasoning:\s*(.*?)(?=Assertion \d+:|$)"
    #     matches = re.finditer(pattern, response, re.DOTALL)
    #     reasoning_dict = {}
    #     for match in matches:
    #         index = int(match.group(1)) - 1
    #         assertion_code = match.group(2).strip()
    #         reasoning = match.group(3).strip()
    #         reasoning_dict[index] = reasoning
    #     # Ensure all reasonings are captured
    #     for i in range(num_assertions):
    #         reasoning_paragraphs.append(reasoning_dict.get(i, ""))
    #     return reasoning_paragraphs

    def parse_reasoning_response(self, response: str, num_assertions: int) -> List[str]:
        """
        Parse batch reasoning response into individual reasonings.
        
        Args:
            response: The complete response from the LLM containing all reasonings
            num_assertions: Expected number of assertions/reasonings
            
        Returns:
            List of individual reasoning strings
        """
        reasonings = []
        pattern = r"Assertion (\d+):\s*```systemverilog\s*(.*?)\s*```\s*Reasoning:\s*(.*?)(?=Assertion \d+:|$)"
        matches = re.finditer(pattern, response, re.DOTALL)
        
        reasoning_dict = {}
        for match in matches:
            index = int(match.group(1)) - 1  # Convert to 0-based index
            reasoning = match.group(3).strip()
            reasoning_dict[index] = reasoning
        
        # Ensure all reasonings are captured in order
        for i in range(num_assertions):
            reasoning = reasoning_dict.get(i, None)
            reasonings.append(reasoning)
        
        return reasonings

    def get_reasoning_for_assertion(self, assertion_hash: str) -> Optional[str]:
        if assertion_hash in self.assertion_rag_info:
            return self.assertion_rag_info[assertion_hash].get("reasoning")
        return None

    def save_reasoning_for_assertion(self, assertion_hash: str, reasoning: str) -> None:
        if assertion_hash not in self.assertion_rag_info:
            self.assertion_rag_info[assertion_hash] = {}
        self.assertion_rag_info[assertion_hash]["reasoning"] = reasoning

    def extract_line_ranges(self, llm_response: str, assertion: str) -> List[Tuple[int, int, int, int]]:
        """Extract line ranges from LLM response"""
        ranges = []
        pattern = r'\((\d+),(\d+)\)-\((\d+),(\d+)\)'
        
        assertion_section = self.find_assertion_section(llm_response, assertion)
        if assertion_section:
            matches = re.finditer(pattern, assertion_section)
            for match in matches:
                start_row = int(match.group(1))
                start_col = int(match.group(2))
                end_row = int(match.group(3))
                end_col = int(match.group(4))
                ranges.append((start_row, start_col, end_row, end_col))
                
        return ranges

    def extract_line_description(self, llm_response: str, assertion: str) -> str:
        """Extract line description from LLM response"""
        assertion_section = self.find_assertion_section(llm_response, assertion)
        if assertion_section:
            # Extract description after "Line description:" marker
            match = re.search(r"Line description:(.*?)(?=Block:|$)", 
                            assertion_section, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def extract_block_description(self, llm_response: str, assertion: str) -> str:
        """Extract block description from LLM response"""
        assertion_section = self.find_assertion_section(llm_response, assertion)
        if assertion_section:
            match = re.search(r"Block description:(.*?)(?=\n\n|$)",
                            assertion_section, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def find_assertion_section(self, response: str, assertion: str) -> Optional[str]:
        """Find the section in LLM response corresponding to an assertion"""
        # Split response by assertions and find matching section
        sections = re.split(r"\nAssertion:", response)
        for section in sections:
            if assertion.strip() in section:
                return section
        return None


    def save_data(self):
        """
        Save the current state of the assertion tracker to disk.
        Includes:
        - Iteration data
        - RAG info
        - Probabilities
        - Coverage data
        """
        # Create directories if they don't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save iteration data and probabilities
        iteration_file = self.save_dir / 'iteration_data.json'
        with open(iteration_file, 'w') as f:
            json.dump({
                'iteration_data': self.iteration_data,
                'group_probabilities': self.group_probabilities,
                'base_probability': self.base_probability,
                'iteration_weight': self.iteration_weight,
            }, f, indent=2)
        
        # Save RAG info
        rag_file = self.save_dir / 'rag_info.json'
        with open(rag_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable_rag = {}
            for hash_val, info in self.assertion_rag_info.items():
                serializable_rag[hash_val] = {
                    key: (list(value) if isinstance(value, set) else value)
                    for key, value in info.items()
                }
            json.dump(serializable_rag, f, indent=2)
        
        # Save coverage data
        coverage_file = self.save_dir / 'coverage_data.json'
        with open(coverage_file, 'w') as f:
            json.dump({
                'uncovered_holes_baseline': self.uncovered_holes_baseline,
                'assertion_coverage_impact': {
                    hash_val: {
                        'total_impact': {
                            'blocks_covered': list(impact['total_impact']['blocks_covered']),
                            'lines_covered': list(impact['total_impact']['lines_covered'])
                        },
                        'iterations': impact['iterations']
                    }
                    for hash_val, impact in self.assertion_coverage_impact.items()
                }
            }, f, indent=2)

    def load_data(self):
        """
        Load previously saved tracker state from disk.
        """
        try:
            # Load iteration data and probabilities
            iteration_file = self.save_dir / 'iteration_data.json'
            if iteration_file.exists():
                with open(iteration_file) as f:
                    data = json.load(f)
                    self.iteration_data = data['iteration_data']
                    self.group_probabilities = data['group_probabilities']
                    self.base_probability = data.get('base_probability', 0.3)
                    self.iteration_weight = data.get('iteration_weight', 0.1)
            
            # Load RAG info
            rag_file = self.save_dir / 'rag_info.json'
            if rag_file.exists():
                with open(rag_file) as f:
                    rag_data = json.load(f)
                    # Convert lists back to sets
                    self.assertion_rag_info = {
                        hash_val: {
                            key: (set(value) if isinstance(value, list) else value)
                            for key, value in info.items()
                        }
                        for hash_val, info in rag_data.items()
                    }
            
            # Load coverage data
            coverage_file = self.save_dir / 'coverage_data.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    self.uncovered_holes_baseline = coverage_data['uncovered_holes_baseline']
                    # Convert lists back to sets for coverage impact
                    self.assertion_coverage_impact = {
                        hash_val: {
                            'total_impact': {
                                'blocks_covered': set(impact['total_impact']['blocks_covered']),
                                'lines_covered': set(impact['total_impact']['lines_covered'])
                            },
                            'iterations': impact['iterations']
                        }
                        for hash_val, impact in coverage_data['assertion_coverage_impact'].items()
                    }
                    
        except Exception as e:
            print(f"Error loading saved data: {e}")
            # Initialize empty if loading fails
            self.iteration_data = {}
            self.assertion_rag_info = {}
            self.group_probabilities = {}
            self.assertion_coverage_impact = {}