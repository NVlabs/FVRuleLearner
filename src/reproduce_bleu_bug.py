#!/usr/bin/env python3
"""
Script to reproduce the BLEU=1.0 but functionality=0 bug using actual evaluation functions

This script now follows the same approach as self_learning.py for functionality evaluation.

To evaluate functionality (Property Equivalence Check):
1. Ensure JasperGold is installed and available in PATH
2. Set check_func=True (currently on line ~320)
3. The script will use FVEval's NL2SVAHumanEvaluator.evaluate_jg method exactly like
   self_learning.py does to verify if the generated assertion is functionally equivalent

Note: Without JasperGold, the script demonstrates the bug by showing high BLEU
scores while functionality remains 0 (not verified).

Key findings:
- The reference answer has trailing spaces after '1'b1' that the generated answer lacks
- BLEU ignores this whitespace difference (tokenizes and compares tokens)
- JasperGold's property equivalence check is sensitive to such differences
"""

import sys
import os
from config import FLAGS
from saver import saver
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up necessary imports
import evaluate
from typing import Dict
from FVEval.fv_eval import utils
from FVEval.fv_eval.data import LMResult, JGEvaluationResult

# Import after mocking
import re
import difflib

# Import InputData from FVEval
from FVEval.fv_eval.data import InputData
from FVEval.fv_eval import fv_tool_execution
from FVEval.fv_eval.evaluation import NL2SVAHumanEvaluator
import tempfile
import shutil

def evaluate_response(response: str, reference: str, row: InputData, check_functionality: bool = False) -> Dict[str, float]:
    """
    Evaluate a single assertion response against a reference.
    Returns comprehensive metrics including BLEU, ROUGE, exact match, and optionally functionality.
    """
    metrics = {}
    
    # Load similarity metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    exact_match_metric = evaluate.load("exact_match")
    
    # Parse the response to extract assertion
    parsed_response = utils.parse_code_response(response)
    parsed_reference = utils.parse_code_response(reference)
    
    # Calculate BLEU score
    bleu_result = bleu_metric.compute(
        predictions=[parsed_response], 
        references=[parsed_reference]
    )
    metrics['bleu'] = bleu_result['bleu']
    
    # Calculate ROUGE score
    rouge_result = rouge_metric.compute(
        predictions=[parsed_response],
        references=[parsed_reference]
    )
    metrics['rouge'] = rouge_result['rougeL']
    
    # Calculate exact match
    exact_result = exact_match_metric.compute(
        predictions=[parsed_response],
        references=[parsed_reference]
    )
    metrics['exact_match'] = exact_result['exact_match']
    
    # Check functionality using JasperGold if requested
    if check_functionality and row is not None:
        try:
            # Create LMResult object for the response
            lm_result = LMResult(
                model_name="test_model",
                experiment_id="bleu_bug_test",
                task_id=row.task_id,
                response=response,
                ref_solution=reference,
                user_prompt=row.prompt,
                output_tb=row.testbench,
                design_rtl = "\n",
                cot_response = "\n",
                # design_name=row.design_name
            )
            
            # Create evaluator instance
            evaluator = NL2SVAHumanEvaluator(
                llm_output_dir="",
                model_name=FLAGS.llm_model,
                temp_dir=os.path.join(saver.logdir,"tmp"),  # Temporary directory for intermediate files
                save_dir=os.path.join(saver.logdir,"save"),  # Directory to save the final results
                parallel_jobs=1,
                cleanup_temp_files=True,  # Keep files for debugging
                debug=FLAGS.debug
            )
            
            # Run JasperGold evaluation - match self_learning.py approach exactly
            jg_results = evaluator.evaluate_jg([lm_result], with_rtl_design=True)
            
            # Extract metrics exactly like self_learning.py does
            functionality_score = jg_results[0].functionality if jg_results else 0
            relaxed_functionality_score = jg_results[0].func_relaxed if jg_results else 0
            syntax_score = jg_results[0].syntax if jg_results else 0
            
            metrics['syntax'] = syntax_score
            metrics['pec'] = functionality_score
            metrics['relax_pec'] = relaxed_functionality_score
            
            # Print metrics like self_learning.py does
            print(f'###Record: \n###pec: {functionality_score}\n###relax_pec: {relaxed_functionality_score}\n###syntax: {syntax_score}')
            
            if FLAGS.debug:
                print(f"@@@DEBUG: jg_results = {jg_results}")
                if jg_results:
                    print(f"@@@DEBUG: syntax = {syntax_score}, pec = {functionality_score}, relax_pec = {relaxed_functionality_score}")
            
        except Exception as e:
            print(f"Error during JasperGold evaluation: {e}")
            import traceback
            traceback.print_exc()
            metrics['syntax'] = 0.0
            metrics['pec'] = 0.0
            metrics['relax_pec'] = 0.0
        finally:
            # Don't clean up temp directory since it's in saver.logdir
            pass
    else:
        # Default values when not checking functionality
        metrics['pec'] = 0.0
        metrics['relax_pec'] = 0.0
        metrics['syntax'] = 1.0
    
    return metrics


def main():

    # Test case: fsm_sequence_1
    task_id = "fsm_sequence_1"

    # Reference answer (ground truth)
    reference_answer = """asrt: assert property (@(posedge clk) disable iff (tb_reset)
        (check_state_legal_precondition && !(|(match_tracker[0]))) !== 1'b1     
    );"""

    # Generated answer that achieved BLEU = 1.0 but functionality = 0
    generated_answer = """asrt: assert property (@(posedge clk) disable iff (tb_reset)
        (check_state_legal_precondition && !(|(match_tracker[0]))) !== 1'b1
    );"""

    # Earlier iteration that had lower BLEU but also functionality = 0
    # earlier_answer = """asrt: assert property (@(posedge clk) disable iff (tb_reset)
    #     check_state_legal_precondition |-> (|match_tracker[0] && !(check_state_legal_precondition && !(|match_tracker[0])))
    # );"""

    # Simplified testbench for the example
    testbench = """module fsm_sequence_tb (
    clk, 
    reset_, 
    fsm_state,
    fsm_sequence
    );
        parameter fsm_width = 2; //actual width of the states in the RTL
        parameter num_of_states=2; //number of states provided in the fsm_sequence
        parameter num_of_times_initial_state_repeats=1; //Number of times the initial state of the "fsm_sequence" is repeated in the "fsm_sequence"  

    input clk;
    input reset_;
    input [fsm_width-1:0]fsm_state;
    input [fsm_width*num_of_states-1:0]fsm_sequence;
    wire tb_reset;
    assign tb_reset = (reset_ == 1'b0);


    wire [fsm_width-1:0] tb_fsm_sequence[num_of_states-1:0]; 

    //match the current "fsm_state" with the states provided in the "fsm_sequence"
    wire [num_of_states-1:0]match_tracker[num_of_times_initial_state_repeats-1:0]; 
    reg [num_of_states-1:0]match_tracker_d1[num_of_times_initial_state_repeats-1:0]; 

    //Track all the states of the "fsm_sequence"
    reg [num_of_states-1:0]state_tracker[num_of_times_initial_state_repeats-1:0]; 

    reg [fsm_width-1:0] fsm_state_d1;
    reg tb_reset_d1;
    wire [fsm_width-1:0] tb_random_state;
    wire [$clog2(num_of_times_initial_state_repeats):0]tb_sequence_seen;


    //storing the states of the fsm_sequence in the correct order
    for (genvar i=num_of_states-1; i >=0; i--) begin : storing_of_fsm_states
        assign  tb_fsm_sequence[num_of_states-1-i] = fsm_sequence[(fsm_width*(i+1))-1 : fsm_width*i];
    end

    //Delayed versions of fsm_state and tb_reset
    always @(posedge clk) begin
        if (!reset_) begin
            fsm_state_d1 <= 'd0;
            tb_reset_d1 <= 1;
        end else begin
            fsm_state_d1 <= fsm_state;
            tb_reset_d1 <= tb_reset;  
        end
    end

    for (genvar n=0; n<num_of_times_initial_state_repeats; n++) begin : matching_of_states_as_per_initial_state_repeat
        if (n==0) begin : matching_of_states_for_certain_cases
            for (genvar i=0; i<num_of_states; i++) begin : matching_of_states_as_per_num_of_states
                if (i==0) begin : matching_of_states_for_first_state 
                    assign match_tracker[n][0] = (fsm_state == tb_fsm_sequence[0]);
                end else begin : matching_of_states_for_other_states 
                    assign match_tracker[n][i] = (fsm_state == tb_fsm_sequence[i]);
                end
            end 
        end else begin : matching_of_states_for_other_cases
            for (genvar i=0; i<num_of_states; i++) begin : matching_of_states_as_per_num_of_states
                if (i==0) begin : matching_of_states_for_first_state 
                assign match_tracker[n][0] = ((fsm_state != fsm_state_d1) && !tb_reset_d1) 
                                                ? (!(|state_tracker[n]) && (|state_tracker[n-1]) && (fsm_state == tb_fsm_sequence[0])) 
                                                : match_tracker[n][0] ;
                end else begin : matching_of_states_for_other_states
                assign match_tracker[n][i] = ((fsm_state != fsm_state_d1) && !tb_reset_d1) 
                                                ? (!state_tracker[n][i] && state_tracker[n][i-1] && (fsm_state == tb_fsm_sequence[i])) 
                                                : match_tracker[n][i] ;
                end
            end 
        end
    end

    reg [$clog2(num_of_times_initial_state_repeats):0] j;

    always @(posedge clk) begin
        if (!reset_) begin
            for (j=0; j< num_of_times_initial_state_repeats; j++) begin
                state_tracker[j] <= 'd0;
                match_tracker_d1[j] <= 'd0;
            end
        end else begin
            for (j=0; j< num_of_times_initial_state_repeats; j++) begin
                match_tracker_d1[j] <= match_tracker[j];
                if (j==0) 
                state_tracker[j] <= (((state_tracker[j]==(match_tracker[j]-1'b1)) || 
                                        (state_tracker[j] == ((match_tracker[j]-1'b1) | match_tracker[j]))) && 
                                        (|match_tracker[j] != 'd0)) 
                                            ? state_tracker[j]|match_tracker[j] 
                                            : ((((|match_tracker[j]) == 0) && (fsm_state == tb_fsm_sequence[0])) 
                                                ? 'd1 
                                                : 'd0
                                            );
                else 
                state_tracker[j] <= (((state_tracker[j]==(match_tracker[j]-1'b1)) || 
                                        (state_tracker[j] == ((match_tracker[j]-1'b1) | match_tracker[j]))) && 
                                        (|match_tracker[j] != 'd0)) 
                                        ? state_tracker[j]|match_tracker[j] 
                                        : 'd0;
            end
        end
    end

    for (genvar n=0; n<num_of_times_initial_state_repeats; n++) begin : fsm_sequence_seen
        assign tb_sequence_seen[n] = state_tracker[n][num_of_states-1];
    end

    reg check_state_legal_precondition;
    always @(posedge clk) begin
        if (!reset_) begin
            check_state_legal_precondition <= 1'b0;
        end else begin
            check_state_legal_precondition <= fsm_state == tb_fsm_sequence[0];
        end
    end

    endmodule"""

    # Create the row object using InputData
    row = InputData(
        design_name="fsm_sequence",  # Added design_name field
        task_id=task_id,
        ref_solution=reference_answer,
        # prompt="Create a SVA assertion that checks: that the current FSM state is legal. Use the signals 'check_state_legal_precondition' and 'match_tracker'.",
        prompt="Create a SVA assertion that checks: that the current FSM state is legal. Use the signals 'check_state_legal_precondition' and 'match_tracker'.",
        testbench=testbench
    )

    print("=" * 80)
    print("Reproducing BLEU=1.0 but functionality=0 bug with actual evaluation")
    print("=" * 80)

    print(f"\nTask ID: {task_id}")
    print("\n1. REFERENCE ANSWER:")
    print("-" * 40)
    print(reference_answer)

    print("\n2. GENERATED ANSWER (Expected: BLEU=1.0, functionality=0):")
    print("-" * 40)
    print(generated_answer)

    # Evaluate the generated answer
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS:")
    print("=" * 80)


    try:
        # Evaluate the generated answer using case-by-case approach
        # Note: Set check_functionality=True to also check property equivalence (requires JasperGold)
        check_func = True  # Set to True if JasperGold is available
        
        # Evaluate the generated answer including JasperGold if enabled
        metrics = evaluate_response(generated_answer, reference_answer, row=row, check_functionality=check_func)
        
        # print(f"\nGenerated Answer Metrics:")
        # print(f"  BLEU Score: {metrics.get('bleu', 'N/A'):.4f}")
        # print(f"  ROUGE-L Score: {metrics.get('rouge', 'N/A'):.4f}")
        # print(f"  Exact Match: {metrics.get('exact_match', 'N/A')}")
        # print(f"  Functionality (PEC): {metrics.get('pec', 'N/A')}")
        # print(f"  Relaxed Functionality: {metrics.get('relax_pec', 'N/A')}")
        # print(f"  Syntax: {metrics.get('syntax', 'N/A')}")
        
        # # Also evaluate the earlier answer for comparison
        # print("\n3. EARLIER ANSWER (Expected: BLEU≈0.586, functionality=0):")
        # print("-" * 40)
        # print(earlier_answer)
        
        # Evaluate the earlier answer including JasperGold if enabled  
        # metrics_earlier = evaluate_response(earlier_answer, reference_answer, row=row, check_functionality=check_func)
        
        # print(f"\nEarlier Answer Metrics:")
        # print(f"  BLEU Score: {metrics_earlier.get('bleu', 'N/A'):.4f}")
        # print(f"  ROUGE-L Score: {metrics_earlier.get('rouge', 'N/A'):.4f}")
        # print(f"  Exact Match: {metrics_earlier.get('exact_match', 'N/A')}")
        # print(f"  Functionality (PEC): {metrics_earlier.get('pec', 'N/A')}")
        # print(f"  Relaxed Functionality: {metrics_earlier.get('relax_pec', 'N/A')}")
        # print(f"  Syntax: {metrics_earlier.get('syntax', 'N/A')}")
        
        # print("\n" + "=" * 80)
        # print("ANALYSIS:")
        # print("=" * 80)
        # print("The results demonstrate that:")
        # print(f"1. Generated answer achieves BLEU={metrics['bleu']:.4f} (close to 1.0)")
        # print(f"2. Earlier answer has BLEU={metrics_earlier['bleu']:.4f} (lower)")
        # print("3. Both have functionality=0 (would fail property equivalence check)")
        # print("4. This confirms BLEU=1.0 doesn't guarantee functional correctness")
        # print("=" * 80)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
