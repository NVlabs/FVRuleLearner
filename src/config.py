from types import SimpleNamespace
from pathlib import Path
from utils import get_user, get_host
from collections import OrderedDict
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autogen import config_list_from_json
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from FVEval.fv_eval import (
    prompts_design2sva,
    prompts_nl2sva_machine,
    prompts_nl2sva_human,
    utils,
)

# global_task = 'inference'
global_task = 'train'
# global_task = 'eval'

# Define the root path (adjust this if necessary)
ROOT = Path(__file__).resolve().parents[2]  # Adjust this number based on actual .git location
debug = True
# debug = False

test_mode = True
# test_mode = False

# Training cases control (only active when debug=True)
# When debug=True and training_cases is not empty, use specified training cases
# for both Examples and Suggestions instead of the default hardcoded training_IDs
# Ratio = 0.2
# training_cases = [156, 281, 238, 231, 171, 122, 20, 74, 17, 269, 137, 176, 70, 229, 175, 54, 216, 258, 192, 222, 268, 53, 189, 7, 255, 48, 120, 164, 249, 152, 35, 131, 146, 270, 91, 87, 129, 259, 29, 111, 275, 55, 0, 1, 19, 118, 42, 223]
# Ratio = 0.4
# training_cases = [156, 20, 238, 231, 17, 122, 245, 49, 281, 171, 41, 74, 269, 137, 176, 70, 229, 175, 54, 216, 258, 23, 116, 198, 294, 105, 286, 121, 252, 123, 254, 192, 222, 268, 53, 189, 7, 255, 48, 120, 99, 2, 158, 37, 247, 71, 284, 266, 145, 200, 164, 249, 152, 35, 131, 146, 270, 91, 87, 129, 259, 29, 194, 136, 135, 107, 13, 224, 212, 139, 111, 275, 55, 0, 1, 19, 118, 42, 223, 235, 115, 240, 190, 61, 226, 187, 3, 265, 62, 181, 149, 288, 221, 15, 108, 79]
# Ratio = 0.6
# training_cases = [156, 20, 238, 231, 17, 122, 245, 49, 281, 171, 41, 74, 137, 252, 105, 175, 294, 23, 54, 50, 258, 116, 254, 286, 70, 269, 216, 123, 176, 198, 229, 121, 192, 222, 268, 53, 189, 7, 255, 48, 120, 99, 2, 158, 37, 247, 71, 284, 266, 145, 200, 250, 15, 106, 166, 160, 12, 290, 18, 181, 164, 249, 152, 35, 131, 146, 270, 91, 87, 129, 259, 29, 194, 136, 135, 107, 13, 224, 212, 139, 104, 38, 151, 79, 215, 178, 291, 177, 108, 169, 182, 218, 111, 275, 55, 0, 1, 19, 118, 42, 223, 235, 115, 240, 190, 61, 226, 187, 3, 265, 62, 8, 10, 246, 211, 6, 271, 184, 133, 102, 264, 193, 298, 101, 170, 277, 30, 92, 96, 288, 113, 84, 162, 199, 232, 126, 11, 66, 72, 295, 127, 56, 234, 205]
# Ratio = 0.8
# training_cases = [156, 20, 238, 231, 17, 122, 245, 49, 281, 171, 41, 74, 137, 252, 105, 175, 294, 23, 54, 50, 258, 116, 254, 286, 70, 269, 216, 123, 176, 198, 229, 121, 192, 222, 268, 53, 189, 7, 255, 48, 120, 99, 2, 158, 37, 247, 71, 284, 266, 145, 200, 250, 15, 106, 166, 160, 12, 290, 18, 181, 221, 66, 96, 199, 197, 298, 205, 110, 84, 11, 164, 249, 152, 35, 131, 146, 270, 91, 87, 129, 259, 29, 194, 136, 135, 107, 13, 224, 212, 139, 104, 38, 151, 79, 215, 178, 291, 177, 108, 169, 182, 218, 202, 149, 30, 208, 288, 295, 127, 101, 111, 275, 55, 0, 1, 19, 118, 42, 223, 235, 115, 240, 190, 61, 226, 187, 3, 265, 62, 8, 10, 246, 211, 6, 271, 184, 133, 102, 173, 85, 236, 82, 237, 34, 9, 142, 206, 24, 277, 77, 162, 170, 68, 234, 161, 113, 264, 126, 299, 193, 92, 72, 56, 201, 232, 4, 274, 167, 51, 58, 230, 154, 262, 40, 279, 57, 25, 292, 117, 186, 64, 93, 183, 180, 260, 242, 22, 285, 251, 14, 31, 147]
# Ratio = 1.0
# training_cases = [239, 290, 50, 167, 243, 7, 135, 266, 111, 54, 240, 173, 17, 258, 27, 242, 29, 86, 25, 269, 227, 177, 291, 262, 209, 74, 178, 119, 252, 222, 142, 264, 274, 31, 87, 206, 146, 91, 187, 15, 138, 18, 116, 156, 85, 299, 38, 158, 200, 45, 168, 160, 241, 297, 104, 285, 101, 130, 171, 232, 35, 265, 267, 12, 55, 257, 184, 70, 98, 246, 22, 179, 151, 102, 65, 286, 221, 211, 58, 194, 294, 51, 195, 84, 107, 139, 110, 155, 215, 199, 238, 9, 82, 66, 14, 2, 152, 92, 175, 62, 259, 61, 105, 166, 208, 89, 162, 136, 147, 260, 133, 137, 237, 245, 23, 261, 201, 272, 270, 63, 295, 292, 268, 231, 109, 83, 174, 154, 13, 40, 72, 186, 233, 205, 223, 224, 19, 235, 131, 53, 37, 212, 127, 41, 93, 180, 284, 108, 26, 192, 288, 121, 64, 198, 126, 48, 229, 141, 11, 244, 76, 256, 148, 190, 189, 183, 80, 129, 149, 123, 145, 213, 230, 20, 77, 216, 193, 210, 30, 271, 115, 181, 57, 218, 3, 134, 202, 279, 10, 1, 6, 71, 4, 8, 170, 251, 164, 117, 24, 88, 282, 283, 106, 42, 236, 161, 275, 176, 281, 298, 287, 34, 28, 226, 120, 169, 255, 79, 254, 196, 75, 143, 118, 49, 197, 234, 96, 277, 113, 249, 247, 159, 68, 248, 0, 99, 250, 182, 122, 56]
training_cases =[]

# task = "nl2sva_human"
# task = "nl2sva_machine"
task = "nl2sva_opencore"
# task = "design2sva_fsm"
# task = "design2sva_pipeline"
# task = "design2sva_training_docs"
# task = "design2sva_assertionbench"

# LLM_gateaway = "ADLR"
# LLM_gateaway = "Mark"
LLM_gateaway = "perflab"
# LLM_gateaway = "original"
# LLM_gateaway = "Finetune"

llm_model = 'o3-mini-20250131'
# llm_model = 'o1-20241217'
# llm_model = "claude-sonnet-4-5-20250929"
# llm_model = "gpt-4o-20241120"
# llm_model = "gpt-4-turbo"
# llm_model = "gpt-4"
# llm_model = 'gpt-4o'
# llm_model = "gpt-3.5-turbo"
# llm_model = "meta/llama3-70b-instruct"
# llm_model = "mixtral_8x7b"
# llm_model = "cursor-agent"

if llm_model == "mixtral_8x7b":
    base_url = "https://chipnemo-nvcf-proxy-rc.sc-paas.nvidia.com/v1/internal/chipnemo/mixtral_8x7b_H100/"
    api_key = "eyJhbGciOiJFUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiI2NjNlYmE1NjllOWE3OGE2ZWQyZDQ0MTEiLCJleHAiOjE3MzE5MDkyMjksInN1YiI6Imxpd2FuIiwidHlwIjoiVVNFUiIsInVzZXJJZCI6Imxpd2FuIiwib3JpZ2luIjoibGl3YW4iLCJqdGkiOiI2NzEzNDk2ZGZhZGQ1YWM1ODIxMzVkYzkiLCJpYXQiOjE3MjkzMTcyMjksImlzcyI6IjYyOTYzMGRkMGJhN2Y3NDEzYTNiNjMwMSJ9.AAkSCApATJFS-iLwtObqIqE45xzkv6soK4Z2zLxxjv0nYYXYZ2LNwnNDLvyd2YKpOtdSBCu4W6dynOmqAXAG2SlOABMML02IHnw1n0aDYS2UpC27u1MSMRSMaG_unHXjZ7NvL92oYDRSRJ75GxxUICpYXp0F43QmBkWuVVOMSKPHyA7p"

elif llm_model == "meta/llama3-70b-instruct":
    base_url = "https://chipnemo-nvcf-proxy-rc.sc-paas.nvidia.com/v1/internal/chipnemo/meta-llama3-70b/"
    api_key = "eyJhbGciOiJFUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiI2NjNlYmE1NjllOWE3OGE2ZWQyZDQ0MTEiLCJleHAiOjE3MzE5MDkyMjksInN1YiI6Imxpd2FuIiwidHlwIjoiVVNFUiIsInVzZXJJZCI6Imxpd2FuIiwib3JpZ2luIjoibGl3YW4iLCJqdGkiOiI2NzEzNDk2ZGZhZGQ1YWM1ODIxMzVkYzkiLCJpYXQiOjE3MjkzMTcyMjksImlzcyI6IjYyOTYzMGRkMGJhN2Y3NDEzYTNiNjMwMSJ9.AAkSCApATJFS-iLwtObqIqE45xzkv6soK4Z2zLxxjv0nYYXYZ2LNwnNDLvyd2YKpOtdSBCu4W6dynOmqAXAG2SlOABMML02IHnw1n0aDYS2UpC27u1MSMRSMaG_unHXjZ7NvL92oYDRSRJ75GxxUICpYXp0F43QmBkWuVVOMSKPHyA7p"

llm_mode = 'agent'
# llm_mode = 'baseline'

if global_task in ['inference', 'train']:
    model = 'agent'
    src_examples = [task]

    if "design2sva" in task:
        cot_strategy = "plan-model-act-in-one"
        num_assertions = 2
        only_assertion = True

        # iterative_optimization = True
        iterative_optimization = False
        use_MCTS = False
        use_multi_and_pruner = True
        # use_multi_and_pruner = False
        
        with_individual_signal = True
        if with_individual_signal:
            all_signals_in_one_shot = True
            # all_signals_in_one_shot = False

        feature_tree = False

        # pre-processing
        # add_AST = True
        add_AST = False
        if add_AST:
            AST_method = "LLM"
            AST_method = "iverilog"
        # add_comment = True
        add_comment = False

        evaluate_models = ['statement', 'branch', 'functional', 'toggle']

        max_iterations = 9999
        target_coverage = 100
        MAX_TIME = 2 * 60 * 60  # 2 hours in seconds

        # with_assumptions = True
        with_assumptions = False
        use_inter_param = False

        # cover & assertion
        # both_cover_and_assertion = True
        both_cover_and_assertion = False

        # Baseline
        # baseline_AutoSVA2 = True
        baseline_AutoSVA2 = False
        # baseline_GOLDMINE = True
        baseline_GOLDMINE = False
        # In config.py:
        # baseline_LAAGRV = True
        baseline_LAAGRV = False
        unified_sample = False

        if baseline_LAAGRV:
            laagrv_max_cex_refinements = 3  # Max 3 refinements per assertion as shown in paper
            max_iterations = 9999

        if use_MCTS:
            max_rollout_depth = 1
            coverage_check_interval = 5
        elif use_multi_and_pruner:
            temperature_list = [0.0]
            assertion_top_k_ratios = 0.5
            assertion_top_k_num = int(assertion_top_k_ratios*len(temperature_list)*num_assertions)
            use_ICRL = True
            # use_ICRL = False
            # with_pruner = True
            with_pruner = False
            post_coverage_calc = True
            coverage_report = True
            # coverage_report = False

            max_assertions = 2
            base_probability = 0.15  # Reduced from 0.3 to 0.15
            iteration_weight = 0.05  # Reduced from 0.1 to 0.05
            min_probability = 0.01   # Reduced from 0.03 to 0.01
            delta = 0.025           # Reduced from 0.05 to 0.025

        elif iterative_optimization:
            use_ICRL = False
            # use_ICRL = False
            coverage_report = True
            # coverage_report = False
        
        coverage_types = ["undetectable", "unprocessed"]


    # agent_arch = "no_agent"  # simplest baseline same as Josh, i.e. no 2 chat, no multi-agent; just simple prompting
    # agent_arch = "two_agents"

    # agent_arch = "three_agents"

    # agent_arch = "three_agents"  # add a serious of Helper functions
    # agent_arch = "four_agents"
    agent_arch = "flexible_agents"

    max_num_syntax_iter = 10
    # max_num_syntax_iter = 40

    # LLM specification
    temperature = 0
    cleanup_temp = True
    num_icl = 3
    # num_icl = 0
    max_token = 20000
    group_id = 0 # For group partition for run.py
    num_group = 1
    start_num = 0

    # split dataset with a certain random seed
    if "nl2sva" in task:
        split_ratios = {
            'test': 0.2,
            'train': 0.8,  # or 0.6 or 0.4
        }
    elif "design2sva" in task:
        split_ratios = {
            'test': 1,
            'train': 0,
        }

    random_seed = 100
    
    # Optional: Subsample QTrees to test different training ratios
    # If you have QTrees built with 0.8 training ratio, use this to test smaller ratios:
    # original_train_ratio = 0.8  # The training ratio used when building the saved QTrees

    # qtree_subsample_ratio = None     # Use all loaded QTrees (default: 0.8 training)
    qtree_subsample_ratio = split_ratios['train']

    if task in src_examples:
        filter_self = False
        # filter_self = True

    # helper_list = []
    # helper_list = [
    #     "You are a SVA expert, trying to ask followup questions to fix some SVAs generated by others. Please double check whether this problem uses the sequential or combinatorial 
    # ions and then rewrite the correct assertions enclosed with ```systemverilog and ```. Only output the code snippet and do NOT output anything else."

    # ]

    # helper_list = [
    #     "Ensure to use |-> for implication instead of |=> for properties where an eventual condition must follow an initial condition.",
    # "Use === for equality checks instead of == to handle X-values correctly in the verification process.",
    # "When specifying that a condition must eventually occur, use the s_eventually operator correctly to capture the intended temporal behavior.",
    # "Correctly apply clock cycle delay using ## before the expression that must hold after the specified cycles.",
    # "Avoid using the |-> ##[1:$] syntax; use s_until for the eventual condition.",
    # "Verify that the temporal operators are appropriately chosen for the specified conditions.",
    # "Ensure that conditions involving multiple signals use logical AND (&&) appropriately to combine them.",
    # "Verify that the signals being compared are of the same width and type.",
    # ]

    # use_JG = False
    use_JG = True

    GPT_retries = 10
    backoff_factor = 2  # decide the wait time of GPT

    if global_task == 'train' and 'nl2sva' in task:
        baseline_textgrad = False
        # baseline_textgrad = True # TODO: this is risky; if this is turned on, will override the suggestion generation and instead use a naive baseline called textgrad
        baseline_finetune = False

        contain_PEC_message = False
        # contain_PEC_message = True

        use_RAG = False  # tricky
        use_JG = False
        RAG_content = ['Suggestions']
        num_iter = 25

        # is_constrain_number = True
        is_constrain_number = False
        # if is_constrain_number:
        #     num_constrain = 5

        suggestion_saving_rule = "none"
        # suggestion_saving_rule = "last_bleu"
        # suggestion_saving_rule = "init_bleu"
        # suggestion_saving_rule = "all"
        train_temp_setting = "random"
        # train_temp_setting = "fixed"
        if train_temp_setting == "random":
            change_only_bleu_same = True
            # change_only_bleu_same = False
            low_temp = 0
            high_temp = 1.0
        if train_temp_setting == "fixed":
            train_temp = temperature
        if debug == True:
            num_iter = 25
            # num_iter = 1 # shorter

        add_ICL_reflection_prompt = False
        # add_ICL_reflection_prompt = True # fancy

        use_QTree = True
        # use_QTree = False
        if use_QTree:
            question_tree_width = 3  # Number of questions per level in Q-Tree
        
        PEC_method = 'JasperGold'
        # only_BLEU = True
        only_BLEU = False

        # carryover = True
        carryover = False

        if carryover == True:
            num_carryover = 1
            # num_carryover = 2
            # num_carryover = 3
            # if num_carryover == 2 or num_carryover == 3:
            #     if (llm_model == "gpt-4") or (LLM_gateaway == "ADLR" and llm_model == "gpt-3.5-turbo"):
            #         num_iter = 5

    elif global_task == 'inference' and 'nl2sva' in task:
        # use_RAG = False
        use_RAG = True
        similarity_str = ["Question"]
        eval_content = ["similarity","jg"]

        # Q-Tree Inference Settings
        # baseline_LFM = True
        baseline_LFM = False
        # baseline_emsemble = True
        baseline_emsemble = False
        # baseline_textgrad = True
        baseline_textgrad = False
        # baseline_finetune = True
        baseline_finetune = False
        # baseline_hybrid_nl2sva = True
        # Hybrid NL2SVA baseline: 
        # 1. Generates initial SVA
        # 2. Extracts operators from initial SVA
        # 3. Appends operator concept-meaning pairs to the prompt
        # 4. Generates final SVA with operator guidance (instead of using suggestions)
        # baseline_hybrid_nl2sva = True
        baseline_hybrid_nl2sva = False

        if baseline_emsemble:
            use_RAG = False
            passk = 10
            # passk = 4
            voting_tech = "LLM"
            # voting_tech = "PEC" # Not Implemented
        if use_RAG:
            # RAG_content = ['Examples']
            # RAG_content = ['Suggestions']
            RAG_content = ['Examples', 'Suggestions']

            all_examples = False
            # all_examples = True

            if 'Examples' in RAG_content:
                prompting_instruction = 0
                Examples_top_k = 1
                # Examples_top_k = 2
                # Examples_top_k = 3
                # Examples_top_k = 4
                # Examples_top_k = 5 

            if 'Suggestions' in RAG_content:
                # Suggestions_top_k = 1
                # Suggestions_top_k = 2
                Suggestions_top_k = 3
                # Suggestions_top_k = 4
                # Suggestions_top_k = 5
                # Suggestions_top_k = 6
                # Suggestions_top_k = 7
                # Suggestions_top_k = 8
                # Suggestions_top_k = 9
                # Suggestions_top_k = 10
                # Suggestions_top_k = 15
                # Suggestions_top_k = 20

                Suggestions_Reasoning = False  # Skip Suggestions_Reasoning Agent (use Q-Tree suggestions directly)
                # Suggestions_Reasoning = True

                # use_qtree_inference = False
                use_qtree_inference = True  # Use Q-Tree based inference instead of traditional RAG
                if use_qtree_inference:
                    # qtree_similarity_top_k = 2  # Number of similar Q-Trees to retrieve

                    # retrieval_on_ranking = False
                    retrieval_on_ranking = True

                    if retrieval_on_ranking:
                        # suggestions_top_k = 10 # the specific number of suggestions to pick up
                        # Add retrieval / generation mechanism
                        qtree_similarity_top_k = 5
                        qtree_ranking_mode = 'prompt' 
                        # qtree_ranking_mode = 'sva'
                        # rule_source = 'retrieve'  # Retrieve rules
                        rule_source = 'generate'  # Generate rules
                        
                        # Hybrid ranking configuration (for 'initial_sva_generation' mode)
                        # llm_score_weight: weight for LLM evaluation (0.0-1.0)
                        # operator_score_weight: weight for operator alignment (0.0-1.0)
                        # Note: llm_score_weight + operator_score_weight should equal 1.0
                        # llm_score_weight = 0.7
                        # operator_score_weight = 0.3

                        llm_score_weight = 0.5
                        operator_score_weight = 0.5
                        
                        # Operator-based pruning (for 'initial_sva_generation' mode)
                        # enable_operator_pruning: whether to prune suggestions based on operator alignment
                        enable_operator_pruning = True
                        # enable_operator_pruning = False
                        operator_pruning_threshold = 0.0  # Minimum operator alignment score (0.0 = include all non-zero)
                        
                        # NL alignment instruction in rule generation prompt
                        # instruct_skip_misaligned_in_rules: tell LLM to output None for traces that don't align with NL
                        instruct_skip_misaligned_in_rules = True
                        # instruct_skip_misaligned_in_rules = False
                    
                # suggestion_link = "Associated" # Use the suggestions provided by the similar questions to append to the prompt
                # suggestion_link = "Direct" # Direct use similar suggestions to append to the prompt
                # suggestion_link = "Dual_Agent" # Use two agents to collaborate with each other to generate the proper suggestions
                temporal_suggestion = False
                # temporal_suggestion = True
                deduplication = True
                # deduplication = False
                # operator_explanation = True
                operator_explanation = False
                filter_functionality = True
                # filter_functionality = False
                retrieve_str = ["Suggestions"]
                if baseline_LFM:
                    retrieve_str = ["Question", "Answer", "Reference Answer"]
                    filter_functionality = False
                    RAG_content = ['Suggestions']
                    deduplication = False
                    Suggestions_top_k = 10
                # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-07-26T00-32-42.887270_pdx-xterm-57_liwan/suggestions.pkl' #  human, llama3-70b-instruct, BLEU+PEC, num_iter = 10, random train_temp,  hange_only_bleu_same, LLM_optimizer, num_carryover = 1 
                # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-07-26T09-52-11.060060_pdx-xterm-153_liwan/suggestions.pkl' # machine, llama3-70b-instruct, BLEU+PEC, num_iter = 10, random train_temp, change_only_bleu_same 
                # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-07-26T13-25-11.933011_pdx-xterm-62_liwan/suggestions.pkl' # machine, gpt-4-turbo, BLEU+PEC, num_iter = 10, fixed train_temp, LLM_optimizer, num_carryover = 1 
                # _______________________________________________________
                # Human:
                if task =="nl2sva_human":
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/backup_logs/train_2024-08-03T20-45-18.088918_pdx-xterm-56_liwan/suggestions.pkl' # llama3-70b-instruct, BLEU+PEC, num_iter = 25, random train_temp, change_only_bleu_same
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-04T09-53-03.763808_pdx-xterm-62_liwan/suggestions.pkl' # llama3-70b-instruct, BLEU+PEC, num_iter = 40, random train_temp, change_only_bleu_same
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-16T13-48-21.561181_pdx-xterm-153_liwan/suggestions.pkl' # 20%
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-16T13-57-39.043413_pdx-xterm-153_liwan/suggestions.pkl' # 20%, V2
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-16T13-48-37.648830_pdx-xterm-153_liwan/suggestions.pkl' # 40%
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-16T13-49-01.355740_pdx-xterm-153_liwan/suggestions.pkl' # 60%
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-16T13-50-35.867869_pdx-xterm-153_liwan/suggestions.pkl' # GPT-4o
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-17T09-51-04.272610_pdx-xterm-153_liwan/suggestions.pkl' # Mixtral
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-04T10-02-33.153812_pdx-xterm-57_liwan/suggestions.pkl' # machine
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-14T15-43-48.623120_pdx-xterm-153_liwan/all_suggestions.pkl' # opencore
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-11T09-01-03.562227_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions.pkl' # llama3-70b-instruct, BLEU+PEC, num_iter = 25, random train_temp, change_only_bleu_same
                    # Now using folder path - pkl files will be appended when loading
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-20T00-01-20.119404_pdx-container-xterm-064.prd.it.nvidia.com_liwan'
                    load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-22T09-03-15.426631_pdx-container-xterm-064.prd.it.nvidia.com_liwan' # Human
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-17T08-56-58.942070_pdx-container-xterm-064.prd.it.nvidia.com_liwan' # Machine
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-13T13-07-19.903685_pdx-container-xterm-064.prd.it.nvidia.com_liwan'
                elif task =="nl2sva_machine":
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-04T10-02-33.153812_pdx-xterm-57_liwan/suggestions.pkl' # llama3-70b-instruct, BLEU+PEC, num_iter = 40, random train_temp, change_only_bleu_same
                    # Now using folder path - pkl files will be appended when loading
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/backup_logs/train_2024-08-03T20-45-18.088918_pdx-xterm-56_liwan' # human
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-22T09-03-15.426631_pdx-container-xterm-064.prd.it.nvidia.com_liwan' # Human
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-22T09-03-15.426631_pdx-container-xterm-064.prd.it.nvidia.com_liwan'
                    load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-17T08-56-58.942070_pdx-container-xterm-064.prd.it.nvidia.com_liwan' # Machine
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-23T14-42-26.354352_pdx-container-xterm-062.prd.it.nvidia.com_liwan' # OpenCore
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-28T09-42-04.137848_pdx-container-xterm-062.prd.it.nvidia.com_liwan' # Machine-suggest-only
                # opencore
                elif task =="nl2sva_opencore":
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-08T21-40-53.479594_pdx-xterm-57_liwan/suggestions.pkl'# llama3-70b-instruct, BLEU+PEC, num_iter = 30, random train_temp, change_only_bleu_same, one time
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-13T21-31-26.756243_pdx-xterm-59_liwan/all_suggestions.pkl' # updated, llama3-70b-instruct, BLEU+PEC, num_iter = 25, random train_temp, change_only_bleu_same
                    # Now using folder path - pkl files will be appended when loading
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/backup_logs/train_2024-08-14T15-43-48.623120_pdx-xterm-153_liwan' # non-JG + JG, llama3-70b-instruct, BLEU+PEC, num_iter = 25, random train_temp, change_only_bleu_same
                    # load_suggestions_path = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-14T17-12-43.034552_pdx-xterm-59_liwan/all_suggestions.pkl' # gpt-4o, BLEU+PEC, num_iter = 25, random train_temp, change_only_bleu_same, LLM_optimizer, num_carryover = 2
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-17T08-56-58.942070_pdx-container-xterm-064.prd.it.nvidia.com_liwan'
                    # load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-27T00-58-28.704284_pdx-container-xterm-062.prd.it.nvidia.com_liwan'
                    load_suggestions_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-23T14-42-26.354352_pdx-container-xterm-062.prd.it.nvidia.com_liwan'

        use_autorater = False
        # use_autorater = True

        if use_autorater:
            # max_autorater_iter = 1
            # max_autorater_iter = 2
            # max_autorater_iter = 3
            max_autorater_iter = 12
            baseline_reflexion = False
            # baseline_reflexion = True

            # autorater_feedback_type = 'yes_or_no'
            # autorater_feedback_type = 'numeric_rating'

    # Dataset specifications
    # if task == "nl2sva_machine":
    #     mode = "machine"
    # elif task == "nl2sva_human":
    #     mode = "human"

    # dataset_path for test
    if task == "nl2sva_human":
        dataset_path = os.path.join(ROOT, "FVEval", "data_nl2sva", "data", "nl2sva_human.csv")
    elif task == "nl2sva_machine":
        # dataset_path = os.path.join(ROOT, "FVEval", "data_nl2sva", "data", "nl2sva_machine.csv")
        dataset_path = os.path.join(ROOT, "FVEval", "data_nl2sva", "data", "nl2sva_machine_updated.csv")
    elif task == "nl2sva_opencore":
        dataset_path = os.path.join(ROOT, "FVEval", "data_1k", "module_sva_nl_manual_editing.csv")
    elif task == "design2sva_fsm":
        dataset_path = os.path.join(ROOT, "FVEval", "data_design2sva", "data", "design2sva_fsm.csv")
    elif task == "design2sva_pipeline":
        dataset_path = os.path.join(ROOT, "FVEval", "data_design2sva", "data", "design2sva_pipeline.csv")
    elif task == "design2sva_training_docs":
        dataset_path = os.path.join(ROOT, "FVEval", "data_design2sva", "data", "design2sva_training_docs.csv")
    elif task == "design2sva_assertionbench":
        dataset_path = os.path.join(ROOT, "FVEval", "data_design2sva", "data", "design2sva_assertionbench.csv")

    # Convert dataset_path to string for SimpleNamespace
    # dataset_path = dataset_path.as_posix()
    # temp_dir = os.path.join(ROOT, "tmp_sva")

    timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
    if task == "nl2sva_machine" or task == "nl2sva_human" or task == "nl2sva_opencore":
        save_dir = ROOT / f"results_nl2sva/{num_icl}/{timestamp_str}"
        save_dir = save_dir.as_posix()
    elif "design2sva" in task:
        save_dir = ROOT/ f"results_design2sva/{cot_strategy}/{timestamp_str}"
        save_dir = save_dir.as_posix()

    if debug == True:
        random_sample_size = 1
        # llm_model = "gpt-3.5-turbo"
        cleanup_temp_files = False
        # only_test_ids = [2,3]  # the specific position during the previous run [91]
        # only_test_ids = [78, 31, 39, 30, 2, 3, 27, 74, 52, 38, 71, 62, 5, 66, 8]  # the specific position during the previous run [91]
        # only_test_ids = [78]
        # only_test_ids = [31]
        # only_test_ids = [39]
        # only_test_ids = [2]
        # only_test_ids = [3]
        # only_test_ids = [30]
        # only_test_ids = [27]
        # Machine Testing
        # only_test_ids = [225, 16, 132, 59, 203, 214, 46, 47, 52, 67, 128, 114, 21, 144, 263, 44, 165, 100, 60, 33, 217, 293, 32, 228, 278, 95, 97, 150, 191, 157, 103, 296, 188, 43, 289, 204, 153, 81, 185, 124, 125, 276, 69, 36, 280, 172, 273, 90, 94, 5, 78, 253, 73, 112, 220, 207, 140, 219, 39, 163]

        # only_test_ids = [64, 47, 114, 44, 67, 143, 32, 78, 38, 54, 35, 124, 130, 123, 36, 94, 5, 70] # Function-zero indices
        # only_test_ids = [217, 103, 280, 44]
        # only_test_ids = [64, 47, 114, 44, 67, 143, 32, 78, 38, 54, 35, 124, 130, 123, 36, 94, 5, 70]
        # only_test_ids = [130]
        # only_test_ids = [225, 16, 132, 59, 203, 214, 46, 47, 52, 67, 128, 114, 21, 144, 263, 44, 165, 100, 60, 33, 217, 293, 32, 228, 278, 95, 97, 150, 191, 157, 103, 296, 188, 43, 289, 204, 153, 81, 185, 124, 125, 276, 69, 36, 280, 172, 273, 90, 94, 5, 78, 253, 73, 112, 220, 207, 140, 219, 39, 163]
        # only_test_ids = [46, 47, 44, 217, 293, 32, 228, 103, 296, 188, 204, 185, 69, 36, 280, 94, 5, 112, 219]
        # only_test_ids = [32, 219]
        # only_test_ids = [46, 52, 60, 81, 57, 140]
        # only_test_ids = [543, 231, 450, 721, 526, 400, 45, 670, 622, 225, 133, 898, 899, 810, 120, 846, 513, 958, 67, 844, 303, 768, 655, 329, 453, 767, 963, 339, 463, 872, 563, 385, 776, 945, 287]
        # only_test_ids = [46, 47, 52, 60, 32, 78, 38, 81, 35, 124, 69, 123, 5, 112, 57, 140, 70]
        # only_test_ids = [214, 46, 47, 114, 60, 32, 228, 188, 204, 81, 185, 124, 69, 36, 280, 273, 94, 5, 112, 220, 207, 140]


        # only_test_ids = [217]
        # only_test_ids = [342, 471, 929, 100, 543, 231, 154, 450, 721, 788, 685, 61, 526, 69, 569, 400, 807, 326, 662, 220, 524, 165, 291, 45, 550, 670, 902, 651, 622, 211, 542, 925, 757, 225, 576, 369, 822, 915, 679, 860, 557, 863, 223, 566, 554, 31, 280, 428, 831, 791, 94, 1, 33, 133, 276, 908, 904, 912, 898, 293, 236, 753, 175, 459, 340, 899, 32, 810, 610, 131, 498, 152, 491, 976, 759, 684, 120, 503, 57, 27, 574, 846, 939, 132, 24, 790, 772, 370, 890, 688, 934, 513, 353, 437, 512, 263, 376, 903, 958, 273, 65, 104, 986, 177, 310, 815, 521, 98, 852, 991, 67, 952, 298, 243, 548, 212, 614, 980, 645, 637, 778, 441, 324, 337, 844, 110, 303, 381, 534, 77, 87, 677, 768, 655, 249, 335, 329, 676, 84, 102, 423, 979, 683, 453, 767, 963, 584, 339, 36, 240, 926, 781, 54, 299, 256, 463, 978, 872, 167, 563, 442, 700, 639, 966, 255, 927, 765, 532, 385, 893, 170, 665, 776, 380, 533, 201, 945, 66, 192, 784, 306, 464, 982, 972, 546, 344, 287, 804, 594, 998, 725, 752, 86, 605, 699, 318, 783, 424, 185, 111]
        # only_test_ids = [36, 38, 50, 70, 83, 95, 110, 111, 117, 121, 127, 134, 141, 143, 167, 194]
        # only_test_ids = [38]
        # only_test_ids = [121, 127]
        # only_test_ids = [342]
        # only_test_ids = [885, 748, 832, 349, 134, 770, 642, 462, 715, 278, 8, 989, 857, 763, 189, 478, 654, 357, 499, 6, 882, 517, 629, 577, 227, 631, 338, 644, 186, 247, 515, 347, 965, 284, 373, 761, 445, 634, 475, 693, 18, 529, 309, 163, 181, 806, 894, 140, 430, 967, 732, 387, 799, 891, 764, 290, 439, 377, 960, 956, 60, 500, 191, 362, 751, 875, 289, 456, 393, 671, 354, 547, 954, 572, 218, 847, 374, 15, 460, 680, 153, 248, 736, 145, 117, 403, 397, 560, 508, 417, 239, 241, 203, 75, 172, 869, 297, 591, 743, 226, 473, 80, 52, 135, 942, 724, 541, 414, 673, 408, 729, 809, 595, 897, 302, 497, 868, 176, 432, 431, 695, 603, 25, 138, 527, 107, 204, 267, 993, 726, 814, 943, 999, 183, 304, 319, 556, 70, 523, 130, 210, 716, 336, 190, 787, 17, 917, 995, 801, 888, 668, 434, 803, 848, 479, 179, 834, 575, 187, 502, 467, 559, 315, 938, 265, 395, 136, 194, 649, 114, 873, 58, 866, 910, 900, 392, 657, 881, 10, 29, 261, 643, 696, 292, 285, 879, 398, 909, 549, 514, 449, 613, 794, 268, 940, 587, 583, 755, 535, 565, 386, 588, 73, 984, 985, 628, 701, 733, 484, 56, 205, 851, 886, 586, 146, 627, 105, 72, 930, 64, 270, 959, 325, 250, 229, 406, 101, 415, 589, 625, 367, 420, 438, 4, 490, 740, 0, 539, 660, 552, 538, 708, 382, 711, 404, 126, 429, 507, 970, 828, 618, 608, 169, 928, 506, 855, 207, 975, 363, 919, 564, 597, 802, 624, 323, 825, 876, 314, 528, 728, 520, 840, 93, 159, 796, 842, 481, 918, 21, 451, 585, 350, 283, 320, 394, 962, 142, 573, 892, 476, 769, 607, 141, 544, 421, 551, 714, 947, 950, 368, 606, 41, 482, 137, 493, 774, 878, 682, 206, 119, 501, 971, 383, 739, 294, 719, 616, 352, 487, 279, 46, 416, 973, 762, 494, 286, 19, 983, 251, 78, 570, 812, 99, 511, 252, 418, 578, 883, 348, 933, 924, 816, 332, 372, 407, 295, 826, 264, 42, 364, 76, 858, 510, 271, 635, 345, 477, 22, 413, 244, 530, 219, 379, 678, 173, 626, 758, 95, 425, 652, 180, 600, 246, 582, 28, 853, 782, 401, 409, 686, 222, 259, 272, 601, 944, 446, 199, 690, 390, 435, 62, 509, 746, 7, 650, 835, 905, 43, 856, 936, 531, 996, 16, 805, 850, 156, 274, 321, 647, 322, 697, 83, 433, 620, 157, 611, 798, 160, 687, 333, 196, 448, 312, 313, 672, 896, 122, 328, 375, 703, 849, 11, 474, 238]
        # only_test_ids = [834, 575, 187, 502, 467, 559, 315, 938, 265, 395, 136, 194, 649, 114, 873, 58, 866, 910, 900, 392, 657, 881, 10, 29, 261, 643, 696, 292, 285, 879, 398, 909, 549, 514, 449, 613, 794, 268, 940, 587, 583, 755, 535, 565, 386, 588, 73, 984, 985, 628, 701, 733, 484, 56, 205, 851, 886, 586, 146, 627, 105, 72, 930, 64, 270, 959, 325, 250, 229, 406, 101, 415, 589, 625, 367, 420, 438, 4, 490, 740, 0, 539, 660, 552, 538, 708, 382, 711, 404, 126, 429, 507, 970, 828, 618, 608, 169, 928, 506, 855, 207, 975, 363, 919, 564, 597, 802, 624, 323, 825, 876, 314, 528, 728, 520, 840, 93, 159, 796, 842, 481, 918, 21, 451, 585, 350, 283, 320, 394, 962, 142, 573, 892, 476, 769, 607, 141, 544, 421, 551, 714, 947, 950, 368, 606, 41, 482, 137, 493, 774, 878, 682, 206, 119, 501, 971, 383, 739, 294, 719, 616, 352, 487, 279, 46, 416, 973, 762, 494, 286, 19, 983, 251, 78, 570, 812, 99, 511, 252, 418, 578, 883, 348, 933, 924, 816, 332, 372, 407, 295, 826, 264, 42, 364, 76, 858, 510, 271, 635, 345, 477, 22, 413, 244, 530, 219, 379, 678, 173, 626, 758, 95, 425, 652, 180, 600, 246, 582, 28, 853, 782, 401, 409, 686, 222, 259, 272, 601, 944, 446, 199, 690, 390, 435, 62, 509, 746, 7, 650, 835, 905, 43, 856, 936, 531, 996, 16, 805, 850, 156, 274, 321, 647, 322, 697, 83, 433, 620, 157, 611, 798, 160, 687, 333, 196, 448, 312, 313, 672, 896, 122, 328, 375, 703, 849, 11, 474, 238, 705, 40, 747, 109, 795, 845, 619, 30, 262, 602, 760, 702, 504, 968, 829, 522, 990, 489, 920, 646, 480, 707, 91, 561, 115, 518, 230, 14, 35, 234, 356, 232, 288, 483, 118, 330, 496, 661, 951, 388, 691, 366, 174, 931, 797, 741, 813, 880, 580, 977, 780, 718, 300, 90, 108, 492, 158, 937, 20, 426, 452, 47, 71, 89, 359, 440, 469, 992, 777, 228, 630, 592, 710, 128, 308, 895, 113, 516, 738, 598, 887, 139, 648, 468, 640, 74, 884, 562, 590, 615, 824, 217, 200, 457, 391, 389, 994, 941, 964, 327, 720, 13, 907, 3, 713, 162, 147, 953, 360, 632, 709, 756, 85, 188, 745, 34, 454, 689, 331, 361, 948, 88, 23, 961, 838, 830, 659, 305, 864, 81, 669, 9, 957, 237, 301, 571, 932, 38, 311, 921, 2, 277, 839, 987, 233, 195, 97, 727, 266, 836, 275, 612, 79, 422, 213, 148, 427, 808, 593, 621, 198, 536, 997, 870, 906, 461, 821, 436, 553, 37, 901, 750, 704, 68, 257, 125, 92, 656, 216, 143, 827, 96, 161, 254, 202, 48, 773, 485, 817, 50, 949, 455, 717, 39, 558, 488, 658, 675, 916, 155, 730, 694, 410, 106, 742, 399, 44, 214, 865, 737, 121, 862, 841, 935, 486, 116, 911, 12, 981, 341, 365, 581, 744, 766, 823, 666, 221, 636, 537, 317, 224, 609, 800, 51, 735, 123, 712, 889, 281, 771, 55, 168, 334, 599, 692, 59, 150, 260, 596, 505, 820, 525, 785, 447, 913, 723, 843, 867, 245, 258, 444, 667, 307, 103, 833, 63, 779, 411, 495, 371, 874, 555, 351, 698, 540, 346, 604, 706, 296, 731, 988, 793, 633, 734, 253, 623, 53, 775, 567, 638, 129, 396, 923, 946, 818, 458, 242, 681, 26, 861, 171, 197, 166, 579, 871, 792, 151, 405, 617, 5, 184, 127, 653, 664, 877, 164, 384, 837, 811, 922, 282, 568, 472, 412, 215, 854, 419, 641, 378, 355, 914, 193, 144, 182, 208, 786, 859, 316, 235, 343, 209, 969, 663, 955, 674, 49, 269, 466, 754, 82, 124, 545, 112, 819, 519, 443, 358, 749, 402, 722, 178, 789, 974, 465, 470, 149]
        # only_test_ids = [234, 356, 232, 288, 483, 118, 330, 496, 661, 951, 388, 691, 366, 174, 931, 797, 741, 813, 880, 580, 977, 780, 718, 300, 90, 108, 492, 158, 937, 20, 426, 452, 47, 71, 89, 359, 440, 469, 992, 777, 228, 630, 592, 710, 128, 308, 895, 113, 516, 738, 598, 887, 139, 648, 468, 640, 74, 884, 562, 590, 615, 824, 217, 200, 457, 391, 389, 994, 941, 964, 327, 720, 13, 907, 3, 713, 162, 147, 953, 360, 632, 709, 756, 85, 188, 745, 34, 454, 689, 331, 361, 948, 88, 23, 961, 838, 830, 659, 305, 864, 81, 669, 9, 957, 237, 301, 571, 932, 38, 311, 921, 2, 277, 839, 987, 233, 195, 97, 727, 266, 836, 275, 612, 79, 422, 213, 148, 427, 808, 593, 621, 198, 536, 997, 870, 906, 461, 821, 436, 553, 37, 901, 750, 704, 68, 257, 125, 92, 656, 216, 143, 827, 96, 161, 254, 202, 48, 773, 485, 817, 50, 949, 455, 717, 39, 558, 488, 658, 675, 916, 155, 730, 694, 410, 106, 742, 399, 44, 214, 865, 737, 121, 862, 841, 935, 486, 116, 911, 12, 981, 341, 365, 581, 744, 766, 823, 666, 221, 636, 537, 317, 224, 609, 800, 51, 735, 123, 712, 889, 281, 771, 55, 168, 334, 599, 692, 59, 150, 260, 596, 505, 820, 525, 785, 447, 913, 723, 843, 867, 245, 258, 444, 667, 307, 103, 833, 63, 779, 411, 495, 371, 874, 555, 351, 698, 540, 346, 604, 706, 296, 731, 988, 793, 633, 734, 253, 623, 53, 775, 567, 638, 129, 396, 923, 946, 818, 458, 242, 681, 26, 861, 171, 197, 166, 579, 871, 792, 151, 405, 617, 5, 184, 127, 653, 664, 877, 164, 384, 837, 811, 922, 282, 568, 472, 412, 215, 854, 419, 641, 378, 355, 914, 193, 144, 182, 208, 786, 859, 316, 235, 343, 209, 969, 663, 955, 674, 49, 269, 466, 754, 82, 124, 545, 112, 819, 519, 443, 358, 749, 402, 722, 178, 789, 974, 465, 470, 149]
        # only_test_ids = [545, 112, 819, 519, 443, 358, 749, 402, 722, 178, 789, 974, 465, 470, 149, 786,
        #  859, 316, 235, 343, 209, 969, 663, 955, 674, 49, 269, 466, 754, 82, 124, 472, 412, 215, 854, 419, 641, 378, 355, 914, 193, 144, 182, 208, 384, 837, 811, 922, 282, 568,
        #  151, 405, 617, 5, 184, 127, 653, 664, 877, 164, 242, 681, 26, 861, 171, 197, 166, 579, 871, 792,
        #  734, 253, 623, 53, 775, 567, 638, 129, 396, 923, 946, 818, 458, 604, 706, 296, 731, 988, 793, 633,
        #  525, 785, 447, 913, 723, 843, 867, 245, 258, 444, 667, 307, 103, 833, 63, 779, 411, 495, 371, 874, 555, 351, 698, 540, 346]
        # only_test_ids = [425]

        # only_test_ids = [223]

        # o3-mini
        # only_test_ids = [
        # 231, 400, 45, 225, 822, 679, 223, 280, 428, 831,
        # 94, 133, 459, 899, 810, 498, 759, 684, 120, 939,
        # 132, 24, 263, 104, 521, 852, 67, 952, 243, 980,
        # 441, 381, 677, 249, 423, 979, 453, 240, 167, 966,
        # 532, 385, 201, 945, 287, 804, 998, 725, 699
        # ]
        # only_test_ids = [94, 131, 498, 120, 263, 952, 243, 980, 441, 423, 167, 532, 784, 699]
        # only_test_ids = [441]

        # only_test_ids = [231, 400, 45, 225, 822, 679, 223, 280, 428, 831, 94, 133, 459, 899, 810, 498, 759, 684, 120, 939, 132, 24, 263, 104, 521, 852, 67, 952, 243, 980, 441, 381, 677, 249, 423, 979, 453, 240, 167, 966, 532, 385, 201, 945, 287, 804, 998, 725, 699]
        # only_test_ids = [231, 400, 45, 225, 679, 223, 94, 133, 459, 498, 684, 120, 939, 263, 104, 852, 67, 952, 243, 441, 381, 677, 249, 423, 979, 167, 532, 385, 201, 945, 287, 804, 998, 725]
        # claude 4.5
        # only_test_ids = [342, 929, 543, 231, 721, 61, 807, 45, 651, 757, 225, 679, 94, 133, 898, 175, 610, 498, 759, 684, 846, 939, 132, 24, 370, 353, 104, 521, 852, 952, 298, 980, 441, 110, 423, 453, 781, 299, 167, 385, 665, 201, 945, 287, 804, 725, 86]
        # gpt-4o
        # only_test_ids = [342, 929, 231, 721, 45, 651, 225, 831, 94, 133, 276, 459, 899, 131, 498, 976, 759, 684, 846, 132, 24, 370, 263, 104, 177, 521, 852, 952, 243, 980, 441, 110, 677, 423, 979, 781, 167, 966, 385, 201, 945, 725, 86, 185]

        # 1125
        # claude 4.5
        # only_test_ids = [342, 929, 231, 721, 61, 69, 220, 45, 757, 225, 576, 860, 133, 236, 175, 810, 131, 498, 684, 503, 574, 370, 890, 353, 104, 852, 952, 980, 441, 110, 423, 781, 872, 167, 385, 665, 201, 945, 784, 804, 725]
        # gpt-4o
        # only_test_ids = [929, 231, 721, 45, 225, 831, 94, 133, 459, 131, 684, 132, 24, 370, 263, 852, 67, 952, 243, 980, 441, 677, 423, 979, 781, 385, 201, 945]
        # o3-mini
        # only_test_ids = [231, 721, 69, 220, 45, 225, 822, 679, 860, 223, 428, 94, 133, 236, 131, 498, 120, 503, 574, 132, 890, 512, 263, 67, 952, 243, 980, 441, 381, 249, 423, 979, 453, 872, 167, 532, 893, 945, 784, 725, 699]
        # only_test_ids = [679, 503, 784, 699]
        # only_test_ids = [503, 699]
        # only_test_ids = [503]
        # only_test_ids = [7, 243, 8, 136, 227, 130, 190, 148, 58, 146, 120, 277, 84, 27, 226, 57, 77, 127, 22, 169, 137, 156, 110, 143, 20, 18, 295, 261, 252, 88, 119, 108, 239, 93, 282, 257, 238, 162, 45, 241, 271, 265, 80, 297, 142, 268, 118, 28, 10, 213, 1, 51, 212, 244, 71, 133, 197, 4, 170, 101, 231, 0, 34, 275, 187, 6, 181, 48, 249, 66, 147, 68, 19, 105, 175, 274, 106, 290, 294, 199, 184, 298, 115, 160, 109, 234, 154, 283, 17, 222, 31, 23, 54, 35, 167, 285, 40, 248, 50, 161, 200, 215, 288, 113, 9, 139, 122, 164, 168, 246, 38, 230, 26, 99, 11, 53, 195, 245, 264, 49, 258, 29, 240, 3, 116, 260, 279, 145, 186, 224, 55, 254, 286, 79, 291, 152, 12, 251, 30, 178, 287, 70, 269, 250, 42, 262, 211, 173, 14, 37, 65, 149, 126, 205, 131, 196, 223, 242, 180, 210, 216, 61, 255, 111, 166, 76, 25, 208, 15, 194, 102, 123, 266, 218, 138, 87, 174, 135, 86, 151, 176, 299, 182, 129, 237, 256, 281, 183, 267, 155, 292, 247, 193, 270, 159, 64, 198, 229, 121, 13, 85, 98, 83, 75, 202, 2, 92, 63, 82, 192, 141, 236, 206, 107, 209, 189, 177, 96, 72, 91, 284, 158, 117, 171, 104, 24, 134, 233, 41, 62, 272, 56, 259, 221, 179, 201, 89, 232, 235, 74]

        only_test_ids = [4]

        # ========== Test qtree_enhanced_retrieval.py ==========
        # Set this to False to skip effectiveness checking during testing
        
        # Test whether only one suggestion is effective
        if test_mode:
            if only_test_ids == [78]:
                suggestions_effectiveness_csv_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-28T00-36-14.172021_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtree_rag_test_summary.csv'
            elif only_test_ids == [2]:
                suggestions_effectiveness_csv_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-28T00-36-49.177945_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtree_rag_test_summary.csv'
            elif only_test_ids == [31]:
                suggestions_effectiveness_csv_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-28T00-38-46.705187_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtree_rag_test_summary.csv'
            elif only_test_ids == [3]:
                suggestions_effectiveness_csv_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-28T16-10-14.095483_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtree_rag_test_summary.csv'
            elif only_test_ids == [30]:
                suggestions_effectiveness_csv_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-27T22-32-13.427873_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtree_rag_test_summary.csv'
            elif only_test_ids == [27]:
                suggestions_effectiveness_csv_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-10-01T14-19-55.390500_pdx-container-xterm-064.prd.it.nvidia.com_liwan/qtree_rag_test_summary.csv'
            else:
                suggestions_effectiveness_csv_path = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-29T11-02-53.963487_pdx-container-xterm-064.prd.it.nvidia.com_liwan copy/qtree_rag_test_summary.csv'

        # only_test_ids = [0,1,4,6]
        # only_test_ids = [50, 22, 77, 58, 18]
        # if only_test_ids == [4]:
        #     goldmine_testbench_path = "/home/scratch.liwan_mobile/repo/fv/baseline/GOLDMINE/design2sva_assertionbench_rxLinkFaultState/design2sva_assertionbench_rxLinkFaultState.sva"
        #     num_assertions = 5
        # elif only_test_ids == [2]:
        #     num_assertions = 11
        # elif only_test_ids == [5]:
        #     goldmine_testbench_path = "/home/scratch.liwan_mobile/repo/fv/baseline/HARM/eth_transmitcontrol.sva"
        #     num_assertions = 2
        # elif only_test_ids == [8]:
        #     goldmine_testbench_path = "/home/scratch.liwan_mobile/repo/fv/baseline/HARM/eth_cop.sva"
        #     num_assertions = 1
        # elif only_test_ids == [9]:
        #     goldmine_testbench_path = "/home/scratch.liwan_mobile/repo/fv/baseline/HARM/eth_macstatus.sva"
        #     num_assertions = 2
        # elif only_test_ids == [11]:
        #     goldmine_testbench_path = "/home/scratch.liwan_mobile/repo/fv/baseline/HARM/ge_1000baseX_syc.sva"
        #     num_assertions = 2
        # elif only_test_ids == [14]:
        #     goldmine_testbench_path = "/home/scratch.liwan_mobile/repo/fv/baseline/HARM/control_unit.sva"
        #     num_assertions = 4
        # elif only_test_ids == [17]:
        #     goldmine_testbench_path = "/home/scratch.liwan_mobile/repo/fv/baseline/HARM/fpu_exceptions.sva"
        #     num_assertions = 1
        # elif only_test_ids == [20]:
        #     num_assertions = 2
        # elif only_test_ids == [90]:
        #     num_assertions = 5
        # Pipeline: [69]
        # num_assertions = 5
    else:
        random_sample_size = -1
        cleanup_temp_files = False

elif global_task == 'eval':
    # folder_to_eval = "/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/inference_2024-09-01T08-35-15.428370_pdx-xterm-62_liwan"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-09-02T14-22-07.612319_pdx-container-xterm-064.prd.it.nvidia.com_liwan"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-10-21T09-30-25.852325_pdx-container-xterm-064.prd.it.nvidia.com_liwan"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-10-27T12-47-47.262230_pdx-container-xterm-062.prd.it.nvidia.com_liwan"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-10-27T12-47-47.262230_pdx-container-xterm-062.prd.it.nvidia.com_liwan"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-10-28T15-40-01.303445_pdx-container-xterm-062.prd.it.nvidia.com_liwan"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/test/update_machine_result/machine"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/test/update_opencore_result/o3-mini"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/test/update_opencore_result/claude-4-5"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/test/update_opencore_result/gpt-4o"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/inference_2025-11-05T21-31-32.608143_pdx-container-xterm-062.prd.it.nvidia.com_liwan"


    # Evaluate Cursor-generated results
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251019_195928" # Use Rules
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251019_203323" # No Rules
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251019_204827" # Use Rules
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251019_205532"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251020_075311"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251020_075332"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251020_080739"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv_Cursor_rules/src/results_nl2sva_human/20251020_080747"

    # 1126
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/test/update_opencore_v3_result/claude-4-5"
    folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/test/update_opencore_v3_result/gpt-4o"
    # folder_to_eval = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/test/update_opencore_v3_result/o3-mini"

else:
    raise NotImplementedError()

if global_task == 'inference' or global_task == 'eval' or global_task == 'train':
    nparallel = 1
    if debug == True:
        nparallel = 1
    if global_task == 'eval':
        eval_content = ["similarity","jg"]
        baseline_finetune = False
        # SVA specification
    if task == "nl2sva_machine":
        tcl_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_only_analysis.tcl")
        tcl_eval_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_nl2sva_machine.tcl")
    elif task == "nl2sva_human":
        tcl_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_only_analysis.tcl")
        tcl_eval_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_nl2sva_human.tcl")
    elif task == "nl2sva_opencore":
        tcl_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_only_syntax_opencore.tcl")
        tcl_eval_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_nl2sva_opencore.tcl")
    elif task in ["design2sva_fsm", "design2sva_pipeline", "design2sva_training_docs"]:
        tcl_coverage_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva.tcl")
        tcl_check_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva_error.tcl")
        tcl_report_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva_report.tcl")
        tcl_history_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva_report.tcl")
    elif task in ["design2sva_assertionbench"]:
        tcl_coverage_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva_assertionbench.tcl")
        tcl_check_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva_error_assertionbench.tcl")
        tcl_report_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva_report_assertionbench.tcl")
        tcl_history_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_design2sva_report_assertionbench.tcl")

user = get_user()
hostname = get_host()

# from fv_tools import evaluate_jg

def load_perflab_config(llm_model, script_dir=None):
    """
    Centralized function to load Perflab configuration based on the LLM model.
    
    Args:
        llm_model: The LLM model name
        script_dir: The directory to look for config files. If None, uses the directory of this file
        
    Returns:
        config_list: The loaded configuration list
    """
    # Use provided directory or default to this file's directory
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine config path based on model
    if llm_model == 'gpt-4o-20241120':
        config_path = os.path.join(script_dir, "OAI_CONFIG/OAI_CONFIG_LIST_Perflab")
        print("Loaded configuration for gpt-4o-20241120")
    elif llm_model == 'claude-3-5-sonnet-20240620':
        config_path = os.path.join(script_dir, "OAI_CONFIG/OAI_CONFIG_LIST_Perflab_Claude35")
        print("Loaded configuration for claude-3-5-sonnet-20240620")
    elif llm_model == 'o1-20241217':
        config_path = os.path.join(script_dir, "OAI_CONFIG/OAI_CONFIG_LIST_O1")
        print("Loaded configuration for o1-20241217")
    elif llm_model == 'o3-mini-20250131':
        config_path = os.path.join(script_dir, "OAI_CONFIG/OAI_CONFIG_LIST_O3mini")
        print("Loaded configuration for o3-mini-20250131")
    elif llm_model == 'claude-sonnet-4-5-20250929':
        config_path = os.path.join(script_dir, "OAI_CONFIG/OAI_CONFIG_LIST_Perflab_Claude45")
        print("Loaded configuration for claude-sonnet-4-5-20250929")
    elif llm_model == 'gpt-4o-20241120':
        config_path = os.path.join(script_dir, "OAI_CONFIG/OAI_CONFIG_LIST_4o")
        print("Loaded configuration for gpt-4o-20241120")
    else:
        config_path = os.path.join(script_dir, "OAI_CONFIG/OAI_CONFIG_LIST_Perflab")
        print(f"Loaded default configuration for model: {llm_model}")
    
    # Load and return the configuration
    config_list = config_list_from_json(env_or_file=config_path)
    return config_list


###################################### Below: no need to touch ######################################

try:
    import git
except Exception as e:
    raise type(e)(f'{e}\nRun pip install gitpython or\nconda install gitpython')
try:
    repo = git.Repo(ROOT)
    repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-1]
    local_branch_name = repo.active_branch.name
    commit_sha = repo.head.object.hexsha
except git.exc.InvalidGitRepositoryError as e:
    raise Exception(f"Invalid Git repository at {ROOT}") from e

proj_dir = ROOT

vars = OrderedDict(vars())
# vars['dataset_path'] = dataset_path  # Ensure dataset_path is included in vars
FLAGS = OrderedDict()
for k, v in vars.items():
    if not k.startswith('__') and type(v) in [
        int,
        float,
        str,
        list,
        dict,
        type(None),
        bool,
    ]:
        FLAGS[k] = v
FLAGS = SimpleNamespace(**FLAGS)
