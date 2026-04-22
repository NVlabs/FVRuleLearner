########################################################
# NVIDIA License
#
# 1. Definitions
#
# “Licensor” means any person or entity that distributes its Work.
# “Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
# The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
# Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.
#
# 2. License Grant
#
# 2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.
#
# 3. Limitations
#
# 3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.
#
# 3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.
#
# 3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.
#
# 3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.
#
# 3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.
#
# 3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.
#
# 4. Disclaimer of Warranty.
#
# THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 
#
# 5. Limitation of Liability.
#
# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
########################################################
from FVEval.run_evaluation import main_eval

from config import FLAGS
from saver import saver

print = saver.log_info

import argparse
import pandas as pd
import os
import glob
import json


def analyze_suggestion_usage_stats(jg_df, folder_to_eval=None):
    """
    Analyze suggestion usage statistics using saver.stats.
    
    Args:
        jg_df: DataFrame with JasperGold results
        folder_to_eval: Directory to save results (optional)
    """
    if folder_to_eval is None:
        folder_to_eval = saver.logdir
    # Get suggestion usage data from saver.stats
    if 'used_suggestions' not in saver.stats or len(saver.stats['used_suggestions']) == 0:
        print("\n⚠️  No suggestion usage data found in saver.stats")
        print("   Run with FLAGS.use_RAG=True and 'Suggestions' in FLAGS.RAG_content to track suggestions")
        return
    
    used_sugg_list = saver.stats['used_suggestions']
    
    # Ensure we have matching number of results
    if len(used_sugg_list) != len(jg_df):
        print(f"\n⚠️  Warning: Suggestion usage data ({len(used_sugg_list)}) doesn't match JG results ({len(jg_df)})")
        # Use minimum length
        min_len = min(len(used_sugg_list), len(jg_df))
        used_sugg_list = used_sugg_list[:min_len]
        jg_df = jg_df.iloc[:min_len]
    
    # Analyze by suggestion usage
    with_sugg_idx = [i for i, used in enumerate(used_sugg_list) if used == 1]
    without_sugg_idx = [i for i, used in enumerate(used_sugg_list) if used == 0]
    
    # Calculate statistics
    stats = {
        'total': len(jg_df),
        'with_suggestions': {
            'count': len(with_sugg_idx),
            'solved': 0,
            'avg_functionality': 0,
            'avg_relax_func': 0,
            'avg_bleu': 0
        },
        'without_suggestions': {
            'count': len(without_sugg_idx),
            'solved': 0,
            'avg_functionality': 0,
            'avg_relax_func': 0,
            'avg_bleu': 0
        }
    }
    
    if len(with_sugg_idx) > 0:
        with_df = jg_df.iloc[with_sugg_idx]
        stats['with_suggestions']['solved'] = int((with_df['functionality'] >= 1.0).sum())
        stats['with_suggestions']['avg_functionality'] = float(with_df['functionality'].mean())
        stats['with_suggestions']['avg_relax_func'] = float(with_df['func_relaxed'].mean())
        stats['with_suggestions']['solve_rate'] = stats['with_suggestions']['solved'] / len(with_sugg_idx)
    
    if len(without_sugg_idx) > 0:
        without_df = jg_df.iloc[without_sugg_idx]
        stats['without_suggestions']['solved'] = int((without_df['functionality'] >= 1.0).sum())
        stats['without_suggestions']['avg_functionality'] = float(without_df['functionality'].mean())
        stats['without_suggestions']['avg_relax_func'] = float(without_df['func_relaxed'].mean())
        stats['without_suggestions']['solve_rate'] = stats['without_suggestions']['solved'] / len(without_sugg_idx)
    
    # Print results
    print("\n" + "="*80)
    print("SUGGESTION USAGE ANALYSIS")
    print("="*80)
    print(f"\nTotal test cases: {stats['total']}")
    print(f"\nCases WITH suggestions: {stats['with_suggestions']['count']}")
    print(f"  - Solved (PEC>=1): {stats['with_suggestions']['solved']}")
    if stats['with_suggestions']['count'] > 0:
        print(f"  - Solve rate: {stats['with_suggestions']['solve_rate']*100:.2f}%")
        print(f"  - Avg functionality: {stats['with_suggestions']['avg_functionality']:.4f}")
        print(f"  - Avg relaxed func: {stats['with_suggestions']['avg_relax_func']:.4f}")
    
    print(f"\nCases WITHOUT suggestions: {stats['without_suggestions']['count']}")
    print(f"  - Solved (PEC>=1): {stats['without_suggestions']['solved']}")
    if stats['without_suggestions']['count'] > 0:
        print(f"  - Solve rate: {stats['without_suggestions']['solve_rate']*100:.2f}%")
        print(f"  - Avg functionality: {stats['without_suggestions']['avg_functionality']:.4f}")
        print(f"  - Avg relaxed func: {stats['without_suggestions']['avg_relax_func']:.4f}")
    
    # Comparison
    if stats['with_suggestions']['count'] > 0 and stats['without_suggestions']['count'] > 0:
        improvement = (stats['with_suggestions']['solve_rate'] - stats['without_suggestions']['solve_rate']) * 100
        print(f"\n{'-'*80}")
        print(f"Improvement with suggestions: {improvement:+.2f} percentage points")
        print(f"{'-'*80}")
    
    # Save to JSON
    output_file = os.path.join(folder_to_eval, "eval", "suggestion_usage_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved analysis to: {output_file}")
    print("="*80 + "\n")
    
    return stats


def eval(folder_to_eval):
    if FLAGS.task == "nl2sva_machine" or FLAGS.task == "nl2sva_human" or FLAGS.task == "nl2sva_opencore":
        parser = argparse.ArgumentParser(description="Run JG-based Evaluation of FVEval-SVAGen Results")
        parser.add_argument(
            "--llm_output_dir",
            "-i",
            type=str,
            help="path to LLM results dir",
            default=folder_to_eval,
        )
        parser.add_argument(
            "--model_name",
            "-m",
            type=str,
            help="specific model name to evaluate for",
            default="",
        )
        parser.add_argument(
            "--save_dir",
            "-o",
            type=str,
            help="path to save JG eval results",
        )
        parser.add_argument(
            "--temp_dir",
            "-t",
            type=str,
            help="path to temp dir",
        )
        parser.add_argument(
            "--cleanup_temp",
            type=bool,
            help="Whether to clean up the temp dir afterwards",
            default=True,
        )
        parser.add_argument(
            "--task",
            type=str,
            help="task you are evaluating for",
            default=FLAGS.task,
        )
        parser.add_argument(
            "--nparallel",
            "-n",
            type=int,
            help="parallel JG jobs",
            default=FLAGS.nparallel,
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="debug ",
            default=FLAGS.debug,
        )

    parser.add_argument("--src_examples", type=str, required=False)
    parser.add_argument("--llm_model", type=str, required=False)
    parser.add_argument("--group_id", type=str, required=False)
    parser.add_argument("--logdir", type=str, required=False)

    args = parser.parse_args()
    main_eval(args)
    if FLAGS.global_task == 'eval':
        postprocess_csv_files(FLAGS.folder_to_eval)
    elif FLAGS.global_task == 'inference':
        postprocess_csv_files(saver.logdir)


def postprocess_csv_files(folder_to_eval):
    if not os.path.isdir(folder_to_eval):
        raise FileNotFoundError(
            f"Eval folder does not exist: {folder_to_eval}. "
            "Set 'folder_to_eval' in src/config.py to a real inference log directory, "
            "or export FVRULELEARNER_EVAL_LOGDIR to override it."
        )
    if "nl2sva" in FLAGS.task:
        jg_file = os.path.join(folder_to_eval, "eval", "*_jg.csv")
        sim_file = os.path.join(folder_to_eval, "eval", "*_sim.csv")

        jg_df = pd.concat([pd.read_csv(f) for f in glob.glob(jg_file)], ignore_index=True)
        sim_df = pd.concat([pd.read_csv(f) for f in glob.glob(sim_file)], ignore_index=True)

        jg_metrics = ["syntax", "functionality", "func_relaxed"]
        sim_metrics = ["bleu", "rouge", "exact_match"]

        results = {"mean": {}, "variance": {}}

        for metric in jg_metrics:
            results["mean"][metric] = jg_df[metric].mean()
            results["variance"][metric] = jg_df[metric].var()

        for metric in sim_metrics:
            results["mean"][metric] = sim_df[metric].mean()
            results["variance"][metric] = sim_df[metric].var()

        # Preparing data for CSV format
        csv_header = "\t".join(sim_metrics + jg_metrics)
        csv_mean = "\t".join([f"{results['mean'][metric]:.6f}" for metric in sim_metrics + jg_metrics])
        csv_variance = "\t".join([f"{results['variance'][metric]:.6f}" for metric in sim_metrics + jg_metrics])

        print("Postprocessing Results:")
        print(csv_header)
        print(csv_mean)
        print(csv_variance)
        
        # Analyze suggestion usage if available
        analyze_suggestion_usage_stats(jg_df, folder_to_eval)