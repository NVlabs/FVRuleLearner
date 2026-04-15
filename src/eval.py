# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
