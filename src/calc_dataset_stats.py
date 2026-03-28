import csv
import os
import sys
import math

from utils_agent import get_tokenizer, count_tokens

def process_prompt_nl2sva(prompt):
    return f"Create a SVA assertion that checks: {prompt}"

def process_prompt_module_sva(prompt):
    return f"Create a SVA assertion that checks: {prompt}"

def process_sva_module_sva(sva):
    return f"assert property({sva});"

def calculate_stats(tokens):
    n = len(tokens)
    if n == 0:
        return {'min': 0, 'max': 0, 'avg': 0, 'std': 0, 'size': 0}
    
    min_val = min(tokens)
    max_val = max(tokens)
    avg = sum(tokens) / n
    
    # Calculate standard deviation
    variance = sum((x - avg) ** 2 for x in tokens) / n
    std = math.sqrt(variance)
    
    return {'min': min_val, 'max': max_val, 'avg': avg, 'std': std, 'size': n}

def process_csv(file_path, prompt_column, ref_column, tokenizer):
    nl_tokens = []
    sva_tokens = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "nl2sva" in file_path:
                prompt = process_prompt_nl2sva(row[prompt_column])
                sva = row[ref_column]
            else:  # module_sva_nl_manual_editing.csv
                prompt = process_prompt_module_sva(row[prompt_column])
                sva = process_sva_module_sva(row[ref_column])
            
            nl_tokens.append(count_tokens(tokenizer, prompt))
            sva_tokens.append(count_tokens(tokenizer, sva))
    
    nl_stats = calculate_stats(nl_tokens)
    sva_stats = calculate_stats(sva_tokens)
    
    return {'nl': nl_stats, 'sva': sva_stats}

def main():
    files = [
        ("/home/scratch.liwan_mobile/repo/fv/FVEval/data_nl2sva/data/nl2sva_human.csv", "prompt", "ref_solution"),
        # ("/home/scratch.liwan_mobile/repo/fv/FVEval/data_nl2sva/data/nl2sva_machine.csv", "prompt", "ref_solution"),
        ("/home/scratch.liwan_mobile/repo/fv/FVEval/data_nl2sva/data/nl2sva_machine_updated.csv", "prompt", "ref_solution"),
        ("/home/scratch.liwan_mobile/repo/fv/FVEval/data_1k/module_sva_nl_manual_editing.csv", "prompt", "ref_solution")
    ]

    tokenizer = get_tokenizer("gpt-3.5-turbo")

    for file_path, prompt_column, ref_column in files:
        if os.path.exists(file_path):
            print(f"Processing {os.path.basename(file_path)}:")
            stats = process_csv(file_path, prompt_column, ref_column, tokenizer)
            
            print(f"  Dataset size: {stats['nl']['size']} samples")
            
            print("  NL (prompt) statistics:")
            print(f"    Minimum tokens: {stats['nl']['min']}")
            print(f"    Maximum tokens: {stats['nl']['max']}")
            print(f"    Average tokens: {stats['nl']['avg']:.2f}")
            print(f"    Standard deviation: {stats['nl']['std']:.2f}")
            
            print("  SVA statistics:")
            print(f"    Minimum tokens: {stats['sva']['min']}")
            print(f"    Maximum tokens: {stats['sva']['max']}")
            print(f"    Average tokens: {stats['sva']['avg']:.2f}")
            print(f"    Standard deviation: {stats['sva']['std']:.2f}")
            
            print()
        else:
            print(f"File not found: {file_path}")
            print()

if __name__ == "__main__":
    main()