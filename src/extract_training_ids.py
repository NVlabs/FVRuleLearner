#!/usr/bin/env python3
"""
Quick script to extract training IDs for nl2sva_machine task.
Run this to get the training IDs that should be hardcoded.
"""

import random
import pandas as pd
from config import FLAGS

def extract_training_ids():
    """Extract training IDs using the same logic as benchmark_launcher"""
    
    # Load dataset
    dataset_path = FLAGS.dataset_path
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    total_samples = len(df)
    print(f"Total samples: {total_samples}")
    
    # Use the same random seed and split ratios
    random.seed(FLAGS.random_seed)
    shuffled_indices = list(range(total_samples))
    random.shuffle(shuffled_indices)
    
    # Split the dataset
    test_size = int(total_samples * FLAGS.split_ratios['test'])
    train_size = int(total_samples * FLAGS.split_ratios['train'])
    
    print(f"Test size: {test_size}")
    print(f"Train size: {train_size}")
    
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:(test_size + train_size+1)]
    
    print(f"\n{'='*80}")
    print(f"TRAINING IDs for {FLAGS.task} (total {len(train_indices)}):")
    print(f"{'='*80}")
    print(train_indices)
    
    print(f"\n{'='*80}")
    print(f"TEST IDs for {FLAGS.task} (total {len(test_indices)}):")
    print(f"{'='*80}")
    print(test_indices)
    
    print(f"\n{'='*80}")
    print("COPY THIS LINE FOR HARDCODING:")
    print(f"{'='*80}")
    print(f"training_IDs = {train_indices}")
    
    return train_indices, test_indices

if __name__ == "__main__":
    print(f"\nTask: {FLAGS.task}")
    print(f"Random seed: {FLAGS.random_seed}")
    print(f"Split ratios: {FLAGS.split_ratios}")
    print()
    
    train_ids, test_ids = extract_training_ids()
    
    print(f"\n✅ Done! Found {len(train_ids)} training IDs and {len(test_ids)} test IDs")

