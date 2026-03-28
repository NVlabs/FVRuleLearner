#!/usr/bin/env python3
"""
Script to merge qtrees.json, qtrees.pkl, and suggestions.pkl files from multiple training log folders.

Usage:
    python merge_training_logs.py folder1 folder2 folder3 [output_folder]
    
Example:
    python merge_training_logs.py \
        /home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-22T13-27-39.296218_pdx-container-xterm-102.prd.it.nvidia.com_liwan \
        /home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-23T11-29-11.120481_pdx-container-xterm-062.prd.it.nvidia.com_liwan \
        /home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-25T23-22-41.508612_pdx-container-xterm-062.prd.it.nvidia.com_liwan \
        /home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-10-27T00-58-28.704284_pdx-container-xterm-062.prd.it.nvidia.com_liwan
"""

import json
import pickle
import sys
import os
from pathlib import Path
from typing import List, Any


def load_json_file(filepath: Path) -> List[Any]:
    """Load a JSON file and return its contents."""
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist, skipping...")
        return []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    else:
        return [data]  # Wrap single items in a list


def load_pickle_file(filepath: Path) -> Any:
    """Load a pickle file and return its contents."""
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist, skipping...")
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def merge_qtrees_json(folders: List[Path]) -> List[Any]:
    """Merge qtrees.json files from multiple folders."""
    print("\n=== Merging qtrees.json files ===")
    merged_data = []
    
    for folder in folders:
        json_file = folder / "qtrees.json"
        print(f"Loading {json_file}...")
        data = load_json_file(json_file)
        print(f"  Found {len(data)} items")
        merged_data.extend(data)
    
    print(f"Total merged items: {len(merged_data)}")
    return merged_data


def merge_pickle_files(folders: List[Path], filename: str) -> Any:
    """Merge pickle files from multiple folders."""
    print(f"\n=== Merging {filename} files ===")
    merged_data = []
    
    for folder in folders:
        pkl_file = folder / filename
        print(f"Loading {pkl_file}...")
        data = load_pickle_file(pkl_file)
        
        if data is None:
            continue
        
        # Handle different data types
        if isinstance(data, list):
            print(f"  Found {len(data)} items (list)")
            merged_data.extend(data)
        elif isinstance(data, dict):
            print(f"  Found {len(data)} items (dict)")
            # For dictionaries, we'll need to merge them
            if not merged_data:
                merged_data = data
            else:
                # Merge dictionaries - you may need to customize this based on your data structure
                if isinstance(merged_data, dict):
                    merged_data.update(data)
                else:
                    merged_data.append(data)
        else:
            print(f"  Found data of type {type(data)}")
            merged_data.append(data)
    
    print(f"Total merged: {len(merged_data) if isinstance(merged_data, (list, dict)) else 'N/A'}")
    return merged_data


def save_merged_files(output_folder: Path, qtrees_json_data: List[Any], 
                      qtrees_pkl_data: Any, suggestions_pkl_data: Any):
    """Save merged files to the output folder."""
    print(f"\n=== Saving merged files to {output_folder} ===")
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Save qtrees.json
    json_output = output_folder / "qtrees.json"
    print(f"Saving {json_output}...")
    with open(json_output, 'w') as f:
        json.dump(qtrees_json_data, f, indent=2)
    print(f"  Saved {len(qtrees_json_data)} items")
    
    # Save qtrees.pkl
    pkl_output = output_folder / "qtrees.pkl"
    print(f"Saving {pkl_output}...")
    with open(pkl_output, 'wb') as f:
        pickle.dump(qtrees_pkl_data, f)
    print(f"  Saved")
    
    # Save suggestions.pkl
    suggestions_output = output_folder / "suggestions.pkl"
    print(f"Saving {suggestions_output}...")
    with open(suggestions_output, 'wb') as f:
        pickle.dump(suggestions_pkl_data, f)
    print(f"  Saved")
    
    print("\n=== Merge complete! ===")


def main():
    # Hardcoded paths as requested
    base_path = Path("/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs")
    
    folder_paths = [
        Path("/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-20T16-01-13.268612_pdx-container-xterm-062.prd.it.nvidia.com_liwan"),
        Path("/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-21T11-51-37.659462_pdx-container-xterm-062.prd.it.nvidia.com_liwan"),
        Path("/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-22T20-30-36.821711_pdx-container-xterm-062.prd.it.nvidia.com_liwan"),
        Path("/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-23T07-33-26.354352_pdx-container-xterm-062.prd.it.nvidia.com_liwan")
    ]
    
    output_folder = base_path / "train_2025-11-23T14-42-26.354352_pdx-container-xterm-062.prd.it.nvidia.com_liwan"
    
    # Convert to Path objects (already done above, but keeping variable name consistent)
    folders = folder_paths
    output = output_folder
    
    # Verify folders exist
    print("=== Checking input folders ===")
    for folder in folders:
        if not folder.exists():
            print(f"Error: Folder {folder} does not exist!")
            sys.exit(1)
        print(f"✓ {folder}")
    
    # Merge qtrees.json
    qtrees_json_merged = merge_qtrees_json(folders)
    
    # Merge qtrees.pkl
    qtrees_pkl_merged = merge_pickle_files(folders, "qtrees.pkl")
    
    # Merge suggestions.pkl
    suggestions_pkl_merged = merge_pickle_files(folders, "suggestions.pkl")
    
    # Save merged files
    save_merged_files(output, qtrees_json_merged, qtrees_pkl_merged, suggestions_pkl_merged)
    
    print(f"\nAll files have been merged and saved to: {output}")


if __name__ == "__main__":
    main()

