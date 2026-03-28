#!/usr/bin/env python3
"""
Extract suggestions with BLEU:1.0 or Functionality:1.0 from suggestions.txt files.
"""

import json
import os
import sys
from pathlib import Path


def extract_high_score_suggestions(input_file_path):
    """
    Extract rows containing either BLEU:1.0 or Functionality:1.0.
    
    Args:
        input_file_path: Path to the suggestions.txt file
    
    Returns:
        List of filtered suggestions
    """
    try:
        # Read the file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        breakpoint()
        
        # Parse JSON array
        data = json.loads(content)
        
        # Filter rows
        filtered_rows = []
        for row in data:
            if isinstance(row, str):
                # Check if the row contains either BLEU:1.0 or Functionality:1.0
                if "<---->BLEU:1.0" in row or "<---->Functionality:1.0" in row:
                    filtered_rows.append(row)
        
        return filtered_rows
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {input_file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")
        return []


def save_filtered_suggestions(filtered_rows, output_file_path):
    """
    Save filtered suggestions to a file in JSON format.
    
    Args:
        filtered_rows: List of filtered suggestion strings
        output_file_path: Path to save the filtered suggestions
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_rows, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(filtered_rows)} filtered suggestions to {output_file_path}")
    except Exception as e:
        print(f"Error saving to {output_file_path}: {e}")


def process_suggestions_file(file_path):
    """
    Process a single suggestions.txt file and save filtered results.
    
    Args:
        file_path: Path to the suggestions.txt file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"Processing: {file_path}")
    
    # Extract high score suggestions
    filtered_rows = extract_high_score_suggestions(file_path)
    
    if filtered_rows:
        # Create output filename in the same directory
        output_filename = file_path.stem + "_high_scores.txt"
        output_path = file_path.parent / output_filename
        
        # Save filtered results
        save_filtered_suggestions(filtered_rows, output_path)
        
        # Also print some statistics
        print(f"  Total rows in original file: Unknown (would need to count)")
        print(f"  Rows with high scores: {len(filtered_rows)}")
        
        # Count BLEU:1.0 and Functionality:1.0 separately
        bleu_count = sum(1 for row in filtered_rows if "<---->BLEU:1.0" in row)
        func_count = sum(1 for row in filtered_rows if "<---->Functionality:1.0" in row)
        both_count = sum(1 for row in filtered_rows if "<---->BLEU:1.0" in row and "<---->Functionality:1.0" in row)
        
        print(f"  - BLEU:1.0 count: {bleu_count}")
        print(f"  - Functionality:1.0 count: {func_count}")
        print(f"  - Both BLEU:1.0 and Functionality:1.0: {both_count}")
    else:
        print(f"  No rows found with BLEU:1.0 or Functionality:1.0")


def main():
    """Main function to handle command line arguments."""
    # Hardcoded file path for now
    # file_path = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-11T09-01-03.562227_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions.txt"
    file_path = "/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-20T00-01-20.119404_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions.txt"

    # Process the hardcoded file
    process_suggestions_file(file_path)


if __name__ == "__main__":
    main()
