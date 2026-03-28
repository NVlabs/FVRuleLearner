# import pickle
# import os

# # Paths to the pickle files
# pkl_files = [
#     "train_2024-08-13T21-31-26.756243_pdx-xterm-59_liwan",
#     "train_2024-08-13T21-33-04.032951_pdx-xterm-59_liwan",
#     "train_2024-08-13T21-33-57.090596_pdx-xterm-153_liwan",
#     "train_2024-08-13T21-35-28.035756_pdx-xterm-58_liwan",
#     "train_2024-08-13T21-36-11.104206_pdx-xterm-58_liwan",
#     "train_2024-08-13T21-37-06.147350_pdx-xterm-58_liwan",
#     "train_2024-08-13T21-37-42.124014_pdx-xterm-58_liwan",
#     "train_2024-08-13T21-39-15.071609_pdx-xterm-57_liwan"
# ]

# # Base directory
# base_dir = "/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs"

# # Full paths to the pickle files
# pkl_paths = [os.path.join(base_dir, pkl_file, "suggestions.pkl") for pkl_file in pkl_files]

# # Directory to save the final merged pickle file
# output_file = os.path.join(base_dir, "train_2024-08-13T21-31-26.756243_pdx-xterm-59_liwan", "all_suggestions.pkl")

# # Initialize a list to store the merged data
# merged_data = []

# # Loop through each pickle file and load the data
# for file_path in pkl_paths:
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#         merged_data.extend(data)

# # Save the merged data to a new pickle file
# with open(output_file, 'wb') as f:
#     pickle.dump(merged_data, f)

# print(f"Merged pickle file saved to {output_file}")

import pickle
import os

# Paths to the pickle files
pkl_files = [
    # "train_2024-08-13T21-31-26.756243_pdx-xterm-59_liwan",
    # "train_2024-08-13T21-33-04.032951_pdx-xterm-59_liwan",
    # "train_2024-08-13T21-33-57.090596_pdx-xterm-153_liwan",
    # "train_2024-08-13T21-35-28.035756_pdx-xterm-58_liwan",
    # "train_2024-08-13T21-36-11.104206_pdx-xterm-58_liwan",
    # "train_2024-08-13T21-37-06.147350_pdx-xterm-58_liwan",
    # "train_2024-08-13T21-37-42.124014_pdx-xterm-58_liwan",
    # "train_2024-08-13T21-39-15.071609_pdx-xterm-57_liwan",
    # "train_2024-08-14T15-43-48.623120_pdx-xterm-153_liwan",
    # "train_2024-08-14T15-44-06.019635_pdx-xterm-153_liwan",
    # "train_2024-08-14T15-45-32.199344_pdx-xterm-153_liwan",
    # "train_2024-08-14T15-45-46.393850_pdx-xterm-153_liwan",
    # "train_2024-08-14T15-52-12.499243_pdx-xterm-58_liwan",
    # "train_2024-08-14T15-52-34.052330_pdx-xterm-58_liwan",
    # "train_2024-08-14T15-52-55.845960_pdx-xterm-58_liwan",
    # "train_2024-08-14T15-53-16.218859_pdx-xterm-58_liwan"
    "train_2024-08-14T17-12-43.034552_pdx-xterm-59_liwan",
    "train_2024-08-14T17-13-23.604542_pdx-xterm-59_liwan",
    "train_2024-08-14T17-13-45.400556_pdx-xterm-59_liwan",
    "train_2024-08-14T17-14-05.300025_pdx-xterm-59_liwan"
]

# Base directory
base_dir = "/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs"

# Full paths to the pickle files
pkl_paths = [os.path.join(base_dir, pkl_file, "suggestions.pkl") for pkl_file in pkl_files]

# Directory to save the final merged pickle file
output_file = os.path.join(base_dir, "train_2024-08-14T17-12-43.034552_pdx-xterm-59_liwan", "all_suggestions.pkl")

# Initialize a list to store the merged data
merged_data = []

# Loop through each pickle file, load the data, and validate
for file_path in pkl_paths:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            merged_data.extend(data)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
        print(f"Error loading {file_path}: {e}")
        continue

# Save the merged data to a new pickle file only if data was successfully loaded
if merged_data:
    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f)
    print(f"Merged pickle file saved to {output_file}")
else:
    print("No valid data to merge, the merged file was not created.")
