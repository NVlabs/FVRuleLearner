import pickle
import os

# Base path for the pickle files
base_path = '/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs'

# Lists of pickle file names
human_files = [
    "train_2024-07-25T23-58-09.727563_pdx-xterm-153_liwan",
    "train_2024-07-25T23-58-59.568975_pdx-xterm-153_liwan",
    "train_2024-07-27T11-00-10.084793_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_0_2024-07-24T18-17-28.493424_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_1_2024-07-24T20-10-24.941423_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_2_2024-07-24T20-10-25.451431_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_3_2024-07-24T21-37-46.518428_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_4_2024-07-24T21-37-47.030203_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_5_2024-07-24T21-37-47.541194_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_6_2024-07-24T22-47-54.887847_pdx-xterm-153_liwan",
    "nl2sva_human_gpt-4_7_2024-07-24T22-47-55.399009_pdx-xterm-153_liwan",
    "train_2024-07-26T00-32-42.887270_pdx-xterm-57_liwan",
    "train_2024-07-24T18-06-46.197448_pdx-xterm-62_liwan",
    "train_2024-07-26T00-36-32.772372_pdx-xterm-57_liwan"
]

machine_files = [
    "train_2024-07-26T13-25-11.933011_pdx-xterm-62_liwan",
    "train_2024-07-27T00-45-16.749433_pdx-xterm-153_liwan",
    "train_2024-07-27T00-46-02.241753_pdx-xterm-153_liwan",
    "train_2024-07-27T00-41-36.469831_pdx-xterm-153_liwan",
    "train_2024-07-27T00-40-49.038595_pdx-xterm-153_liwan",
    "train_2024-07-24T20-24-07.141673_pdx-xterm-62_liwan",
    "train_2024-07-26T09-52-11.060060_pdx-xterm-153_liwan",
    "train_2024-07-24T21-49-12.644918_pdx-xterm-57_liwan",
    "train_2024-07-26T09-58-31.469401_pdx-xterm-62_liwan"
]

# Function to load pickle files and merge the data
def load_and_merge_pickle_files(file_list, base_path):
    merged_data = []
    for file_name in file_list:
        file_name = os.path.join(file_name, 'suggestions.pkl')
        file_path = os.path.join(base_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                merged_data.append(data)
        else:
            print(f"Skipping {file_path}: Not a valid file.")
            if os.path.isdir(file_path):
                print(f"{file_path} is a directory.")
            else:
                print(f"{file_path} does not exist.")
    return merged_data

# Load and merge data
human_data = load_and_merge_pickle_files(human_files, base_path)
machine_data = load_and_merge_pickle_files(machine_files, base_path)
mixed_data = human_data + machine_data

# Path to save the merged pickle files
save_path = '/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/suggestions_pkl'
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

# Save merged data into separate pickle files
with open(os.path.join(save_path, 'human_only.pkl'), 'wb') as f:
    pickle.dump(human_data, f)

with open(os.path.join(save_path, 'machine_only.pkl'), 'wb') as f:
    pickle.dump(machine_data, f)

with open(os.path.join(save_path, 'human_machine_mixed.pkl'), 'wb') as f:
    pickle.dump(mixed_data, f)

print("Pickle files have been merged and saved successfully to suggestions.pkl.")
