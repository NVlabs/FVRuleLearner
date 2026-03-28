import pickle
import json

# Define the path to your pickle file
# pkl_filename = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-03T13-10-05.086890_pdx-xterm-153_liwan/suggestions.pkl'
# pkl_filename  = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-17T07-20-54.564842_pdx-xterm-153_liwan/suggestions.pkl'
# pkl_filename = f'/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/backup_logs/train_2024-08-03T20-45-18.088918_pdx-xterm-56_liwan/suggestions.pkl'
# pkl_filename = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-11T09-01-03.562227_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions.pkl'
# pkl_filename = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-20T00-01-20.119404_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions.pkl'
pkl_filename = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-28T09-42-04.137848_pdx-container-xterm-062.prd.it.nvidia.com_liwan/suggestions.pkl'

# Load the pickle file
with open(pkl_filename, "rb") as pkl_file:
    data = pickle.load(pkl_file)

# Print the content of the pickle file
print(data)
print(f"data length: {len(data)}")
if len(data) >= 1:
    print(f"data[0]: {data[0]}")

# Convert the data to a JSON string for better readability
readable_data = json.dumps(data, indent=4)

# Define the path to the output text file
# output_filename = '/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/readable_suggestions.txt'
# output_filename = '/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/logs/train_2024-08-03T13-10-05.086890_pdx-xterm-153_liwan/readable_suggestions.txt'
# output_filename = '/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/backup_logs/train_2024-08-03T20-45-18.088918_pdx-xterm-56_liwan/suggestions.txt'
# output_filename = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-11T09-01-03.562227_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions.txt'
output_filename = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-28T09-42-04.137848_pdx-container-xterm-062.prd.it.nvidia.com_liwan/suggestions.txt'

# Write the readable data to the text file
with open(output_filename, "w") as output_file:
    output_file.write(readable_data)

print(f"Readable data has been written to {output_filename}")
