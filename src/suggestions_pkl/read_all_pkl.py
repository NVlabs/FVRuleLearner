import pickle
import json
import sys

# Define the path to your pickle file
pkl_filename = f'/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-28T09-42-04.137848_pdx-container-xterm-062.prd.it.nvidia.com_liwan/suggestions.pkl'

# Initialize a list to hold all data found in the file
all_data = []

print(f"Reading file: {pkl_filename} ...")

with open(pkl_filename, "rb") as pkl_file:
    # Loop continuously to read multiple pickle objects if they exist
    while True:
        try:
            # Try to load the next object
            obj = pickle.load(pkl_file)
            all_data.append(obj)
        except EOFError:
            # "End Of File" Reached - We are done reading
            break
        except pickle.UnpicklingError:
            # Catch file corruption issues
            print("Error: The file appears to be corrupted or incomplete.")
            break

# INTELLIGENT FLATTENING:
# If the file contained only ONE big object (standard case), we pull it out of the list
# so your JSON doesn't have an unnecessary extra set of brackets [ ].
if len(all_data) == 1:
    final_data = all_data[0]
    print("Detected single object in file.")
else:
    final_data = all_data
    print(f"Detected {len(all_data)} separate objects in file (append-style).")

# Print the content summary
print(f"Total items loaded: {len(final_data) if isinstance(final_data, list) else 1}")

# Convert to JSON string
try:
    readable_data = json.dumps(final_data, indent=4)
except TypeError as e:
    print(f"JSON Serialization Error: {e}")
    print("The data might contain non-serializable types (like Sets or Custom Classes).")
    print("Attempting to force string conversion...")
    readable_data = str(final_data)

# Define the path to the output text file
output_filename = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-11-28T09-42-04.137848_pdx-container-xterm-062.prd.it.nvidia.com_liwan/all_suggestions.txt'

# Write the readable data to the text file
with open(output_filename, "w") as output_file:
    output_file.write(readable_data)

print(f"Readable data has been written to {output_filename}")