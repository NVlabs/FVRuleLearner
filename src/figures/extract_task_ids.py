import re

# Read the suggestions file
with open('/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/logs/train_2025-09-11T09-01-03.562227_pdx-container-xterm-064.prd.it.nvidia.com_liwan/suggestions.txt', 'r') as f:
    content = f.read()

# Extract all Task IDs
task_ids = re.findall(r'Task ID:([^<]+)', content)

# Count unique task IDs
unique_task_ids = set(task_ids)

print("Total entries: {}".format(len(task_ids)))
print("Unique task IDs: {}".format(len(unique_task_ids)))
print("\nTask ID frequency:")

# Count frequency of each task ID
from collections import Counter
task_id_counts = Counter(task_ids)

for task_id, count in sorted(task_id_counts.items()):
    print("{}: {} times".format(task_id, count))
