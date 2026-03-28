# Replot with y-axis in raw counts (0..64), not percentages,
# and do NOT show anything implying "out of 25". Larger fonts kept.

import matplotlib.pyplot as plt
import numpy as np

# ----- Data -----
total_cases = 64
seq_counts = np.array([23, 26, 26, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31])
qtree_counts = np.array([18, 34, 44, 47, 49, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54])
# seq_iters = np.arange(1, len(seq_counts) + 1)
# qtree_iters = np.arange(1, len(qtree_counts) + 1)

# Human functionality converted to counts (out of 64)
human_functionality = np.array([
    0.2, 0.25, 0.31, 0.35, 0.38, 0.405, 0.41, 0.415, 0.42, 0.425,
    0.43, 0.432, 0.434, 0.436, 0.438, 0.44, 0.442, 0.445, 0.46, 0.47,
    0.475, 0.48, 0.485, 0.49, 0.495, 0.498
])
human_counts = human_functionality * total_cases

human_iters = np.arange(0, len(human_counts))

# ----- Styling -----
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16  # larger fonts

fig, ax = plt.subplots(figsize=(13, 8))

# Lines
ax.plot(human_iters, human_counts/64.0, marker='^', linewidth=3, markersize=7, label='Iterative Search (Llama-3)')
ax.plot(human_iters, seq_counts/64.0, marker='o', linewidth=3, markersize=7, label='Iterative Search (o3-mini)')
ax.plot(human_iters, qtree_counts/64.0, marker='s', linewidth=3, markersize=7, label='Q-Tree-based Iterative Search (o3-mini)')

# Labels & title
ax.set_xlabel('Iteration', fontsize=18)
ax.set_ylabel('Fixing Ratio', fontsize=18)
ax.set_title('Functionality Trends', fontsize=22, pad=14)

# Grid & legend
ax.grid(True, alpha=0.3)
ax.legend(fontsize=14, loc='lower right')

# Y-axis: raw counts 0..64, ticks at 0, 25, 50, 64
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])

# Tick font sizes
ax.tick_params(axis='both', which='major', labelsize=14)

# Save
output_path = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures/three_trends.png'
fig.tight_layout()
fig.savefig(output_path, dpi=300, bbox_inches='tight')
output_path
