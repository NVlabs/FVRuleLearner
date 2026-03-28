# Replot with y-axis in raw counts (0..64), not percentages,
# and do NOT show anything implying "out of 25". Larger fonts kept.

import matplotlib.pyplot as plt
import numpy as np

# ----- Data -----
total_cases = 800
# seq_counts = np.array([23, 26, 26, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31])
qtree_count_1 = np.array([368, 106, 38, 19, 16, 15, 14, 13, 12, 11, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])
# print(len(qtree_count_1))
qtree_count_2 = np.array([65, 13, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# print(len(qtree_count_2))
qtree_count_3 = np.array([366, 95, 24, 13, 11, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
# print(len(qtree_count_3))

# opencore functionality converted to counts (out of 64)
opencore_functionality = qtree_count_1 + qtree_count_2 + qtree_count_3
opencore_counts =total_cases - opencore_functionality

opencore_iters = np.arange(0, len(opencore_counts))

# ----- Styling -----
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16  # larger fonts

fig, ax = plt.subplots(figsize=(13, 8))

# Lines
# ax.plot(opencore_iters, opencore_counts/800.0, marker='^', linewidth=3, markersize=7, label='Iterative Search (Llama-3)')
# ax.plot(opencore_iters, seq_counts/800.0, marker='o', linewidth=3, markersize=7, label='Iterative Search (o3-mini)')
ax.plot(opencore_iters, opencore_counts/800.0, marker='s', linewidth=3, markersize=7, label='Q-Tree-based Iterative Search (o3-mini)')

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
output_path = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures/three_trends_opencore.png'
fig.tight_layout()
fig.savefig(output_path, dpi=300, bbox_inches='tight')
output_path



