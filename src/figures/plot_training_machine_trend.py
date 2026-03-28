# Replot with y-axis in raw counts (0..64), not percentages,
# and do NOT show anything implying "out of 25". Larger fonts kept.

import matplotlib.pyplot as plt
import numpy as np

# ----- Data -----
total_cases = 240
# seq_counts = np.array([23, 26, 26, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31])
qtree_counts = np.array([116, 116, 203, 210, 217, 220, 223, 225, 226, 226,
 227, 227, 227, 228, 228, 229, 230, 230, 230, 230,
 231, 231, 231, 231, 231, 231])
# seq_iters = np.arange(1, len(seq_counts) + 1)
# qtree_iters = np.arange(1, len(qtree_counts) + 1)

# Machine functionality converted to counts (out of 64)
machine_functionality = np.array([
    0.36, 0.53, 0.60, 0.65, 0.68, 0.71, 0.72, 0.73, 0.735, 0.74,
    0.745, 0.747, 0.749, 0.75, 0.752, 0.754, 0.755, 0.757, 0.758, 0.759,
    0.760, 0.761, 0.762, 0.763, 0.764, 0.765, 
    # 0.766, 0.767, 0.768, 0.769,
    # 0.770, 0.771, 0.772, 0.773, 0.774, 0.775, 0.776, 0.777, 0.778, 0.779, 0.780
])
machine_counts = machine_functionality * total_cases

machine_iters = np.arange(0, len(machine_counts))

# ----- Styling -----
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16  # larger fonts

fig, ax = plt.subplots(figsize=(13, 8))

# Lines
ax.plot(machine_iters, machine_counts/240.0, marker='^', linewidth=3, markersize=7, label='Iterative Search (Llama-3)')
# ax.plot(machine_iters, seq_counts/240.0, marker='o', linewidth=3, markersize=7, label='Iterative Search (o3-mini)')
ax.plot(machine_iters, qtree_counts/240.0, marker='s', linewidth=3, markersize=7, label='Q-Tree-based Iterative Search (o3-mini)')

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
output_path = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures/three_trends_machine.png'
fig.tight_layout()
fig.savefig(output_path, dpi=300, bbox_inches='tight')
output_path
