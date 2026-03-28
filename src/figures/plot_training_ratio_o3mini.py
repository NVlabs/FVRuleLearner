# Plot Syntax/Functionality vs Training Data Ratio for o3-mini model
# Based on TABLE IV: Effect of Training Data Ratio on FVRuleELearner Performance on NL2SVA-MACHINE

import matplotlib.pyplot as plt
import numpy as np

# ----- Data from TABLE IV -----
training_ratio = np.array([0, 0.16, 0.32, 0.48, 0.64, 0.80])
bleu = np.array([0.6696, 0.7808, 0.7698, 0.7774, 0.7918, 0.6980])
syn = np.array([0.9500, 0.9667, 0.9667, 0.9667, 0.9667, 0.9667])
func = np.array([0.4833, 0.6833, 0.6833, 0.7667, 0.7500, 0.7833])
r_func = np.array([0.5833, 0.7833, 0.7500, 0.8667, 0.8000, 0.8333])

# ----- Styling -----
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16  # larger fonts

fig, ax = plt.subplots(figsize=(13, 8))

# Lines
ax.plot(training_ratio, syn, marker='o', linewidth=3, markersize=10, label='Syntax (SYN)', color='#2E86AB')
ax.plot(training_ratio, func, marker='s', linewidth=3, markersize=10, label='Functionality (FUNC)', color='#A23B72')
ax.plot(training_ratio, r_func, marker='^', linewidth=3, markersize=10, label='Relaxed Functionality (R.FUNC)', color='#F18F01')

# Labels & title
ax.set_xlabel('Training Data Ratio', fontsize=18)
ax.set_ylabel('Score', fontsize=18)
ax.set_title('o3-mini Performance vs Training Data Ratio', fontsize=22, pad=14)

# Grid & legend
ax.grid(True, alpha=0.3)
ax.legend(fontsize=14, loc='lower right')

# Y-axis: 0 to 1.0
ax.set_ylim(0.4, 1.0)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# X-axis
ax.set_xlim(-0.05, 0.85)
ax.set_xticks([0, 0.16, 0.32, 0.48, 0.64, 0.80])

# Tick font sizes
ax.tick_params(axis='both', which='major', labelsize=14)

# Save
output_path = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures/o3mini_training_ratio.png'
fig.tight_layout()
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PDF
output_path_pdf = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures/o3mini_training_ratio.pdf'
fig.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path_pdf}")

plt.show()

