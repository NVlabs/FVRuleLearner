import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# ----- Data -----
training_ratio = np.array([0, 0.16, 0.32, 0.48, 0.64, 0.80])

# o3-mini (Functionality)
func_o3mini = np.array([0.4833, 0.6833, 0.6833, 0.7667, 0.7500, 0.7833])

# claude (Functionality - 5 points)
func_claude = np.array([0.4667, 0.7000, 0.7000, 0.7500, 0.7500, 0.7500])

# 4o (Functionality)
func_4o = np.array([0.4333, 0.7167, 0.7167, 0.7667, 0.7500, 0.7500])

# ----- Styling (Matching the reference) -----
sns.set_style("whitegrid", {
    'grid.linestyle': '--', 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.linewidth': 1.0, 'axes.edgecolor': 'black',
})
sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 2.0})

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 12 
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Create figure with the reference size (5, 3)
fig, ax = plt.subplots(figsize=(5, 3))


# Plot GPT-4o
ax.plot(training_ratio, func_4o, 
        marker='^', linewidth=2.0, markersize=6, linestyle='-',
        markeredgewidth=1.0, markeredgecolor='white', alpha=0.95,
        label='GPT-4o', color='#F18F01')

# Plot Claude (Use first 5 x-values)
ax.plot(training_ratio[:len(func_claude)], func_claude, 
        marker='s', linewidth=2.0, markersize=6, linestyle='--',
        markeredgewidth=1.0, markeredgecolor='white', alpha=0.95,
        label='Claude 4.5 sonnet', color='#A23B72')

# Plot o3-mini
ax.plot(training_ratio, func_o3mini, 
        marker='o', linewidth=2.0, markersize=6, linestyle='-.', 
        markeredgewidth=1.0, markeredgecolor='white', alpha=0.95,
        label='o3-mini', color='#2E86AB')

# Labels
ax.set_xlabel('Training Data Ratio', labelpad=5)
ax.set_ylabel('Functionality Score', labelpad=5)

# Legend
ax.legend(fontsize=10, loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc')

# Y-axis: 0.4 to 1.05
ax.set_ylim(0.4, 0.81)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_yticklabels(['0.40', '0.50', '0.60', '0.70', '0.80'])

# X-axis
ax.set_xlim(-0.05, 0.85)
ax.set_xticks([0, 0.16, 0.32, 0.48, 0.64, 0.80])

# Layout
fig.tight_layout()

# Save
output_dir = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

output_path = output_dir + 'llm_functionality_comparison_resized.png'
output_path_pdf = output_dir + 'llm_functionality_comparison_resized.pdf'

fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(output_path_pdf, dpi=300, bbox_inches='tight', format='pdf')

print(f"Figure saved to: {output_path}")
print(f"Figure saved to: {output_path_pdf}")

# plt.show()