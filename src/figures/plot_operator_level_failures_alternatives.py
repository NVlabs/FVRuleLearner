"""
Operator-level failure analysis - MULTIPLE PUBLICATION-QUALITY ALTERNATIVES.
Generates 3 clearer visualization options for academic papers:
1. Horizontal grouped bar chart (easier to read category names)
2. Heatmap with annotations (ultra-compact)
3. Stacked percentage bar chart (shows proportions)

All options are more publication-appropriate than 3D plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns

# Configure for IEEE publication standards
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'

# Data
categories = [
    'Temporal Implication Operators',
    'Temporal Delay Operators',
    'Temporal Sampling Operators',
    'Combinational Logic Operators',
    'Temporal Liveness Operators',
    'Miscellaneous Operators'
]

human_counts = [4, 2, 0, 0, 1, 1]      # Total: 8
machine_counts = [19, 1, 3, 2, 4, 2]
opencore_counts = [41, 20, 2, 0, 0, 1] # Total: 64

dataset_names = ['NL2SVA-Human', 'NL2SVA-Machine', 'NL2SVA-OpenCore']
dataset_colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

output_dir = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures'

# ============================================================================
# OPTION 1: HORIZONTAL GROUPED BAR CHART (BEST FOR LONG LABELS)
# ============================================================================
print("Generating Option 1: Horizontal Grouped Bar Chart...")

fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')

y = np.arange(len(categories))
height = 0.25

bars1 = ax.barh(y - height, human_counts, height, label='NL2SVA-Human', 
                color=dataset_colors[0], edgecolor='white', linewidth=0.8)
bars2 = ax.barh(y, machine_counts, height, label='NL2SVA-Machine', 
                color=dataset_colors[1], edgecolor='white', linewidth=0.8)
bars3 = ax.barh(y + height, opencore_counts, height, label='NL2SVA-OpenCore', 
                color=dataset_colors[2], edgecolor='white', linewidth=0.8)

# Add value labels
def add_horizontal_labels(bars, counts):
    for bar, count in zip(bars, counts):
        if count > 0:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                   f'{int(count)}',
                   ha='left', va='center', fontsize=9, fontweight='bold')

add_horizontal_labels(bars1, human_counts)
add_horizontal_labels(bars2, machine_counts)
add_horizontal_labels(bars3, opencore_counts)

ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=10)
ax.set_xlabel('Number of Failures', fontsize=11, fontweight='bold')
# ax.set_title('Operator-Level Failure Analysis (Horizontal Layout)', 
#              fontsize=12, fontweight='bold', pad=12)
ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax.xaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save
output_path = f'{output_dir}/operator_failures_horizontal.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', format='png')
print(f"✓ Horizontal bar chart saved: {output_path}")

output_path_pdf = f'{output_dir}/operator_failures_horizontal.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', format='pdf')
print(f"✓ PDF saved: {output_path_pdf}")

plt.close()

# ============================================================================
# OPTION 2: HEATMAP WITH ANNOTATIONS (ULTRA-COMPACT)
# ============================================================================
print("\nGenerating Option 2: Heatmap with Annotations...")

fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='white')

# Prepare data matrix
data_matrix = np.array([human_counts, machine_counts, opencore_counts])

# Create heatmap with custom colormap
cmap = sns.color_palette("YlGnBu", as_cmap=True)  # Yellow-Green-Blue instead of Yellow-Orange-Red
im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=np.max(data_matrix))

# Set ticks and labels
ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(dataset_names)))
ax.set_xticklabels([cat.replace(' Operators', '').replace(' Differences', '') 
                     for cat in categories], rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(['Human\n(n=8)', 'Machine\n(n=31)', 'OpenCore\n(n=64)'], fontsize=10)

# Add text annotations
for i in range(len(dataset_names)):
    for j in range(len(categories)):
        value = data_matrix[i, j]
        # Choose text color based on background intensity
        text_color = 'white' if value > np.max(data_matrix) * 0.5 else 'black'
        text = ax.text(j, i, int(value),
                      ha="center", va="center", color=text_color,
                      fontsize=11, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Number of Failures', rotation=270, labelpad=20, fontweight='bold')

# ax.set_title('Operator-Level Failure Heatmap', fontsize=12, fontweight='bold', pad=12)

plt.tight_layout()

# Save
output_path = f'{output_dir}/operator_failures_heatmap.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', format='png')
print(f"✓ Heatmap saved: {output_path}")

output_path_pdf = f'{output_dir}/operator_failures_heatmap.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', format='pdf')
print(f"✓ PDF saved: {output_path_pdf}")

plt.close()

# ============================================================================
# OPTION 3: STACKED PERCENTAGE BAR CHART (SHOWS PROPORTIONS)
# ============================================================================
print("\nGenerating Option 3: Stacked Percentage Bar Chart...")

fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')

# Calculate percentages
human_pcts = (np.array(human_counts) / np.sum(human_counts) * 100)
machine_pcts = (np.array(machine_counts) / np.sum(machine_counts) * 100)
opencore_pcts = (np.array(opencore_counts) / np.sum(opencore_counts) * 100)

x = np.arange(3)
width = 0.6

# Category colors - balanced palette without excessive red
category_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#7f7f7f']

# Create stacked bars
bottom_human = np.zeros(1)
bottom_machine = np.zeros(1)
bottom_opencore = np.zeros(1)

for i, (cat, color) in enumerate(zip(categories, category_colors)):
    if human_pcts[i] > 0 or machine_pcts[i] > 0 or opencore_pcts[i] > 0:
        # Human
        bar = ax.bar(0, human_pcts[i], width, bottom=bottom_human, 
                    color=color, edgecolor='white', linewidth=1.5,
                    label=cat.replace(' Operators', '').replace(' Differences', '') if i == 0 else "")
        if human_pcts[i] > 5:
            ax.text(0, bottom_human + human_pcts[i]/2, f'{human_pcts[i]:.0f}%',
                   ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        bottom_human += human_pcts[i]
        
        # Machine
        bar = ax.bar(1, machine_pcts[i], width, bottom=bottom_machine, 
                    color=color, edgecolor='white', linewidth=1.5)
        if machine_pcts[i] > 5:
            ax.text(1, bottom_machine + machine_pcts[i]/2, f'{machine_pcts[i]:.0f}%',
                   ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        bottom_machine += machine_pcts[i]
        
        # OpenCore
        bar = ax.bar(2, opencore_pcts[i], width, bottom=bottom_opencore, 
                    color=color, edgecolor='white', linewidth=1.5)
        if opencore_pcts[i] > 5:
            ax.text(2, bottom_opencore + opencore_pcts[i]/2, f'{opencore_pcts[i]:.0f}%',
                   ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        bottom_opencore += opencore_pcts[i]

# Customize
ax.set_ylabel('Percentage of Failures (%)', fontsize=11, fontweight='bold')
# ax.set_title('Operator-Level Failure Distribution by Dataset', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(['NL2SVA-Human', 'NL2SVA-Machine', 'NL2SVA-OpenCore'])
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_axisbelow(True)

# Create custom legend at bottom with 3 rows × 2 columns
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='white', linewidth=1.5)
                   for color in category_colors]
legend_labels = categories  # Use full original names
ax.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
         ncol=2, frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)

plt.tight_layout()

# Save
output_path = f'{output_dir}/operator_failures_stacked.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', format='png')
print(f"✓ Stacked percentage bar chart saved: {output_path}")

output_path_pdf = f'{output_dir}/operator_failures_stacked.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', format='pdf')
print(f"✓ PDF saved: {output_path_pdf}")

plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("THREE SUPERIOR ALTERNATIVES TO 3D PLOTS")
print("="*80)
print("\n1. HORIZONTAL GROUPED BAR CHART (7\" × 5\")")
print("   ✓ Best for: Long category names, easy to read")
print("   ✓ Clarity: Excellent - no label rotation needed")
print("   ✓ Use case: When category names are important")
print("   → Files: operator_failures_horizontal.{png,pdf}")

print("\n2. HEATMAP WITH ANNOTATIONS (6\" × 4.5\")")
print("   ✓ Best for: Ultra-compact display, pattern recognition")
print("   ✓ Clarity: Excellent - color + numbers show magnitude")
print("   ✓ Use case: When space is extremely limited")
print("   → Files: operator_failures_heatmap.{png,pdf}")

print("\n3. STACKED PERCENTAGE BAR CHART (7\" × 4\")")
print("   ✓ Best for: Showing proportional composition")
print("   ✓ Clarity: Excellent - reveals relative importance")
print("   ✓ Use case: When comparing proportions across datasets")
print("   → Files: operator_failures_stacked.{png,pdf}")

print("\n" + "="*80)
print("WHY NOT 3D PLOTS?")
print("="*80)
print("❌ Perspective distortion makes values hard to read")
print("❌ Not IEEE/publication-compliant (journals discourage 3D)")
print("❌ Bars occlude each other")
print("❌ Adds visual complexity without information gain")
print("❌ Appears unprofessional in academic papers")
print("\n✓ Stick with 2D visualizations for maximum clarity and professionalism")
print("="*80)

