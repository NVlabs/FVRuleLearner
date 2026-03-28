"""
Operator-level failure analysis across three datasets - COMPACT BAR CHART VERSION.
Generates a publication-quality grouped bar chart showing 6 NON-OVERLAPPING categories 
based on refined SystemVerilog Assertion (SVA) operator taxonomy (IEEE 1800 standard).

More space-efficient than pie charts for academic publication.

6 Non-Overlapping SVA Operator Categories:
1. Temporal Implication Operators: |-> / |=> / phase / structure / added implications
2. Temporal Delay Operators: ##k / ranges / repetition / indexing
3. Temporal Sampling Operators: $past / $fell / $stable / signal history queries
4. Combinational Logic Operators: boolean structure, !, !==, ^, &, |
5. Temporal Liveness Operators: strong / weak / eventually / s_eventually / s_always
6. Miscellaneous Differences: syntax / binding / normalization / arithmetic operators

IEEE Publication-Quality Specifications:
- Figure size: 6.5" × 3.5" (compact, fits IEEE single-column)
- Font sizes: 10-11pt (clear and readable)
- Color scheme: Industrial engineering palette (darker, professional, colorblind-safe)
- Fonts: Type 42 (TrueType) Helvetica/Arial embedded for compatibility
- Resolution: 600 DPI (exceeds IEEE minimum of 300 DPI)
- Formats: PNG (raster), PDF (vector), and EPS (IEEE Xplore preferred)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns

# Set publication-quality style for IEEE format
sns.set_style("whitegrid")
sns.set_context("paper")

# Configure for IEEE publication standards
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['axes.labelweight'] = 'normal'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 10
rcParams['pdf.fonttype'] = 42  # TrueType fonts required by IEEE
rcParams['ps.fonttype'] = 42
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'

# Data for all three datasets
categories = [
    'Temporal\nImplication',
    'Temporal\nDelay',
    'Temporal\nSampling',
    'Combinational\nLogic',
    'Temporal\nLiveness',
    'Miscellaneous'
]

# Shorter labels for x-axis (can customize)
categories_short = [
    'Implication\nOps',
    'Delay\nOps',
    'Sampling\nOps',
    'Logic\nOps',
    'Liveness\nOps',
    'Misc'
]

# Data arrays (Human, Machine, OpenCore)
human_counts = [5, 2, 0, 0, 1, 0]      # Total: 8
machine_counts = [15, 1, 4, 6, 4, 1]   # Total: 31
opencore_counts = [32, 16, 2, 1, 0, 3] # Total: 54

# Dataset colors (professional, colorblind-safe)
dataset_colors = {
    'Human': '#2ca02c',      # Green
    'Machine': '#ff7f0e',    # Orange
    'OpenCore': '#1f77b4'    # Blue
}

# Create figure
fig, ax = plt.subplots(figsize=(6.5, 3.5), facecolor='white')

# Set bar width and positions
x = np.arange(len(categories))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, human_counts, width, label='NL2SVA-Human (n=8)', 
               color=dataset_colors['Human'], edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x, machine_counts, width, label='NL2SVA-Machine (n=31)', 
               color=dataset_colors['Machine'], edgecolor='white', linewidth=0.8)
bars3 = ax.bar(x + width, opencore_counts, width, label='OpenCore (n=54)', 
               color=dataset_colors['OpenCore'], edgecolor='white', linewidth=0.8)

# Add value labels on bars (only if value > 0)
def add_value_labels(bars, counts):
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

add_value_labels(bars1, human_counts)
add_value_labels(bars2, machine_counts)
add_value_labels(bars3, opencore_counts)

# Customize axes
ax.set_ylabel('Number of Failures', fontsize=11, fontweight='bold')
ax.set_xlabel('SVA Operator Category', fontsize=11, fontweight='bold')
ax.set_title('Operator-Level Failure Analysis: SVA Operator Categories', 
             fontsize=12, fontweight='bold', pad=12)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels(categories_short, fontsize=9)

# Set y-axis limits with some headroom for labels
ax.set_ylim(0, max(opencore_counts) * 1.15)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False,
          framealpha=0.95, edgecolor='#CCCCCC', fontsize=10)

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# Save outputs
output_dir = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures'

# High-resolution PNG
output_path_png = f'{output_dir}/operator_level_failures_barplot.png'
plt.savefig(output_path_png, dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='png',
            pil_kwargs={'optimize': True})
print(f"IEEE-compliant PNG (600 DPI) saved: {output_path_png}")

# Vector PDF
output_path_pdf = f'{output_dir}/operator_level_failures_barplot.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='pdf',
            metadata={'Creator': 'Matplotlib', 
                     'Title': 'Operator-Level Failure Analysis - Bar Chart',
                     'Author': 'IEEE Submission'})
print(f"IEEE-compliant PDF (vector) saved: {output_path_pdf}")

# EPS for IEEE Xplore
output_path_eps = f'{output_dir}/operator_level_failures_barplot.eps'
plt.savefig(output_path_eps, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='eps',
            dpi=600)
print(f"IEEE-compliant EPS saved: {output_path_eps}")

plt.show()

# Print summary statistics
print("\n" + "="*80)
print("COMPACT BAR CHART - OPERATOR-LEVEL FAILURE ANALYSIS")
print("="*80)
print("Figure dimensions: 6.5\" × 3.5\" (40% more compact than pie chart version)")
print("Space savings: ~40% compared to three-pie-chart layout")
print("Advantages: Direct comparison across datasets, easier to read exact values")
print("="*80)

for i, cat in enumerate(categories):
    cat_name = cat.replace('\n', ' ')
    print(f"\n{cat_name}:")
    print(f"  Human: {human_counts[i]:2d} ({human_counts[i]/8*100:5.1f}%)")
    print(f"  Machine: {machine_counts[i]:2d} ({machine_counts[i]/31*100:5.1f}%)")
    print(f"  OpenCore: {opencore_counts[i]:2d} ({opencore_counts[i]/54*100:5.1f}%)")

print("="*80)

