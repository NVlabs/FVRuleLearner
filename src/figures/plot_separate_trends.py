# Combined plot showing all functionality trends across different benchmarks

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----- Human Benchmark Data (64 total cases) -----
total_cases_human = 64
# Iterative training data
seq_counts_human = np.array([23, 26, 26, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31])
# Q-tree data (kept for reference)
qtree_counts_human = np.array([18, 34, 44, 47, 49, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54])

human_functionality = np.array([
    0.2, 0.25, 0.31, 0.35, 0.38, 0.405, 0.41, 0.415, 0.42, 0.425,
    0.43, 0.432, 0.434, 0.436, 0.438, 0.44, 0.442, 0.445, 0.46, 0.47,
    0.475, 0.48, 0.485, 0.49, 0.495, 0.498
])
human_counts = human_functionality * total_cases_human
human_iters = np.arange(0, len(human_counts))

# ----- OpenCore Benchmark Data (800 total cases) -----
total_cases_opencore = 800
# Q-tree data (kept for reference)
# qtree_count_1 = np.array([106, 38, 19, 16, 15, 14, 13, 12, 11, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])
# qtree_count_2 = np.array([13, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# qtree_count_3 = np.array([95, 24, 13, 11, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
opencore_functionality = np.array([271, 114, 68, 56, 50, 46, 45, 44, 44, 44, 43, 42, 42, 41, 41, 41, 41, 41, 41, 41, 40, 40, 40, 39, 39, 39])
# Iterative training data: 800 - [error counts]
opencore_error_counts = np.array([280, 187, 169, 163, 160, 160, 160, 159, 159, 158, 156, 154, 153, 153, 152, 152, 150, 150, 150, 149, 148, 148, 147, 146, 146, 146])
opencore_counts = total_cases_opencore - opencore_error_counts
opencore_iters = np.arange(0, len(opencore_counts))

# ----- Machine Benchmark Data (240 total cases) -----
total_cases_machine = 240
# Q-tree data (kept for reference)
qtree_counts_machine = np.array([116, 203, 210, 217, 220, 223, 225, 226, 226,
                                  227, 227, 227, 228, 228, 229, 230, 230, 230, 230,
                                  231, 231, 231, 231, 231, 231, 231])
# Iterative training data: functionality ratios * 240
# machine_functionality = np.array([
#     0.36, 0.53, 0.60, 0.65, 0.68, 0.71, 0.72, 0.73, 0.735, 0.74,
#     0.745, 0.747, 0.749, 0.75, 0.752, 0.754, 0.755, 0.757, 0.758, 0.759,
#     0.760, 0.761, 0.762, 0.763, 0.764, 0.765
# ])
machine_functionality = 240-np.array([122, 63, 60, 59, 58, 58, 58, 58, 58, 57, 56, 55, 55, 55, 55, 54, 53, 53, 53, 53, 53, 53, 53, 52, 52, 52])
machine_counts = machine_functionality * total_cases_machine
machine_iters = np.arange(0, len(machine_counts))

# ----- Styling with Seaborn -----
# Set Seaborn style for publication-quality plots
sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
})
sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 2.0})

# Industrial color palette - darker, more professional colors
colors = {
    'human': '#1f77b4',      # Industrial blue (steel blue)
    'opencore': '#2ca02c',   # Industrial green (technical green)
    'machine': '#d62728',    # Industrial red (engineering red)
}

# IEEE publication font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'  # For consistent math rendering

# Helper function to create individual benchmark plots
from matplotlib.lines import Line2D

def plot_benchmark(iters, qtree_counts, iter_counts, total_cases, color, benchmark_name, output_base_path):
    """
    Create a compact figure for a single benchmark showing both Q-Tree and Iterative Training.
    
    Args:
        iters: Iteration indices
        qtree_counts: Q-Tree counts
        iter_counts: Iterative training counts
        total_cases: Total number of test cases for this benchmark
        color: Color for the plot
        benchmark_name: Name of the benchmark (for title and filename)
        output_base_path: Base path for output files
    """
    # Compact figure size: same width (7 inches), reduced height (2.5 inches)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    # Plot Iterative Training (square, dashed line)
    ax.plot(iters, iter_counts/total_cases, 
            marker='s', linewidth=2.0, markersize=6, 
            linestyle='--', 
            color=color, markeredgewidth=1.0, 
            markeredgecolor='white', alpha=0.7, zorder=2,
            clip_on=False)
    
    # Plot Q-Tree (circle, solid line)
    ax.plot(iters, qtree_counts/total_cases, 
            marker='o', linewidth=2.0, markersize=6, 
            linestyle='-', 
            color=color, markeredgewidth=1.0, 
            markeredgecolor='white', alpha=0.95, zorder=3,
            clip_on=False)
    
    # Labels with IEEE-style formatting
    ax.set_xlabel('Iteration', fontsize=11, labelpad=6)
    ax.set_ylabel('Fixing Ratio', fontsize=11, labelpad=6)
    
    # Grid is already set by seaborn style
    ax.set_axisbelow(True)
    
    # Single legend: Methods (markers and line styles)
    method_legend_elements = [
        Line2D([0], [0], marker='o', color=color, linewidth=2.0, linestyle='-', 
               label='Op-Tree', markersize=6, markeredgecolor='white', markeredgewidth=1.0),
        Line2D([0], [0], marker='s', color=color, linewidth=2.0, linestyle='--', 
               label='Non-Op-Tree', markersize=6, markeredgecolor='white', markeredgewidth=1.0),
    ]
    
    # Add legend in the best location
    legend = ax.legend(handles=method_legend_elements, fontsize=9, loc='lower right', 
                       frameon=False, labelspacing=0.4, handlelength=1.8, handletextpad=0.4)
    
    # Y-axis: Focus on data range for better use of space
    ax.set_ylim(0.25, 1.03)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'])
    
    # X-axis limits and ticks
    ax.set_xlim(-0.8, 25.8)
    ax.set_xticks(range(0, 26, 5))
    
    # Tick styling for IEEE format
    ax.tick_params(axis='both', which='major', labelsize=10, 
                   width=1.0, length=4, direction='out',
                   pad=4)
    
    # Clean white background for IEEE publications
    ax.set_facecolor('white')
    
    # Tight layout with minimal padding
    fig.tight_layout(pad=0.2)
    
    # Save in multiple formats
    output_png = f'{output_base_path}_{benchmark_name.lower()}.png'
    output_pdf = f'{output_base_path}_{benchmark_name.lower()}.pdf'
    output_eps = f'{output_base_path}_{benchmark_name.lower()}.eps'
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                format='png', transparent=False)
    
    fig.savefig(output_pdf, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                format='pdf', transparent=False)
    
    fig.savefig(output_eps, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                format='eps', transparent=False)
    
    plt.close(fig)
    
    return output_png, output_pdf, output_eps


# Base output path
output_base_path = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures/trends'

# Create three separate figures

# 1. Human Benchmark
human_png, human_pdf, human_eps = plot_benchmark(
    human_iters, 
    qtree_counts_human, 
    seq_counts_human, 
    total_cases_human, 
    colors['human'], 
    'human',
    output_base_path
)

# 2. Machine Benchmark
# Note: machine_functionality is already in counts, not ratios
machine_png, machine_pdf, machine_eps = plot_benchmark(
    machine_iters, 
    qtree_counts_machine, 
    machine_functionality,  # Already in counts 
    total_cases_machine, 
    colors['machine'], 
    'machine',
    output_base_path
)

# 3. OpenCore Benchmark
# Calculate Q-Tree counts from functionality
opencore_qtree_counts = total_cases_opencore - opencore_functionality
opencore_png, opencore_pdf, opencore_eps = plot_benchmark(
    opencore_iters, 
    opencore_qtree_counts, 
    opencore_counts, 
    total_cases_opencore, 
    colors['opencore'], 
    'opencore',
    output_base_path
)

print(f"✓ IEEE-style plots saved successfully!")
print(f"\n1. Human Benchmark:")
print(f"  PNG (300 DPI): {human_png}")
print(f"  PDF (vector):  {human_pdf}")
print(f"  EPS (vector):  {human_eps}")
print(f"\n2. Machine Benchmark:")
print(f"  PNG (300 DPI): {machine_png}")
print(f"  PDF (vector):  {machine_pdf}")
print(f"  EPS (vector):  {machine_eps}")
print(f"\n3. OpenCore Benchmark:")
print(f"  PNG (300 DPI): {opencore_png}")
print(f"  PDF (vector):  {opencore_pdf}")
print(f"  EPS (vector):  {opencore_eps}")
print(f"\nFigure specifications:")
print(f"  - Font: Times/Times New Roman (IEEE standard)")
print(f"  - Size: 7 x 2.5 inches (compact for academic papers)")
print(f"  - Colors: Industrial palette (blue=human, red=machine, green=opencore)")

# plt.show()  # Uncomment to display interactively

