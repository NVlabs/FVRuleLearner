import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# [Data Definitions - Same as before]
total_cases_human = 64
seq_counts_human = np.array([23, 26, 26, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31])
qtree_counts_human = np.array([18, 34, 44, 47, 49, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54])
human_iters = np.arange(0, len(seq_counts_human))

total_cases_opencore = 800
opencore_functionality = np.array([271, 114, 68, 56, 50, 46, 45, 44, 44, 44, 43, 42, 42, 41, 41, 41, 41, 41, 41, 41, 40, 40, 40, 39, 39, 39])
opencore_error_counts = np.array([280, 187, 169, 163, 160, 160, 160, 159, 159, 158, 156, 154, 153, 153, 152, 152, 150, 150, 150, 149, 148, 148, 147, 146, 146, 146])
opencore_counts = total_cases_opencore - opencore_error_counts
opencore_qtree_counts = total_cases_opencore - opencore_functionality
opencore_iters = np.arange(0, len(opencore_counts))

total_cases_machine = 240
qtree_counts_machine = np.array([116, 203, 210, 217, 220, 223, 225, 226, 226, 227, 227, 227, 228, 228, 229, 230, 230, 230, 230, 231, 231, 231, 231, 231, 231, 231])
machine_functionality = 240-np.array([122, 63, 60, 59, 58, 58, 58, 58, 58, 57, 56, 55, 55, 55, 55, 54, 53, 53, 53, 53, 53, 53, 53, 52, 52, 52])
machine_iters = np.arange(0, len(machine_functionality))

# [Style Settings]
sns.set_style("whitegrid", {
    'grid.linestyle': '--', 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.linewidth': 1.0, 'axes.edgecolor': 'black',
})
sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 2.0})
colors = {'human': '#1f77b4', 'opencore': '#2ca02c', 'machine': '#d62728'}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 12 
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

def plot_benchmark_v2(iters, qtree_counts, iter_counts, total_cases, color, benchmark_name, output_base_path):
    # Square aspect ratio (5x3.8) to fit 3 columns
    fig, ax = plt.subplots(figsize=(5, 3))
    
    y_iter = iter_counts/total_cases
    y_qtree = qtree_counts/total_cases

    ax.plot(iters, y_iter, marker='s', linewidth=2.0, markersize=6, linestyle='--', 
            color=color, markeredgewidth=1.0, markeredgecolor='white', alpha=0.7, label='General Rules Learning')

    # Plot Lines    
    ax.plot(iters, y_qtree, marker='o', linewidth=2.0, markersize=6, linestyle='-', 
            color=color, markeredgewidth=1.0, markeredgecolor='white', alpha=0.95, label='FVRuleLearner')

    ax.set_xlabel('Iteration', labelpad=5)
    ax.set_ylabel('Fixing Ratio', labelpad=5)
    
    # --- Annotations ---
    # 1. Saturation Point
    # sat_idx = 5
    # sat_x = iters[sat_idx]
    # sat_y = y_qtree[sat_idx]
    
    # ax.annotate('Saturation', xy=(sat_x, sat_y), xytext=(sat_x + 3, sat_y - 0.15),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6),
    #             fontsize=10, color='#333333')

    # 2. Discrepancy (Gap between lines at the end)
    end_x = iters[-2]
    end_y1 = y_qtree[-2]
    end_y2 = y_iter[-2]
    mid_y = (end_y1 + end_y2) / 2
    
    ax.annotate('', xy=(end_x, end_y1), xytext=(end_x, end_y2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    # ax.text(end_x - 1, mid_y, 'Discrepancy', rotation=90, va='center', ha='right', fontsize=10, color='#333333')

    # Legend
    ax.legend(fontsize=10, loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
    
    # Axes limits
    ax.set_ylim(0.25, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    # ax.set_yticklabels(['25%', '50%', '75%', '100%'])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'])
    ax.set_xlim(-1, 26)
    
    fig.tight_layout()
    
    # --- SAVE FILES ---
    # 1. Save as PDF (Vector graphic for LaTeX)
    output_pdf = f'{output_base_path}_{benchmark_name.lower()}_v2.pdf'
    fig.savefig(output_pdf, bbox_inches='tight', format='pdf')
    
    # 2. Save as PNG (Raster graphic for slides/web)
    output_png = f'{output_base_path}_{benchmark_name.lower()}_v2.png'
    fig.savefig(output_png, bbox_inches='tight', format='png', dpi=300, facecolor='white')
    
    plt.close(fig)
    return output_pdf, output_png

# Run Plotting
output_base_path = 'trends' # Adjust to your desired path/filename prefix

print("Generating figures...")
h_pdf, h_png = plot_benchmark_v2(human_iters, qtree_counts_human, seq_counts_human, total_cases_human, colors['human'], 'human', output_base_path)
m_pdf, m_png = plot_benchmark_v2(machine_iters, qtree_counts_machine, machine_functionality, total_cases_machine, colors['machine'], 'machine', output_base_path)
o_pdf, o_png = plot_benchmark_v2(opencore_iters, opencore_qtree_counts, opencore_counts, total_cases_opencore, colors['opencore'], 'opencore', output_base_path)

print(f"Done! Generated:\n1. {h_pdf} & {h_png}\n2. {m_pdf} & {m_png}\n3. {o_pdf} & {o_png}")