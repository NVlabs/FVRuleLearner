import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ==========================================
# IEEE/DAC SINGLE-COLUMN CONFIGURATION
# ==========================================
# Standard column width is ~3.5 inches. 
# Height set to 1.8 inches to keep it compact.
rcParams['figure.figsize'] = (3.5, 1.8)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
rcParams['font.size'] = 6              # Reduced for single column
rcParams['lines.linewidth'] = 0.8
rcParams['axes.linewidth'] = 0.6
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

colors = {
    'Imp': '#1f77b4', 'Delay': '#2ca02c', 'Samp': '#9467bd',
    'Comb': '#ff7f0e', 'Live': '#17becf', 'Misc': '#7f7f7f'
}

# Shortened labels to save horizontal space
cats = ['Temp. Implication', 'Temp. Delay', 'Temp. Sampling', 
        'Combinational', 'Temp. Liveness', 'Miscellaneous']
short_cats = ['Temp. Implication', 'Temp. Delay', 'Temp. Sampling', 
              'Combinational', 'Temp. Liveness', 'Miscellaneous']

# ==========================================
# DATA (Verified)
# ==========================================
h_base = np.array([4, 2, 0, 0, 1, 1])
h_ours = np.array([1, 0, 0, 0, 0, 1]) 
m_base = np.array([19, 1, 3, 2, 4, 2])
m_ours = np.array([4, 0, 2, 2, 3, 1])
o_base = np.array([41, 20, 2, 0, 0, 1])
o_ours = np.array([5,  9, 1, 0, 0, 1])

datasets = [('Human', h_base, h_ours), ('Machine', m_base, m_ours), ('OpenCore', o_base, o_ours)]

# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(1, 3, sharey=True)
# Specific margins for 3.5 inch width:
# Left needs space for labels (0.28), Right is tight (0.98)
plt.subplots_adjust(wspace=0.1, left=0.28, right=0.98, top=0.82, bottom=0.15)

y_pos = np.arange(len(cats))
height = 0.75

for i, (ax, (name, base, ours)) in enumerate(zip(axes, datasets)):
    # 1. Ghost Bar (Baseline) - No Labels to save space
    ax.barh(y_pos, base, height, color='#e0e0e0', edgecolor='#999999', 
            hatch='///', linewidth=0.5, label='Baseline')
    
    # 2. Solid Bar (FVRuleGen)
    bar_colors = [colors['Imp'], colors['Delay'], colors['Samp'], 
                  colors['Comb'], colors['Live'], colors['Misc']]
    ax.barh(y_pos, ours, height, color=bar_colors, edgecolor='black', 
            linewidth=0.6, alpha=0.9, label='FVRuleGen')

    # 3. Annotations - Only Ours (Small fonts)
    max_val = max(base)
    for j, (b, o) in enumerate(zip(base, ours)):
        if o > 0:
            # Check if bar is large enough for text inside
            if o > max_val * 0.25:
                ax.text(o/2, j, str(o), va='center', ha='center', 
                        fontweight='bold', color='white', fontsize=5)
            else:
                # Place outside, offset slightly
                ax.text(o + (max_val * 0.1), j, str(o), va='center', ha='center', 
                        fontweight='bold', color='black', fontsize=5)

    # Formatting
    ax.set_title(name, fontsize=7, fontweight='bold', pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(i == 0) # Only first plot gets y-axis line
    ax.tick_params(left=False)            # Hide ticks
    
    # X-Axis: Minimalist (0, Max) to save space
    ax.set_xlim(0, max_val * 1.4)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2)) # Max 2 ticks (e.g. 0, 20)
    ax.tick_params(axis='x', labelsize=5, pad=2)
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.5)

# Y-Labels (Leftmost only)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(short_cats, fontsize=6)
axes[0].invert_yaxis() # Top-down order

# Legend - Compact, Top Center
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e0e0e0', hatch='///', edgecolor='#999999', label='FVEval'),
    Patch(facecolor='#1f77b4', label='FVRuleGen')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.62, 1.02), 
           ncol=2, frameon=False, fontsize=6, handlelength=1.5, handleheight=0.6)

# Save
output_path = 'dac_comparison_clean.pdf'
plt.savefig(output_path, bbox_inches='tight', dpi=600)
plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=600)
print(f"Saved figure to {output_path} and .png")