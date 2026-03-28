"""
Operator-level failure analysis across three datasets.
Generates three publication-quality pie charts showing 6 NON-OVERLAPPING categories 
based on refined SystemVerilog Assertion (SVA) operator taxonomy (IEEE 1800 standard).

Enhanced for academic publication with professional typography, industrial engineering
color palette (darker, professional tones), and high visual clarity.

6 Non-Overlapping SVA Operator Categories:
1. Temporal Implication Operators: |-> / |=> / phase / structure / added implications
2. Temporal Delay Operators: ##k / ranges / repetition / indexing
3. Temporal Sampling Operators: $past / $fell / $stable / signal history queries
4. Combinational Logic Operators: boolean structure, !, !==, ^, &, |
5. Temporal Liveness Operators: strong / weak / eventually / s_eventually / s_always
6. Miscellaneous Differences: syntax / binding / normalization / arithmetic operators

IEEE Publication-Quality Specifications:
- Figure size: 10" × 4.0" (optimized for double-column IEEE format)
- Font sizes: 12-13pt (enhanced for maximum readability in print and digital)
- Color scheme: Industrial engineering palette (darker, professional, colorblind-safe)
- Fonts: Type 42 (TrueType) Helvetica/Arial embedded for compatibility
- Resolution: 600 DPI (exceeds IEEE minimum of 300 DPI)
- Formats: PNG (raster), PDF (vector), and EPS (IEEE Xplore preferred)
- Visual enhancements: Large percentage labels (13pt), concise formal category names, subtle slice separation
"""

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.patheffects
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# Set publication-quality style for IEEE format
sns.set_style("whitegrid")
sns.set_context("paper")

# Configure for IEEE publication standards - professional academic typography
# Use Helvetica/Arial for clean, professional look
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans']
rcParams['font.size'] = 10  # Readable base font size
rcParams['axes.linewidth'] = 1.0
rcParams['patch.linewidth'] = 0.8
rcParams['axes.labelweight'] = 'normal'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['pdf.fonttype'] = 42  # TrueType fonts required by IEEE
rcParams['ps.fonttype'] = 42
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'

# Define NON-OVERLAPPING categories based on refined SVA operator taxonomy
# 6 categories aligned with SystemVerilog Assertion operator-level failure patterns
#
# Category Definitions (based on IEEE 1800 SVA standard):
# 1. Temporal Implication Operators: |-> / |=> / phase / structure / added implications
# 2. Temporal Delay Operators: ##k / ranges / repetition / indexing
# 3. Temporal Sampling Operators: $past / $fell / $stable / signal history
# 4. Combinational Logic Operators: boolean structure, !, !==, ^, &, |
# 5. Temporal Liveness Operators: strong / weak / eventually / s_eventually / s_always
# 6. Miscellaneous Differences: syntax / binding / normalization / arithmetic

# Human (NL2SVA-Human) - Total: 8 cases
human_data = {
    'Temporal Implication Operators': 5,                  # |->/|=>/phase operator mismatches
    'Temporal Delay Operators': 2,           # ##k / ranges
    'Temporal Sampling Operators': 0,           # None
    'Combinational Logic Operators': 0,          # None
    'Temporal Liveness Operators': 1,                    # strong/weak/eventually
    'Miscellaneous Differences': 0,             # None
}

# Machine (NL2SVA-Machine) - Total: 31 failures
machine_data = {
    'Temporal Implication Operators': 15,                 # |->/|=>/added/phase mismatches
    'Temporal Delay Operators': 1,           # ##k / ranges
    'Temporal Sampling Operators': 4,           # $past/$fell/$stable
    'Combinational Logic Operators': 6,          # Boolean structure, logical operators
    'Temporal Liveness Operators': 4,                    # s_eventually / s_always / strong
    'Miscellaneous Differences': 1,             # Other issues
}

# OpenCore - Total: 54 failures
opencore_data = {
    'Temporal Implication Operators': 32,                 # |->/|=>/structure/phase mismatches
    'Temporal Delay Operators': 16,          # ##k, ranges, repetition, indexing
    'Temporal Sampling Operators': 2,           # $past/$stable/...
    'Combinational Logic Operators': 1,          # Boolean structure
    'Temporal Liveness Operators': 0,                    # None
    'Miscellaneous Differences': 3,             # Syntax/binding/normalization
}

# Industrial, colorblind-friendly palette for academic publication (6 categories)
# Darker, more professional engineering colors for technical publications
category_colors = {
    'Temporal Implication Operators': '#d62728',                  # Industrial red (engineering red) - most critical
    'Temporal Delay Operators': '#2ca02c',           # Industrial green (technical green) - timing
    'Temporal Sampling Operators': '#1f77b4',           # Industrial blue (steel blue) - sampling
    'Combinational Logic Operators': '#ff7f0e',          # Industrial orange (safety orange) - logic
    'Temporal Liveness Operators': '#9467bd',                    # Industrial purple (deep purple) - properties
    'Miscellaneous Differences': '#7f7f7f',             # Industrial gray (steel gray) - misc
}

# Create figure with three subplots
# Professional layout optimized for academic papers (IEEE double-column width)
fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), facecolor='white')
fig.patch.set_facecolor('white')

# Function to create a publication-quality pie chart
def create_pie(ax, data, title, total_failures):
    # Filter out zero values
    categories = [cat for cat, val in data.items() if val > 0]
    values = [val for val in data.values() if val > 0]
    colors = [category_colors[cat] for cat in categories]
    
    # Calculate percentages for better control
    total = sum(values)
    percentages = [100 * v / total for v in values]
    
    # Slight explode for visual separation (emphasize smaller slices)
    explode = [0.03 if pct < 8 else 0.0 for pct in percentages]
    
    # Create pie chart with professional styling for IEEE
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # We'll use legend instead
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',  # Show percentages >5%
        startangle=140,  # Rotated for better visual balance
        explode=explode,
        textprops={'fontsize': 13, 'weight': 'bold'},
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white', 'antialiased': True},
        pctdistance=0.75,
        labeldistance=1.1
    )
    
    # Style percentage text - high contrast for readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
        autotext.set_path_effects([
            matplotlib.patheffects.withStroke(linewidth=2, foreground='black', alpha=0.3)
        ])
    
    # Enhanced title with sample size
    ax.set_title(f'{title}', 
                fontsize=12, fontweight='bold', pad=15,
                linespacing=1.2)
    
    # Equal aspect ratio ensures circular pie chart
    ax.axis('equal')
    
    # Clean background
    ax.set_facecolor('white')
    
    return wedges

# Create the three pie charts
create_pie(axes[0], human_data, 'NL2SVA-Human', 8)
create_pie(axes[1], machine_data, 'NL2SVA-Machine', 31)
create_pie(axes[2], opencore_data, 'NL2SVA-OpenCore', 54)

# Create a single unified legend for all 6 SVA operator categories
# Ordered by IEEE 1800 SVA taxonomy (Miscellaneous at end)
all_categories = [
    'Temporal Implication Operators',
    'Temporal Delay Operators',
    'Temporal Sampling Operators',
    'Combinational Logic Operators',
    'Temporal Liveness Operators',
    'Miscellaneous Differences'
]

# Create legend handles with professional styling
legend_handles = [matplotlib.patches.Patch(
                    facecolor=category_colors[cat], 
                    edgecolor='white', 
                    linewidth=1.5) 
                 for cat in all_categories]

# Place legend at the bottom center - 2 rows × 3 cols (6 items)
# Enhanced styling for publication quality with larger font
fig.legend(legend_handles, all_categories, 
          loc='lower center', 
          ncol=3, 
          fontsize=12,
          frameon=True,
          fancybox=False,
          shadow=False,
          framealpha=0.95,
          edgecolor='#CCCCCC',
          facecolor='white',
          bbox_to_anchor=(0.5, -0.03),
          columnspacing=2.0,
          handlelength=2.0,
          handleheight=1.5,
          borderpad=0.8,
          labelspacing=0.6)

# Adjust layout for publication-quality figure
# Balanced spacing with room for titles, pies, and legend
plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.22, wspace=0.15)

# Save the figure with IEEE publication-quality settings
output_dir = '/home/scratch.liwan_research/Documents/fv/hardware-agent-marco/src/figures'

# High-resolution PNG (IEEE requires minimum 300 DPI, we use 600 for quality)
output_path_png = f'{output_dir}/operator_level_failures_pie_charts.png'
plt.savefig(output_path_png, dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='png',
            pil_kwargs={'optimize': True})
print(f"IEEE-compliant PNG (600 DPI) saved: {output_path_png}")

# Vector PDF for publication (Type 42 fonts embedded)
output_path_pdf = f'{output_dir}/operator_level_failures_pie_charts.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='pdf',
            metadata={'Creator': 'Matplotlib', 
                     'Title': 'Operator-Level Failure Analysis',
                     'Author': 'IEEE Submission'})
print(f"IEEE-compliant PDF (vector) saved: {output_path_pdf}")

# EPS for IEEE Xplore compatibility (preferred format for IEEE)
output_path_eps = f'{output_dir}/operator_level_failures_pie_charts.eps'
plt.savefig(output_path_eps, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='eps',
            dpi=600)  # DPI for rasterized elements in EPS
print(f"IEEE-compliant EPS saved: {output_path_eps}")

# Show the plot
plt.show()

# Print IEEE-compliant publication summary statistics
print("\n" + "="*80)
print("OPERATOR-LEVEL FAILURE ANALYSIS - IEEE 1800 SVA TAXONOMY")
print("6 Non-Overlapping Categories Based on Refined SVA Operator Classification")
print("PUBLICATION-QUALITY FIGURE - ENHANCED FOR ACADEMIC PAPERS")
print("="*80)
print("Figure dimensions: 10\" × 4.0\" (optimized for IEEE double-column format)")
print("Font sizes: 12-13pt (enhanced for maximum readability in print and digital)")
print("Color scheme: Industrial engineering palette (darker, professional, colorblind-safe)")
print("Visual enhancements: Large percentage labels (13pt), concise formal legend")
print("Output formats: PNG (600 DPI), PDF (vector), EPS (IEEE Xplore preferred)")
print("="*80)

datasets = [
    ("NL2SVA-Human", human_data, 8, "8 cases"),
    ("NL2SVA-Machine", machine_data, 31, "31 failures"),
    ("OpenCore", opencore_data, 54, "54 failures")
]

for dataset_name, data, total_failures, desc in datasets:
    print(f"\n{dataset_name} ({desc}):")
    print("-" * 80)
    total_issues = sum(data.values())
    for cat, count in sorted(data.items(), key=lambda x: x[1], reverse=True):
        if count > 0:  # Only show non-zero categories
            pct = count/total_issues*100
            print(f"  {cat:30s}: {count:3d} issues ({pct:5.1f}%)")

print("\n" + "="*80)
print("KEY FINDINGS - IEEE 1800 SVA OPERATOR TAXONOMY (6 CATEGORIES)")
print("="*80)
print("1. Temporal Implication Operators (|->/|=>/phase/structure): Most critical across all datasets")
print(f"   - Human: {human_data.get('Temporal Implication Operators', 0)} ({human_data.get('Temporal Implication Operators', 0)/sum(human_data.values())*100:.1f}%)")
print(f"   - Machine: {machine_data.get('Temporal Implication Operators', 0)} ({machine_data.get('Temporal Implication Operators', 0)/sum(machine_data.values())*100:.1f}%)")  
print(f"   - OpenCore: {opencore_data.get('Temporal Implication Operators', 0)} ({opencore_data.get('Temporal Implication Operators', 0)/sum(opencore_data.values())*100:.1f}%)")
print()
print("2. Sequence Delay/Repetition (##k/ranges/repetition/indexing): Critical in OpenCore")
print(f"   - Human: {human_data.get('Temporal Delay Operators', 0)} | Machine: {machine_data.get('Temporal Delay Operators', 0)} | OpenCore: {opencore_data.get('Temporal Delay Operators', 0)} ({opencore_data.get('Temporal Delay Operators', 0)/sum(opencore_data.values())*100:.1f}%)")
print()
print("3. Boolean/Equality Operators (boolean structure): Mainly in Machine dataset")
print(f"   - Human: {human_data.get('Combinational Logic Operators', 0)} | Machine: {machine_data.get('Combinational Logic Operators', 0)} ({machine_data.get('Combinational Logic Operators', 0)/sum(machine_data.values())*100:.1f}%) | OpenCore: {opencore_data.get('Combinational Logic Operators', 0)}")
print()
print("4. Temporal Sampling Operators ($past/$fell/$stable): Mainly in Machine dataset")
print(f"   - Human: {human_data.get('Temporal Sampling Operators', 0)} | Machine: {machine_data.get('Temporal Sampling Operators', 0)} ({machine_data.get('Temporal Sampling Operators', 0)/sum(machine_data.values())*100:.1f}%) | OpenCore: {opencore_data.get('Temporal Sampling Operators', 0)}")
print()
print("5. Temporal Liveness Operators (strong/weak/eventually/s_eventually/s_always): In Machine/Human")
print(f"   - Human: {human_data.get('Temporal Liveness Operators', 0)} | Machine: {machine_data.get('Temporal Liveness Operators', 0)} ({machine_data.get('Temporal Liveness Operators', 0)/sum(machine_data.values())*100:.1f}%) | OpenCore: {opencore_data.get('Temporal Liveness Operators', 0)}")
print()
print("6. Miscellaneous Differences (syntax/binding/normalization): Mainly in OpenCore")
print(f"   - Human: {human_data.get('Miscellaneous Differences', 0)} | Machine: {machine_data.get('Miscellaneous Differences', 0)} | OpenCore: {opencore_data.get('Miscellaneous Differences', 0)} ({opencore_data.get('Miscellaneous Differences', 0)/sum(opencore_data.values())*100:.1f}%)")
print("="*80)

