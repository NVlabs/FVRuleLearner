import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# Create the dataset with the correct interpretation
data = [
    (1, 1, 0.6677, 1.0000, 0.6000, 0.6000),
    (1, 2, 0.6754, 1.0000, 0.5333, 0.6000),
    (1, 3, 0.6555, 1.0000, 0.6000, 0.6000),
    (1, 4, 0.6595, 1.0000, 0.6000, 0.6000),
    (1, 5, 0.6749, 1.0000, 0.6667, 0.6667),
    (2, 1, 0.6753, 1.0000, 0.7333, 0.7333),
    (2, 2, 0.6631, 1.0000, 0.6667, 0.6667),
    (2, 3, 0.6685, 1.0000, 0.7333, 0.7333),
    (2, 4, 0.6483, 1.0000, 0.6667, 0.7333),
    (2, 5, 0.6796, 1.0000, 0.6000, 0.6667),
    (3, 1, 0.6844, 1.0000, 0.7333, 0.7333),
    (3, 2, 0.6798, 1.0000, 0.6000, 0.6667),
    (3, 3, 0.6787, 1.0000, 0.6667, 0.6667),
    (3, 4, 0.6500, 1.0000, 0.6000, 0.6000),
    (3, 5, 0.6678, 1.0000, 0.6667, 0.6667),
    (4, 1, 0.6794, 1.0000, 0.6000, 0.6667),
    (4, 2, 0.6652, 1.0000, 0.6667, 0.7333),
    (4, 3, 0.7010, 1.0000, 0.6667, 0.7333),
    (4, 4, 0.6844, 1.0000, 0.6000, 0.6667),
    (4, 5, 0.6985, 1.0000, 0.6667, 0.7333),
    (5, 1, 0.6802, 1.0000, 0.6000, 0.6667),
    (5, 2, 0.6754, 1.0000, 0.6000, 0.6667),
    (5, 3, 0.6957, 1.0000, 0.6667, 0.7333),
    (5, 4, 0.7001, 1.0000, 0.7333, 0.7333),
    (5, 5, 0.6988, 1.0000, 0.6667, 0.7333)
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['K', 'K_prime', 'BLEU', 'Syntax', 'Func', 'R.Func'])

# Specify the save path
save_path = '/home/scratch.liwan_mobile/repo/fv/hardware-agent-marco/src/figures'

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

# Set up the plot style
try:
    plt.style.use('ggplot')
except Exception as e:
    print(f"Warning: Could not use 'ggplot' style. Using default style. Error: {e}")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create a single figure with four heatmap subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
fig.suptitle('Bird\'s-Eye View of Metrics Across K and K\'', fontsize=24, y=0.95)

metrics = ['BLEU', 'Syntax', 'Func', 'R.Func']
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
color_maps = ['viridis', 'plasma', 'inferno', 'magma']

for (metric, label, cmap), ax in zip(zip(metrics, subplot_labels, color_maps), axs.ravel()):
    # Pivot the data for heatmap
    pivot_data = df.pivot(index='K_prime', columns='K', values=metric)
    
    # Create the heatmap
    sns.heatmap(pivot_data, ax=ax, cmap=cmap, annot=True, fmt='.4f', cbar_kws={'label': f'{metric} Score'})
    
    # Customize the plot
    ax.set_title(f'{label} {metric} Score', fontsize=18, pad=20)
    ax.set_xlabel('K (Examples Size)', fontsize=12)
    ax.set_ylabel('K\' (Suggestions Size)', fontsize=12)
    
    # Rotate the tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()

# Save the figure as PDF
file_path = os.path.join(save_path, 'birds_eye_metrics_visualization.pdf')
with PdfPages(file_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
plt.close()

print(f"Saved bird's-eye view metrics visualization figure to {file_path}")