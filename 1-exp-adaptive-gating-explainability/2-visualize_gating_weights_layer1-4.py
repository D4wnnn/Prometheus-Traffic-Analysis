"""
visualize_gating_weights.py
Visualize gating weight heatmaps and distribution (Conference Styling).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# ================= 1. Global Font Settings (Academic Style) =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
# ============================================================================

def load_gating_data(npz_path):
    """Load saved gating weight data."""
    data = np.load(npz_path, allow_pickle=True)
    
    # Rebuild dictionary structure
    gating_weights = {}
    for key in data.files:
        if key.startswith('gating_weights'):
            gating_dict = data[key].item()
            for layer_idx, weights in gating_dict.items():
                gating_weights[layer_idx] = weights
            break
    
    return {
        'gating_weights': gating_weights,
        'labels': data['labels'],
        'predictions': data['predictions'],
        'class_names': list(data['class_names']),
        'head_names': list(data['head_names']),
        'num_layers': int(data['num_layers']),
        'num_samples': int(data['num_samples']),
        'metrics': data['metrics'].item()
    }

def plot_weight_distribution(gating_weights, head_names, layer_idx, save_path=None):
    """
    Plot head weight distribution for a single layer (Violin Plot - Conference Style).
    """
    weights = gating_weights[layer_idx]
    num_heads = len(head_names)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data: list of arrays
    data_to_plot = [weights[:, i] for i in range(num_heads)]
    
    # Plot violin plots
    parts = ax.violinplot(data_to_plot, positions=range(num_heads), 
                          showmeans=True, showmedians=True)
    
    # ================= 2. Custom Color Scheme =================
    # Core colors extracted from bar charts (visual weight order)
    paper_colors = [
        "#FAD06F",  # Yellow (NetMamba style)
        "#E5604B",  # Red (YaTC style)
        "#836085",  # Purple (ET-BERT style)
        "#1D6BAC",  # Medium Blue (TrafficFormer style)
        "#24376E"   # Dark Blue (Ours style)
    ]
    
    # Cycle colors if heads exceed 5
    colors = paper_colors * (num_heads // len(paper_colors) + 1)
    
    # Set violin body colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(colors[i]) 
        pc.set_alpha(0.8)           
    
    # Professional styling for statistical lines
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('#404040') # Dark grey lines
            vp.set_linewidth(1.2)
    # =========================================================
    
    # Set axis configuration
    ax.set_xticks(range(num_heads))
    if not head_names:
        head_names = [f'Head {i}' for i in range(num_heads)]
        
    ax.set_xticklabels(head_names, fontsize=25, fontname='Times New Roman')
    ax.set_xlabel('Attention Head', fontsize=25, fontname='Times New Roman')
    ax.set_ylabel('Gate Weight', fontsize=25, fontname='Times New Roman')
    
    # Tick styling
    ax.tick_params(axis='y', labelsize=25)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        
    ax.set_ylim(0, 1.1) 
    
    # Add grid lines for readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close(fig) 
    
    return fig


def main(args):
    """Main execution function"""
    print(f"Loading data from {args.data_path}...")
    data = load_gating_data(args.data_path)
    
    gating_weights = data['gating_weights']
    labels = data['labels']
    class_names = data['class_names']
    head_names = data['head_names']
    
    print(f"Loaded {data['num_samples']} samples, {data['num_layers']} layers")
    print(f"Classes: {class_names}")
    print(f"Heads: {head_names}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate distribution plots
    print("\nGenerating weight distribution plots with paper styling...")
    for layer_idx in gating_weights.keys():
        plot_weight_distribution(
            gating_weights, head_names, layer_idx,
            save_path=output_dir / f'weight_distribution_layer_{layer_idx}.pdf'
        )
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Adaptive Gating Weights (Violin)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the .npz file containing gating weights')
    parser.add_argument('--output_dir', type=str, default='./gating_vis_violin',
                        help='Output directory for figures')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Maximum samples to show (unused in violin but kept for compat)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    args = parser.parse_args()
    main(args)