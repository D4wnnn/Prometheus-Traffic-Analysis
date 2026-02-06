import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # For custom colormaps
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
    gating_weights = {}
    for key in data.files:
        if key.startswith('gating_weights'):
            gating_dict = data[key].item()
            for layer_idx, weights in gating_dict.items():
                gating_weights[layer_idx] = weights
            break
    return {
        'gating_weights': gating_weights,
        'head_names': list(data['head_names'])
    }

def plot_single_layer_heatmap(layer_idx, weights, head_names, sample_indices, save_path=None):
    """
    Plot single-layer heatmap (using custom academic-style color palette).
    Args:
        weights: Array of shape (Num_Selected_Samples, Num_Heads)
        sample_indices: List of real indices for selected samples (used for X-axis labels)
    """
    # 1. Prepare data: Transpose -> (Heads, Samples)
    data_to_plot = weights.T
    
    # 2. Dynamically adjust canvas size
    num_heads, num_samples = data_to_plot.shape
    fig_width = max(8, num_samples * 0.9) 
    fig_height = max(4, num_heads * 1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ================= 3. Define Custom Colormap =================
    # Hex codes extracted from paper bar charts:
    # NetMamba (Yellow) -> YaTC (Red) -> ET-BERT (Purple) -> Ours (Dark Blue)
    # This gradient ensures low weights are light (Yellow) and high weights are dark (Ours Blue)
    paper_colors = [
        "#FAD06F",  # 0.0 - Yellow (NetMamba)
        "#E5604B",  # 0.33 - Red (YaTC)
        "#836085",  # 0.66 - Purple (ET-BERT)
        "#24376E"   # 1.0 - Dark Blue (Ours)
    ]
    
    # Create linear segmented colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("PaperStyle", paper_colors)
    # ===================================================================

    # 4. Plot Heatmap
    hm = sns.heatmap(
        data_to_plot,
        ax=ax,
        annot=True,            # Enable text annotations
        fmt=".2f",              # Two decimal places
        cmap=custom_cmap,      # Use custom cmap
        vmin=0, vmax=1,        # Fixed range 0-1
        cbar=True,
        linewidths=0.5,        # Grid borders
        linecolor='white',     # White borders for better clarity against dark blocks
        annot_kws={"size": 22, "fontfamily": "Times New Roman"}, 
    )
    
    # Configure Colorbar
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    # 5. Set Axis Labels
    # Y-axis: Head Names
    if head_names and len(head_names) == num_heads:
        display_heads = head_names
    else:
        display_heads = ['Con','Time','Size','Dir','TCP']
        
    ax.set_yticklabels(display_heads, rotation=0, fontsize=25, fontname='Times New Roman')
    
    # X-axis: Real Sample Index
    ax.set_xlabel("Sample Index", fontsize=25, fontname='Times New Roman')
    
    # Directly use passed sample_indices as labels to display real IDs
    ax.set_xticklabels(sample_indices, fontsize=25, fontname='Times New Roman')

    # 6. Save
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Layer {layer_idx} heatmap to {save_path}")
        plt.close(fig)

def main(args):
    print(f"Loading data from {args.data_path}...")
    data = load_gating_data(args.data_path)
    
    gating_weights = data['gating_weights']
    head_names = data['head_names'] 
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_layers = sorted(gating_weights.items())
    
    # ================= Global Uniform Random Sampling =================
    # Get total sample count
    first_layer_weights = sorted_layers[0][1]
    total_samples = first_layer_weights.shape[0]
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Generate random indices
    if total_samples > args.num_vis_samples:
        selected_indices = np.random.choice(total_samples, args.num_vis_samples, replace=False)
        selected_indices = np.sort(selected_indices)
    else:
        selected_indices = np.arange(total_samples)
        
    print(f"Selected Sample Indices: {selected_indices}")
    # ===================================================================

    for layer_idx, weights in sorted_layers:
        # Extract data using selected indices
        weights_subset = weights[selected_indices]
        
        save_name = output_dir / f'layer_{layer_idx}_heatmap.pdf'
        
        plot_single_layer_heatmap(
            layer_idx=layer_idx,
            weights=weights_subset,
            head_names=head_names,
            sample_indices=selected_indices, # Pass real indices
            save_path=save_name
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Adaptive Gating Weights (Paper Style)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the .npz file containing gating weights')
    parser.add_argument('--output_dir', type=str, default='./gating_vis_paper_style',
                        help='Output directory for figures')
    parser.add_argument('--num_vis_samples', type=int, default=15,
                        help='Number of random samples to visualize')
    parser.add_argument('--seed', type=int, default=1037,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)