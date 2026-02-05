"""
visualize_gating_weights.py
Visualize heatmaps of gating weights.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def load_gating_data(npz_path):
    """Load saved gating weight data."""
    data = np.load(npz_path, allow_pickle=True)
    
    # Rebuild dictionary structure
    gating_weights = {}
    for key in data.files:
        if key.startswith('gating_weights'):
            # Handle nested dictionary structures
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


def plot_heatmap_by_sample(gating_weights, head_names, layer_idx, 
                            max_samples=500, save_path=None):
    """
    Plot gating weight heatmap for a single layer.
    X-axis: Sample Index
    Y-axis: Different attention heads
    """
    weights = gating_weights[layer_idx]
    
    # Limit the number of samples for better visualization
    if weights.shape[0] > max_samples:
        # Uniform sampling
        indices = np.linspace(0, weights.shape[0]-1, max_samples, dtype=int)
        weights = weights[indices]
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    sns.heatmap(
        weights.T,  # Transpose to put heads on the y-axis
        ax=ax,
        cmap='RdYlBu_r',  # Red-Yellow-Blue palette, high weights in red
        vmin=0, vmax=1,
        yticklabels=head_names,
        cbar_kws={'label': 'Gate Weight'}
    )
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Attention Head')
    ax.set_title(f'Layer {layer_idx} - Adaptive Gating Weights')
    
    # Adjust x-axis ticks
    num_samples = weights.shape[0]
    tick_positions = np.linspace(0, num_samples-1, min(10, num_samples), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_all_layers_heatmap(gating_weights, head_names, max_samples=500, save_path=None):
    """
    Plot gating weight heatmaps for all layers (stacked subplots).
    """
    num_layers = len(gating_weights)
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(16, 3*num_layers))
    
    if num_layers == 1:
        axes = [axes]
    
    for idx, (layer_idx, weights) in enumerate(sorted(gating_weights.items())):
        ax = axes[idx]
        
        # Limit sample count
        if weights.shape[0] > max_samples:
            indices = np.linspace(0, weights.shape[0]-1, max_samples, dtype=int)
            weights_plot = weights[indices]
        else:
            weights_plot = weights
        
        im = ax.imshow(
            weights_plot.T,
            aspect='auto',
            cmap='RdYlBu_r',
            vmin=0, vmax=1,
            interpolation='nearest'
        )
        
        ax.set_yticks(range(len(head_names)))
        ax.set_yticklabels(head_names)
        ax.set_ylabel(f'Layer {layer_idx}')
        
        if idx == num_layers - 1:
            ax.set_xlabel('Sample Index')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Weight' if idx == 0 else '')
    
    plt.suptitle('Adaptive Gating Weights Across All Layers', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_heatmap_by_class(gating_weights, labels, class_names, head_names, 
                          layer_idx, save_path=None):
    """
    Gating weight heatmap grouped by class.
    X-axis: Traffic Class
    Y-axis: Different attention heads
    Displays the average gating weight for each class.
    """
    weights = gating_weights[layer_idx]
    num_classes = len(class_names)
    num_heads = len(head_names)
    
    # Calculate average and standard deviation of weights for each class
    class_avg_weights = np.zeros((num_heads, num_classes))
    class_std_weights = np.zeros((num_heads, num_classes))
    
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_weights = weights[mask]
            class_avg_weights[:, class_idx] = class_weights.mean(axis=0)
            class_std_weights[:, class_idx] = class_weights.std(axis=0)
    
    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.8), 6))
    
    sns.heatmap(
        class_avg_weights,
        ax=ax,
        cmap='RdYlBu_r',
        vmin=0, vmax=1,
        annot=True,
        fmt='.3f',
        xticklabels=class_names,
        yticklabels=head_names,
        cbar_kws={'label': 'Average Gate Weight'}
    )
    for idx, cn in enumerate(class_names):
        print(f"{cn}:", f"{class_avg_weights[:, idx].round(3)}")
    ax.set_xlabel('Traffic Class')
    ax.set_ylabel('Attention Head')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, class_avg_weights


def plot_head_importance_bar(gating_weights, head_names, save_path=None):
    """
    Plot bar chart of average importance for each head across layers.
    """
    num_layers = len(gating_weights)
    num_heads = len(head_names)
    
    # Collect data
    layer_indices = sorted(gating_weights.keys())
    importance_matrix = np.zeros((num_layers, num_heads))
    
    for i, layer_idx in enumerate(layer_indices):
        importance_matrix[i] = gating_weights[layer_idx].mean(axis=0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(num_layers)
    width = 0.15
    
    colors = plt.cm.Set2(np.linspace(0, 1, num_heads))
    
    for head_idx, head_name in enumerate(head_names):
        offset = (head_idx - num_heads/2 + 0.5) * width
        bars = ax.bar(x + offset, importance_matrix[:, head_idx], 
                      width, label=head_name, color=colors[head_idx])
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Average Gate Weight')
    ax.set_title('Head Importance Across Layers')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {i}' for i in layer_indices])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_weight_distribution(gating_weights, head_names, layer_idx, save_path=None):
    """
    Plot the distribution of weights for each head in a single layer (Violin Plot).
    """
    weights = gating_weights[layer_idx]
    num_heads = len(head_names)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    data_to_plot = [weights[:, i] for i in range(num_heads)]
    
    parts = ax.violinplot(data_to_plot, positions=range(num_heads), 
                          showmeans=True, showmedians=True)
    
    # Custom colors
    colors = plt.cm.Set2(np.linspace(0, 1, num_heads))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels(head_names)
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Gate Weight')
    ax.set_title(f'Layer {layer_idx} - Gate Weight Distribution')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def main(args):
    """Main execution function"""
    # Load data
    print(f"Loading data from {args.data_path}...")
    data = load_gating_data(args.data_path)
    
    gating_weights = data['gating_weights']
    labels = data['labels']
    class_names = data['class_names']
    head_names = data['head_names']
    
    print(f"Loaded {data['num_samples']} samples, {data['num_layers']} layers")
    print(f"Classes: {class_names}")
    print(f"Heads: {head_names}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot heatmaps for all layers by sample
    print("\n[1/5] Generating all-layers heatmap...")
    plot_all_layers_heatmap(
        gating_weights, head_names, 
        max_samples=args.max_samples,
        save_path=output_dir / 'heatmap_all_layers.pdf'
    )
    
    # 2. Plot detailed heatmap for each layer
    print("\n[2/5] Generating per-layer heatmaps...")
    for layer_idx in gating_weights.keys():
        plot_heatmap_by_sample(
            gating_weights, head_names, layer_idx,
            max_samples=args.max_samples,
            save_path=output_dir / f'heatmap_layer_{layer_idx}.pdf'
        )
    
    # 3. Plot heatmaps grouped by class
    print("\n[3/5] Generating class-grouped heatmaps...")
    for layer_idx in gating_weights.keys():
        plot_heatmap_by_class(
            gating_weights, labels, class_names, head_names, layer_idx,
            save_path=output_dir / f'heatmap_by_class_layer_{layer_idx}.pdf'
        )
    
    # 4. Plot head importance bar chart
    print("\n[4/5] Generating head importance bar chart...")
    plot_head_importance_bar(
        gating_weights, head_names,
        save_path=output_dir / 'head_importance_bar.pdf'
    )
    
    # 5. Plot weight distribution
    print("\n[5/5] Generating weight distribution plots...")
    for layer_idx in gating_weights.keys():
        plot_weight_distribution(
            gating_weights, head_names, layer_idx,
            save_path=output_dir / f'weight_distribution_layer_{layer_idx}.pdf'
        )
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Show plots (optional)
    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Adaptive Gating Weights')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the .npz file containing gating weights')
    parser.add_argument('--output_dir', type=str, default='./gating_visualizations',
                        help='Output directory for figures')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Maximum samples to show in heatmap')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    args = parser.parse_args()
    main(args)