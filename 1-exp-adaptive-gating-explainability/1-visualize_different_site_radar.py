import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D # For custom legend elements

# --- 0. Global Style Settings (Academic Publication Standards) ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # Ensure PDF fonts are editable
plt.rcParams['ps.fonttype'] = 42

# --- 1. Data Preparation ---
labels = ['Content', 'Temporal', 'Size', 'Direction', 'TCP']
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Color and Marker Configuration
categories = {
    # --- Group 1 (Modified colors for contrast) ---
    "Periodic": {
        "file_suffix": "periodic",
        "color": "#C0392B",  # Pomegranate Red - deep and professional
        "color_light": "#F9EBEA", # Corresponding light background
        "marker": "o",
        "desc": "High Time",
        "sites": [
            {"name": "google",      "data": [0.016, 0.987, 0.909, 0.646, 0.081]},
            {"name": "adblockplus", "data": [0.064, 0.995, 0.936, 0.684, 0.081]},
            {"name": "ubuntu",      "data": [0.028, 0.999, 0.881, 0.834, 0.028]}
        ]
    },
    "Static": {
        "file_suffix": "static",
        "color": "#2980B9",  # Belize Hole Blue - solid presence
        "color_light": "#EAF2F8", 
        "marker": "s",
        "desc": "Size Dominant",
        "sites": [
            {"name": "steamstatic", "data": [0.267, 0.056, 0.866, 0.733, 0.377]},
            {"name": "typekit",     "data": [0.155, 0.073, 0.960, 0.757, 0.323]},
            {"name": "waze",        "data": [0.258, 0.021, 0.959, 0.687, 0.548]}
        ]
    },
    
    # --- Group 2 ---
    "Complex": {
        "file_suffix": "complex",
        "color": "#5C7AEA", 
        "color_light": "#E8ECFC",
        "marker": "^",
        "desc": "Active Mixed",
        "sites": [
            {"name": "cloudfront",  "data": [0.762, 0.022, 0.762, 0.511, 0.653]},
            {"name": "ctfassets",   "data": [0.700, 0.011, 0.777, 0.764, 0.793]},
            {"name": "trustarc",    "data": [0.602, 0.027, 0.677, 0.665, 0.660]}
        ]
    },
    "Sparse": {
        "file_suffix": "micro",
        "color": "#A66DD4", 
        "color_light": "#F3E8FC",
        "marker": "D",
        "desc": "Sparse",
        "sites": [
            {"name": "cookielaw",   "data": [0.131, 0.004, 0.308, 0.158, 0.185]},
            {"name": "garmin",      "data": [0.028, 0.014, 0.210, 0.296, 0.151]},
            {"name": "hsadspixel",  "data": [0.171, 0.135, 0.317, 0.266, 0.203]}
        ]
    }
}

# --- 2. Output Path ---
output_dir = "imgs/different_sites"
os.makedirs(output_dir, exist_ok=True)

# --- 3. Grouping Config ---
cat_items = list(categories.items())
groups = [
    (cat_items[0], cat_items[1]), 
    (cat_items[2], cat_items[3])
]

# --- 4. Plotting Loop ---
for g_idx, group in enumerate(groups):
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    
    # Adjust background (using pure white for academic clarity)
    ax.set_facecolor('white') 
    
    # Set start angle (Content at the top)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # --- Optimization A: Adjust outer circle labels ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=25, fontweight='bold', color='#2C3E50') 
    ax.tick_params(axis='x', pad=10) 

    # --- Optimization B: Adjust radial ticks (0.2, 0.4...) ---
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # Offset labels to 25 degrees to avoid overlap with vertical axis
    ax.set_rlabel_position(25) 
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                        size=18, color='#95A5A6', fontweight='normal')

    # --- Optimization C: Grid lines (Mimic Figure 12 style) ---
    ax.yaxis.grid(True, linestyle=(0, (5, 5)), linewidth=1.0, alpha=0.4, color='#BDC3C7')
    ax.xaxis.grid(True, linestyle=(0, (5, 5)), linewidth=1.0, alpha=0.4, color='#BDC3C7')
    
    # Set polar spine
    ax.spines['polar'].set_color('#BDC3C7')
    ax.spines['polar'].set_linewidth(1.0)
    ax.spines['polar'].set_linestyle('--')

    # ==============================================================================
    # --- New: Add category text annotations ---
    # Manually position labels based on cluster characteristics
    bbox_args = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=4) 

    if g_idx == 0: # Comparison 1: Periodic vs. Static
        # 1. Periodic: Cluster towards Content/Temporal (Top-Right quadrant)
        ax.text(np.deg2rad(100), 0.5, group[0][0], 
                color=group[0][1]['color'], fontsize=25, fontweight='bold', 
                ha='center', va='center', zorder=10, bbox=bbox_args)

        # 2. Static: Cluster towards Size/Direction/TCP (Bottom quadrants)
        ax.text(np.deg2rad(250), 0.5, group[1][0], 
                color=group[1][1]['color'], fontsize=25, fontweight='bold', 
                ha='center', va='center', zorder=10, bbox=bbox_args)

    elif g_idx == 1: # Comparison 2: Complex vs. Sparse
        # 3. Complex: Overall high values (Outer circles)
        ax.text(np.deg2rad(315), 0.42, group[0][0], 
                color=group[0][1]['color'], fontsize=25, fontweight='bold', 
                ha='center', va='center', zorder=10, bbox=bbox_args)

        # 4. Sparse: Overall low values (Inner circles)
        ax.text(np.deg2rad(90), 0.4, group[1][0], 
                color=group[1][1]['color'], fontsize=25, fontweight='bold', 
                ha='center', va='center', zorder=10, bbox=bbox_args)
    # ==============================================================================
    
    # Plot data
    line_styles = ['-', '--', ':'] 
    legend_elements = []

    for cat_idx, (cat_name, cat_data) in enumerate(group):
        base_color = cat_data["color"]
        marker = cat_data["marker"]
        
        for site_idx, site in enumerate(cat_data["sites"]):
            values = site["data"].copy()
            values += values[:1]
            
            # Plot lines
            ax.plot(angles, values, 
                   linewidth=2.0, 
                   linestyle=line_styles[site_idx],
                   color=base_color, 
                   alpha=0.9,
                   label=f'{site["name"]}',
                   zorder=3)
            
            # Plot markers
            ax.scatter(angles[:-1], values[:-1], 
                      color=base_color, 
                      s=60, 
                      marker=marker,
                      edgecolors='white',
                      linewidths=1.5,
                      alpha=1.0,
                      zorder=4)
            
            # Fill area (low alpha to avoid visual clutter)
            ax.fill(angles, values, 
                   color=base_color, 
                   alpha=0.05, 
                   zorder=1)
            
            # Collect legend info
            legend_elements.append(
                Line2D([0], [0], 
                       color=base_color, 
                       linewidth=2.0,
                       linestyle=line_styles[site_idx],
                       marker=marker,
                       markersize=9,
                       markerfacecolor=base_color,
                       markeredgecolor='white',
                       markeredgewidth=1.2,
                       label=site["name"]) 
            )
    
    # --- Optimization D: Legend Layout ---
    legend = ax.legend(handles=legend_elements,
                       loc='upper center', 
                       bbox_to_anchor=(0.5, -0.10), 
                       frameon=False,
                       fancybox=False, 
                       shadow=False,
                       ncol=3, 
                       fontsize=22,
                       columnspacing=1,
                       handletextpad=0.6,
                       labelspacing=0.5,
                       edgecolor='#E0E0E0',
                       facecolor='white')
    
    legend.get_frame().set_linewidth(0.8)

    # Save
    suffix1 = group[0][1]['file_suffix']
    suffix2 = group[1][1]['file_suffix']
    filename = f"overlay_comparison_{g_idx+1}_{suffix1}_vs_{suffix2}.pdf"
    save_path = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9) 
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    
    plt.close(fig)

print("\n✅ Overlay comparison figures generated successfully!")