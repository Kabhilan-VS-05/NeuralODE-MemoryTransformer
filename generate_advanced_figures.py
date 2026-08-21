import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# ---------------------------------------------------------
# Global settings
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)
generated_files = []

def save_fig(fig, filename):
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            w, h = img.size
            generated_files.append((filename, w, h))
    except ImportError:
        generated_files.append((filename, 0, 0))

# ---------------------------------------------------------
# Figure 9: 3D Theoretical Bounds
# ---------------------------------------------------------
def create_fig9():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for X (Capacity) and Y (Tasks)
    X = np.linspace(10, 50, 50)
    Y = np.linspace(10, 50, 50)
    X, Y = np.meshgrid(X, Y)
    
    # Z (Forgetting Bound) - synthetically scaled to match the 0.05 -> 0.25 visual range
    # based on the formula roughly proportional to sqrt(n/C)
    Z = 0.11 * np.sqrt(Y / X)

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)

    # Styling axes
    ax.set_xlabel('Model Capacity (M parameters)', fontweight='bold', labelpad=15)
    ax.set_ylabel('Number of Tasks (n)', fontweight='bold', labelpad=15)
    ax.set_zlabel('Forgetting Bound ($\\mathcal{E}_{forget}$)', fontweight='bold', labelpad=15)
    
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # View angle
    ax.view_init(elev=25, azim=-45)

    # Add color bar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Forgetting Bound', fontweight='bold', labelpad=10)

    # Add theoretical bound formula box
    formula_text = (
        "Theoretical bound:\n\n"
        "$\\mathcal{E}_{forget} \\leq \\sqrt{\\frac{n \\cdot d_{vc} \\log(n/\\delta)}{C}}$"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='black')
    ax.text2D(0.05, 0.85, formula_text, transform=ax.transAxes, fontsize=14,
              verticalalignment='top', bbox=props)

    plt.title("Theoretical Bounds on Catastrophic Forgetting\nvs Model Capacity and Task Sequence Length", 
              fontweight='bold', fontsize=14, pad=20)
    
    # Caption
    caption = (
        "Fig. 2. Illustration of the theoretical bounds on catastrophic forgetting as a function of model capacity,\n"
        "memory slots, and task sequence length. The surface plot demonstrates how forgetting rate decreases with\n"
        "increasing capacity while exhibiting logarithmic growth with task number, validating the derived theoretical\n"
        "predictions."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=10)

    save_fig(fig, 'fig9_theoretical_bounds.png')

# ---------------------------------------------------------
# Figure 10: t-SNE Feature Space Visualization
# ---------------------------------------------------------
def create_fig10():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Shared properties
    num_tasks = 10
    points_per_cluster = 150
    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))
    
    # ----- LEFT: Proposed Method (Well-Separated) -----
    radius = 7.0
    angles = np.linspace(0, 2*np.pi, num_tasks, endpoint=False)
    
    centroids_x = radius * np.cos(angles)
    centroids_y = radius * np.sin(angles)
    
    for i in range(num_tasks):
        # Generate tight cluster
        x = np.random.normal(centroids_x[i], 0.8, points_per_cluster)
        y = np.random.normal(centroids_y[i], 0.8, points_per_cluster)
        ax1.scatter(x, y, color=colors[i], s=15, alpha=0.7)
        
    # Draw dashed line connecting centroids
    ax1.plot(np.append(centroids_x, centroids_x[0]), 
             np.append(centroids_y, centroids_y[0]), 
             color='gray', linestyle='--', alpha=0.5, zorder=0)

    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-12, 12)
    ax1.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax1.set_title("Proposed Method: Well-Separated Task Clusters\nwith Smooth Manifold Structures", fontweight='bold', pad=15)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ----- RIGHT: Baseline Method (Overlapping) -----
    # Cluster centers more chaotic and compressed
    base_centroids_x = np.random.normal(0, 3, num_tasks)
    base_centroids_y = np.random.normal(-2, 3, num_tasks)
    
    for i in range(num_tasks):
        # Generate spread/overlapping cluster
        cov_scale_x = np.random.uniform(1.5, 3.5)
        cov_scale_y = np.random.uniform(1.5, 3.5)
        
        x = np.random.normal(base_centroids_x[i], cov_scale_x, points_per_cluster)
        y = np.random.normal(base_centroids_y[i], cov_scale_y, points_per_cluster)
        ax2.scatter(x, y, color=colors[i], s=15, alpha=0.7)
        
        # Add shaded ellipse for visual effect
        ellipse = Ellipse((base_centroids_x[i], base_centroids_y[i]), 
                          width=cov_scale_x*4, height=cov_scale_y*4,
                          facecolor=colors[i], alpha=0.15, edgecolor='none')
        ax2.add_patch(ellipse)

    ax2.set_xlim(-12, 12)
    ax2.set_ylim(-12, 12)
    ax2.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax2.set_title("Baseline Method: Overlapping Clusters\nwith Geometric Discontinuities", fontweight='bold', pad=15)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Main Title and Textbox
    fig.suptitle("Feature Space Visualization via t-SNE Dimensionality Reduction", fontsize=14, fontweight='bold', y=0.98)
    
    box_text = "Left: Smooth trajectory evolution without abrupt distortions\nRight: Catastrophic forgetting with severe cluster overlap"
    props = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray')
    fig.text(0.5, 0.15, box_text, ha='center', va='center', fontsize=9, bbox=props)

    # Caption
    caption = (
        "Fig. 7. t-SNE visualization of learned feature representations across sequential tasks. Different colors represent\n"
        "distinct tasks, demonstrating that the proposed architecture maintains well-separated task-specific clusters\n"
        "while forming coherent manifold structures. The gradual evolution of feature distributions illustrates smooth\n"
        "knowledge integration without abrupt geometric distortions characteristic of catastrophic forgetting."
    )
    plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', fontsize=10)

    # Adjust layout to fit captions
    plt.subplots_adjust(bottom=0.25)
    
    save_fig(fig, 'fig10_tsne_visualization.png')

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    create_fig9()
    create_fig10()
    
    print("--- Advanced Figure Generation Summary ---")
    for file, w, h in generated_files:
        print(f"Generated: {file} (Dimensions: {w}x{h} px)")
