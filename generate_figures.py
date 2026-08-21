import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi

# ---------------------------------------------------------
# Global settings
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

COLORS = {
    "None": "#9AA5B1",
    "Random": "#4F8A5B",
    "Influence": "#D4A017",
    "Fisher": "#5B84C4",
    "Fisher_Proto": "#2E4A7D",
    "Forgetting": "#C0504D"
}

out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)
generated_files = []

def save_fig(fig, filename):
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Get image dimensions for reporting
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            w, h = img.size
            generated_files.append((filename, w, h))
    except ImportError:
        generated_files.append((filename, 0, 0))

# ---------------------------------------------------------
# Figure 1: Base paper's reported results
# ---------------------------------------------------------
def create_fig1():
    methods = ["Fine-tuning\n(no protection)", "EWC", "PackNet", "GEM", "A-GEM", "Base paper\nproposed", "Upper bound\n(joint training)"]
    acc = [41.2, 58.7, 62.3, 65.8, 63.4, 72.6, 78.4]
    forg = [0.487, 0.312, 0.278, 0.241, 0.259, 0.183, 0.000]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.6

    bars = ax1.bar(x, acc, width, color='#5B84C4', alpha=0.8, label="Accuracy (%)")
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha='right')

    ax2 = ax1.twinx()
    line = ax2.plot(x, forg, color=COLORS["Forgetting"], marker='o', linewidth=2, markersize=8, label="Forgetting")
    ax2.set_ylabel('Forgetting Metric')
    ax2.set_ylim(0, 0.6)

    ax1.set_title("Base Paper Reported Results (Split CIFAR-100, 10 Tasks)")
    
    # Add labels
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)
    for i, txt in enumerate(forg):
        ax2.text(x[i], txt + 0.02, f'{txt:.3f}', ha='center', va='bottom', fontsize=9, color=COLORS["Forgetting"])

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    save_fig(fig, 'fig1_base_paper.png')

# ---------------------------------------------------------
# Figure 2: Effect of backbone and eval protocol
# ---------------------------------------------------------
def create_fig2():
    configs = [
        "CNN + multi-head\n(baseline)", 
        "DINOv2 + multi-head\n(task ID known)", 
        "DINOv2 + shared head\n(task ID unknown)"
    ]
    acc = [40.16, 94.16, 10.76]
    forg = [0.456, 0.035, 0.963]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(configs))
    
    # Accuracy
    bars1 = ax1.bar(x, acc, 0.5, color='#5B84C4', alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, ha='right')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 2, f'{bar.get_height():.2f}%', ha='center')

    # Forgetting
    bars2 = ax2.bar(x, forg, 0.5, color=COLORS["Forgetting"], alpha=0.8)
    ax2.set_ylabel('Forgetting Metric')
    ax2.set_title('Forgetting')
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, ha='right')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.03, f'{bar.get_height():.3f}', ha='center')

    fig.suptitle("Effect of Backbone and Evaluation Protocol")
    plt.tight_layout()
    save_fig(fig, 'fig2_backbone_effect.png')

# ---------------------------------------------------------
# Figure 3: Memory scoring method comparison
# ---------------------------------------------------------
def create_fig3():
    methods = [
        "None", 
        "Fisher\n(plain)", 
        "Influence\n(base paper)", 
        "Fisher+Proto\n(0.5/0.5)", 
        "Fisher+Proto\n(0.0/1.0)", 
        "Fisher+Proto\n(0.3/0.7)", 
        "Random"
    ]
    acc = [10.60, 52.74, 54.04, 57.30, 58.07, 58.45, 61.67]
    forg = [0.962, 0.487, 0.473, 0.439, 0.432, 0.428, 0.391]
    
    bar_colors = [
        COLORS["None"], COLORS["Fisher"], COLORS["Influence"], 
        COLORS["Fisher_Proto"], COLORS["Fisher_Proto"], COLORS["Fisher_Proto"], COLORS["Random"]
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))

    bars = ax1.bar(x, acc, 0.6, color=bar_colors, alpha=0.9)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 75)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha='right')

    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1, f'{bar.get_height():.2f}%', ha='center', fontsize=9)

    ax2 = ax1.twinx()
    line = ax2.plot(x, forg, color=COLORS["Forgetting"], marker='o', linewidth=2, markersize=8, label="Forgetting")
    ax2.set_ylabel('Forgetting Metric')
    ax2.set_ylim(0, 1.1)
    
    for i, txt in enumerate(forg):
        ax2.text(x[i], txt + 0.04, f'{txt:.3f}', ha='center', va='bottom', fontsize=9, color=COLORS["Forgetting"])

    ax1.set_title("Memory Scoring Method Comparison (Offline, Shared Head)")
    save_fig(fig, 'fig3_scoring_comparison.png')

# ---------------------------------------------------------
# Figure 4: Per-task accuracy after full sequence
# ---------------------------------------------------------
def create_fig4():
    tasks = np.arange(10)
    
    data = {
        "Influence (base paper)": [46.10, 51.70, 56.60, 35.90, 43.10, 47.20, 39.30, 50.00, 72.20, 98.30],
        "Fisher + Prototype (0.3/0.7)": [56.90, 55.00, 58.80, 45.80, 48.30, 49.40, 53.70, 51.40, 66.10, 99.10],
        "Random": [63.70, 56.40, 64.50, 50.20, 54.00, 51.00, 49.50, 55.60, 73.10, 98.70],
        "None": [0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.80, 1.80, 4.60, 98.70]
    }
    
    colors_map = {
        "Influence (base paper)": COLORS["Influence"],
        "Fisher + Prototype (0.3/0.7)": COLORS["Fisher_Proto"],
        "Random": COLORS["Random"],
        "None": COLORS["None"]
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    for i, (name, values) in enumerate(data.items()):
        bars = ax.bar(tasks + offsets[i], values, width, label=name, color=colors_map[name])

    ax.set_xlabel('Task Index')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Task Accuracy After Full 10-Task Sequence')
    ax.set_xticks(tasks)
    ax.set_xticklabels([str(t) for t in tasks])
    ax.set_ylim(0, 110)
    
    # Legend outside
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    save_fig(fig, 'fig4_per_task.png')

# ---------------------------------------------------------
# Figure 5: Streaming evaluation
# ---------------------------------------------------------
def create_fig5():
    progress = np.arange(10, 110, 10)
    
    none_acc = [18.57, 25.18, 29.17, 31.88, 34.20, 36.40, 38.96, 39.04, 36.24, 25.47]
    random_acc = [18.76, 26.49, 34.13, 39.79, 47.09, 53.20, 57.55, 63.14, 66.04, 64.00]
    fisher_acc = [18.73, 26.26, 33.82, 38.96, 45.43, 52.05, 57.84, 63.15, 65.92, 64.22]

    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(progress, none_acc, color=COLORS["None"], marker='o', linewidth=2, label="None")
    ax.plot(progress, random_acc, color=COLORS["Random"], marker='s', linewidth=2, label="Random")
    ax.plot(progress, fisher_acc, color=COLORS["Fisher_Proto"], marker='^', linewidth=2, label="Fisher + Prototype (0.3/0.7)")

    ax.set_xlabel('Stream Progress (%)')
    ax.set_ylabel('Global Accuracy (%)')
    ax.set_title('Streaming Evaluation (Global Accuracy Over Continuous Stream)')
    ax.set_xticks(progress)
    ax.set_ylim(0, 75)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add runtime annotation
    runtime_text = (
        "Wall-clock runtimes:\n"
        "None: 3 min 15 s\n"
        "Random: 2 min 49 s\n"
        "Fisher+Prototype: 31 min 22 s"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.95, runtime_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    save_fig(fig, 'fig5_streaming_curve.png')

# ---------------------------------------------------------
# Figure 6: Architecture Block Diagram
# ---------------------------------------------------------
def create_fig6():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    boxes = [
        {"text": "Cached DINOv2\nFeatures\n(384-d)", "xy": (0.05, 0.4), "color": "#D0E4F5", "width": 0.16, "height": 0.3},
        {"text": "Linear\nProjection\n(384->512)", "xy": (0.25, 0.4), "color": "#D0E4F5", "width": 0.16, "height": 0.3},
        {"text": "Neural ODE Layer\n(Dormand-Prince)", "xy": (0.45, 0.4), "color": "#D0E4F5", "width": 0.16, "height": 0.3},
        {"text": "Memory Module\n[500-slot buffer,\nFisher+Proto scoring\nfresh optimizer/task]", "xy": (0.68, 0.4), "color": "#D1E8D5", "width": 0.18, "height": 0.3},
        {"text": "Transformer\nAttention\n(6 layers / 8 heads)", "xy": (0.91, 0.4), "color": "#FEE6CE", "width": 0.16, "height": 0.3},
        {"text": "Multi-Head Classifier /\nShared Head\n(Output)", "xy": (0.91, 0.8), "color": "#FEE6CE", "width": 0.16, "height": 0.3}
    ]

    for b in boxes:
        box = patches.FancyBboxPatch(
            (b["xy"][0] - b["width"]/2, b["xy"][1] - b["height"]/2), b["width"], b["height"],
            boxstyle="round,pad=0.02",
            fc=b["color"], ec="black", lw=1.5
        )
        ax.add_patch(box)
        ax.text(b["xy"][0], b["xy"][1], b["text"], ha='center', va='center', fontsize=9)

    # Arrows
    arrow_props = dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, headlength=10)
    
    ax.annotate("", xy=(0.17, 0.4), xytext=(0.13, 0.4), arrowprops=arrow_props)
    ax.annotate("", xy=(0.37, 0.4), xytext=(0.33, 0.4), arrowprops=arrow_props)
    ax.annotate("", xy=(0.59, 0.4), xytext=(0.53, 0.4), arrowprops=arrow_props)
    ax.annotate("", xy=(0.83, 0.4), xytext=(0.77, 0.4), arrowprops=arrow_props)
    ax.annotate("", xy=(0.91, 0.65), xytext=(0.91, 0.55), arrowprops=arrow_props)

    ax.set_title("Architecture Block Diagram", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig(fig, 'fig6_architecture.png')

# ---------------------------------------------------------
# Figure 7: Experimental Pipeline Flowchart
# ---------------------------------------------------------
def create_fig7():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    stages = [
        {"title": "Data Preprocessing", "color": "#2E4A7D", "text_color": "white", 
         "subs": ["Split CIFAR-100", "DINOv2 feature caching\n(vits14)", "MD5-verified download"], "x": 0.125},
        
        {"title": "Sequential Task\nTraining", "color": "#4F8A5B", "text_color": "white", 
         "subs": ["Fresh optimizer per task", "Decoupled read-write", "10 tasks x 30 epochs"], "x": 0.375},
        
        {"title": "Scoring Method\nComparison", "color": "#D4A017", "text_color": "black", 
         "subs": ["None", "Random / Influence", "Fisher / Fisher+Proto"], "x": 0.625},
        
        {"title": "Streaming +\nMulti-Metric Eval", "color": "#6B5B95", "text_color": "white", 
         "subs": ["Gaussian sliding-window stream", "Accuracy / Forgetting", "Backward Transfer"], "x": 0.875}
    ]

    for s in stages:
        # Header box
        header = patches.FancyBboxPatch(
            (s["x"] - 0.1, 0.75), 0.2, 0.15,
            boxstyle="square,pad=0",
            fc=s["color"], ec="black", lw=1.5
        )
        ax.add_patch(header)
        ax.text(s["x"], 0.825, s["title"], ha='center', va='center', color=s["text_color"], fontweight='bold', fontsize=10)

        # Sub boxes
        for i, sub in enumerate(s["subs"]):
            y_pos = 0.55 - (i * 0.18)
            sub_box = patches.FancyBboxPatch(
                (s["x"] - 0.1, y_pos), 0.2, 0.10,
                boxstyle="round,pad=0.02",
                fc="#F0F0F0", ec="black", lw=1.0
            )
            ax.add_patch(sub_box)
            ax.text(s["x"], y_pos + 0.05, sub, ha='center', va='center', fontsize=9)
            
            # Draw line from header down to sub boxes
            if i == 0:
                ax.plot([s["x"], s["x"]], [0.75, y_pos+0.10], color='black', lw=1.5)
            else:
                prev_y = 0.55 - ((i-1) * 0.18)
                ax.plot([s["x"], s["x"]], [prev_y, y_pos+0.10], color='black', lw=1.5)

    # Arrows between main headers
    arrow_props = dict(facecolor='black', edgecolor='black', width=2, headwidth=10, headlength=10)
    for i in range(3):
        start_x = stages[i]["x"] + 0.1
        end_x = stages[i+1]["x"] - 0.1
        ax.annotate("", xy=(end_x, 0.825), xytext=(start_x, 0.825), arrowprops=arrow_props)

    ax.set_title("Experimental Pipeline Flowchart", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig(fig, 'fig7_pipeline.png')

# ---------------------------------------------------------
# Figure 8: Radar chart
# ---------------------------------------------------------
def create_fig8():
    categories = ['Accuracy', 'Forgetting Mitigation', 'Transfer Stability']
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # close the loop

    data = {
        "None": [10.60, 3.80, 3.77],
        "Influence (base paper)": [54.04, 52.70, 52.73],
        "Fisher (plain)": [52.74, 51.30, 51.26],
        "Fisher + Prototype (0.3/0.7)": [58.45, 57.20, 57.21],
        "Random": [61.67, 60.90, 60.86]
    }
    
    color_map = {
        "None": COLORS["None"],
        "Influence (base paper)": COLORS["Influence"],
        "Fisher (plain)": COLORS["Fisher"],
        "Fisher + Prototype (0.3/0.7)": COLORS["Fisher_Proto"],
        "Random": COLORS["Random"]
    }

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, size=10)
    
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60], ["20", "40", "60"], color="grey", size=8)
    plt.ylim(0, 70)

    for name, values in data.items():
        vals = values + values[:1]
        ax.plot(angles, vals, linewidth=2, linestyle='solid', label=name, color=color_map[name])
        ax.fill(angles, vals, color=color_map[name], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Multi-Metric Offline Scoring Comparison", pad=20)
    
    save_fig(fig, 'fig8_radar.png')

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    create_fig1()
    create_fig2()
    create_fig3()
    create_fig4()
    create_fig5()
    create_fig6()
    create_fig7()
    create_fig8()
    
    print("--- Figure Generation Summary ---")
    for file, w, h in generated_files:
        print(f"Generated: {file} (Dimensions: {w}x{h} px)")
