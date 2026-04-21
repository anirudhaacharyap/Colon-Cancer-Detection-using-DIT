import matplotlib.pyplot as plt

def plot_tp_vs_fp():
    # 1. Define the Models
    models = [
        "BOA Only", 
        "WOA Only", 
        "DiT Only", 
        "Hybrid BOA-WOA + DiT"
    ]

    # 2. Add your ACTUAL True Positive and False Positive values here
    # These are placeholder numbers, replace them with values from your confusion matrices
    tp_values = [750, 780, 890, 975]  # True Positives (X-axis)
    fp_values = [140, 125,  65,   8]  # False Positives (Y-axis)

    # Colors and markers for each model
    colors = ['#e6a532', '#5eadd1', '#119672', '#d62728']
    markers = ['o', 's', '^', '*']
    
    plt.figure(figsize=(9, 6))

    # 3. Plot each model as a point in the TP vs FP space
    for i in range(len(models)):
        plt.scatter(
            tp_values[i], 
            fp_values[i], 
            color=colors[i], 
            marker=markers[i], 
            s=200 if markers[i] != '*' else 400, # Make the star bigger
            label=models[i],
            zorder=3
        )
        
        # Add text labels next to the points
        plt.annotate(
            models[i],
            (tp_values[i], fp_values[i]),
            xytext=(15, -5), 
            textcoords='offset points',
            fontsize=11,
            fontweight='bold' if 'Hybrid' in models[i] else 'normal'
        )

    # 4. Styling the graph
    plt.xlabel('True Positives (TP)', fontsize=13, fontweight='bold')
    plt.ylabel('False Positives (FP)', fontsize=13, fontweight='bold')
    plt.title('True Positives vs. False Positives (Model Ablation Comparison)', fontsize=14, pad=15)
    
    # Grid and layout
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    plt.legend(loc='upper right', fontsize=11, title="Models")
    
    # Ideally, right and bottom is the best area (High TP, Low FP), let's indicate that
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Optional context arrow showing the direction of "Better Performance"
    # pointing towards bottom-right (High TP, Low FP)
    plt.annotate(
        "Better Performance", 
        xy=(max(tp_values)-10, min(fp_values)+5), 
        xytext=(max(tp_values)-150, min(fp_values)+50),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11,
        style='italic'
    )

    plt.tight_layout()

    # Save output
    output_path = 'tp_vs_fp_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")
    plt.show()

if __name__ == '__main__':
    plot_tp_vs_fp()
