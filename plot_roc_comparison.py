"""
CHANGE 8: Combined ROC Comparison — loads real prediction data from all 4 ablation
result folders and plots overlaid ROC curves with a zoom inset.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import roc_curve, auc


# Ablation configurations: (label, log_dir, color, linewidth)
ABLATIONS = [
    ("BOA Only",              "./logs/ablation_boa_only/", '#5eadd1', 2.0),
    ("WOA Only",              "./logs/ablation_woa_only/", '#e6a532', 2.0),
    ("DiT Only",              "./logs/ablation_dit_only/", '#119672', 2.0),
    ("Hybrid BOA-WOA + DiT",  "./logs/full_run/",          '#d62728', 2.5),
]


def load_predictions(log_dir):
    """Load predictions from CSV saved by evaluate.py."""
    csv_path = os.path.join(log_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        return None, None, None
    df = pd.read_csv(csv_path)
    return df['target'].values, df['pred'].values, df['prob'].values


def plot_comparison_roc():
    fig, ax_main = plt.subplots(figsize=(9, 7))

    curves = []
    has_data = False

    for label, log_dir, color, lw in ABLATIONS:
        targets, preds, probs = load_predictions(log_dir)
        if targets is None:
            print(f"  [SKIP] No predictions found in {log_dir}")
            continue

        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        curves.append((label, fpr, tpr, roc_auc, color, lw))
        has_data = True

        ax_main.plot(fpr, tpr, color=color, lw=lw,
                     label=f'{label} (AUC={roc_auc:.4f})')

    if not has_data:
        print("No prediction data found in any ablation folder.")
        print("Run evaluate.py for each ablation first, e.g.:")
        print("  python evaluate.py --ablation dit_only")
        print("  python evaluate.py --ablation boa_only")
        print("  python evaluate.py --ablation woa_only")
        print("  python evaluate.py")
        return

    # Diagonal
    ax_main.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random')

    # Main axes formatting
    ax_main.set_xlim([-0.02, 1.02])
    ax_main.set_ylim([-0.02, 1.02])
    ax_main.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax_main.set_title('ROC Curve Comparison (Model Ablation)', fontsize=15, pad=12)
    ax_main.grid(True, linestyle='--', alpha=0.5)
    ax_main.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # Zoom inset: FPR 0.0–0.02, TPR 0.98–1.0
    ax_inset = ax_main.inset_axes([0.35, 0.08, 0.4, 0.4])  # [x, y, width, height] in axes fraction
    for label, fpr, tpr, roc_auc, color, lw in curves:
        ax_inset.plot(fpr, tpr, color=color, lw=lw)
    ax_inset.set_xlim([0.0, 0.02])
    ax_inset.set_ylim([0.98, 1.0])
    ax_inset.set_xlabel('FPR', fontsize=8)
    ax_inset.set_ylabel('TPR', fontsize=8)
    ax_inset.tick_params(axis='both', which='major', labelsize=7)
    ax_inset.grid(True, linestyle=':', alpha=0.5)
    ax_inset.set_title('Zoom: High-TPR Region', fontsize=9)
    
    # Mark the zoom region on main plot
    ax_main.indicate_inset_zoom(ax_inset, edgecolor='black', alpha=0.6, lw=1.5)

    plt.tight_layout()

    os.makedirs('logs', exist_ok=True)
    out_path = os.path.join('logs', 'roc_curve_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC comparison saved to {out_path}")


if __name__ == '__main__':
    plot_comparison_roc()
