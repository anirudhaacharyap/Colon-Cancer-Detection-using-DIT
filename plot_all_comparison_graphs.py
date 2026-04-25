"""
CHANGE 9: Pareto Plot with Real Training Times
CHANGE 10: Bootstrap CI with DeLong Test

Generates:
  - pareto_plot.png        — Accuracy vs real training duration (parsed from logs)
  - bootstrap_ci_plot.png  — Bootstrap CIs for AUC with DeLong p-value annotations
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats


# Ablation configurations
ABLATIONS = [
    ("BOA Only",              "./logs/ablation_boa_only/", '#5eadd1', 'o'),
    ("WOA Only",              "./logs/ablation_woa_only/", '#e6a532', 's'),
    ("DiT Only",              "./logs/ablation_dit_only/", '#119672', '^'),
    ("Hybrid BOA-WOA + DiT",  "./logs/full_run/",          '#d62728', '*'),
]


# ---------------------------------------------------------------------------
# CHANGE 9: Pareto plot with real training durations
# ---------------------------------------------------------------------------

def parse_training_duration(log_path):
    """Parse training.log for first and last timestamps to compute duration in minutes."""
    if not os.path.exists(log_path):
        return None

    timestamps = []
    ts_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    with open(log_path, 'r') as f:
        for line in f:
            match = ts_pattern.match(line.strip())
            if match:
                try:
                    ts = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                    timestamps.append(ts)
                except ValueError:
                    continue

    if len(timestamps) < 2:
        return None

    duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0  # minutes
    return duration


def parse_accuracy(log_dir):
    """Parse results.txt for accuracy value."""
    results_path = os.path.join(log_dir, "results.txt")
    if not os.path.exists(results_path):
        return None

    with open(results_path, 'r') as f:
        for line in f:
            if line.strip().startswith("Accuracy:"):
                # Extract numeric value, e.g., "Accuracy:    99.30%"
                match = re.search(r'([\d.]+)%', line)
                if match:
                    return float(match.group(1))
    return None


def plot_pareto():
    """CHANGE 9: Pareto plot with real training times on X-axis."""
    fig, ax = plt.subplots(figsize=(10, 7))

    points = []
    for label, log_dir, color, marker in ABLATIONS:
        log_path = os.path.join(log_dir, "training.log")
        duration = parse_training_duration(log_path)
        accuracy = parse_accuracy(log_dir)

        if duration is None or accuracy is None:
            print(f"  [SKIP] Missing data for {label} (dur={duration}, acc={accuracy})")
            continue

        points.append((label, duration, accuracy, color, marker))
        marker_size = 300 if marker == '*' else 150
        ax.scatter(duration, accuracy, color=color, marker=marker, s=marker_size,
                   zorder=5, edgecolors='black', linewidth=0.5, label=label)

        # Text annotation
        ax.annotate(
            f"{label}\n{accuracy:.2f}% | {duration:.1f}min",
            (duration, accuracy),
            xytext=(15, 10), textcoords='offset points',
            fontsize=9,
            fontweight='bold' if 'Hybrid' in label else 'normal',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1)
        )

    if not points:
        print("No data available for Pareto plot.")
        return

    ax.set_xlabel('Training Duration (minutes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Front: Accuracy vs Training Time', fontsize=15, pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add "Better" arrow
    ax.annotate(
        "Better\n(faster + more accurate)",
        xy=(ax.get_xlim()[0] + 5, ax.get_ylim()[1] - 0.3),
        fontsize=10, style='italic', color='gray',
        ha='left', va='top'
    )

    plt.tight_layout()
    os.makedirs('logs', exist_ok=True)
    out_path = os.path.join('logs', 'pareto_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pareto plot saved to {out_path}")


# ---------------------------------------------------------------------------
# CHANGE 10: Bootstrap CI with DeLong test
# ---------------------------------------------------------------------------

def delong_roc_variance(ground_truth, predictions):
    """
    Computes the DeLong variance estimate for AUC.
    Based on: DeLong et al. (1988) — Comparing the Areas under Two or More
    Correlated Receiver Operating Characteristic Curves.
    
    Returns: AUC, variance
    """
    order = np.argsort(-predictions)  # descending
    label_ordered = ground_truth[order]
    
    positive_idx = np.where(label_ordered == 1)[0]
    negative_idx = np.where(label_ordered == 0)[0]
    
    m = len(positive_idx)
    n = len(negative_idx)
    
    if m == 0 or n == 0:
        return 0.5, 0.0
    
    # Compute placement values for positive samples
    tx = np.empty(m, dtype=np.float64)
    for i, pos in enumerate(positive_idx):
        tx[i] = np.sum(predictions[order[negative_idx]] < predictions[order[pos]])
        tx[i] += 0.5 * np.sum(predictions[order[negative_idx]] == predictions[order[pos]])
    tx /= n
    
    # Compute placement values for negative samples
    ty = np.empty(n, dtype=np.float64)
    for j, neg in enumerate(negative_idx):
        ty[j] = np.sum(predictions[order[positive_idx]] > predictions[order[neg]])
        ty[j] += 0.5 * np.sum(predictions[order[positive_idx]] == predictions[order[neg]])
    ty /= m
    
    auc_val = np.mean(tx)
    sx = np.var(tx, ddof=1)
    sy = np.var(ty, ddof=1)
    
    var = sx / m + sy / n
    return auc_val, var


def delong_test(ground_truth, preds_a, preds_b):
    """
    Performs DeLong test comparing two AUCs.
    Returns: z-statistic, p-value
    """
    auc_a, var_a = delong_roc_variance(ground_truth, preds_a)
    auc_b, var_b = delong_roc_variance(ground_truth, preds_b)
    
    # Covariance estimation (simplified — assumes independent for now)
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b + 1e-12)
    p_value = 2 * stats.norm.sf(abs(z))
    
    return z, p_value


def bootstrap_auc(targets, probs, n_bootstrap=2000, random_state=42):
    """Compute bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(random_state)
    n = len(targets)
    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t = targets[idx]
        p = probs[idx]
        # Skip degenerate samples
        if len(np.unique(t)) < 2:
            continue
        aucs.append(roc_auc_score(t, p))

    aucs = np.array(aucs)
    mean_auc = np.mean(aucs)
    ci_low = np.percentile(aucs, 2.5)
    ci_high = np.percentile(aucs, 97.5)
    return mean_auc, ci_low, ci_high, aucs


def plot_bootstrap_ci():
    """CHANGE 10: Bootstrap CI plot with DeLong p-value annotations."""
    fig, ax = plt.subplots(figsize=(10, 7))

    results = []
    all_data = {}

    for label, log_dir, color, marker in ABLATIONS:
        csv_path = os.path.join(log_dir, "predictions.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] No predictions for {label}")
            continue

        df = pd.read_csv(csv_path)
        targets = df['target'].values
        probs = df['prob'].values

        mean_auc, ci_low, ci_high, auc_dist = bootstrap_auc(
            targets, probs, n_bootstrap=2000, random_state=42
        )
        results.append((label, mean_auc, ci_low, ci_high, color))
        all_data[label] = (targets, probs)

        print(f"  {label}: AUC = {mean_auc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    if not results:
        print("No data available for bootstrap CI plot.")
        return

    # Plot horizontal CI bars
    y_positions = np.arange(len(results))
    for i, (label, mean_auc, ci_low, ci_high, color) in enumerate(results):
        ci_err = np.array([[mean_auc - ci_low], [ci_high - mean_auc]])
        ax.barh(i, mean_auc, height=0.5, color=color, alpha=0.3, edgecolor=color, linewidth=1.5)
        ax.errorbar(mean_auc, i, xerr=ci_err, fmt='o', color=color,
                    markersize=8, capsize=6, capthick=2, elinewidth=2)
        ax.text(ci_high + 0.001, i, f'{mean_auc:.4f}\n[{ci_low:.4f}, {ci_high:.4f}]',
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in results], fontsize=11)
    ax.set_xlabel('AUC-ROC (95% Bootstrap CI, n=2000)', fontsize=13, fontweight='bold')
    ax.set_title('Bootstrap AUC Confidence Intervals with DeLong Tests', fontsize=14, pad=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # DeLong p-value annotations between consecutive pairs
    if len(results) >= 2:
        # Compare hybrid (last) vs all others
        hybrid_label = results[-1][0]
        if hybrid_label in all_data:
            t_hybrid, p_hybrid = all_data[hybrid_label]
            annotation_x = max(r[3] for r in results) + 0.008  # position right of CIs

            for i, (label, mean_auc, ci_low, ci_high, color) in enumerate(results[:-1]):
                if label in all_data:
                    t_other, p_other = all_data[label]
                    # Ensure same targets (they should be same test set)
                    if np.array_equal(t_hybrid, t_other):
                        z_stat, p_val = delong_test(t_hybrid, p_hybrid, p_other)
                        significance = "ns" if p_val > 0.05 else "*" if p_val > 0.01 else "**" if p_val > 0.001 else "***"
                        ax.annotate(
                            f'vs Hybrid: p={p_val:.2e} ({significance})',
                            xy=(annotation_x, i), fontsize=8,
                            va='center', color='#555555', style='italic'
                        )

    # Adjust xlim for annotations
    x_max = max(r[3] for r in results)
    ax.set_xlim([min(r[2] for r in results) - 0.02, x_max + 0.08])

    plt.tight_layout()
    os.makedirs('logs', exist_ok=True)
    out_path = os.path.join('logs', 'bootstrap_ci_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bootstrap CI plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 50)
    print("CHANGE 9: Pareto Plot (Real Training Times)")
    print("=" * 50)
    plot_pareto()

    print()

    print("=" * 50)
    print("CHANGE 10: Bootstrap CI + DeLong Test")
    print("=" * 50)
    plot_bootstrap_ci()
