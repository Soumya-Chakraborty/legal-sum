"""
Training Log Visualization and Performance Analyzer

This script parses training log text files produced during Counterfactual REINFORCE training,
extracts key metrics (rewards, validation F-scores, learning rates, and entropy coefficients)
using regular expressions, and renders a high-quality 4-panel diagnostic plot.

Example usage:
    python plot_training_logs.py log/summe-counterfactual-s0/log_train.txt outputs/training_analysis.png
"""

import re
import sys
import math
import matplotlib
# Use standard 'Agg' backend to allow generating plots without a GUI environment
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def parse_log(log_path):
    """
    Parses a training log file to extract epoch-wise metrics.

    Args:
        log_path (str): Path to the log_train.txt file.

    Returns:
        dict: A dictionary containing lists of epochs, rewards, lrs, entropies,
              val_epochs, val_f_scores, and the start epoch of Phase 2.
    """
    epochs, rewards, lrs, entropies = [], [], [], []
    val_epochs, val_f_scores = [], []
    current_epoch = None
    phase2_start = None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Regex to match training metrics line: "epoch X/Y\t reward R\t entropy_coef E\t lr L"
        m = re.search(
            r'epoch (\d+)/(\d+)\s+reward ([\d\.\-]+)\s+entropy_coef ([\de\.\-\+]+)\s+lr ([\de\.\-\+]+)',
            line
        )
        if m:
            current_epoch = int(m.group(1))
            total_epochs = int(m.group(2))
            epochs.append(current_epoch)
            rewards.append(float(m.group(3)))
            entropies.append(float(m.group(4)))
            lrs.append(float(m.group(5)))

        # Detect transition marker from Phase 1 (exploration) to Phase 2 (exploitation)
        if 'PHASE 2' in line and phase2_start is None and current_epoch is not None:
            phase2_start = current_epoch + 1

        # Regex to match validation checkpoint evaluations (e.g. "Average F-score 42.5%")
        m2 = re.search(r'Average F-score ([\d\.]+)%', line)
        if m2 and current_epoch is not None:
            val_epochs.append(current_epoch)
            val_f_scores.append(float(m2.group(1)))

    # Deduplicate validation checkpoints (keeping only the latest record per epoch)
    unique_val = {}
    for ep, fs in zip(val_epochs, val_f_scores):
        unique_val[ep] = fs
    val_epochs = sorted(unique_val.keys())
    val_f_scores = [unique_val[ep] for ep in val_epochs]

    return {
        'epochs': epochs, 'rewards': rewards,
        'lrs': lrs, 'entropies': entropies,
        'val_epochs': val_epochs, 'val_f_scores': val_f_scores,
        'phase2_start': phase2_start,
    }


def plot_all(data, out_path):
    """
    Renders and saves a 4-panel visualization of training metrics.

    Args:
        data (dict): Dictionary populated with parsed metrics from parse_log.
        out_path (str): Target output file path (.png).
    """
    epochs = data['epochs']
    rewards = data['rewards']
    lrs = data['lrs']
    entropies = data['entropies']
    val_epochs = data['val_epochs']
    val_f_scores = data['val_f_scores']
    phase2_start = data['phase2_start']

    # ── Beautiful Custom Color Palette ──────────────────────────────────────
    C_PHASE1 = '#4C72B0'  # Steel blue
    C_PHASE2 = '#DD8452'  # Orange/amber
    C_VAL    = '#55A868'  # Grass green
    C_ENT    = '#C44E52'  # Soft red
    C_LR     = '#8172B2'  # Muted purple

    # Set up 2x2 grid subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Counterfactual REINFORCE — Training Analysis\n'
                 'Enhanced DSN (Bi-LSTM + Multi-Head Attention) on SumMe',
                 fontsize=15, fontweight='bold', y=1.01)

    # ── Plot 1. REINFORCE Reward Convergence ──────────────────────────────────
    ax = axes[0, 0]
    if phase2_start and phase2_start in epochs:
        p2_idx = epochs.index(phase2_start)
    elif phase2_start:
        p2_idx = next((i for i, e in enumerate(epochs) if e >= phase2_start), len(epochs))
    else:
        p2_idx = len(epochs)

    # Plot Phase 1 reward curve
    ax.plot(epochs[:p2_idx], rewards[:p2_idx], color=C_PHASE1, lw=2.5,
            marker='o', ms=3, label='Phase 1 (Exploration)')
    
    # Plot Phase 2 reward curve if available
    if p2_idx < len(epochs):
        ax.plot(epochs[p2_idx:], rewards[p2_idx:], color=C_PHASE2, lw=2.5,
                marker='o', ms=3, label='Phase 2 (Exploitation)')
        ax.axvline(x=epochs[p2_idx], color='gray', lw=1.5, linestyle='--', alpha=0.7)
        ax.text(epochs[p2_idx] + 0.3, min(rewards) + 0.02, 'Phase 2 →',
                color='gray', fontsize=9)

    ax.set_title('REINFORCE Reward Convergence', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Expected Reward', fontsize=11, color=C_PHASE1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Compute and plot moving average trend line
    if len(rewards) > 5:
        window = max(3, len(rewards) // 10)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        xs = epochs[window//2: window//2 + len(smoothed)]
        ax.plot(xs, smoothed, color='black', lw=2, linestyle='--', alpha=0.5,
                label='Smoothed trend')
        ax.legend(fontsize=9)

    # ── Plot 2. Validation F-score (SumMe) ──────────────────────────────────
    ax = axes[0, 1]
    if val_epochs:
        ax.plot(val_epochs, val_f_scores, color=C_VAL, lw=3,
                marker='s', ms=9, label='Val F-score (ensemble)')
        best_idx = np.argmax(val_f_scores)
        # Highlight the best score with a star marker
        ax.scatter([val_epochs[best_idx]], [val_f_scores[best_idx]],
                   color='gold', s=200, zorder=5, marker='*', label='Best')
        # Annotate score values on top of each marker
        for x, y in zip(val_epochs, val_f_scores):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                        xytext=(0, 12), ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', fc='#E8F5E9', alpha=0.8))
        # Draw baseline SOTA reference line
        sota_line = 41.4
        ax.axhline(y=sota_line, color='red', lw=1.5, linestyle='-.', alpha=0.7)
        ax.text(val_epochs[0], sota_line + 0.5, f'SOTA baseline (41.4%)',
                color='red', fontsize=8, alpha=0.8)

    ax.set_title('Validation F-score — SumMe (MC-Dropout Ensemble)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('F-score (%)', fontsize=11, color=C_VAL)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 3. Learning Rate Schedule ───────────────────────────────────────
    ax = axes[1, 0]
    ax.semilogy(epochs, lrs, color=C_LR, lw=2.5, marker='', label='LR (log scale)')
    if phase2_start and p2_idx < len(epochs):
        ax.axvline(x=epochs[p2_idx], color='gray', lw=1.5, linestyle='--', alpha=0.7)
        ax.text(epochs[p2_idx] + 0.3, max(lrs)*0.5, 'Phase 2', color='gray', fontsize=9)
    ax.set_title('Cosine Annealing LR Schedule (Both Phases)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate (log)', fontsize=11, color=C_LR)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)

    # ── Plot 4. Entropy Coefficient Decay ─────────────────────────────────────
    ax = axes[1, 1]
    ax.semilogy(epochs, entropies, color=C_ENT, lw=2.5,
                marker='', label='Entropy coeff (log)')
    if phase2_start and p2_idx < len(epochs):
        ax.axvline(x=epochs[p2_idx], color='gray', lw=1.5, linestyle='--', alpha=0.7)
    ax.fill_between(epochs, entropies, alpha=0.15, color=C_ENT)
    ax.set_title('Adaptive Entropy Decay Schedule\n(Exploration → Exploitation)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Entropy Coefficient (log)', fontsize=11, color=C_ENT)
    ax.grid(True, alpha=0.3, which='both')

    # Add explanatory annotations about entropy decay mechanism
    ax.text(0.02, 0.97,
            '① High entropy = broad exploration\n'
            '② Low entropy = confident selection\n'
            '③ Exponential decay bridges both',
            transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_path}")
    return data


if __name__ == '__main__':
    # Parse inputs from command line arguments or use default paths
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'log/summe-counterfactual-s0/log_train.txt'
    out_file = sys.argv[2] if len(sys.argv) > 2 else 'log/summe-counterfactual-s0/training_analysis.png'

    data = parse_log(log_file)
    print(f"Parsed {len(data['epochs'])} training epochs, "
          f"{len(data['val_epochs'])} validation checkpoints")
    if data['val_f_scores']:
        print(f"Best F-score: {max(data['val_f_scores']):.1f}% at epoch "
              f"{data['val_epochs'][int(np.argmax(data['val_f_scores']))]}")
    plot_all(data, out_file)
