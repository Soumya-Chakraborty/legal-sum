#!/usr/bin/env python3
"""
train_cpu.py — LegalSum Comprehensive CPU Training Script
==========================================================
Identical to train_gpu.py in logging depth and plotting, but:
  • Defaults tuned for CPU (smaller batch, fewer episodes, fewer MC-Dropout passes)
  • Uses torch.set_num_threads() to saturate all CPU cores
  • No CUDA calls — pure CPU execution
  • Shows per-epoch time estimates appropriate for CPU speeds
  • All 12 diagnostic plots identical to GPU version

Usage:
    python train_cpu.py \\
        -d datasets/eccv16_dataset_summe_google_pool5.h5 \\
        -s datasets/splits/summe_splits.json \\
        -m summe \\
        --save-dir log/cpu_run \\
        --max-epoch 60 \\
        --eval-every 5

CPU-tuned defaults vs GPU defaults:
    --max-epoch      60  (GPU: 100)  — fewer epochs OK on CPU
    --num-episode     3  (GPU: 5)    — fewer MC samples per video
    --ensemble-k      5  (GPU: 10)   — fewer MC-Dropout eval passes
    --hidden-dim    128  (GPU: 256)  — smaller model fits CPU RAM
    --num-threads     0  (auto-detect all cores)
"""

from __future__ import print_function
import os, os.path as osp, sys, time, csv, json, math, argparse, datetime
import numpy as np, h5py, scipy.stats

import torch, torch.nn as nn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[WARN] tqdm not installed. Install with: pip install tqdm")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from models import DSN, DSN_Transformer, DualPathwayDSN
from rewards import (compute_reward, compute_per_frame_attribution,
                     compute_contrastive_bonus, compute_legal_coherence_reward)
import vsum_tools
from utils import read_json, save_checkpoint

# ══════════════════════════════════════════════════════════════════════════════
# ARGS  (CPU-tuned defaults)
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser("LegalSum CPU Training — Detailed Logging")
parser.add_argument('-d','--dataset',   type=str, required=True)
parser.add_argument('-s','--split',     type=str, required=True)
parser.add_argument('--split-id',       type=int, default=0)
parser.add_argument('-m','--metric',    type=str, required=True, choices=['tvsum','summe'])
parser.add_argument('--model-type',     type=str, default='enhanced',
                    choices=['original','enhanced','transformer','dual'])
parser.add_argument('--input-dim',      type=int, default=1024)
parser.add_argument('--hidden-dim',     type=int, default=128,    # CPU: smaller
                    help="Hidden dim (CPU default: 128, GPU: 256)")
parser.add_argument('--num-layers',     type=int, default=2)
parser.add_argument('--num-heads',      type=int, default=4,      # CPU: fewer heads
                    help="Attention heads (CPU default: 4, GPU: 8)")
parser.add_argument('--rnn-cell',       type=str, default='lstm')
parser.add_argument('--dropout',        type=float, default=0.25)
parser.add_argument('--lr',             type=float, default=1e-4)
parser.add_argument('--weight-decay',   type=float, default=1e-5)
parser.add_argument('--max-epoch',      type=int, default=60,     # CPU: fewer epochs
                    help="Total Phase-1 epochs (CPU default: 60, GPU: 100)")
parser.add_argument('--lr-scheduler',   type=str, default='cosine', choices=['step','cosine'])
parser.add_argument('--stepsize',       type=int, default=20)
parser.add_argument('--gamma',          type=float, default=0.1)
parser.add_argument('--num-episode',    type=int, default=3,      # CPU: fewer episodes
                    help="REINFORCE episodes per video (CPU default: 3, GPU: 5)")
parser.add_argument('--beta',           type=float, default=0.01)
parser.add_argument('--entropy-start',  type=float, default=0.10)
parser.add_argument('--entropy-end',    type=float, default=0.001)
parser.add_argument('--phase2-epochs',  type=int, default=20,     # CPU: fewer
                    help="Phase-2 epochs (CPU default: 20, GPU: 30)")
parser.add_argument('--ensemble-k',     type=int, default=5,      # CPU: fewer passes
                    help="MC-Dropout passes at eval (CPU default: 5, GPU: 10)")
parser.add_argument('--contrastive-weight', type=float, default=0.05)
parser.add_argument('--action-lock-start',  type=float, default=0.95)
parser.add_argument('--action-lock-end',    type=float, default=0.85)
parser.add_argument('--use-counterfactual', action='store_true', default=True)
parser.add_argument('--no-counterfactual',  dest='use_counterfactual', action='store_false')
parser.add_argument('--use-curriculum',     action='store_true', default=True)
parser.add_argument('--no-curriculum',      dest='use_curriculum', action='store_false')
parser.add_argument('--use-legal-reward',   action='store_true', default=False)
parser.add_argument('--eval-every',     type=int, default=5)
parser.add_argument('--save-dir',       type=str, default='log/cpu_run')
parser.add_argument('--seed',           type=int, default=1)
parser.add_argument('--num-threads',    type=int, default=0,
                    help="CPU threads (0 = auto, uses all available cores)")
parser.add_argument('--verbose',        action='store_true', default=True)
args = parser.parse_args()

# ── CPU Setup ─────────────────────────────────────────────────────────────────
torch.manual_seed(args.seed); np.random.seed(args.seed)
USE_GPU = False  # always CPU

if args.num_threads == 0:
    n_cores = os.cpu_count() or 4
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(max(1, n_cores // 2))
else:
    torch.set_num_threads(args.num_threads)

n_threads = torch.get_num_threads()

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(osp.join(args.save_dir, 'plots'), exist_ok=True)

R='\033[91m'; G='\033[92m'; Y='\033[93m'; B='\033[94m'; C='\033[96m'; W='\033[0m'

class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            try: f.write(obj); f.flush()
            except: pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except: pass

_log_file = open(osp.join(args.save_dir, 'log_cpu_train.txt'), 'w')
sys.stdout = _Tee(sys.__stdout__, _log_file)

_csv_path = osp.join(args.save_dir, 'metrics_cpu.csv')
_csv_file = open(_csv_path, 'w', newline='')
_csv_writer = csv.writer(_csv_file)
_csv_writer.writerow([
    'epoch','phase','reward','reward_var','entropy_coef','lr','lock_pct','grad_norm',
    'f_score','precision','recall','spearman','kendall',
    'event_coverage','speaker_consistency','best_f_score','elapsed_sec',
    'epoch_time_sec','cpu_threads'
])

HIST = {
    'epoch':[], 'phase':[], 'reward':[], 'reward_var':[],
    'entropy':[], 'lr':[], 'lock_pct':[], 'grad_norm':[], 'epoch_time':[],
    'eval_epoch':[], 'f_score':[], 'precision':[], 'recall':[],
    'spearman':[], 'kendall':[], 'event_coverage':[], 'speaker_consistency':[],
    'per_video_fscores':[], 'best_f_score':[]
}

COLORS = {
    'p1':'#4C72B0','p2':'#DD8452','green':'#55A868','red':'#C44E52',
    'purple':'#8172B2','cyan':'#64B5CD','orange':'#F0953A','magenta':'#C77CCC','grey':'#888888'
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_model():
    kw = dict(in_dim=args.input_dim, hid_dim=args.hidden_dim,
               num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout)
    if args.model_type == 'transformer':
        return DSN_Transformer(in_dim=args.input_dim, hid_dim=args.hidden_dim*2,
                               num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout)
    elif args.model_type == 'dual':
        return DualPathwayDSN(**kw, cell=args.rnn_cell)
    else:
        return DSN(**kw, cell=args.rnn_cell)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _load(dataset, key):
    """Load features — CPU only, no .cuda() calls."""
    seq = torch.from_numpy(dataset[key]['features'][...]).unsqueeze(0).float()
    acoustic = semantic = event_mask = speaker_mask = None
    if 'acoustic' in dataset[key]:
        acoustic = torch.from_numpy(dataset[key]['acoustic'][...]).unsqueeze(0).float()
    if 'semantic' in dataset[key]:
        semantic = torch.from_numpy(dataset[key]['semantic'][...]).unsqueeze(0).float()
    if 'event_mask' in dataset[key]:
        event_mask = torch.from_numpy(dataset[key]['event_mask'][...]).unsqueeze(0).float()
    if 'speaker_mask' in dataset[key]:
        speaker_mask = torch.from_numpy(dataset[key]['speaker_mask'][...]).unsqueeze(0).long()
    return seq, acoustic, semantic, event_mask, speaker_mask

def _smooth(arr, w=5):
    if len(arr) < w: return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid').tolist()

def _epochs_for_smooth(epochs, arr, w=5):
    sm = _smooth(arr, w)
    ep = epochs[len(epochs)-len(sm):]
    return ep, sm

def build_curriculum_order(train_keys, reward_writers, local_ep, warmup=10):
    if local_ep < warmup:
        idx = np.arange(len(train_keys)); np.random.shuffle(idx); return idx
    diffs = [float(np.var(reward_writers.get(k,[])[-10:])) if len(reward_writers.get(k,[])) >= 3 else 0.0
             for k in train_keys]
    noisy = np.array(diffs) + np.random.normal(0, 0.01, len(diffs))
    return np.argsort(noisy)

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, dataset, test_keys, phase_name='eval'):
    eval_metric = 'avg' if args.metric == 'tvsum' else 'max'
    model.train()
    fms, fms_avg, fms_max, precs, recs, spearmans, kendalls, event_covs, speaker_cons, vnames = [], [], [], [], [], [], [], [], [], []
    t0_eval = time.time()
    with torch.no_grad():
        it = tqdm(test_keys, desc=f"  Eval [{phase_name}]", ncols=90, leave=False) if HAS_TQDM else test_keys
        for key in it:
            seq, acoustic, semantic, event_mask, speaker_mask = _load(dataset, key)
            probs = np.mean([model(seq, acoustic, semantic, speaker_mask, event_mask)
                             .data.squeeze().numpy() for _ in range(args.ensemble_k)], axis=0)
            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_sum = dataset[key]['user_summary'][...]
            machine_sum = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm_avg, fm_max, prec, rec = vsum_tools.evaluate_summary(machine_sum, user_sum, 'all')
            fm = fm_avg if eval_metric == 'avg' else fm_max
            fms.append(fm); fms_avg.append(fm_avg); fms_max.append(fm_max)
            precs.append(prec); recs.append(rec); vnames.append(key)
            r_sp = r_kd = 0.0
            if 'gtscore' in dataset[key]:
                gt = dataset[key]['gtscore'][...]
                p_a = probs if len(probs)==len(gt) else np.interp(
                    np.linspace(0,1,len(gt)), np.linspace(0,1,len(probs)), probs)
                r_sp, _ = scipy.stats.spearmanr(p_a, gt)
                r_kd, _ = scipy.stats.kendalltau(p_a, gt)
                if np.isnan(r_sp): r_sp = 0.0
                if np.isnan(r_kd): r_kd = 0.0
            spearmans.append(r_sp); kendalls.append(r_kd)
            ev_cov = 0.0; sp_con = 1.0
            picks = np.where(machine_sum == 1)[0]
            if 'event_mask' in dataset[key] and len(picks) > 0:
                em = dataset[key]['event_mask'][...]
                active = np.where(em.sum(axis=0) > 0)[0]
                if len(active) > 0:
                    covered = np.where(em[picks].sum(axis=0) > 0)[0]
                    ev_cov = len(np.intersect1d(covered, active)) / len(active)
                else: ev_cov = 1.0
            if 'speaker_mask' in dataset[key] and len(picks) > 1:
                sm = dataset[key]['speaker_mask'][...]
                switches = (sm[picks[:-1]] != sm[picks[1:]]).sum()
                sp_con = 1.0 - switches/(len(picks)-1)
            event_covs.append(ev_cov); speaker_cons.append(sp_con)

    eval_time = time.time() - t0_eval
    mean_fm=np.mean(fms); mean_fm_avg=np.mean(fms_avg); mean_fm_max=np.mean(fms_max)
    mean_prec=np.mean(precs); mean_rec=np.mean(recs)
    mean_sp=np.mean(spearmans); mean_kd=np.mean(kendalls)
    mean_ec=np.mean(event_covs); mean_sc=np.mean(speaker_cons)

    if args.verbose:
        print(f"\n  {'Video':<30} {'F%':>6} {'Prec%':>6} {'Rec%':>6} {'Spear':>7} {'Kend':>7} {'EvCov%':>8} {'SpCon%':>8}")
        print("  " + "─"*85)
        for i,k in enumerate(vnames):
            flag = f"{G}★{W}" if fms[i]==max(fms) else (" " if fms[i]>=mean_fm else f"{R}▼{W}")
            print(f"  {flag} {k:<29} {fms[i]*100:>5.1f}% {precs[i]*100:>5.1f}% "
                  f"{recs[i]*100:>5.1f}% {spearmans[i]:>7.3f} {kendalls[i]:>7.3f} "
                  f"{event_covs[i]*100:>7.1f}% {speaker_cons[i]*100:>7.1f}%")
        print("  " + "─"*85)
        print(f"  {'MEAN':<31} {mean_fm*100:>5.1f}% {mean_prec*100:>5.1f}% "
              f"{mean_rec*100:>5.1f}% {mean_sp:>7.3f} {mean_kd:>7.3f} "
              f"{mean_ec*100:>7.1f}% {mean_sc*100:>7.1f}%")
        print(f"  Eval time: {eval_time:.1f}s  (K={args.ensemble_k} MC passes)\n")

    return dict(f_score=float(mean_fm), f_score_avg=float(mean_fm_avg), f_score_max=float(mean_fm_max),
                precision=float(mean_prec), recall=float(mean_rec),
                spearman=float(mean_sp), kendall=float(mean_kd),
                event_coverage=float(mean_ec), speaker_consistency=float(mean_sc),
                per_video_fscores=fms)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING  (identical to GPU version)
# ══════════════════════════════════════════════════════════════════════════════
def save_all_plots(hist, save_dir, best_f=None, sota=41.4):
    pd = osp.join(save_dir, 'plots')
    os.makedirs(pd, exist_ok=True)
    epochs = hist['epoch']; eval_epochs = hist['eval_epoch']
    plt.rcParams.update({'font.size': 11})

    # 1. Reward Convergence
    fig, ax = plt.subplots(figsize=(9,4))
    rew = hist['reward']; phases = hist['phase']
    p1 = [(e,r) for e,r,ph in zip(epochs,rew,phases) if ph==1]
    p2 = [(e,r) for e,r,ph in zip(epochs,rew,phases) if ph==2]
    if p1: ax.plot(*zip(*p1), color=COLORS['p1'], lw=2, marker='o', ms=3, label='Phase 1 – Exploration')
    if p2:
        ax.plot(*zip(*p2), color=COLORS['p2'], lw=2, marker='o', ms=3, label='Phase 2 – Exploitation')
        ax.axvline(x=p2[0][0], color='grey', lw=1.5, ls='--', alpha=0.7, label='Phase 2 start')
    if len(rew)>5:
        se, sm = _epochs_for_smooth(epochs, rew, w=max(3,len(rew)//10))
        ax.plot(se, sm, 'k--', lw=1.5, alpha=0.5, label='Trend (smooth)')
    ax.set(title='1. REINFORCE Reward Convergence  [CPU]', xlabel='Epoch', ylabel='Mean Reward')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(osp.join(pd,'01_reward_convergence.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    if not eval_epochs: return

    # 2. F-Score
    fig, ax = plt.subplots(figsize=(9,4))
    fs = [x*100 for x in hist['f_score']]
    ax.plot(eval_epochs, fs, color=COLORS['green'], lw=2.5, marker='s', ms=7, label='F-Score (%)')
    bi = int(np.argmax(fs))
    ax.scatter([eval_epochs[bi]], [fs[bi]], color='gold', s=220, zorder=5, marker='*',
               label=f"Best: {fs[bi]:.1f}%")
    for xe,ye in zip(eval_epochs, fs):
        ax.annotate(f'{ye:.1f}', (xe,ye), textcoords='offset points', xytext=(0,10),
                    ha='center', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='#E8F5E9', alpha=0.7))
    ax.axhline(sota, color='red', lw=1.5, ls='-.', alpha=0.7, label=f'SOTA ({sota}%)')
    if best_f: ax.axhline(best_f*100, color='gold', lw=1, ls=':', alpha=0.7, label='Current Best')
    ax.set(title='2. Validation F-Score (MC-Dropout Ensemble)  [CPU]', xlabel='Epoch', ylabel='F-Score (%)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(osp.join(pd,'02_fscore.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 3. Precision & Recall
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(eval_epochs, [x*100 for x in hist['precision']], color=COLORS['purple'], lw=2, marker='^', label='Precision (%)')
    ax.plot(eval_epochs, [x*100 for x in hist['recall']], color=COLORS['cyan'], lw=2, marker='v', label='Recall (%)')
    ax.set(title='3. Precision & Recall  [CPU]', xlabel='Epoch', ylabel='%')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(osp.join(pd,'03_precision_recall.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 4. Correlations
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(eval_epochs, hist['spearman'], color=COLORS['red'], lw=2, marker='D', label='Spearman ρ')
    ax.plot(eval_epochs, hist['kendall'],  color=COLORS['orange'], lw=2, marker='P', label='Kendall τ')
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.set(title='4. Rank Correlations vs. Ground-Truth  [CPU]', xlabel='Epoch', ylabel='Correlation')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(osp.join(pd,'04_correlations.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 5. Event Coverage
    fig, ax = plt.subplots(figsize=(9,4))
    ec = [x*100 for x in hist['event_coverage']]
    ax.fill_between(eval_epochs, ec, alpha=0.15, color=COLORS['magenta'])
    ax.plot(eval_epochs, ec, color=COLORS['magenta'], lw=2.5, marker='o', label='Event Coverage (%)')
    ax.set(title='5. Courtroom Event Coverage (Domain-Specific)  [CPU]', xlabel='Epoch', ylabel='Coverage (%)')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(osp.join(pd,'05_event_coverage.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 6. Speaker Consistency
    fig, ax = plt.subplots(figsize=(9,4))
    sc = [x*100 for x in hist['speaker_consistency']]
    ax.fill_between(eval_epochs, sc, alpha=0.15, color=COLORS['cyan'])
    ax.plot(eval_epochs, sc, color=COLORS['cyan'], lw=2.5, marker='x', ms=8, label='Speaker Consistency (%)')
    ax.set(title='6. Speaker Consistency (Domain-Specific)  [CPU]', xlabel='Epoch', ylabel='Consistency (%)')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(osp.join(pd,'06_speaker_consistency.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 7. Entropy Decay
    if hist['entropy']:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.semilogy(epochs, hist['entropy'], color=COLORS['red'], lw=2.5)
        ax.fill_between(epochs, hist['entropy'], alpha=0.1, color=COLORS['red'])
        ax.set(title='7. Adaptive Policy Entropy Decay  [CPU]', xlabel='Epoch', ylabel='Entropy Coefficient (log)')
        ax.grid(True, alpha=0.3, which='both'); fig.tight_layout()
        fig.savefig(osp.join(pd,'07_entropy_decay.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)

    # 8. LR Schedule
    if hist['lr']:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.semilogy(epochs, hist['lr'], color=COLORS['purple'], lw=2.5)
        ax.set(title='8. Learning Rate Schedule  [CPU]', xlabel='Epoch', ylabel='LR')
        ax.grid(True, alpha=0.3, which='both'); fig.tight_layout()
        fig.savefig(osp.join(pd,'08_lr_schedule.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)

    # 9. Gradient Norm
    if hist['grad_norm']:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(epochs, hist['grad_norm'], color=COLORS['orange'], lw=1.5, alpha=0.6, label='Grad L2-norm')
        if len(hist['grad_norm']) > 5:
            se, sm = _epochs_for_smooth(epochs, hist['grad_norm'], w=5)
            ax.plot(se, sm, 'k--', lw=2, alpha=0.8, label='Smoothed')
        ax.axhline(5.0, color='red', lw=1.2, ls=':', alpha=0.7, label='Clip threshold (5.0)')
        ax.set(title='9. Gradient L2-Norm per Epoch  [CPU]', xlabel='Epoch', ylabel='Grad Norm')
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(osp.join(pd,'09_grad_norm.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)

    # 10. Per-Video Box-Plot
    if len(hist['per_video_fscores']) >= 2:
        fig, ax = plt.subplots(figsize=(max(9, len(eval_epochs)*0.8), 4))
        data = [[x*100 for x in row] for row in hist['per_video_fscores']]
        xlabels = [f"ep{e}" for e in eval_epochs]
        ax.boxplot(data, tick_labels=xlabels, patch_artist=True,
                   boxprops=dict(facecolor=COLORS['p1'], alpha=0.4),
                   medianprops=dict(color='darkblue', lw=2))
        ax.set(title='10. Per-Video F-Score Distribution (Box-Plot)  [CPU]',
               xlabel='Eval Epoch', ylabel='F-Score (%)')
        ax.grid(True, alpha=0.3); plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(osp.join(pd,'10_per_video_boxplot.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)

    # 11. Reward Variance
    if hist['reward_var']:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(epochs, hist['reward_var'], color=COLORS['grey'], lw=1.5, alpha=0.7, label='Reward Variance')
        if len(hist['reward_var']) > 5:
            se, sm = _epochs_for_smooth(epochs, hist['reward_var'], w=5)
            ax.plot(se, sm, color=COLORS['p2'], lw=2.5, label='Smoothed')
        ax.set(title='11. Reward Variance per Epoch (Curriculum Difficulty Proxy)  [CPU]',
               xlabel='Epoch', ylabel='Variance')
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(osp.join(pd,'11_reward_variance.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)

    # 12. Epoch Time (CPU-specific — helps profiling)
    if hist['epoch_time']:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(epochs, hist['epoch_time'], color=COLORS['grey'], lw=1.5, alpha=0.7, label='Epoch time (s)')
        if len(hist['epoch_time']) > 5:
            se, sm = _epochs_for_smooth(epochs, hist['epoch_time'], w=5)
            ax.plot(se, sm, color='black', lw=2, label='Smoothed')
        ax.set(title='12. CPU Epoch Time  (profiling — helps spot slow episodes)',
               xlabel='Epoch', ylabel='Time (seconds)')
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(osp.join(pd,'12_epoch_time.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)

    # Also save the Dashboard panel
    fig = plt.figure(figsize=(18,10))
    fig.suptitle('LegalSum — CPU Training Dashboard', fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    panels = [
        (gs[0,0], epochs, hist['reward'], 'Reward', COLORS['p1']),
        (gs[0,1], eval_epochs, [x*100 for x in hist['f_score']], 'F-Score (%)', COLORS['green']),
        (gs[0,2], eval_epochs, [x*100 for x in hist['event_coverage']], 'Event Coverage (%)', COLORS['magenta']),
        (gs[1,0], eval_epochs, hist['spearman'], 'Spearman ρ', COLORS['red']),
        (gs[1,1], eval_epochs, hist['kendall'],  'Kendall τ', COLORS['orange']),
        (gs[1,2], eval_epochs, [x*100 for x in hist['speaker_consistency']], 'Speaker Con. (%)', COLORS['cyan']),
    ]
    for spec, xe, ye, title, color in panels:
        ax = fig.add_subplot(spec)
        ax.plot(xe, ye, color=color, lw=2, marker='o', ms=4)
        if xe and ye:
            bi = int(np.argmax(ye))
            ax.scatter([xe[bi]], [ye[bi]], color='gold', s=150, zorder=5, marker='*')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(osp.join(pd,'00_dashboard.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"  {G}[PLOTS]{W} 12 plots saved → {pd}/")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PHASE
# ══════════════════════════════════════════════════════════════════════════════
def train_phase(model, optimizer, scheduler, dataset, train_keys, test_keys,
                num_epochs, baselines, reward_writers, phase_id,
                entropy_start, entropy_end, start_epoch=0,
                action_lock_start=0.95, action_lock_end=0.85):

    best_fm=0.0; best_epoch=0; best_state=None
    baseline_step = {k:0 for k in train_keys}
    t0_total = time.time()

    print(f"\n{'═'*72}")
    print(f"  {Y}PHASE {phase_id}{W}  [{G}CPU{W}]  threads={n_threads}  "
          f"epochs={num_epochs}  entropy {entropy_start:.3f}→{entropy_end:.4f}")
    print(f"{'═'*72}\n")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        local_ep  = epoch - start_epoch
        progress  = local_ep / max(1, num_epochs-1)
        ent_coef  = entropy_start * (entropy_end / entropy_start) ** progress
        lock_pct  = action_lock_start - (action_lock_start - action_lock_end) * progress
        cur_lr    = optimizer.param_groups[0]['lr']

        idxs = (build_curriculum_order(train_keys, reward_writers, local_ep)
                if args.use_curriculum else np.random.permutation(len(train_keys)))

        ep_rewards=[]; ep_grad_norms=[]
        t0_ep = time.time()

        vid_iter = (tqdm(idxs, desc=f"  Ep {epoch+1:03d}/{start_epoch+num_epochs}", ncols=90, leave=False)
                    if HAS_TQDM else idxs)

        for idx in vid_iter:
            key = train_keys[idx]
            seq, acoustic, semantic, event_mask, speaker_mask = _load(dataset, key)
            probs = model(seq, acoustic, semantic, speaker_mask, event_mask)
            length_pen = args.beta * (probs.mean() - 0.15)**2
            m = Bernoulli(probs)
            cost = length_pen - ent_coef * m.entropy().mean()

            epis_r = []
            for _ in range(args.num_episode):
                actions = m.sample()
                if args.use_legal_reward:
                    full_r = compute_legal_coherence_reward(
                        seq, actions, use_gpu=False, acoustic=acoustic, semantic_boost=semantic)
                    attr = None
                else:
                    attr, full_r = compute_per_frame_attribution(
                        seq, actions, use_gpu=False,
                        acoustic=acoustic, semantic_boost=semantic,
                        event_mask=event_mask, speaker_mask=speaker_mask)
                if args.contrastive_weight > 0:
                    full_r = full_r + args.contrastive_weight * compute_contrastive_bonus(seq, actions)
                bv = baselines[key]
                if attr is not None:
                    shaped = attr - bv
                    std = shaped.std()
                    shaped = shaped / std if std > 1e-5 else shaped
                    cost -= (m.log_prob(actions).squeeze() * shaped).mean()
                else:
                    cost -= m.log_prob(actions).mean() * (full_r - bv)
                epis_r.append(full_r.item() if hasattr(full_r,'item') else float(full_r))

            optimizer.zero_grad()
            cost.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
            optimizer.step()

            baseline_step[key] += 1
            t_ = baseline_step[key]
            raw_b = 0.9 * baselines[key] + 0.1 * np.mean(epis_r)
            baselines[key] = float(np.clip(raw_b / (1 - 0.9**t_), -2.0, 2.0))
            reward_writers[key].append(np.mean(epis_r))
            ep_rewards.append(np.mean(epis_r)); ep_grad_norms.append(gn)

        if scheduler: scheduler.step()

        epoch_reward = float(np.mean(ep_rewards))
        epoch_var    = float(np.var(ep_rewards))
        epoch_grad   = float(np.mean(ep_grad_norms))
        ep_time      = time.time() - t0_ep
        elapsed      = time.time() - t0_total
        remaining    = (start_epoch + num_epochs) - (epoch+1)
        eta          = (elapsed / (local_ep+1)) * remaining

        # Per-epoch timing for CPU profiling
        vids_per_sec = len(train_keys) / ep_time

        print(f"  Ep {B}{epoch+1:03d}{W}/{start_epoch+num_epochs}"
              f"  rew={Y}{epoch_reward:+.4f}{W}"
              f"  var={epoch_var:.4f}"
              f"  grad={epoch_grad:.3f}"
              f"  ent={ent_coef:.4f}"
              f"  lock={lock_pct*100:.0f}%"
              f"  lr={cur_lr:.2e}"
              f"  {ep_time:.0f}s ({vids_per_sec:.2f} vids/s)"
              f"  ETA {datetime.timedelta(seconds=int(eta))}")

        HIST['epoch'].append(epoch+1); HIST['phase'].append(phase_id)
        HIST['reward'].append(epoch_reward); HIST['reward_var'].append(epoch_var)
        HIST['entropy'].append(ent_coef); HIST['lr'].append(cur_lr)
        HIST['lock_pct'].append(lock_pct*100); HIST['grad_norm'].append(epoch_grad)
        HIST['epoch_time'].append(ep_time)

        do_eval = ((epoch+1) % args.eval_every == 0 or
                   epoch == start_epoch or
                   epoch == start_epoch + num_epochs - 1)

        if do_eval:
            print(f"\n  {C}{'─'*60}{W}")
            print(f"  {C}Evaluation @ epoch {epoch+1}{W}")
            mets = evaluate(model, dataset, test_keys, phase_name=f"P{phase_id}")
            HIST['eval_epoch'].append(epoch+1)
            for k_ in ('f_score','precision','recall','spearman','kendall',
                       'event_coverage','speaker_consistency'):
                HIST[k_].append(mets[k_])
            HIST['per_video_fscores'].append(mets['per_video_fscores'])
            HIST['best_f_score'].append(best_fm)

            print(f"  F={G}{mets['f_score']*100:.2f}%{W}  "
                  f"Prec={mets['precision']*100:.1f}%  "
                  f"Rec={mets['recall']*100:.1f}%  "
                  f"Spear={mets['spearman']:.4f}  "
                  f"Kend={mets['kendall']:.4f}  "
                  f"EvCov={mets['event_coverage']*100:.1f}%  "
                  f"SpkCon={mets['speaker_consistency']*100:.1f}%")

            if mets['f_score'] > best_fm:
                best_fm=mets['f_score']; best_epoch=epoch+1
                best_state = {k_: v.clone() for k_,v in
                              (model.module if hasattr(model,'module') else model).state_dict().items()}
                save_checkpoint(best_state, osp.join(args.save_dir,'model_best.pth.tar'))
                print(f"  {G}★ NEW BEST  F={best_fm*100:.2f}%  epoch={best_epoch}{W}")

            _csv_writer.writerow([
                epoch+1, phase_id, f"{epoch_reward:.6f}", f"{epoch_var:.6f}",
                f"{ent_coef:.6f}", f"{cur_lr:.8f}", f"{lock_pct*100:.1f}", f"{epoch_grad:.4f}",
                f"{mets['f_score']:.6f}", f"{mets['precision']:.6f}", f"{mets['recall']:.6f}",
                f"{mets['spearman']:.6f}", f"{mets['kendall']:.6f}",
                f"{mets['event_coverage']:.6f}", f"{mets['speaker_consistency']:.6f}",
                f"{best_fm:.6f}", f"{elapsed:.1f}", f"{ep_time:.1f}", n_threads
            ])
            _csv_file.flush()
            save_all_plots(HIST, args.save_dir, best_f=best_fm)
            print(f"  {C}{'─'*60}{W}\n")
        else:
            _csv_writer.writerow([
                epoch+1, phase_id, f"{epoch_reward:.6f}", f"{epoch_var:.6f}",
                f"{ent_coef:.6f}", f"{cur_lr:.8f}", f"{lock_pct*100:.1f}", f"{epoch_grad:.4f}",
                '','','','','','','',
                f"{best_fm:.6f}", f"{elapsed:.1f}", f"{ep_time:.1f}", n_threads
            ])
            _csv_file.flush()
            save_all_plots(HIST, args.save_dir, best_f=best_fm)

    return best_fm, best_epoch, best_state, baselines, reward_writers

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'═'*72}")
    print(f"  {G}LegalSum CPU Training Script{W}")
    print(f"  PyTorch: {torch.__version__}  |  CPU threads: {n_threads}")
    print(f"  Model: {args.model_type}  hidden_dim={args.hidden_dim}  "
          f"num_heads={args.num_heads}  episodes={args.num_episode}  "
          f"ensemble_k={args.ensemble_k}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  {Y}TIP:{W} Use train_gpu.py if you have a GPU for ~10-50x speed.")
    print(f"{'═'*72}\n")

    dataset    = h5py.File(args.dataset, 'r')
    splits     = read_json(args.split)
    split      = splits[args.split_id]
    train_keys = [str(k) for k in split['train_keys']]
    test_keys  = [str(k) for k in split['test_keys']]
    print(f"  Split {args.split_id}: {len(train_keys)} train | {len(test_keys)} test")

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  (CPU-optimized smaller model)")

    baselines      = {k: 0.0 for k in train_keys}
    reward_writers = {k: []  for k in train_keys}

    # Phase 1
    optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched1 = (lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.max_epoch, eta_min=args.lr*0.01)
              if args.lr_scheduler == 'cosine'
              else lr_scheduler.StepLR(optimizer1, step_size=args.stepsize, gamma=args.gamma))

    best_fm1, best_epoch1, best_state1, baselines, reward_writers = train_phase(
        model, optimizer1, sched1, dataset, train_keys, test_keys,
        num_epochs=args.max_epoch, baselines=baselines, reward_writers=reward_writers,
        phase_id=1, entropy_start=args.entropy_start, entropy_end=args.entropy_end,
        action_lock_start=args.action_lock_start, action_lock_end=args.action_lock_end)

    print(f"\n  {Y}Phase 1 done.{W}  Best F={G}{best_fm1*100:.2f}%{W} @ epoch {best_epoch1}")
    if best_state1:
        (model.module if hasattr(model,'module') else model).load_state_dict(best_state1)

    # Phase 2
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr*0.1, weight_decay=args.weight_decay)
    sched2 = (lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.phase2_epochs, eta_min=args.lr*0.001)
              if args.lr_scheduler == 'cosine'
              else lr_scheduler.StepLR(optimizer2, step_size=args.stepsize, gamma=args.gamma))

    best_fm2, best_epoch2, best_state2, baselines, reward_writers = train_phase(
        model, optimizer2, sched2, dataset, train_keys, test_keys,
        num_epochs=args.phase2_epochs, baselines=baselines, reward_writers=reward_writers,
        phase_id=2, entropy_start=args.entropy_end, entropy_end=args.entropy_end*0.1,
        start_epoch=args.max_epoch,
        action_lock_start=args.action_lock_end, action_lock_end=args.action_lock_end)

    overall_best = max(best_fm1, best_fm2)
    save_all_plots(HIST, args.save_dir, best_f=overall_best)

    report = dict(device='cpu', cpu_threads=n_threads,
                  phase1_best_f=best_fm1, phase1_best_epoch=best_epoch1,
                  phase2_best_f=best_fm2, phase2_best_epoch=best_epoch2,
                  overall_best_f=overall_best, args=vars(args))
    with open(osp.join(args.save_dir,'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'═'*72}")
    print(f"  {G}TRAINING COMPLETE{W}  [CPU]")
    print(f"  Overall Best F-Score : {G}{overall_best*100:.2f}%{W}")
    print(f"  CSV Metrics          : {_csv_path}")
    print(f"  Plots (12 charts)    : {osp.join(args.save_dir,'plots')}/")
    print(f"  Report JSON          : {osp.join(args.save_dir,'training_report.json')}")
    print(f"{'═'*72}\n")

    dataset.close(); _csv_file.close(); _log_file.close()

if __name__ == '__main__':
    main()
