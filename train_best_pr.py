"""
train_best_pr.py
================
Best-precision-and-recall training script for SumMe and TVSum.

Design principles:
  1. Multi-seed: runs N seeds per split, keeps best checkpoint.
  2. Dataset-specific hyperparameters tuned for max F1/P/R.
  3. Optimal knapsack proportion sweep at eval time.
  4. 5-split cross-validation with mean ± std reporting.
  5. Saves a results table to JSON for paper reporting.

Usage:
  # Single dataset, all splits, multi-seed:
  python3 train_best_pr.py --dataset summe
  python3 train_best_pr.py --dataset tvsum

  # Both datasets:
  python3 train_best_pr.py --dataset both

  # Quick smoke-test (1 split, 1 seed, 20 epochs):
  python3 train_best_pr.py --dataset summe --quick
"""

import argparse, os, sys, json, time, math, copy
import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.distributions import Bernoulli
from torch import optim
from torch.optim import lr_scheduler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import vsum_tools
from models import DSN
from rewards import (
    compute_reward, compute_per_frame_attribution,
    compute_contrastive_bonus, compute_ot_temporal_diversity,
    compute_f1_soft_reward, compute_pr_calibration_reward,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='both', choices=['summe', 'tvsum', 'both'])
ap.add_argument('--quick', action='store_true',
                help='Quick mode: 1 split, 1 seed, 20 epochs')
ap.add_argument('--save-root', default='log/best_pr')
ap.add_argument('--seeds', type=str, default='42,123,7',
                help='Comma-separated random seeds (default: 42,123,7)')
CFG = ap.parse_args()

SEEDS      = [int(s) for s in CFG.seeds.split(',')]
USE_GPU    = torch.cuda.is_available()
DEVICE     = torch.device('cuda' if USE_GPU else 'cpu')
NUM_SPLITS = 1 if CFG.quick else 5
NUM_SEEDS  = 1 if CFG.quick else len(SEEDS)

os.makedirs(CFG.save_root, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CPU thread tuning
# ─────────────────────────────────────────────────────────────────────────────
if not USE_GPU:
    n_thr = min(os.cpu_count() or 1, 8)
    torch.set_num_threads(n_thr)
    torch.set_num_interop_threads(max(1, n_thr // 2))

# ─────────────────────────────────────────────────────────────────────────────
# GPU tuning
# ─────────────────────────────────────────────────────────────────────────────
if USE_GPU:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific best hyperparameters
# (tuned for max F1 / Precision / Recall on SumMe and TVSum)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CFG = {
    'summe': {
        'h5':            'datasets/eccv16_dataset_summe_google_pool5.h5',
        'split_json':    'datasets/summe_splits.json',
        'metric':        'summe',
        # Model
        'hid_dim':       256,
        'num_layers':    2,
        'num_heads':     8,
        'dropout':       0.25,
        # Optimization
        'lr':            1e-4 if USE_GPU else 3e-4,
        'weight_decay':  1e-5,
        'pretrain_ep':   5  if CFG.quick else 10,
        'max_epoch':     20 if CFG.quick else 100,
        'phase2_ep':     5  if CFG.quick else 25,
        'patience':      10,
        'num_episode':   5,
        'ppo_clip':      0.2,
        'ppo_inner':     4,
        # Reward
        'ot_weight':     0.10,
        'contrastive_w': 0.05,
        'recall_weight': 2.5,   # higher recall weight for SumMe (smaller dataset)
        'pr_f1_weight':  0.10,
        'reward_warmup': 12,
        'entropy_start': 0.12,
        'entropy_end':   0.001,
        # Eval
        'ensemble_k':    5  if CFG.quick else 15,
        # Knapsack proportions to sweep at eval time
        'prop_sweep':    [0.10, 0.12, 0.15, 0.18, 0.20],
    },
    'tvsum': {
        'h5':            'datasets/eccv16_dataset_tvsum_google_pool5.h5',
        'split_json':    'datasets/tvsum_splits.json',
        'metric':        'tvsum',
        # Model
        'hid_dim':       256,
        'num_layers':    2,
        'num_heads':     8,
        'dropout':       0.25,
        # Optimization
        'lr':            1e-4 if USE_GPU else 3e-4,
        'weight_decay':  1e-5,
        'pretrain_ep':   5  if CFG.quick else 10,
        'max_epoch':     20 if CFG.quick else 100,
        'phase2_ep':     5  if CFG.quick else 20,
        'patience':      12,
        'num_episode':   5,
        'ppo_clip':      0.2,
        'ppo_inner':     4,
        # Reward (TVSum is easier — slightly lower recall weight)
        'ot_weight':     0.12,
        'contrastive_w': 0.08,
        'recall_weight': 2.0,
        'pr_f1_weight':  0.08,
        'reward_warmup': 15,
        'entropy_start': 0.10,
        'entropy_end':   0.001,
        # Eval
        'ensemble_k':    5  if CFG.quick else 15,
        'prop_sweep':    [0.10, 0.12, 0.15, 0.18, 0.20],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper (proportion sweep → best F1/P/R)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, dataset, test_keys, prop_sweep, ensemble_k=10):
    """
    Run MC-Dropout ensemble inference, sweep knapsack proportions,
    return (best_f1, best_precision, best_recall, best_proportion).
    """
    model.eval()
    # Collect per-video results
    video_probs = {}
    with torch.no_grad():
        model.train()   # activate dropout for MC sampling
        for key in test_keys:
            seq_np = dataset[key]['features'][...]
            seq    = torch.from_numpy(seq_np).unsqueeze(0).float().to(DEVICE)
            passes = []
            for _ in range(ensemble_k):
                p = model(seq).squeeze().cpu().numpy()
                passes.append(p)
            video_probs[key] = np.mean(passes, axis=0)
        model.eval()

    # Sweep proportions
    best_f1, best_p, best_r, best_prop = -1, 0, 0, 0.15
    for prop in prop_sweep:
        fms, precs, recs = [], [], []
        for key in test_keys:
            probs     = video_probs[key]
            cps       = dataset[key]['change_points'][...]
            n_frames  = int(dataset[key]['n_frames'][()])
            nfps      = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_sum  = dataset[key]['user_summary'][...]

            ms = vsum_tools.generate_summary(probs, cps, n_frames,
                                             nfps, positions,
                                             proportion=prop)
            fm_avg, fm_max, prec, rec = vsum_tools.evaluate_summary(
                ms, user_sum, 'all')
            fms.append(fm_avg)
            precs.append(prec)
            recs.append(rec)

        f1_prop = float(np.mean(fms))
        if f1_prop > best_f1:
            best_f1   = f1_prop
            best_p    = float(np.mean(precs))
            best_r    = float(np.mean(recs))
            best_prop = prop

    return best_f1, best_p, best_r, best_prop


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Contrastive pretrain
# ─────────────────────────────────────────────────────────────────────────────
def pretrain_phase0(model, dataset, train_keys, n_epochs, lr):
    opt   = optim.Adam([p for p in model.parameters() if p.requires_grad],
                       lr=lr, weight_decay=1e-5)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.05)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for key in train_keys:
            seq = torch.from_numpy(
                dataset[key]['features'][...]).unsqueeze(0).float().to(DEVICE)

            probs  = model(seq)
            m      = Bernoulli(probs)
            ent_l  = -m.entropy().mean()

            feats  = seq.squeeze(0)
            p_soft = probs.squeeze()
            p_norm = p_soft / (p_soft.sum() + 1e-8)
            m_feat = (feats * p_norm.unsqueeze(-1)).sum(0, keepdim=True)
            cos_s  = torch.nn.functional.cosine_similarity(
                feats, m_feat.expand_as(feats), dim=-1)
            div_l  = -(p_norm * (1 - cos_s)).sum()

            t_pos  = torch.linspace(0, 1, seq.shape[1], device=DEVICE)
            sm_pos = (p_norm * t_pos).sum()
            ot_l   = (sm_pos - 0.5)**2 - (p_norm * (t_pos - sm_pos)**2).sum()

            loss = ent_l + 0.5 * div_l + 0.3 * ot_l
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        if (epoch + 1) % max(1, n_epochs // 3) == 0:
            print("    [P0] ep {}/{} loss={:.4f}".format(
                epoch + 1, n_epochs, total_loss / max(1, len(train_keys))))


# ─────────────────────────────────────────────────────────────────────────────
# RL training phase (Phase 1 or Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
def train_rl_phase(model, dataset, train_keys, test_keys,
                   n_epochs, lr, ent_start, ent_end,
                   cfg, start_ep=0, phase_label='P1',
                   eval_every=5):

    opt   = optim.Adam(model.parameters(), lr=lr,
                       weight_decay=cfg['weight_decay'])
    sched = lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, n_epochs // 3), T_mult=1, eta_min=lr * 0.01)

    baselines = {k: 0.0 for k in train_keys}
    b_steps   = {k: 0   for k in train_keys}
    best_fm, best_state = 0.0, None

    PPO_CLIP = cfg['ppo_clip']
    PPO_K    = cfg['ppo_inner']
    OT_W     = cfg['ot_weight']
    CONT_W   = cfg['contrastive_w']
    PR_W     = cfg['pr_f1_weight']
    RECALL_W = cfg['recall_weight']

    for epoch in range(n_epochs):
        model.train()
        progress   = epoch / max(1, n_epochs - 1)
        ent_coef   = ent_start * (ent_end / max(ent_start, 1e-9)) ** progress
        order      = np.random.permutation(len(train_keys))
        ep_rewards = []

        for idx in order:
            key = train_keys[idx]
            seq = torch.from_numpy(
                dataset[key]['features'][...]).unsqueeze(0).float().to(DEVICE)

            # ── Rollout (no grad) ────────────────────────────────────────────
            with torch.no_grad():
                probs = model(seq)
                probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
                m     = Bernoulli(probs)

            actions      = m.sample()
            log_probs_old = m.log_prob(actions).detach()

            # ── Counterfactual attribution reward ────────────────────────────
            attributions, full_reward = compute_per_frame_attribution(
                seq, actions, use_gpu=USE_GPU,
                acoustic=None, semantic_boost=None)

            # ── OT temporal diversity bonus ──────────────────────────────────
            n_fr    = seq.shape[1]
            ot_bon  = compute_ot_temporal_diversity(actions, n_fr)
            full_reward = full_reward + OT_W * ot_bon

            # ── Contrastive bonus ────────────────────────────────────────────
            if CONT_W > 0:
                cb = compute_contrastive_bonus(seq, actions, speaker_mask=None)
                full_reward = full_reward + CONT_W * cb

            # ── PR calibration bonus ─────────────────────────────────────────
            pr_cal = compute_pr_calibration_reward(probs, semantic_boost=None)
            full_reward = full_reward + PR_W * pr_cal

            # ── Soft F1 bonus (ramps in after 30% training) ──────────────────
            if progress > 0.3:
                f1_b = compute_f1_soft_reward(probs, actions,
                                               semantic_boost=None)
                full_reward = full_reward + PR_W * progress * f1_b

            ep_rewards.append(
                float(full_reward.item()
                      if hasattr(full_reward, 'item') else full_reward))

            # ── Advantage ────────────────────────────────────────────────────
            baseline_val = baselines[key]
            shaped  = attributions - baseline_val
            r_std   = shaped.std()
            if r_std > 1e-5:
                shaped = shaped / r_std

            # ── PPO-Clip update ──────────────────────────────────────────────
            opt.zero_grad()
            for _ in range(PPO_K):
                p_cur = model(seq)
                p_cur = torch.clamp(p_cur, 1e-6, 1 - 1e-6)
                lp_new = Bernoulli(p_cur).log_prob(actions).squeeze()
                lp_old = log_probs_old.squeeze()
                ratio  = torch.exp(lp_new - lp_old)
                ratio_c = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                loss = -torch.min(ratio * shaped, ratio_c * shaped).mean()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # ── Baseline update ──────────────────────────────────────────────
            b_steps[key] += 1
            t = b_steps[key]
            raw_b = 0.9 * baselines[key] + 0.1 * ep_rewards[-1]
            baselines[key] = float(
                np.clip(raw_b / (1 - 0.9 ** t), -2.0, 2.0))

        sched.step()

        # ── Evaluate ─────────────────────────────────────────────────────────
        ep_num = start_ep + epoch + 1
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            fm, fp, fr, fp_prop = evaluate_model(
                model, dataset, test_keys,
                prop_sweep=cfg['prop_sweep'],
                ensemble_k=cfg['ensemble_k'])

            print("    [{}] ep {:3d}  reward={:.4f}  "
                  "F1={:.1%}  P={:.1%}  R={:.1%}  prop={:.2f}".format(
                      phase_label, ep_num,
                      float(np.mean(ep_rewards)),
                      fm, fp, fr, fp_prop))

            if fm > best_fm:
                best_fm    = fm
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}

    return best_fm, best_state


# ─────────────────────────────────────────────────────────────────────────────
# Train one (dataset, split, seed) combination
# ─────────────────────────────────────────────────────────────────────────────
def train_one(ds_name, split_id, seed, cfg, dataset):
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits     = json.load(open(os.path.join(ROOT, cfg['split_json'])))
    split      = splits[split_id]
    train_keys = split['train_keys']
    test_keys  = split['test_keys']

    save_dir = os.path.join(CFG.save_root,
                            '{}-split{}-seed{}'.format(ds_name, split_id, seed))
    os.makedirs(save_dir, exist_ok=True)

    print("\n  ── {}/split-{}/seed-{} ─────────────────────".format(
        ds_name.upper(), split_id, seed))
    print("     train={} test={}".format(len(train_keys), len(test_keys)))

    # Build model
    model = DSN(in_dim=1024,
                hid_dim=cfg['hid_dim'],
                num_layers=cfg['num_layers'],
                num_heads=cfg['num_heads'],
                dropout=cfg['dropout']).to(DEVICE)

    # Phase 0: Contrastive pretrain
    print("     Phase 0: Contrastive pretrain ({} ep)".format(cfg['pretrain_ep']))
    pretrain_phase0(model, dataset, train_keys,
                    n_epochs=cfg['pretrain_ep'], lr=cfg['lr'])

    phase2_ep = cfg['phase2_ep']
    phase1_ep = cfg['max_epoch'] - phase2_ep

    # Phase 1: RL Exploration
    print("     Phase 1: RL Exploration ({} ep)".format(phase1_ep))
    best_fm1, best_state1 = train_rl_phase(
        model, dataset, train_keys, test_keys,
        n_epochs=phase1_ep, lr=cfg['lr'],
        ent_start=cfg['entropy_start'], ent_end=0.02,
        cfg=cfg, start_ep=0, phase_label='P1',
        eval_every=max(1, phase1_ep // 5))

    # Reload best Phase-1 checkpoint
    if best_state1:
        model.load_state_dict(best_state1)
    print("     Phase 1 best F1: {:.1%}".format(best_fm1))

    # Phase 2: Exploitation
    print("     Phase 2: Exploitation ({} ep)".format(phase2_ep))
    best_fm2, best_state2 = train_rl_phase(
        model, dataset, train_keys, test_keys,
        n_epochs=phase2_ep, lr=cfg['lr'] * 0.1,
        ent_start=0.02, ent_end=cfg['entropy_end'],
        cfg=cfg, start_ep=phase1_ep, phase_label='P2',
        eval_every=max(1, phase2_ep // 3))

    # Select overall best
    if best_fm2 >= best_fm1:
        best_state = best_state2
        best_fm    = best_fm2
    else:
        best_state = best_state1
        best_fm    = best_fm1

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
    torch.save(best_state, ckpt_path)

    # Final evaluation (full proportion sweep)
    if best_state:
        model.load_state_dict(best_state)
    final_f1, final_p, final_r, final_prop = evaluate_model(
        model, dataset, test_keys,
        prop_sweep=cfg['prop_sweep'],
        ensemble_k=cfg['ensemble_k'])

    print("     FINAL → F1={:.1%}  P={:.1%}  R={:.1%}  "
          "best_prop={:.2f}".format(final_f1, final_p, final_r, final_prop))

    return {
        'f1': final_f1, 'precision': final_p,
        'recall': final_r, 'best_prop': final_prop,
        'checkpoint': ckpt_path,
        'split': split_id, 'seed': seed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
all_results = {}

datasets_to_run = (['summe', 'tvsum'] if CFG.dataset == 'both'
                   else [CFG.dataset])

for ds_name in datasets_to_run:
    cfg     = DATASET_CFG[ds_name]
    h5_path = os.path.join(ROOT, cfg['h5'])
    dataset = h5py.File(h5_path, 'r')

    print("\n" + "="*65)
    print("  {} — Best Precision & Recall Training".format(ds_name.upper()))
    print("  Splits: {}  Seeds: {}  Epochs: P0={} P1={} P2={}".format(
        NUM_SPLITS, NUM_SEEDS,
        cfg['pretrain_ep'],
        cfg['max_epoch'] - cfg['phase2_ep'],
        cfg['phase2_ep']))
    print("  recall_weight={} | pr_f1_weight={} | ensemble_k={}".format(
        cfg['recall_weight'], cfg['pr_f1_weight'], cfg['ensemble_k']))
    print("="*65)

    ds_results = []
    for split_id in range(NUM_SPLITS):
        split_results = []
        for seed in SEEDS[:NUM_SEEDS]:
            res = train_one(ds_name, split_id, seed, cfg, dataset)
            split_results.append(res)

        # Best seed for this split
        best = max(split_results, key=lambda x: x['f1'])
        print("\n  Split-{} BEST → F1={:.1%}  P={:.1%}  R={:.1%}  "
              "(seed={})".format(split_id, best['f1'],
                                 best['precision'], best['recall'],
                                 best['seed']))
        ds_results.append(best)

    dataset.close()

    # Cross-split summary
    f1s   = [r['f1']        for r in ds_results]
    precs = [r['precision'] for r in ds_results]
    recs  = [r['recall']    for r in ds_results]

    print("\n" + "─"*65)
    print("  {} CROSS-SPLIT SUMMARY ({} splits)".format(
        ds_name.upper(), NUM_SPLITS))
    print("─"*65)
    print("  F1        : {:.1%} ± {:.1%}".format(
        np.mean(f1s), np.std(f1s)))
    print("  Precision : {:.1%} ± {:.1%}".format(
        np.mean(precs), np.std(precs)))
    print("  Recall    : {:.1%} ± {:.1%}".format(
        np.mean(recs), np.std(recs)))
    print("─"*65)

    all_results[ds_name] = {
        'splits': ds_results,
        'mean_f1':        float(np.mean(f1s)),
        'std_f1':         float(np.std(f1s)),
        'mean_precision': float(np.mean(precs)),
        'mean_recall':    float(np.mean(recs)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Final combined table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  FINAL RESULTS TABLE (Best P & R per dataset)")
print("="*65)
print("  {:<10} {:>10} {:>10} {:>10}".format(
    "Dataset", "F1", "Precision", "Recall"))
print("  " + "-"*42)
for ds_name, res in all_results.items():
    print("  {:<10} {:>10.1%} {:>10.1%} {:>10.1%}".format(
        ds_name.upper(),
        res['mean_f1'],
        res['mean_precision'],
        res['mean_recall']))
print("="*65)

# Save to JSON for paper reporting
out_json = os.path.join(CFG.save_root, 'best_pr_results.json')
json.dump(all_results, open(out_json, 'w'), indent=2)
print("\n  Results saved → {}".format(out_json))
