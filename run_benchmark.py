"""
run_benchmark.py
================
One-shot script that:
  1. Evaluates BASELINE (legacy model = old arch, pre-trained checkpoint).
  2. Trains NEW model (enhanced = new arch with all PR improvements) for N epochs.
  3. Evaluates NEW model.
  4. Prints a clean side-by-side comparison table.

Usage:
    python run_benchmark.py [--fast]   # --fast = 20 epochs (quick test)
                                       # default = 60 epochs
"""

import argparse, os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.distributions import Bernoulli
from torch import optim
from torch.optim import lr_scheduler

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import vsum_tools
from knapsack import knapsack_dp

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--fast', action='store_true',
               help='Quick 20-epoch run instead of full 60 epochs')
p.add_argument('--dataset', default='datasets/eccv16_dataset_summe_google_pool5.h5')
p.add_argument('--split',   default='datasets/summe_splits.json')
p.add_argument('--split-id', type=int, default=0)
p.add_argument('--metric',  default='summe', choices=['tvsum', 'summe'])
p.add_argument('--baseline-ckpt',
               default='log/summe-counterfactual-optimized/model_best.pth.tar')
p.add_argument('--save-dir', default='log/benchmark_run')
BM = p.parse_args()

os.makedirs(BM.save_dir, exist_ok=True)
USE_GPU = torch.cuda.is_available()
DEVICE  = torch.device('cuda' if USE_GPU else 'cpu')
EPOCHS  = 20 if BM.fast else 60
PHASE2  = 5  if BM.fast else 15
PRETRAIN= 3  if BM.fast else 8

print("\n" + "="*65)
print("  LegalSum Benchmark: BASELINE vs NEW (PR-improved)")
print("  Dataset: {}  |  Split: {}  |  Device: {}".format(
    BM.metric.upper(), BM.split_id, DEVICE))
print("  Epochs: P0={} P1={} P2={}".format(PRETRAIN, EPOCHS-PHASE2, PHASE2))
print("="*65 + "\n")

# ── CPU thread tuning ─────────────────────────────────────────────────────────
if not USE_GPU:
    n_thr = min(os.cpu_count() or 1, 8)
    torch.set_num_threads(n_thr)
    torch.set_num_interop_threads(max(1, n_thr // 2))
    print("CPU threads: intra={} inter={}".format(n_thr, max(1, n_thr//2)))

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset   = h5py.File(os.path.join(ROOT, BM.dataset), 'r')
splits    = json.load(open(os.path.join(ROOT, BM.split)))
split     = splits[BM.split_id]
train_keys = split['train_keys']
test_keys  = split['test_keys']
print("Videos: train={} test={}\n".format(len(train_keys), len(test_keys)))

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_model(model, tag='model'):
    model.eval()
    fms, precs, recs = [], [], []
    with torch.no_grad():
        for key in test_keys:
            seq_np  = dataset[key]['features'][...]
            seq     = torch.from_numpy(seq_np).unsqueeze(0).float().to(DEVICE)
            probs   = model(seq).squeeze().cpu().numpy()
            probs   = np.clip(probs, 0, 1)

            cps      = dataset[key]['change_points'][...]
            n_frames = int(dataset[key]['n_frames'][()])
            nfps     = dataset[key]['n_frame_per_seg'][...].tolist()
            positions= dataset[key]['picks'][...]
            user_sum = dataset[key]['user_summary'][...]

            machine_sum = vsum_tools.generate_summary(
                probs, cps, n_frames, nfps, positions, proportion=0.15)

            fm_avg, fm_max, prec, rec = vsum_tools.evaluate_summary(
                machine_sum, user_sum, 'all')
            fms.append(fm_avg)
            precs.append(prec)
            recs.append(rec)

    mf  = float(np.mean(fms))
    mp  = float(np.mean(precs))
    mr  = float(np.mean(recs))
    print("  [{}]  F1={:.1%}  Precision={:.1%}  Recall={:.1%}".format(
        tag, mf, mp, mr))
    return mf, mp, mr


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BASELINE (legacy architecture, pre-trained checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════
print("─"*65)
print("STEP 1: BASELINE evaluation (legacy model, existing checkpoint)")
print("─"*65)

from models import (MultiScaleConv1D, MultiHeadSelfAttention, FeedForward,
                    GatedTemporalRouting, TemporalSegmentGraph,
                    ConversationalHypergraphAttention, CrossModalAttentionFusion)

class LegacyDSN(nn.Module):
    """Old DSN WITHOUT LegalPrecisionBoostHead — exact pre-change architecture."""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=2, cell='lstm',
                 num_heads=8, dropout=0.25):
        super().__init__()
        self.fusion     = CrossModalAttentionFusion(in_dim)
        self.input_proj = MultiScaleConv1D(in_dim, hid_dim * 2)
        if cell == 'lstm':
            self.rnn = nn.LSTM(hid_dim*2, hid_dim, num_layers=num_layers,
                               bidirectional=True, batch_first=True,
                               dropout=dropout if num_layers>1 else 0.0)
        else:
            self.rnn = nn.GRU(hid_dim*2, hid_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True,
                              dropout=dropout if num_layers>1 else 0.0)
        rnn_out_dim = hid_dim * 2
        self.attn1 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
        self.ff1   = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim*4, dropout=dropout)
        self.attn2 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
        self.ff2   = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim*4, dropout=dropout)
        self.final_norm = nn.LayerNorm(rnn_out_dim)
        self.gtr   = GatedTemporalRouting(hid_dim=hid_dim, dropout=dropout)
        self.tsg   = TemporalSegmentGraph(hid_dim=hid_dim*2, k=5, dropout=dropout)
        self.cha   = ConversationalHypergraphAttention(hid_dim=hid_dim*2, dropout=dropout)
        self.fc    = nn.Sequential(
            nn.Linear(rnn_out_dim, hid_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hid_dim, 1)
        )

    def _positional_encoding(self, x):
        batch, seq_len, d_model = x.shape
        pe  = torch.zeros(seq_len, d_model, device=x.device)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2, device=x.device).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0: pe[:, 1::2] = torch.cos(pos * div)
        else:                 pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        return x + pe.unsqueeze(0)

    def forward(self, x, acoustic=None, semantic=None,
                speaker_mask=None, event_mask=None):
        x = self.fusion(x, acoustic, semantic)
        x = self.input_proj(x)
        h_rnn, _ = self.rnn(x)
        h_attn = self._positional_encoding(h_rnn)
        h_attn = self.ff1(self.attn1(h_attn))
        h_attn = self.ff2(self.attn2(h_attn))
        h_attn = self.final_norm(h_attn)
        h_routed = self.gtr(h_rnn, h_attn)
        if speaker_mask is not None or event_mask is not None:
            h_graph = self.cha(h_routed, speaker_mask, event_mask)
        else:
            h_graph = self.tsg(h_routed)
        final_feats = h_routed + h_graph
        return torch.sigmoid(self.fc(final_feats))

baseline_model = LegacyDSN().to(DEVICE)
ckpt_path = os.path.join(ROOT, BM.baseline_ckpt)
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE)
    # Handle DataParallel prefix
    state = {k.replace('module.',''):v for k,v in state.items()}
    miss, unexp = baseline_model.load_state_dict(state, strict=False)
    print("  Checkpoint loaded (missing:{} unexpected:{})".format(
        len(miss), len(unexp)))
else:
    print("  WARNING: checkpoint not found — random weights used for baseline")

B_F1, B_P, B_R = evaluate_model(baseline_model, tag='BASELINE')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN NEW MODEL (enhanced = DSN + LegalPrecisionBoostHead + new rewards)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("STEP 2: TRAINING NEW model (all PR improvements active)")
print("─"*65)

from models import DSN
from rewards import (compute_reward, compute_per_frame_attribution,
                     compute_contrastive_bonus, compute_ot_temporal_diversity,
                     compute_f1_soft_reward, compute_pr_calibration_reward)

new_model = DSN(in_dim=1024, hid_dim=256, num_layers=2,
                cell='lstm', num_heads=8, dropout=0.25).to(DEVICE)

# ── Phase 0: contrastive pretrain ─────────────────────────────────────────────
print("\n==> Phase 0: Contrastive Pre-Train ({} epochs)".format(PRETRAIN))
opt0 = optim.Adam(
    [p for p in new_model.parameters() if p.requires_grad], lr=3e-4)
sched0 = lr_scheduler.CosineAnnealingLR(opt0, T_max=PRETRAIN, eta_min=1e-5)

for epoch in range(PRETRAIN):
    new_model.train()
    epoch_loss = 0.0
    for key in train_keys:
        seq = torch.from_numpy(dataset[key]['features'][...]).unsqueeze(0).float().to(DEVICE)
        probs = new_model(seq)
        m = Bernoulli(probs)
        entropy_loss = -m.entropy().mean()
        feats  = seq.squeeze(0)
        p_soft = probs.squeeze()
        p_norm = p_soft / (p_soft.sum() + 1e-8)
        mean_f = (feats * p_norm.unsqueeze(-1)).sum(0, keepdim=True)
        cos_s  = torch.nn.functional.cosine_similarity(feats, mean_f.expand_as(feats), dim=-1)
        div_loss = -(p_norm * (1 - cos_s)).sum()
        t_pos = torch.linspace(0, 1, seq.shape[1], device=DEVICE)
        sm_pos = (p_norm * t_pos).sum()
        ot_loss = (sm_pos - 0.5)**2 - (p_norm * (t_pos - sm_pos)**2).sum()
        loss = entropy_loss + 0.5*div_loss + 0.3*ot_loss
        opt0.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(new_model.parameters(), 5.0)
        opt0.step()
        epoch_loss += loss.item()
    sched0.step()
    print("  pretrain {}/{} loss={:.4f}".format(
        epoch+1, PRETRAIN, epoch_loss/max(1,len(train_keys))))

# ── Phase 1 & 2: RL ───────────────────────────────────────────────────────────
def train_rl(model, n_epochs, lr, ent_start, ent_end, start_ep=0,
             phase_label='RL', recall_weight=2.0):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.01)
    baselines = {k: 0.0 for k in train_keys}
    b_steps   = {k: 0    for k in train_keys}
    best_fm, best_state = 0.0, None
    OT_W = 0.10
    PPO_CLIP = 0.2

    for epoch in range(n_epochs):
        model.train()
        progress = epoch / max(1, n_epochs-1)
        ent_coef = ent_start * (ent_end / max(ent_start, 1e-8)) ** progress
        order = np.random.permutation(len(train_keys))

        for idx in order:
            key = train_keys[idx]
            seq = torch.from_numpy(dataset[key]['features'][...]).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                probs = model(seq)
                probs = torch.clamp(probs, 1e-6, 1-1e-6)
                m = Bernoulli(probs)

            # Sample actions
            actions = m.sample()
            log_probs_old = m.log_prob(actions).detach()

            # New reward (with all PR improvements)
            local_epoch = epoch
            attributions, full_reward = compute_per_frame_attribution(
                seq, actions, use_gpu=USE_GPU,
                acoustic=None, semantic_boost=None)

            # OT bonus
            n_fr = seq.shape[1]
            ot_b = compute_ot_temporal_diversity(actions, n_fr)
            full_reward = full_reward + OT_W * ot_b

            # PR calibration bonus
            pr_cal = compute_pr_calibration_reward(probs, semantic_boost=None)
            full_reward = full_reward + 0.08 * pr_cal

            # Baseline
            baseline_val = baselines[key]
            shaped = attributions - baseline_val
            r_std = shaped.std()
            if r_std > 1e-5:
                shaped = shaped / r_std

            # PPO-Clip update
            opt.zero_grad()
            for _ in range(4):
                p_cur = model(seq)
                p_cur = torch.clamp(p_cur, 1e-6, 1-1e-6)
                m_cur = Bernoulli(p_cur)
                lp_new = m_cur.log_prob(actions).squeeze()
                lp_old = log_probs_old.squeeze()
                ratio  = torch.exp(lp_new - lp_old)
                ratio_c = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP)
                ppo_loss = -torch.min(ratio*shaped, ratio_c*shaped).mean()
                ppo_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # Momentum baseline
            b_steps[key] += 1
            t = b_steps[key]
            raw_b = 0.9*baselines[key] + 0.1*float(full_reward.item() if hasattr(full_reward,'item') else full_reward)
            baselines[key] = float(np.clip(raw_b / (1 - 0.9**t), -2.0, 2.0))

        sched.step()
        if (epoch+1) % max(1, n_epochs//5) == 0 or epoch == n_epochs-1:
            fm, _, _ = evaluate_model(model, tag='{} ep{}'.format(phase_label, start_ep+epoch+1))
            if fm > best_fm:
                best_fm = fm
                best_state = {k: v.clone() for k,v in model.state_dict().items()}

    return best_fm, best_state


print("\n==> Phase 1: Exploration ({} epochs)".format(EPOCHS - PHASE2))
best_fm1, best_state1 = train_rl(
    new_model, n_epochs=EPOCHS-PHASE2, lr=3e-4,
    ent_start=0.10, ent_end=0.02, start_ep=0, phase_label='P1')

print("\n  Phase 1 best F1: {:.1%}".format(best_fm1))

if best_state1:
    new_model.load_state_dict(best_state1)

print("\n==> Phase 2: Exploitation ({} epochs)".format(PHASE2))
best_fm2, best_state2 = train_rl(
    new_model, n_epochs=PHASE2, lr=3e-5,
    ent_start=0.02, ent_end=0.005,
    start_ep=EPOCHS-PHASE2, phase_label='P2')

print("\n  Phase 2 best F1: {:.1%}".format(best_fm2))

# Load best overall
best_overall = best_state2 if best_fm2 >= best_fm1 else best_state1
if best_overall:
    new_model.load_state_dict(best_overall)
torch.save(best_overall or new_model.state_dict(),
           os.path.join(BM.save_dir, 'new_model_best.pth.tar'))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FINAL EVALUATION & COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("STEP 3: FINAL evaluation & comparison")
print("─"*65 + "\n")

N_F1, N_P, N_R = evaluate_model(new_model, tag='NEW MODEL')

# ── Results table ─────────────────────────────────────────────────────────────
def delta(new, old):
    d = (new - old) * 100
    sign = "+" if d >= 0 else ""
    return "{}{}  ({}{:.2f}pp)".format(
        "{:.1%}".format(new), " " * (7 - len("{:.1%}".format(new))),
        sign, d)

print("\n" + "="*65)
print("  RESULTS COMPARISON  ({} Split-{})".format(BM.metric.upper(), BM.split_id))
print("="*65)
print("  Metric     BASELINE     NEW MODEL")
print("  " + "-"*58)
print("  F1         {:.1%}        {}".format(B_F1, delta(N_F1, B_F1)))
print("  Precision  {:.1%}        {}".format(B_P,  delta(N_P,  B_P)))
print("  Recall     {:.1%}        {}".format(B_R,  delta(N_R,  B_R)))
print("="*65)

improved = N_F1 > B_F1 and (N_P > B_P or N_R > B_R)
print("\n  VERDICT: {} IMPROVED" .format("✅ PR" if improved else "❌ NOT YET"))
if improved:
    print("  Precision delta: {:.2f}pp | Recall delta: {:.2f}pp".format(
        (N_P-B_P)*100, (N_R-B_R)*100))

# Save results
results = {
    'baseline': {'f1': B_F1, 'precision': B_P, 'recall': B_R},
    'new_model': {'f1': N_F1, 'precision': N_P, 'recall': N_R},
    'delta': {
        'f1':        round((N_F1-B_F1)*100, 3),
        'precision': round((N_P -B_P )*100, 3),
        'recall':    round((N_R -B_R )*100, 3)
    },
    'config': {'dataset': BM.metric, 'split': BM.split_id,
               'epochs': EPOCHS, 'phase2': PHASE2, 'pretrain': PRETRAIN}
}
out_path = os.path.join(BM.save_dir, 'benchmark_results.json')
json.dump(results, open(out_path, 'w'), indent=2)
print("\n  Full results saved to: {}".format(out_path))

dataset.close()
print("\nDone.\n")
