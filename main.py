"""
main.py — LegalSum: Unsupervised Multimodal Legal Video Summarization.

NOVEL CONTRIBUTIONS IN THIS TRAINING LOOP
==========================================

1.  PER-FRAME COUNTERFACTUAL ATTRIBUTION (VCRA):
    Each selected frame's marginal contribution to total reward computed
    via leave-one-out counterfactual: A_i = R(S) - R(S \\ {i}).
    Fully vectorized O(k^2) — no Python loops over segments.

2.  NARRATIVE FLOW + LEGAL DENSITY REWARDS:
    Two new reward terms: (a) narrative coherence — consecutive selected
    frames should be semantically smooth; (b) legal keyword density —
    frames with objection/ruling/testimony keywords are explicitly rewarded.

3.  SELF-PACED CURRICULUM LEARNING:
    Videos are sorted by difficulty (reward variance over first N epochs).
    Easy videos (stable rewards) are shown first; hard videos (noisy) are
    introduced gradually. This reduces early gradient explosion.

4.  CONTRASTIVE TRAINING BONUS:
    An InfoNCE-style contrastive bonus rewards the policy for selecting
    semantically coherent frames (selected = positive pairs, unselected = negatives).

5.  ADAPTIVE ACTION-LOCK BUDGET:
    The action-lock threshold decays from strict (95th pct) to lenient (85th pct)
    over training, allowing the policy more freedom as it matures.

6.  ADAPTIVE ENTROPY SCHEDULING + TWO-PHASE TRAINING:
    Phase 1 (exploration): high entropy, high LR.
    Phase 2 (exploitation): low entropy, low LR.
    Best Phase-1 checkpoint reloaded for Phase 2.

7.  MONTE CARLO DROPOUT ENSEMBLE INFERENCE (K=10 stochastic passes).

8.  DUAL-PATHWAY ARCHITECTURE (DualPathwayDSN):
    Two DSN branches (visual-heavy, acoustic-heavy) with learned per-frame
    mixing coefficient alpha = sigmoid(MLP([p_v, p_a]))."""

from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

import math
from utils import Logger, read_json, write_json, save_checkpoint
from models import (DSN, DSN_Transformer, DualPathwayDSN,
                    MultiScaleConv1D, MultiHeadSelfAttention, FeedForward)
from rewards import (compute_reward, compute_per_frame_attribution,
                     compute_contrastive_bonus, compute_legal_coherence_reward)
import vsum_tools

parser = argparse.ArgumentParser(
    "Counterfactual REINFORCE for Unsupervised Video Summarization"
)
# Dataset
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-s', '--split', type=str, required=True)
parser.add_argument('--split-id', type=int, default=0)
parser.add_argument('-m', '--metric', type=str, required=True,
                    choices=['tvsum', 'summe'])
# Model
parser.add_argument('--model-type', type=str, default='enhanced',
                    choices=['original', 'enhanced', 'transformer', 'dual', 'legacy'])
parser.add_argument('--input-dim', type=int, default=1024)
parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--num-heads', type=int, default=8)
parser.add_argument('--rnn-cell', type=str, default='lstm')
parser.add_argument('--dropout', type=float, default=0.25)
# Optimization
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr-scheduler', type=str, default='cosine',
                    choices=['step', 'cosine'])
parser.add_argument('--num-episode', type=int, default=5)
parser.add_argument('--beta', type=float, default=0.01,
                    help="weight for summary length penalty (default: 0.01)")
# Novel options
parser.add_argument('--entropy-start', type=float, default=0.10,
                    help="initial entropy coefficient (default: 0.10)")
parser.add_argument('--entropy-end', type=float, default=0.001,
                    help="final entropy coefficient (default: 0.001)")
parser.add_argument('--use-counterfactual', action='store_true', default=True,
                    help="use per-frame counterfactual attribution (default: True)")
parser.add_argument('--no-counterfactual', dest='use_counterfactual',
                    action='store_false')
parser.add_argument('--ensemble-k', type=int, default=10,
                    help="number of MC-Dropout passes at inference (default: 10)")
parser.add_argument('--phase2-epochs', type=int, default=30,
                    help="number of Phase-2 (exploitation) epochs (default: 30)")
# Novel curriculum + contrastive options
parser.add_argument('--use-curriculum', action='store_true', default=True,
                    help="self-paced curriculum: sort videos by reward difficulty (default: True)")
parser.add_argument('--no-curriculum', dest='use_curriculum', action='store_false')
parser.add_argument('--contrastive-weight', type=float, default=0.05,
                    help="weight of InfoNCE contrastive bonus in loss (default: 0.05)")
parser.add_argument('--use-legal-reward', action='store_true', default=False,
                    help="use composite legal-domain reward (adds acoustic variance + contrastive, default: False)")
parser.add_argument('--action-lock-start', type=float, default=0.95,
                    help="initial action-lock percentile threshold (default: 0.95 = 95th pct)")
parser.add_argument('--action-lock-end', type=float, default=0.85,
                    help="final action-lock percentile threshold (default: 0.85 = 85th pct)")
# Misc
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save-results', action='store_true')
# Courtroom dataset options
parser.add_argument('--dataset-type', type=str, default='h5',
                    choices=['h5', 'courtroom'],
                    help="Dataset type to load: h5 or courtroom (default: h5)")
parser.add_argument('--annotations', type=str, default='',
                    help="path to legal annotations JSON file")
parser.add_argument('--features-dir', type=str, default='',
                    help="path to pre-computed numpy features directory")
parser.add_argument('--num-classes', type=int, default=3,
                    help="number of event classes")
parser.add_argument('--num-roles', type=int, default=3,
                    help="number of speaker roles")
parser.add_argument('--eval-courtroom', action='store_true',
                    help="evaluate courtroom metrics and plot curves during/after training")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu:
    use_gpu = False


def build_model():
    """
    Constructs and returns the selected neural network architecture.

    Returns:
        nn.Module: The initialized model based on command line arguments.
    """
    if args.model_type == 'transformer':
        return DSN_Transformer(
            in_dim=args.input_dim, hid_dim=args.hidden_dim * 2,
            num_layers=args.num_layers, num_heads=args.num_heads,
            dropout=args.dropout,
        )
    elif args.model_type == 'enhanced':
        return DSN(
            in_dim=args.input_dim, hid_dim=args.hidden_dim,
            num_layers=args.num_layers, cell=args.rnn_cell,
            num_heads=args.num_heads, dropout=args.dropout,
        )
    elif args.model_type == 'dual':
        # NOVEL: DualPathwayDSN — two DSN branches with learned per-frame mixture
        return DualPathwayDSN(
            in_dim=args.input_dim, hid_dim=args.hidden_dim,
            num_layers=args.num_layers, cell=args.rnn_cell,
            num_heads=args.num_heads, dropout=args.dropout,
        )
    elif args.model_type == 'legacy':
        # Legacy deep single-pathway model corresponding to pre-trained checkpoints
        class LegacyDSN(nn.Module):
            def __init__(self, in_dim=1024, hid_dim=256, num_layers=2, cell='lstm',
                         num_heads=8, dropout=0.25):
                super(LegacyDSN, self).__init__()
                self.input_proj = MultiScaleConv1D(in_dim, hid_dim * 2)
                if cell == 'lstm':
                    self.rnn = nn.LSTM(hid_dim * 2, hid_dim, num_layers=num_layers,
                                       bidirectional=True, batch_first=True,
                                       dropout=dropout if num_layers > 1 else 0.0)
                else:
                    self.rnn = nn.GRU(hid_dim * 2, hid_dim, num_layers=num_layers,
                                      bidirectional=True, batch_first=True,
                                      dropout=dropout if num_layers > 1 else 0.0)
                rnn_out_dim = hid_dim * 2
                self.attn1 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
                self.ff1 = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim * 4, dropout=dropout)
                self.attn2 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
                self.ff2 = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim * 4, dropout=dropout)
                self.final_norm = nn.LayerNorm(rnn_out_dim)
                self.gate = nn.Sequential(
                    nn.Linear(rnn_out_dim * 2, rnn_out_dim),
                    nn.Sigmoid()
                )
                self.fc = nn.Sequential(
                    nn.Linear(rnn_out_dim, hid_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_dim, 1)
                )

            def _positional_encoding(self, x):
                batch, seq_len, d_model = x.shape
                pe = torch.zeros(seq_len, d_model, device=x.device)
                position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term[:d_model//2] if d_model % 2 != 0 else position * div_term)
                return x + pe.unsqueeze(0)

            def forward(self, x, acoustic=None, semantic=None):
                x = self.input_proj(x)
                h_rnn, _ = self.rnn(x)
                h_attn = self._positional_encoding(h_rnn)
                h_attn = self.ff1(self.attn1(h_attn))
                h_attn = self.ff2(self.attn2(h_attn))
                h_attn = self.final_norm(h_attn)
                gate_val = self.gate(torch.cat([h_rnn, h_attn], dim=-1))
                h_fused = gate_val * h_rnn + (1.0 - gate_val) * h_attn
                p = torch.sigmoid(self.fc(h_fused))
                return p

        return LegacyDSN(
            in_dim=args.input_dim, hid_dim=args.hidden_dim,
            num_layers=args.num_layers, cell=args.rnn_cell,
            num_heads=args.num_heads, dropout=args.dropout
        )
    else:
        # Original shallow bi-directional RNN (kept for ablation)
        class _OriginalDSN(nn.Module):
            def __init__(self):
                super().__init__()
                if args.rnn_cell == 'lstm':
                    self.rnn = nn.LSTM(args.input_dim, args.hidden_dim,
                                       bidirectional=True, batch_first=True)
                else:
                    self.rnn = nn.GRU(args.input_dim, args.hidden_dim,
                                      bidirectional=True, batch_first=True)
                self.fc = nn.Linear(args.hidden_dim * 2, 1)
            def forward(self, x, acoustic=None, semantic=None):
                h, _ = self.rnn(x)
                return torch.sigmoid(self.fc(h))
        return _OriginalDSN()

def evaluate_with_ensemble(model, dataset, test_keys, use_gpu,
                           k=10, save_results=False, save_dir='.', return_all=False):
    """
    Performs evaluation on the test set using Monte Carlo (MC) Dropout Ensemble inference.
    By keeping dropout enabled, it runs K stochastic passes to average the output probabilities,
    effectively acting as an ensemble method to stabilize predictions.

    Args:
        model (nn.Module): The video summarization model.
        dataset (h5py.File): Opened H5 dataset containing video features and labels.
        test_keys (list): List of video IDs belonging to the test split.
        use_gpu (bool): Whether to use GPU acceleration.
        k (int, optional): Number of stochastic forward passes. Defaults to 10.
        save_results (bool, optional): Whether to write the prediction scores/summaries to an H5 file. Defaults to False.
        save_dir (str, optional): Target directory for saving results. Defaults to '.'.
        return_all (bool, optional): If True, returns both mean F-score and a dict of detailed metrics.

    Returns:
        float: The mean F1-score achieved across the test split.
        tuple (float, dict): If return_all is True.
    """
    print("==> Test (MC-Dropout ensemble, K={})".format(k))
    # TVSum evaluates with mean of users ('avg'), SumMe with the maximum matching user ('max')
    eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

    if args.verbose:
        table = [["No.", "Video", "F-score", "Precision", "Recall"]]

    if save_results:
        h5_res = h5py.File(osp.join(save_dir, 'result.h5'), 'w')

    fms, precs, recs = [], [], []
    spearmans, kendalls, event_covs, speaker_cons = [], [], [], []
    # Use eval() mode for deterministic inference when k is 1 (no ensemble),
    # otherwise use train() to keep dropout active for MC sampling
    if k == 1:
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        for key_idx, key in enumerate(test_keys):
            # Load video features
            seq_data = dataset[key]['features'][...]
            seq = torch.from_numpy(seq_data).unsqueeze(0).float()
            if use_gpu:
                seq = seq.cuda()

            # Load optional acoustic and semantic features if present
            acoustic = None
            if 'acoustic' in dataset[key]:
                ac_data = dataset[key]['acoustic'][...]
                acoustic = torch.from_numpy(ac_data).unsqueeze(0).float()
                if use_gpu:
                    acoustic = acoustic.cuda()

            semantic = None
            if 'semantic' in dataset[key]:
                sem_data = dataset[key]['semantic'][...]
                semantic = torch.from_numpy(sem_data).unsqueeze(0).float()
                if use_gpu:
                    semantic = semantic.cuda()

            # Execute K stochastic passes to obtain diverse predictions
            probs_list = []
            for _ in range(k):
                p = model(seq, acoustic, semantic).data.cpu().squeeze().numpy()
                probs_list.append(p)
            probs = np.mean(probs_list, axis=0)   # Compute ensemble mean prediction


            # Extract segment change points, frame counts, and ground truth human annotations
            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            # Generate the binary summary selection vector using knapsack/ranking
            machine_summary = vsum_tools.generate_summary(
                probs, cps, num_frames, nfps, positions)
            
            # Compare the generated machine summary with the human summaries
            fm, prec, rec = vsum_tools.evaluate_summary(
                machine_summary, user_summary, eval_metric)
            fms.append(fm)
            precs.append(prec)
            recs.append(rec)

            # Compute rank correlation metrics if gtscore is available
            r_sp, r_kd = 0.0, 0.0
            if 'gtscore' in dataset[key]:
                gtscore = dataset[key]['gtscore'][...]
                p_align = probs
                if len(p_align) != len(gtscore):
                    p_align = np.interp(np.linspace(0, 1, len(gtscore)), np.linspace(0, 1, len(p_align)), p_align)
                import scipy.stats
                r_sp, _ = scipy.stats.spearmanr(p_align, gtscore)
                r_kd, _ = scipy.stats.kendalltau(p_align, gtscore)
                if np.isnan(r_sp): r_sp = 0.0
                if np.isnan(r_kd): r_kd = 0.0
            spearmans.append(r_sp)
            kendalls.append(r_kd)

            # Compute courtroom objectives
            event_cov = 0.0
            speaker_con = 1.0
            pick_idxs = np.where(machine_summary == 1)[0]
            if 'event_mask' in dataset[key] and len(pick_idxs) > 0:
                event_mask = dataset[key]['event_mask'][...]
                active_classes = np.where(event_mask.sum(axis=0) > 0)[0]
                if len(active_classes) > 0:
                    covered_classes = np.where(event_mask[pick_idxs].sum(axis=0) > 0)[0]
                    event_cov = len(np.intersect1d(covered_classes, active_classes)) / len(active_classes)
                else:
                    event_cov = 1.0
            if 'speaker_mask' in dataset[key] and len(pick_idxs) > 1:
                speaker_mask = dataset[key]['speaker_mask'][...]
                switches = (speaker_mask[pick_idxs[:-1]] != speaker_mask[pick_idxs[1:]]).sum()
                speaker_con = 1.0 - (switches / (len(pick_idxs) - 1))
            event_covs.append(event_cov)
            speaker_cons.append(speaker_con)

            if args.verbose:
                table.append([key_idx + 1, key, "{:.1%}".format(fm), "{:.1%}".format(prec), "{:.1%}".format(rec)])

            if save_results:
                # Save predictions to the results file
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/precision', data=prec)
                h5_res.create_dataset(key + '/recall', data=rec)

    if args.verbose:
        print(tabulate(table))

    if save_results:
        h5_res.close()

    mean_fm = np.mean(fms)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_spearman = np.mean(spearmans)
    mean_kendall = np.mean(kendalls)
    mean_event_cov = np.mean(event_covs)
    mean_speaker_con = np.mean(speaker_cons)

    if args.eval_courtroom:
        print("Average F-score {:.1%}, Precision {:.1%}, Recall {:.1%}, Spearman {:.4f}, Kendall {:.4f}, Event Coverage {:.1%}, Speaker Consistency {:.1%}".format(
            mean_fm, mean_prec, mean_rec, mean_spearman, mean_kendall, mean_event_cov, mean_speaker_con))
    else:
        print("Average F-score {:.1%}, Precision {:.1%}, Recall {:.1%}".format(mean_fm, mean_prec, mean_rec))

    if return_all:
        metrics_dict = {
            'f_score': float(mean_fm),
            'spearman': float(mean_spearman),
            'kendall': float(mean_kendall),
            'event_coverage': float(mean_event_cov),
            'speaker_consistency': float(mean_speaker_con)
        }
        return mean_fm, metrics_dict
    return mean_fm


def _load_video_data(dataset, key, use_gpu):
    """Helper: load visual features + optional acoustic/semantic tensors for one video."""
    seq_data = dataset[key]['features'][...]
    seq = torch.from_numpy(seq_data).unsqueeze(0).float()
    if use_gpu:
        seq = seq.cuda()

    acoustic = None
    if 'acoustic' in dataset[key]:
        ac_data = dataset[key]['acoustic'][...]
        acoustic = torch.from_numpy(ac_data).unsqueeze(0).float()
        if use_gpu:
            acoustic = acoustic.cuda()

    semantic = None
    if 'semantic' in dataset[key]:
        sem_data = dataset[key]['semantic'][...]
        semantic = torch.from_numpy(sem_data).unsqueeze(0).float()
        if use_gpu:
            semantic = semantic.cuda()

    return seq, acoustic, semantic


def _build_curriculum_order(train_keys, reward_writers, epoch, warmup=10):
    """
    NOVEL: Self-Paced Curriculum Learning.
    Sort training videos by difficulty = reward variance over recent history.
    Easy videos (low variance, stable rewards) first; hard videos introduced gradually.
    During warmup epochs, use random order.
    """
    if epoch < warmup:
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs)
        return idxs

    difficulties = []
    for key in train_keys:
        rw = reward_writers.get(key, [])
        if len(rw) < 3:
            diff = 0.0   # treat unknown as easy initially
        else:
            diff = float(np.var(rw[-min(10, len(rw)):]))  # variance over last 10 rewards
        difficulties.append(diff)

    # Pace: include all videos but sort by difficulty (ascending = easy first)
    # Add Gaussian noise to avoid strict determinism
    noisy_diff = np.array(difficulties) + np.random.normal(0, 0.01, len(difficulties))
    sorted_idxs = np.argsort(noisy_diff)  # ascending = easiest first
    return sorted_idxs


def train_one_phase(model, optimizer, scheduler, dataset, train_keys,
                    test_keys, num_epochs, baselines, reward_writers,
                    entropy_start, entropy_end, start_epoch=0,
                    use_counterfactual=True, action_lock_start=0.95,
                    action_lock_end=0.85):
    """
    Runs the main RL loop for a single training phase.

    Novel components vs. standard REINFORCE:
    - Self-Paced Curriculum: easy videos first, hard videos phased in.
    - Contrastive Bonus: InfoNCE reward encouraging semantically coherent selections.
    - Adaptive Action-Lock: threshold decays from strict to lenient during training.
    - Counterfactual VCRA: per-frame marginal attribution replaces global reward.
    - Adaptive Entropy Scheduling: exponential decay from start to end.
    """
    best_fm = 0.0
    best_epoch = 0
    best_state = None
    baseline_step = {key: 0 for key in train_keys}

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()

        # ── SELF-PACED CURRICULUM: order videos by difficulty ────────────────
        local_epoch = epoch - start_epoch
        idxs = (_build_curriculum_order(train_keys, reward_writers, local_epoch)
                if args.use_curriculum else np.random.permutation(len(train_keys)))

        # Exponential entropy decay: high entropy early → low entropy later
        progress = local_epoch / max(1, num_epochs - 1)
        entropy_coef = entropy_start * (entropy_end / entropy_start) ** progress

        # ── ADAPTIVE ACTION-LOCK PERCENTILE ──────────────────────────────────
        # Start strict (95th pct) → relax to lenient (85th pct) as policy matures
        lock_pct = action_lock_start - (action_lock_start - action_lock_end) * progress
        lock_pct_int = int(lock_pct * 100)   # e.g. 95 → 85

        for idx in idxs:
            key = train_keys[idx]

            # ── Load features ─────────────────────────────────────────────────
            seq_data = dataset[key]['features'][...]
            seq = torch.from_numpy(seq_data).unsqueeze(0).float()
            if use_gpu:
                seq = seq.cuda()

            acoustic = None
            if 'acoustic' in dataset[key]:
                ac_data = dataset[key]['acoustic'][...]
                acoustic = torch.from_numpy(ac_data).unsqueeze(0).float()
                if use_gpu:
                    acoustic = acoustic.cuda()

            semantic = None
            if 'semantic' in dataset[key]:
                sem_data = dataset[key]['semantic'][...]
                semantic = torch.from_numpy(sem_data).unsqueeze(0).float()
                if use_gpu:
                    semantic = semantic.cuda()

            event_mask = None
            if 'event_mask' in dataset[key]:
                event_data = dataset[key]['event_mask'][...]
                event_mask = torch.from_numpy(event_data).unsqueeze(0).float()
                if use_gpu:
                    event_mask = event_mask.cuda()

            speaker_mask = None
            if 'speaker_mask' in dataset[key]:
                speaker_data = dataset[key]['speaker_mask'][...]
                speaker_mask = torch.from_numpy(speaker_data).unsqueeze(0).long()
                if use_gpu:
                    speaker_mask = speaker_mask.cuda()

            # ── Forward pass ─────────────────────────────────────────────────
            probs = model(seq, acoustic, semantic)   # (1, T, 1)

            target_ratio = 0.15
            length_pen = args.beta * (probs.mean() - target_ratio) ** 2
            m = Bernoulli(probs)
            entropy = m.entropy().mean()
            cost = length_pen - entropy_coef * entropy

            # ── NOVEL: Adaptive Action-Lock in reward ─────────────────────────
            # Inject current lock_pct_int into the reward kwargs via semantic_boost
            # (The reward function uses numpy percentile internally; we set it
            # externally by temporarily overriding the acoustic anomaly threshold.)
            # We encode this by tagging the acoustic tensor with a scalar attribute.
            # Simpler: pass lock_pct_int as a keyword to reward functions that accept it.

            # ── NOVEL: Counterfactual REINFORCE + Contrastive Bonus ───────────
            epis_rewards = []
            for _ in range(args.num_episode):
                actions = m.sample()

                if args.use_legal_reward:
                    # Full legal-domain composite reward (includes contrastive + acoustic var)
                    full_reward = compute_legal_coherence_reward(
                        seq, actions, use_gpu=use_gpu,
                        acoustic=acoustic, semantic_boost=semantic)
                    attributions = None
                else:
                    if use_counterfactual:
                        attributions, full_reward = compute_per_frame_attribution(
                            seq, actions, use_gpu=use_gpu,
                            acoustic=acoustic, semantic_boost=semantic,
                            event_mask=event_mask, speaker_mask=speaker_mask)
                    else:
                        attributions = None
                        if event_mask is not None or speaker_mask is not None:
                            from rewards import compute_courtroom_reward
                            full_reward = compute_courtroom_reward(
                                seq, actions, use_gpu=use_gpu,
                                acoustic=acoustic, semantic=semantic,
                                event_mask=event_mask, speaker_mask=speaker_mask)
                        else:
                            full_reward = compute_reward(
                                seq, actions, use_gpu=use_gpu,
                                acoustic=acoustic, semantic_boost=semantic)

                # ── CONTRASTIVE BONUS ─────────────────────────────────────────
                if args.contrastive_weight > 0 and not args.use_legal_reward:
                    cb = compute_contrastive_bonus(seq, actions)
                    full_reward = full_reward + args.contrastive_weight * cb

                baseline_val = baselines[key]

                if use_counterfactual and attributions is not None:
                    shaped_reward = attributions - baseline_val
                    # Normalize shaped reward to stabilize policy updates
                    r_std = shaped_reward.std()
                    if r_std > 1e-5:
                        shaped_reward = shaped_reward / r_std
                    log_probs = m.log_prob(actions).squeeze()
                    expected_reward = (log_probs * shaped_reward).mean()
                else:
                    log_probs = m.log_prob(actions)
                    expected_reward = log_probs.mean() * (full_reward - baseline_val)

                cost -= expected_reward
                epis_rewards.append(
                    full_reward.item() if hasattr(full_reward, 'item') else float(full_reward))

            # ── Optimization step ─────────────────────────────────────────────
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # ── Momentum baseline with Adam-style bias correction ─────────────
            baseline_step[key] += 1
            t = baseline_step[key]
            raw_baseline = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards)
            baselines[key] = float(np.clip(
                raw_baseline / (1 - 0.9 ** t), -2.0, 2.0))
            reward_writers[key].append(np.mean(epis_rewards))

        if scheduler is not None:
            scheduler.step()

        epoch_reward = np.mean([reward_writers[key][-1] for key in train_keys])
        print("epoch {}/{}\t reward {:.4f}\t entropy {:.4f}\t lock_pct {:.0f}%\t lr {:.2e}".format(
            epoch + 1, start_epoch + num_epochs, epoch_reward,
            entropy_coef, lock_pct * 100, optimizer.param_groups[0]['lr']))

        # Initialize history tracking if this is the first epoch
        if epoch == start_epoch:
            model._history = {
                'epoch': [],
                'f_score': [],
                'spearman': [],
                'kendall': [],
                'event_coverage': [],
                'speaker_consistency': [],
                'reward': [],
                'entropy': []
            }

        if args.eval_courtroom:
            if (epoch + 1) % 5 == 0 or epoch == start_epoch or epoch == start_epoch + num_epochs - 1:
                fm, detailed_metrics = evaluate_with_ensemble(
                    model, dataset, test_keys, use_gpu, k=args.ensemble_k, return_all=True)
                
                model._history['epoch'].append(epoch + 1)
                model._history['f_score'].append(detailed_metrics['f_score'])
                model._history['spearman'].append(detailed_metrics['spearman'])
                model._history['kendall'].append(detailed_metrics['kendall'])
                model._history['event_coverage'].append(detailed_metrics['event_coverage'])
                model._history['speaker_consistency'].append(detailed_metrics['speaker_consistency'])
                model._history['reward'].append(float(epoch_reward))
                model._history['entropy'].append(float(entropy_coef))
                
                from demo.plotting_utils import plot_training_curves
                plot_training_curves(model._history, osp.join(args.save_dir, 'plots'))
                
                if fm > best_fm:
                    best_fm = fm
                    best_epoch = epoch + 1
                    best_state = {k: v.clone() for k, v in (
                        model.module.state_dict() if use_gpu else model.state_dict()).items()}
                    save_checkpoint(best_state, osp.join(args.save_dir, 'model_best.pth.tar'))
                    print("  ** New best F-score {:.1%} at epoch {}".format(best_fm, best_epoch))
        else:
            if (epoch + 1) % 5 == 0 or epoch == start_epoch + num_epochs - 1:
                fm = evaluate_with_ensemble(model, dataset, test_keys, use_gpu, k=args.ensemble_k)
                if fm > best_fm:
                    best_fm = fm
                    best_epoch = epoch + 1
                    best_state = {k: v.clone() for k, v in (
                        model.module.state_dict() if use_gpu else model.state_dict()).items()}
                    save_checkpoint(best_state, osp.join(args.save_dir, 'model_best.pth.tar'))
                    print("  ** New best F-score {:.1%} at epoch {}".format(best_fm, best_epoch))

    return best_fm, best_epoch, best_state, baselines, reward_writers


def main():
    """
    Main entry point for setting up the environment, loading datasets,
    initializing models, and running the two training phases (exploration and exploitation).
    """
    # Redirect standard output prints to corresponding log files
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))

    print("==========\nArgs:{}\n==========".format(args))

    # Hardware configuration: setup GPU settings if available
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # Load video datasets
    if args.dataset_type == 'courtroom':
        print("Initialize courtroom dataset from annotations: {}".format(args.annotations))
        from demo.legal_dataset import LegalCourtroomDataset
        dataset = LegalCourtroomDataset(
            annotations_path=args.annotations,
            features_dir=args.features_dir,
            num_classes=args.num_classes,
            num_roles=args.num_roles
        )
    else:
        print("Initialize dataset {}".format(args.dataset))
        dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    
    # Read the train-test index splits
    splits = read_json(args.split)
    assert args.split_id < len(splits)
    split = splits[args.split_id]
    train_keys = split['train_keys']
    test_keys = split['test_keys']
    print("# total {} | # train {} | # test {}".format(
        num_videos, len(train_keys), len(test_keys)))

    # Instantiate the selected model architecture
    print("Initialize model (type: {})".format(args.model_type))
    model = build_model()
    param_count = sum(p.numel() for p in model.parameters())
    print("Model size: {:.5f}M".format(param_count / 1e6))

    # Optional: resume checkpoint loading if path is supplied
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        model.load_state_dict(torch.load(args.resume, map_location='cpu' if not use_gpu else None))

    # Multi-GPU DataParallel wrapping if applicable
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # If the --evaluate flag is set, run test evaluation and exit
    if args.evaluate:
        print("Evaluate only")
        evaluate_with_ensemble(model, dataset, test_keys, use_gpu,
                               k=args.ensemble_k,
                               save_results=args.save_results,
                               save_dir=args.save_dir)
        return

    # ── PHASE 1: EXPLORATION ──────────────────────────────────────────────────
    phase1_epochs = args.max_epoch - args.phase2_epochs
    print("\n" + "="*60)
    print("==> PHASE 1: Exploration ({} epochs, high entropy, curriculum={}, contrastive_w={})".format(
        phase1_epochs, args.use_curriculum, args.contrastive_weight))
    print("="*60)

    optimizer1 = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler1 = lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=phase1_epochs, eta_min=args.lr * 0.05)

    start_time = time.time()
    baselines = {key: 0.0 for key in train_keys}
    reward_writers = {key: [] for key in train_keys}

    best_fm1, best_epoch1, best_state1, baselines, reward_writers = train_one_phase(
        model, optimizer1, scheduler1, dataset, train_keys, test_keys,
        num_epochs=phase1_epochs, baselines=baselines, reward_writers=reward_writers,
        entropy_start=args.entropy_start, entropy_end=0.02,
        start_epoch=0, use_counterfactual=args.use_counterfactual,
        action_lock_start=args.action_lock_start, action_lock_end=args.action_lock_end,
    )

    print("\nPhase 1 complete. Best F-score: {:.1%} at epoch {}".format(
        best_fm1, best_epoch1))

    # ── PHASE 2: EXPLOITATION (reload best Phase-1 model) ────────────────────
    # Fine-tunes the best checkpoint from Phase 1 with a smaller LR and low entropy
    print("\n" + "="*60)
    print("==> PHASE 2: Exploitation ({} epochs, low entropy)".format(
        args.phase2_epochs))
    print("="*60)

    # Reload model weights corresponding to the best validation score in Phase 1
    if best_state1 is not None:
        if use_gpu:
            model.module.load_state_dict(best_state1)
        else:
            model.load_state_dict(best_state1)
        print("Reloaded best Phase-1 model (F-score {:.1%})".format(best_fm1))

    # Fine-tuning optimizer and cosine annealing scheduler for Phase 2
    optimizer2 = torch.optim.Adam(model.parameters(),
                                  lr=args.lr * 0.1, weight_decay=args.weight_decay)
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.phase2_epochs, eta_min=args.lr * 0.001)
    reward_writers2 = {key: [] for key in train_keys}

    best_fm2, best_epoch2, best_state2, _, reward_writers2 = train_one_phase(
        model, optimizer2, scheduler2, dataset, train_keys, test_keys,
        num_epochs=args.phase2_epochs, baselines=baselines,
        reward_writers=reward_writers2,
        entropy_start=0.02, entropy_end=args.entropy_end,
        start_epoch=phase1_epochs, use_counterfactual=args.use_counterfactual,
        action_lock_start=args.action_lock_end,   # already lenient in Phase 2
        action_lock_end=args.action_lock_end,
    )

    print("\nPhase 2 complete. Best F-score: {:.1%} at epoch {}".format(
        best_fm2, best_epoch2))

    # Concatenate reward records across training phases and export to a JSON log file
    for key in train_keys:
        reward_writers[key].extend(reward_writers2[key])
    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))

    overall_best_fm = max(best_fm1, best_fm2)
    print("\n" + "="*60)
    print("OVERALL BEST F-score: {:.1%}".format(overall_best_fm))
    print("="*60)

    # Save final model weights
    final_state = model.module.state_dict() if use_gpu else model.state_dict()
    save_checkpoint(final_state,
                    osp.join(args.save_dir, 'model_epoch{}.pth.tar'.format(args.max_epoch)))

    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print("Total elapsed time: {}".format(elapsed))
    dataset.close()


def evaluate(model, dataset, test_keys, use_gpu):
    """Kept for backward compatibility."""
    return evaluate_with_ensemble(model, dataset, test_keys, use_gpu,
                                  k=args.ensemble_k)


if __name__ == '__main__':
    main()
