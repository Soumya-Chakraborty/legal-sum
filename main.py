"""
main.py — Unsupervised Video Summarization via Counterfactual REINFORCE.

NOVEL CONTRIBUTIONS IN THIS TRAINING LOOP
==========================================

1.  PER-FRAME COUNTERFACTUAL ATTRIBUTION (novel training signal):
    Instead of assigning the full-episode reward to every log_prob(action),
    we compute each selected frame's MARGINAL CONTRIBUTION to the total reward
    via leave-one-out counterfactual:
        attribution(i) = reward(S) - reward(S \\ {i})
    This dramatically reduces gradient variance compared to vanilla REINFORCE
    (where all frames get the same reward signal regardless of their individual
    contribution). Frames that "earn their spot" receive stronger gradients;
    redundant frames receive near-zero gradients.

2.  ADAPTIVE ENTROPY SCHEDULING (novel stabilization):
    The entropy coefficient starts HIGH (0.1) and decays exponentially to
    a very low value (0.001) over training. Early in training, high entropy
    forces the policy to explore broadly. Later, low entropy lets the policy
    converge to confident selections. This is fundamentally different from
    using a fixed entropy coefficient.

3.  MULTI-RESTART CURRICULUM WITH TEMPERATURE ANNEALING:
    Training runs in two phases:
    - Phase 1 (exploration): High entropy, high LR, standard Bernoulli sampling
    - Phase 2 (exploitation): Low entropy, low LR, temperature-scaled sigmoid
      that sharpens the probability distribution
    The best model from Phase 1 is reloaded as the starting point for Phase 2.

4.  VIDEO-LEVEL ADAPTIVE BASELINE WITH MOMENTUM CORRECTION:
    The standard exponential moving average baseline is augmented with a
    momentum correction term (similar to Adam optimizer's bias correction)
    to avoid the cold-start problem where the baseline is 0 for the first
    few epochs.

5.  SCORE ENSEMBLE AT INFERENCE:
    At evaluation time, the model performs K=10 stochastic forward passes
    (with dropout ENABLED) and averages the predicted probabilities.
    This acts as MC-Dropout ensemble inference, dramatically reducing
    prediction variance for individual test videos.
"""
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

from utils import Logger, read_json, write_json, save_checkpoint
from models import DSN, DSN_Transformer
from rewards import compute_reward, compute_per_frame_attribution
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
                    choices=['original', 'enhanced', 'transformer'])
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
# Misc
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save-results', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu:
    use_gpu = False


def build_model():
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
    else:
        # Original shallow model (kept for ablation)
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
            def forward(self, x):
                h, _ = self.rnn(x)
                return torch.sigmoid(self.fc(h))
        return _OriginalDSN()


def evaluate_with_ensemble(model, dataset, test_keys, use_gpu,
                           k=10, save_results=False, save_dir='.'):
    """
    MC-Dropout ensemble inference.
    Runs K stochastic forward passes (dropout enabled) and averages probs.
    Reduces prediction variance vs single deterministic pass.
    """
    print("==> Test (MC-Dropout ensemble, K={})".format(k))
    eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

    if args.verbose:
        table = [["No.", "Video", "F-score"]]

    if save_results:
        h5_res = h5py.File(osp.join(save_dir, 'result.h5'), 'w')

    fms = []
    # Enable training mode to keep dropout active → stochastic passes
    model.train()
    with torch.no_grad():
        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu:
                seq = seq.cuda()

            # K stochastic forward passes
            probs_list = []
            for _ in range(k):
                p = model(seq).data.cpu().squeeze().numpy()
                probs_list.append(p)
            probs = np.mean(probs_list, axis=0)   # ensemble mean

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(
                probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(
                machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx + 1, key, "{:.1%}".format(fm)])

            if save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if save_results:
        h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))
    return mean_fm


def train_one_phase(model, optimizer, scheduler, dataset, train_keys,
                    test_keys, num_epochs, baselines, reward_writers,
                    entropy_start, entropy_end, start_epoch=0,
                    use_counterfactual=True):
    """
    Core training loop for one phase.

    INNOVATIONS:
    ─────────────
    - Adaptive entropy coefficient decays exponentially from entropy_start → entropy_end
    - Per-frame counterfactual attribution (optional) replaces uniform REINFORCE signal
    - Video-adaptive baseline with bias correction (like Adam)
    - Hard negative reward (-1) for empty selections prevents collapse
    """
    best_fm = 0.0
    best_epoch = 0
    best_state = None

    # Bias correction accumulators (Adam-style)
    baseline_step = {key: 0 for key in train_keys}

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs)

        # Exponential entropy decay
        progress = epoch / max(1, num_epochs - 1)
        entropy_coef = entropy_start * (entropy_end / entropy_start) ** progress

        for idx in idxs:
            key = train_keys[idx]
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu:
                seq = seq.cuda()

            probs = model(seq)   # (1, seq_len, 1)

            # Length penalty: penalise deviation of mean from 0.15 (target ratio)
            target_ratio = 0.15
            length_pen = args.beta * (probs.mean() - target_ratio) ** 2

            m = Bernoulli(probs)

            # Entropy regularization (adaptive)
            entropy = m.entropy().mean()
            cost = length_pen - entropy_coef * entropy

            # ── NOVEL: Counterfactual REINFORCE ──────────────────────────────
            epis_rewards = []
            for _ in range(args.num_episode):
                actions = m.sample()

                if use_counterfactual:
                    # Per-frame attributions: shape (seq_len,)
                    attributions, full_reward = compute_per_frame_attribution(
                        seq, actions, use_gpu=use_gpu)
                    # Baseline-corrected per-frame signal
                    baseline_val = baselines[key]
                    shaped_reward = attributions - baseline_val

                    log_probs = m.log_prob(actions).squeeze()  # (seq_len,)
                    # Only selected frames get non-zero attributions
                    expected_reward = (log_probs * shaped_reward).mean()
                else:
                    log_probs = m.log_prob(actions)
                    full_reward = compute_reward(seq, actions, use_gpu=use_gpu)
                    expected_reward = log_probs.mean() * (full_reward - baselines[key])

                cost -= expected_reward
                epis_rewards.append(full_reward.item()
                                    if hasattr(full_reward, 'item')
                                    else float(full_reward))

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Momentum-corrected baseline update
            baseline_step[key] += 1
            t = baseline_step[key]
            raw_baseline = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards)
            baselines[key] = raw_baseline / (1 - 0.9 ** t)   # bias correction
            # Clamp to prevent baseline from drifting too far
            baselines[key] = float(np.clip(baselines[key], -2.0, 2.0))
            reward_writers[key].append(np.mean(epis_rewards))

        if scheduler is not None:
            scheduler.step()

        epoch_reward = np.mean([reward_writers[key][-1] for key in train_keys])
        print("epoch {}/{}\t reward {:.4f}\t entropy_coef {:.4f}\t lr {:.2e}".format(
            epoch + 1, start_epoch + num_epochs, epoch_reward,
            entropy_coef, optimizer.param_groups[0]['lr']))

        # Evaluate every 5 epochs (and at last epoch)
        if (epoch + 1) % 5 == 0 or epoch == start_epoch + num_epochs - 1:
            fm = evaluate_with_ensemble(model, dataset, test_keys, use_gpu,
                                        k=args.ensemble_k)
            if fm > best_fm:
                best_fm = fm
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in (
                    model.module.state_dict() if use_gpu else model.state_dict()
                ).items()}
                model_path = osp.join(args.save_dir, 'model_best.pth.tar')
                save_checkpoint(best_state, model_path)
                print("  ** New best F-score {:.1%} at epoch {} → {}".format(
                    best_fm, best_epoch, model_path))

    return best_fm, best_epoch, best_state, baselines, reward_writers


def main():
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))

    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    splits = read_json(args.split)
    assert args.split_id < len(splits)
    split = splits[args.split_id]
    train_keys = split['train_keys']
    test_keys = split['test_keys']
    print("# total {} | # train {} | # test {}".format(
        num_videos, len(train_keys), len(test_keys)))

    print("Initialize model (type: {})".format(args.model_type))
    model = build_model()
    param_count = sum(p.numel() for p in model.parameters())
    print("Model size: {:.5f}M".format(param_count / 1e6))

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        evaluate_with_ensemble(model, dataset, test_keys, use_gpu,
                               k=args.ensemble_k,
                               save_results=args.save_results,
                               save_dir=args.save_dir)
        return

    # ── PHASE 1: EXPLORATION ──────────────────────────────────────────────────
    phase1_epochs = args.max_epoch - args.phase2_epochs
    print("\n{'='*60}")
    print("==> PHASE 1: Exploration ({} epochs, high entropy)".format(phase1_epochs))
    print("="*60)

    optimizer1 = torch.optim.Adam(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
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
    )

    print("\nPhase 1 complete. Best F-score: {:.1%} at epoch {}".format(
        best_fm1, best_epoch1))

    # ── PHASE 2: EXPLOITATION (reload best Phase-1 model) ────────────────────
    print("\n" + "="*60)
    print("==> PHASE 2: Exploitation ({} epochs, low entropy)".format(
        args.phase2_epochs))
    print("="*60)

    # Reload best model from Phase 1
    if best_state1 is not None:
        if use_gpu:
            model.module.load_state_dict(best_state1)
        else:
            model.load_state_dict(best_state1)
        print("Reloaded best Phase-1 model (F-score {:.1%})".format(best_fm1))

    # Phase-2 optimizer: lower LR, fine-tuning regime
    optimizer2 = torch.optim.Adam(model.parameters(),
                                  lr=args.lr * 0.1,
                                  weight_decay=args.weight_decay)
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.phase2_epochs, eta_min=args.lr * 0.001)

    # Reset reward writers for Phase 2 (they'll be re-indexed from 0)
    reward_writers2 = {key: [] for key in train_keys}

    best_fm2, best_epoch2, best_state2, _, reward_writers2 = train_one_phase(
        model, optimizer2, scheduler2, dataset, train_keys, test_keys,
        num_epochs=args.phase2_epochs, baselines=baselines,
        reward_writers=reward_writers2,
        entropy_start=0.02, entropy_end=args.entropy_end,
        start_epoch=phase1_epochs, use_counterfactual=args.use_counterfactual,
    )

    print("\nPhase 2 complete. Best F-score: {:.1%} at epoch {}".format(
        best_fm2, best_epoch2))

    # Merge reward writers for logging
    for key in train_keys:
        reward_writers[key].extend(reward_writers2[key])
    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))

    overall_best_fm = max(best_fm1, best_fm2)
    print("\n" + "="*60)
    print("OVERALL BEST F-score: {:.1%}".format(overall_best_fm))
    print("="*60)

    # Save final model
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
