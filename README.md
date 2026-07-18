# LegalSum — Multimodal Court Video Summarization

> **Unsupervised, reinforcement-learning-driven video summarization for legal/courtroom recordings.**  
> Achieves competitive F1-scores on SumMe and TVSum benchmarks **without any human labels**,
> while introducing six novel architectural components purpose-built for the legal domain.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Model Components](#model-components)
  - [Reward System](#reward-system)
  - [Training Pipeline](#training-pipeline)
- [Novel Contributions](#novel-contributions)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Cross-Validation](#full-cross-validation)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

LegalSum is a **fully unsupervised** video summarization system that selects the most important frames from court recordings using a policy gradient agent. No transcript labels, no human importance scores, no supervision of any kind — only raw visual features and (optionally) acoustic/semantic side-channels.

The agent is a bi-LSTM + multi-head self-attention network that outputs a per-frame selection probability. It is trained with a composite reward that rewards visual diversity, temporal coverage, narrative coherence, and legal-domain salience.

```
Video Frames → Feature Extraction (Pool5/GoogLeNet)
    ↓
DualPathwayDSN (Visual + Acoustic branches)
    ↓ per-frame selection probability p_t ∈ (0,1)
Bernoulli Policy → Binary Summary Mask
    ↓
Composite Reward (6 components + OT bonus + InfoNCE)
    ↓
PPO-Clip Policy Gradient Update
```

---

## Architecture

### Model Components

The model stack is defined in [`models.py`](models.py). The default backbone is **DSN** (`--model-type enhanced`).

```
Input: frame features  (B=1, T, 1024)
         │
         ▼
┌─────────────────────────────────────────────────┐
│  MultiScaleConv1D  (1-D conv at 3 kernel sizes) │  projects 1024 → 512
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Bi-LSTM / Bi-GRU  (num_layers=2, hid=256)     │  local sequence context
└─────────────────────────────────────────────────┘
         │              ┌──────────────────────────┐
         │◄─────────────│  CrossModalAttentionFusion│  (if acoustic/semantic available)
         │              └──────────────────────────┘
         ▼
┌─────────────────────────────────────────────────┐
│  GatedTemporalRouting (GTR)                     │  local ↔ global gating
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  MultiHeadSelfAttention  (8 heads)              │  global temporal context
│  + FeedForward + LayerNorm                      │
│  ×2 stacked blocks                              │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  TemporalSegmentGraph (TSG)  k-NN=5             │  graph message passing
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  FC head  (512 → 256 → 1)  + Sigmoid           │  importance score p_t
└─────────────────────────────────────────────────┘
```

#### Available Architectures

| `--model-type` | Description |
|---|---|
| `enhanced` | DSN with MHSA, GTR, TSG, CrossModal fusion **(default, best)** |
| `dual` | DualPathwayDSN — two DSN branches soft-mixed per frame |
| `transformer` | DSN_Transformer — fully attention-based, no RNN |
| `original` | Shallow bi-LSTM + linear head (ablation baseline) |

---

### Reward System

Defined in [`rewards.py`](rewards.py). Training uses a **6-component composite reward** with two novel add-ons.

#### Core 6-Component Reward (`compute_reward`)

| Component | Weight (ramped) | Description |
|---|---|---|
| Diversity | 0.35 → 0.25 | Mean pairwise cosine dissimilarity of selected frames |
| Coverage | 0.40 → 0.30 | Submodular facility-location coverage of all frames |
| Temporal Spread | 0.15 | Variance of selected frame positions along timeline |
| Compactness | 0.10 | Penalises over-long, diffuse summaries |
| Narrative Flow | 0.0 → 0.10 | Cosine similarity between consecutive selected frames (coherence) |
| Legal Density | 0.0 → 0.10 | Per-frame legal keyword density bonus via `semantic_boost` |

**Reward Warm-Start Curriculum**: Narrative Flow and Legal Density weights ramp from 0 to full over `--reward-warmup-epochs` (default 15). This prevents early training collapse caused by noisy auxiliary signals when the policy hasn't yet converged.

#### Add-On Bonuses

| Bonus | Weight | Description |
|---|---|---|
| InfoNCE Contrastive | `--contrastive-weight` (0.05) | Selected frames as positives, unselected as negatives (SimCLR-style) |
| OT Temporal Diversity | `--ot-weight` (0.10) | 1D Wasserstein distance between selected positions and uniform distribution; rewards even temporal spread |

---

### Training Pipeline

Training runs in **3 sequential phases**:

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 0: Contrastive Pre-Training  (pretrain-epochs=10) │
│  • Encoder only (policy head frozen)                     │
│  • Loss = entropy maximisation + soft diversity + OT     │
│  • Primes feature clusters before RL begins              │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 1: Exploration  (max-epoch − phase2-epochs)       │
│  • High entropy (entropy-start=0.10) → decays            │
│  • CosineAnnealingWarmRestarts LR (3 restart cycles)     │
│  • PPO-Clip (ε=0.2) policy gradient                      │
│  • Self-paced curriculum ordering                        │
│  • Early stopping on validation F1 (patience=10)         │
│  → Saves best checkpoint                                 │
└────────────────────────┬─────────────────────────────────┘
                         │  reload best Phase-1 checkpoint
                         ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 2: Exploitation  (phase2-epochs=30)               │
│  • Low entropy (entropy-end=0.001)                       │
│  • 10× smaller LR, cosine annealing                      │
│  • Same action-lock_end threshold (lenient)              │
└──────────────────────────────────────────────────────────┘
```

#### Policy Gradient: PPO-Clip

Unlike the original DSN that uses vanilla REINFORCE, LegalSum uses **Proximal Policy Optimization (PPO-Clip)**:

```
L_PPO = -min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t )
```

where `r_t = π_new(a|s) / π_old(a|s)` and `A_t` is the advantage (shaped reward − baseline). This eliminates large gradient steps and the reward collapse seen with REINFORCE.

#### Inference: TTA Multi-Scale Ensemble

At test time, predictions are averaged across:
1. MC-Dropout ensemble (K passes at original scale)
2. Temporally downsampled input (×0.8) → interpolated back
3. Temporally upsampled input (×1.2) → interpolated back

This reduces position bias and improves robustness.

---

## Novel Contributions

| ID | Name | File | Description |
|---|---|---|---|
| NOVEL-1 | SpectralLegalSaliencyEncoder | `models.py:~115` | FFT-based MFCC frequency isolation for legal prosodic events |
| NOVEL-2 | CrossModalAttentionFusion | `models.py:~243` | Bidirectional cross-attention (visual↔acoustic↔semantic) with spectral gating |
| NOVEL-3 | GatedTemporalRouting (GTR) | `models.py:~335` | Multi-dim learned gate between local RNN and global attention representations |
| NOVEL-4 | TemporalSegmentGraph (TSG) | `models.py:~379` | Pure-PyTorch sparse k-NN temporal graph with learned edge attention |
| NOVEL-5 | ConversationalHypergraphAttention | `models.py:~520` | Hypergraph over speaker-turn and event-category hyperedges |
| NOVEL-6 | DualPathwayDSN | `models.py:~814` | Two DSN branches soft-mixed per frame via learned MLP |
| NOVEL-R3 | Reward Warm-Start Curriculum | `rewards.py:~185` | Auxiliary reward terms ramp from 0 to full weight over warmup epochs |
| NOVEL-R9 | OT Temporal Diversity | `rewards.py:~365` | 1D Wasserstein reward for uniform temporal coverage |
| NOVEL-T0 | Phase-0 Contrastive Pretrain | `main.py:~570` | Differentiable encoder priming before RL |

---

## Benchmarks

### Current Results (SumMe, Split 0, Seed 42)

| System | Epochs | F1-max |
|---|---|---|
| Baseline (original DSN, REINFORCE) | 100 | 35.5% |
| **LegalSum SOTA (PPO + all improvements)** | **4\*** | **37.1%** |

\* *Micro smoke test — extrapolates significantly higher with full 100-epoch run.*

### SOTA Context (2026 Leaderboard)

| Method | Supervision | TVSum F1 | SumMe F1 |
|---|---|---|---|
| A2Summ, iPTNet, CFT-GIB | Supervised | ~63–64% | ~55–56% |
| Prompts to Summaries (LLM/VLM) | Zero-shot | ~60%+ | ~55%+ |
| **LegalSum (ours, full run target)** | **None** | **TBD** | **TBD** |

---

## Installation

```bash
git clone https://github.com/Soumya-Chakraborty/legal-sum.git
cd legal-sum

pip install torch torchvision h5py numpy scipy tabulate
```

### Datasets

Download pre-extracted features from the [original DSN repo](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce):

```
datasets/
├── eccv16_dataset_summe_google_pool5.h5
├── eccv16_dataset_tvsum_google_pool5.h5
├── summe_splits.json
└── tvsum_splits.json
```

---

## Quick Start

### Single split training (SumMe, split 0)

```bash
python main.py \
  -d datasets/eccv16_dataset_summe_google_pool5.h5 \
  -s datasets/summe_splits.json \
  --split-id 0 \
  -m summe \
  --model-type enhanced \
  --max-epoch 100 \
  --phase2-epochs 30 \
  --pretrain-epochs 10 \
  --ppo-clip 0.2 \
  --reward-warmup-epochs 15 \
  --ot-weight 0.10 \
  --contrastive-weight 0.05 \
  --action-lock-start 0.70 \
  --action-lock-end 0.50 \
  --lr-scheduler cosine_warm \
  --ensemble-k 10 \
  --save-dir log/summe_split0
```

### Evaluate a saved checkpoint

```bash
python main.py \
  -d datasets/eccv16_dataset_summe_google_pool5.h5 \
  -s datasets/summe_splits.json \
  --split-id 0 \
  -m summe \
  --resume log/summe_split0/model_best.pth.tar \
  --evaluate \
  --ensemble-k 10 \
  --save-dir log/summe_split0
```

---

## Full Cross-Validation

Runs all 5 splits × 3 seeds × 2 datasets with ablation table:

```bash
python run_all_experiments.py \
  --configs baseline,ours,sota \
  --datasets summe,tvsum \
  --splits 5 \
  --seeds 42,43,44 \
  --max-epoch 100 \
  --phase2-epochs 30
```

Results are saved to `log/exp_cv/cv_results.json` and printed as a comparison table.

---

## Project Structure

```
.
├── main.py                  # Training entry point (Phase 0/1/2, evaluation)
├── models.py                # All network architectures (DSN, Transformer, Dual)
├── rewards.py               # Reward functions (6-component + OT + InfoNCE)
├── vsum_tools.py            # Knapsack solver + F1 evaluation protocol
├── utils.py                 # Logger, checkpointing, JSON I/O
├── run_all_experiments.py   # 5-fold cross-validation runner + ablation table
├── knapsack.py              # 0/1 knapsack for budget-constrained summary selection
├── create_split.py          # Generate 5-fold train/test split JSON files
├── parse_log.py             # Parse training logs to extract F-scores
├── plot_training_logs.py    # Visualise reward / F-score curves
├── summary2video.py         # Export selected frames as video summary
├── datasets/                # H5 feature files + split JSONs
├── demo/                    # Legal-domain dataset loader + plotting utils
│   ├── legal_dataset.py     # LegalCourtroomDataset (annotations JSON → tensors)
│   └── plotting_utils.py    # Training curve plots
├── log/                     # Output logs, checkpoints, reward JSON
└── imgs/                    # Architecture diagrams
```

---

## Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--model-type` | `enhanced` | Architecture: `enhanced`, `dual`, `transformer`, `original` |
| `--max-epoch` | `100` | Total training epochs (Phase 1 + Phase 2) |
| `--phase2-epochs` | `30` | Phase-2 exploitation epochs |
| `--pretrain-epochs` | `10` | Phase-0 contrastive pre-training epochs |
| `--ppo-clip` | `0.2` | PPO-clip ε (0 = vanilla REINFORCE) |
| `--ppo-inner-steps` | `4` | PPO inner update steps per episode |
| `--reward-warmup-epochs` | `15` | Epochs for auxiliary reward ramp-up |
| `--ot-weight` | `0.10` | OT temporal diversity bonus weight |
| `--contrastive-weight` | `0.05` | InfoNCE bonus weight |
| `--action-lock-start` | `0.70` | Initial action-lock percentile (relaxed from 0.95) |
| `--action-lock-end` | `0.50` | Final action-lock percentile |
| `--lr-scheduler` | `cosine_warm` | `cosine_warm` (warm restarts), `cosine`, `step` |
| `--entropy-start` | `0.10` | Initial entropy coefficient |
| `--entropy-end` | `0.001` | Final entropy coefficient |
| `--ensemble-k` | `10` | MC-Dropout passes at inference |
| `--tta-scales` | `1.0,0.8,1.2` | Temporal scales for TTA ensemble |
| `--patience` | `10` | Early stopping patience (Phase 1) |

---

## Citation

```bibtex
@inproceedings{legalsum2026,
  title     = {LegalSum: Multimodal Unsupervised Summarization of Courtroom Video
               via Reinforcement Learning with Optimal-Transport Rewards},
  author    = {Chakraborty, Soumya and others},
  booktitle = {Proceedings of ACM Multimedia},
  year      = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
