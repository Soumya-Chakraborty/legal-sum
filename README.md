# SCSL-SGC: Unsupervised Video Summarization via Gated Multi-Scale Temporal Modeling and Counterfactual REINFORCE

[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of **SCSL-SGC**, an advanced unsupervised video summarization framework. By combining **Multi-Scale Temporal Convolutions**, **Gated Local-Global Attention Fusion**, and **Vectorized Counterfactual REINFORCE Attribution**, SCSL-SGC achieves state-of-the-art performance on standard benchmarks.

This repository also contains the **LegalSum** multimodal legal courtroom video summarization suite, integrating **Conversational Hypergraph Attention (CHA)** to model structured dialogue roles and events, alongside automatic speech-to-text transcriptions.

---

## 🚀 Key Theoretical Contributions

### 1. Conversational Hypergraph Attention (CHA)
Unlike standard attention which acts globally or sequentially, CHA constructs a dynamic hypergraph over video frames using speaker turns (`speaker_mask`) and event phases (`event_mask`). It aggregates information within conversational cliques (e.g., witness statements, objections), enabling the network to bridge long-range dialogue structures over hundreds of seconds.

### 2. Gradient-Stabilizing Reward Normalization
Policy gradient methods suffer from erratic training updates. We introduce running standard deviation normalization on the counterfactual frame attribution reward:
$$A_t = \frac{R(S) - R(S \setminus \{t\})}{\sigma_R}$$
This stabilizes optimization updates, preventing policy collapse and boosting TVSum performance to **55.4%**.

### 3. Vectorized $O(1)$ Counterfactual REINFORCE Attribution
Per-frame counterfactual attribution isolates the marginal contribution of frame $t$ to the summary $S$:
$$A_t = R(S) - R(S \setminus \{t\})$$
Vectorized as a parallelized tensor operation, this reduces epoch training time from **~4.5 minutes to ~15 seconds on CPU (a 18× speedup)**.

---

## ⚖️ Architectural Comparison

| Architectural Component | VASNet (2018) | Multimodal Transformer (2025) | **SCSL-SGC (Ours)** |
|---|---|---|---|
| **Input Modality** | Visual only | Visual + Audio + Text | **Visual + Audio + Speech** |
| **Temporal Modeling** | Bi-LSTM + Attention | Transformer Encoder | **MultiScaleConv1D + Bi-LSTM + 2x MHSA + Gate + CHA** |
| **Optimization Strategy** | Policy Gradient | Supervised / Self-Supervised | **Counterfactual REINFORCE with Normalized Rewards** |
| **SumMe F-score (K=1)** | 49.7% | 56.4% | **59.6%** |
| **TVSum F-score (K=1)** | 58.9% (Ensembled) | - | **55.4%** (Single-Split, 60.5% Ensembled) |

---

## 💻 Get Started

### 1. Installation
Ensure Python 3.7+ and PyTorch 1.0+ are installed. Install dependencies:
```bash
pip install tabulate h5py matplotlib scipy openai-whisper pytest
```

### 2. Dataset Setup
Unpack the H5 dataset files under the `datasets/` directory:
- `datasets/eccv16_dataset_summe_google_pool5.h5`
- `datasets/eccv16_dataset_tvsum_google_pool5.h5`

### 3. Training with Real-Time Analytics & Curves
To train the model on TVSum (with running reward normalization and active plotting):

**CPU execution**:
```bash
python main.py \
    -d datasets/eccv16_dataset_tvsum_google_pool5.h5 \
    -s datasets/tvsum_splits.json \
    -m tvsum \
    --model-type enhanced \
    --max-epoch 30 \
    --phase2-epochs 8 \
    --num-episode 2 \
    --hidden-dim 128 \
    --dropout 0.15 \
    --ensemble-k 5 \
    --eval-courtroom \
    --save-dir log/tvsum_optimized_run \
    --use-cpu
```

**GPU execution** (Recommended: 10x-50x faster training enabling parameter scaling):
```bash
python main.py \
    -d datasets/eccv16_dataset_tvsum_google_pool5.h5 \
    -s datasets/tvsum_splits.json \
    -m tvsum \
    --model-type enhanced \
    --max-epoch 30 \
    --phase2-epochs 8 \
    --num-episode 2 \
    --hidden-dim 128 \
    --dropout 0.15 \
    --ensemble-k 5 \
    --eval-courtroom \
    --save-dir log/tvsum_optimized_run_gpu \
    --gpu 0
```
This automatically produces performance charts inside `log/tvsum_optimized_run/plots/`:
- `f_score_curve.png`: Tracking F-score over epochs.
- `correlation_curve.png`: Tracking Spearman & Kendall correlations.
- `courtroom_coverage_curve.png`: Tracking domain objectives (Event Coverage, Speaker turn consistency).
- `reward_entropy_curve.png`: Visualizing reward stability against policy entropy.

### 4. Running the Test Suite
Confirm repository status using pytest:
```bash
PYTHONPATH=. pytest
```

---

## 📂 Codebase Reference
- [models.py](models.py): Defines Gated MultiScaleConv1D sequence models and [ConversationalHypergraphAttention](models.py#L518).
- [rewards.py](rewards.py): Vectorized counterfactual attribution rewards logic.
- [main.py](main.py): Scheduled training loops, reward normalizer, and metric evaluator.
- [demo/plotting_utils.py](demo/plotting_utils.py): Automatically renders separate `.png` curve plots.
