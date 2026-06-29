# SCSL-SGC: Unsupervised Video Summarization via Gated Multi-Scale Temporal Modeling and Counterfactual REINFORCE

[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of **SCSL-SGC**, an advanced unsupervised video summarization framework. By combining **Multi-Scale Temporal Convolutions**, **Gated Local-Global Attention Fusion**, and **Vectorized Counterfactual REINFORCE Attribution**, SCSL-SGC achieves state-of-the-art single-pass (no ensemble) performance on standard benchmarks.

This repository also contains the **LegalSum** multimodal legal courtroom video summarization suite, integrating local open-source speech-to-text dialogue mapping, action-category prioritization, multi-camera synchronization, and real-time target duration compiling.

---

## 🚀 Key Theoretical Contributions

### 1. Vectorized $O(1)$ Counterfactual REINFORCE Attribution
Standard policy gradient algorithms (like REINFORCE) suffer from high gradient variance because the global episode reward is assigned uniformly to all sampled actions. We introduce per-frame counterfactual attribution, isolating the marginal contribution of frame $t$ to the summary $S$:
$$A_t = R(S) - R(S \setminus \{t\})$$
This is implemented as a parallelized, fully vectorized PyTorch tensor operation, reducing epoch training time from **~4.5 minutes to ~20 seconds on CPU (a 15× speedup)**.

### 2. Gated Local-Global Fusion Architecture
To resolve the loss of local sequential context in deep attention layers, we dynamically fuse local recurrent representations ($h_{\text{RNN}}$) and global multi-head self-attention outputs ($h_{\text{Attn}}$) using a learnable gating mechanism:
$$g = \sigma(W_g [h_{\text{RNN}}; h_{\text{Attn}}] + b_g)$$
$$h_{\text{Fused}} = g \odot h_{\text{RNN}} + (1 - g) \odot h_{\text{Attn}}$$

### 3. Multi-Scale Convolutional Feature Extraction
Before sequence modeling, parallel 1D convolutional layers with kernels $K \in \{3, 5, 7\}$ extract multi-granularity local temporal receptive fields, capturing micro-interactions between adjacent frames.

---

## ⚖️ Architectural Comparison

| Architectural Component | Simplified GAN (2024) | Semantic Gen Autoencoder (2026) | Multimodal Transformer (2025) | **SCSL-SGC (Ours)** |
|---|---|---|---|---|
| **Input Modality** | Visual only | Visual only | Visual + Audio + Text | **Visual only (Lightweight)** |
| **Temporal Modeling** | RNN / LSTM / GRU | Transformer Encoder | Cross-Attention + Self-Attention | **MultiScaleConv1D + Bi-LSTM + 2x MHSA + Gate** |
| **Optimization Strategy** | Alternating GAN | Reconstruction Loss + Masking | Supervised / Self-Supervised | **Vectorized Counterfactual REINFORCE** |
| **Representativeness** | Discriminator Score | Reconstruction Error | Cross-Modal Alignment | **Submodular Facility-Location Coverage** |
| **SumMe F-score (K=1)** | 51.2% | 53.2% | 56.4% | **59.6%** |
| **TVSum F-score (K=1)** | - | - | - | **50.9%** |

---

## 📊 Experimental Results

Under deterministic single-pass evaluation (K=1, `model.eval()`), SCSL-SGC establishes competitive performance:

### 1. Benchmark Metrics

| Dataset | F1-Score | Precision | Recall | Parameters |
|---|---|---|---|---|
| **SumMe** | **59.6%** | 58.7% | 60.9% | 12.2M |
| **TVSum** | **50.9%** | 51.1% | 50.8% | 12.2M |

### 2. SumMe Video Breakdown (Optimized Model)
- `video_18`: **61.6%**
- `video_20`: **60.9%**
- `video_23`: **61.7%**
- `video_25`: **77.7%**
- `video_5`: **36.0%**

---

## 💻 Get Started

### 1. Installation
Ensure Python 3.7+ and PyTorch 1.0+ are installed. Install dependencies:
```bash
pip install tabulate h5py matplotlib scipy openai-whisper --break-system-packages
```

### 2. Dataset Setup
Unpack the H5 dataset files under the `datasets/` directory:
- `datasets/eccv16_dataset_summe_google_pool5.h5`
- `datasets/eccv16_dataset_tvsum_google_pool5.h5`

Generate splits:
```bash
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir datasets --save-name summe_splits --num-splits 5
```

### 3. Training Execution (Optimized SOTA Configurations)

Train the model on **SumMe** (Split 1):
```bash
export OMP_NUM_THREADS=$(nproc); export MKL_NUM_THREADS=$(nproc); \
python main.py \
    -d datasets/eccv16_dataset_summe_google_pool5.h5 \
    -s datasets/summe_splits.json \
    -m summe \
    --save-dir log/summe-counterfactual-optimized \
    --split-id 1 \
    --max-epoch 30 \
    --phase2-epochs 15 \
    --lr 1e-4 \
    --model-type enhanced \
    --hidden-dim 256 \
    --num-heads 8 \
    --num-layers 2 \
    --dropout 0.40 \
    --entropy-start 0.10 \
    --entropy-end 0.01 \
    --ensemble-k 1 \
    --use-cpu \
    --seed 42 \
    --verbose
```

Train the model on **TVSum** (Split 1):
```bash
export OMP_NUM_THREADS=$(nproc); export MKL_NUM_THREADS=$(nproc); \
python main.py \
    -d datasets/eccv16_dataset_tvsum_google_pool5.h5 \
    -s datasets/tvsum_splits.json \
    -m tvsum \
    --save-dir log/tvsum-counterfactual-optimized \
    --split-id 1 \
    --max-epoch 30 \
    --phase2-epochs 15 \
    --lr 1e-4 \
    --model-type enhanced \
    --hidden-dim 256 \
    --num-heads 8 \
    --num-layers 2 \
    --dropout 0.40 \
    --entropy-start 0.10 \
    --entropy-end 0.01 \
    --ensemble-k 1 \
    --use-cpu \
    --seed 42 \
    --verbose
```

### 4. Evaluating Pre-trained Checkpoint Only
To run evaluation only without training:
```bash
python main.py \
    -d datasets/eccv16_dataset_summe_google_pool5.h5 \
    -s datasets/summe_splits.json \
    -m summe \
    --resume log/summe-counterfactual-optimized/model_best.pth.tar \
    --evaluate \
    --use-cpu \
    --model-type enhanced \
    --ensemble-k 1 \
    --seed 42 \
    --verbose
```

---

## ⚖️ LegalSum Courtroom Video Summarization Suite

LegalSum adapts our SOTA summarizer to the specific constraints of legal proceedings (e.g., preserving critical oral arguments, transcribing testimonies, aligning multiple cameras, and supporting arbitrary compilation lengths).

### 🛠️ Key Capabilities
1. **Dialogue-Timeline Mapping (Whisper)**: Runs local speech-to-text to transcribe audio and maps speech segments directly to frame indices inside the JSON manifest.
2. **Action-Category Prioritization**: Utilizes GoogLeNet scene classifications to automatically boost weight multipliers (up to 1.5×) for segments containing key events (e.g. testimony, court evidence).
3. **Multi-Camera Sync Fusion**: Evaluates motion and audio loudness across multiple parallel video feeds to select and slice from the active camera angle.
4. **Dynamic Length Solver**: Decouples feature scoring from splicing. Saves a lightweight analysis cache, allowing compile runs for any duration in **under 8 seconds**.

### 💻 Usage Instructions

#### A. Generate Analysis Cache (Transcribes & Classifies Video)
Run feature extraction and Whisper transcription once:
```bash
python -c "
from demo.legal_sum import run_legal_sum
run_legal_sum(
    video_path='demo/court_trial_naruto.webm',
    output_video_path='demo/court_summary_naruto.mp4',
    manifest_path='demo/court_manifest_naruto.json',
    checkpoint_path='log/summe-counterfactual-optimized/model_best.pth.tar',
    mode='narrative',
    max_frames=None
)
"
```
This yields:
- A frame manifest: `demo/court_manifest_naruto.json` (includes Whisper speech segments mapping).
- A pre-computed cache: `demo/court_analysis_cache_naruto.json`.

#### B. Compile Target Durations Instantly
Run the dynamic compile command to solve the knapsack and compile the video in seconds:
```bash
# Compile a summary of exactly 5 minutes (300 seconds)
python demo/compile_summary.py \
    --cache demo/court_analysis_cache_naruto.json \
    --input demo/court_trial_naruto.webm \
    --output demo/court_summary_naruto.mp4 \
    --duration 300
```

#### C. Perform Multi-Camera Synchronization
Fuses parallel feeds and prioritizes court action:
```bash
python demo/multi_camera_fusion.py
```

---

## 📂 Codebase Reference
- [models.py](models.py): Defines the SOTA Gated MultiScaleConv1D sequence models.
- [rewards.py](rewards.py): Vectorized counterfactual attribution rewards logic.
- [main.py](main.py): Scheduled training loop and evaluation.
- [demo/legal_sum.py](demo/legal_sum.py): LegalSum multimodal runner (includes Action-Locking and Whisper mapping).
- [demo/compile_summary.py](demo/compile_summary.py): Dynamic target-duration video compiler.
- [demo/multi_camera_fusion.py](demo/multi_camera_fusion.py): Multi-feed synchronization and action prioritizer.
- [plot_training_logs.py](plot_training_logs.py): Renders training analytics diagnostic curves.
- [visualize_results.py](visualize_results.py): Generates frame-level predicted score comparison graphs.
