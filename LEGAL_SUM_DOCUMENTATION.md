# Architecture Deep-Dive

> Detailed technical documentation of every component in LegalSum.  
> For usage and benchmarks see [README.md](README.md).

---

## Table of Contents

- [System Architecture Overview](#system-architecture-overview)
- [Model: `models.py`](#model-modelspy)
  - [Shared Primitives](#shared-primitives)
  - [NOVEL-1 SpectralLegalSaliencyEncoder](#novel-1-spectrallegalsaliencyencoder)
  - [NOVEL-2 CrossModalAttentionFusion](#novel-2-crossmodalattentionfusion)
  - [NOVEL-3 GatedTemporalRouting](#novel-3-gatedtemporalrouting)
  - [NOVEL-4 TemporalSegmentGraph](#novel-4-temporalsegmentgraph)
  - [NOVEL-5 ConversationalHypergraphAttention](#novel-5-conversationalhypergraphattention)
  - [DSN — Main Backbone](#dsn--main-backbone)
  - [DSN_Transformer](#dsn_transformer)
  - [NOVEL-6 DualPathwayDSN](#novel-6-dualpathwaydsn)
- [Rewards: `rewards.py`](#rewards-rewardspy)
  - [Core 6-Component Reward](#core-6-component-reward)
  - [NOVEL-R3 Reward Warm-Start Curriculum](#novel-r3-reward-warm-start-curriculum)
  - [NOVEL-R9 OT Temporal Diversity](#novel-r9-ot-temporal-diversity)
  - [InfoNCE Contrastive Bonus](#infonce-contrastive-bonus)
  - [Counterfactual Per-Frame Attribution](#counterfactual-per-frame-attribution)
- [Training: `main.py`](#training-mainpy)
  - [NOVEL-T0 Phase-0 Contrastive Pretrain](#novel-t0-phase-0-contrastive-pretrain)
  - [Phase-1 Exploration](#phase-1-exploration)
  - [Phase-2 Exploitation](#phase-2-exploitation)
  - [PPO-Clip Policy Gradient](#ppo-clip-policy-gradient)
  - [Adaptive Action-Lock](#adaptive-action-lock)
  - [Self-Paced Curriculum](#self-paced-curriculum)
  - [TTA Multi-Scale Ensemble Inference](#tta-multi-scale-ensemble-inference)
- [Evaluation: `vsum_tools.py`](#evaluation-vsum_toolspy)
- [Data Flow End-to-End](#data-flow-end-to-end)

---

## System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INPUT                                          │
│  video.mp4 → GoogLeNet pool5 features → (1, T, 1024) tensor           │
│  (optionally) acoustic MFCC → (1, T, 128), legal keyword vec (1,T,D)  │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PHASE 0 — PRETRAIN  (main.py)                     │
│  Differentiable objectives on encoder only (policy head frozen):       │
│  • Entropy maximisation (explore)                                      │
│  • Soft cosine diversity (feature spread)                              │
│  • OT temporal coverage (soft Wasserstein)                             │
└────────────────────────┬───────────────────────────────────────────────┘
                         │ weights warm-started
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│              DSN BACKBONE  (models.py → DSN.forward)                  │
│                                                                        │
│  ┌──────────────────┐    ┌──────────────────────────────────────────┐  │
│  │ MultiScaleConv1D │    │    CrossModalAttentionFusion             │  │
│  │ 1024 → 512       │◄───│  (visual ↔ acoustic ↔ semantic)          │  │
│  └────────┬─────────┘    │  + SpectralLegalSaliencyEncoder          │  │
│           │              └──────────────────────────────────────────┘  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │  Bi-LSTM (×2)    │  hidden=256, bidirectional → output (1,T,512)   │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ GatedTemporalRou-│  multi-dim gate: local RNN ↔ global attention   │
│  │ ting  (GTR)      │                                                  │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ MHSA (×2 blocks) │  8 heads, pre-LN, FFN(dim×4), residual         │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ TemporalSegment  │  k-NN=5 sparse graph, 2-round message passing   │
│  │ Graph  (TSG)     │                                                  │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ FC head + Sigmoid│  512 → 256 → 1  →  p_t ∈ (0,1) per frame      │
│  └────────┬─────────┘                                                  │
└───────────┼────────────────────────────────────────────────────────────┘
            │  p_t  (importance probability vector, shape T)
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    RL TRAINING LOOP  (main.py)                        │
│                                                                        │
│  Bernoulli(p_t) → actions a_t ∈ {0,1}                                 │
│                                                                        │
│  Reward = compute_reward(seq, actions, epoch=E)                        │
│         + contrastive_weight × InfoNCE(seq, actions)                  │
│         + ot_weight × OT_Diversity(actions)                            │
│                                                                        │
│  PPO-Clip Loss: -min(r_t·A_t,  clip(r_t,1-ε,1+ε)·A_t)               │
│  + length penalty β·(mean(p)-0.15)²                                   │
│  - entropy_coef · H(Bernoulli(p_t))                                    │
│                                                                        │
│  Gradient clipping (max_norm=5.0) → Adam optimizer                    │
└───────────┬────────────────────────────────────────────────────────────┘
            │  best Phase-1 checkpoint
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                PHASE 2 — FINE-TUNING  (main.py)                       │
│  Reload best Phase-1 model. Low entropy, 10× smaller LR.              │
└───────────┬────────────────────────────────────────────────────────────┘
            │  final model weights
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│              TTA INFERENCE  (evaluate_with_ensemble)                  │
│  Run model at scales {1.0, 0.8, 1.2}, average importance scores      │
│  Knapsack solve → binary summary mask → F1 vs human annotations       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Model: `models.py`

### Shared Primitives

| Class | Input | Output | Purpose |
|---|---|---|---|
| `MultiHeadSelfAttention` | (B,T,D) | (B,T,D) | Scaled dot-product MHSA with pre-LN |
| `FeedForward` | (B,T,D) | (B,T,D) | Two-layer MLP with GELU, dropout |
| `MultiScaleConv1D` | (B,T,1024) | (B,T,512) | 3 parallel Conv1D at kernels 1,3,5, concat |

---

### NOVEL-1: SpectralLegalSaliencyEncoder

**File**: `models.py` ~line 115  
**Input**: MFCC tensor `(B, T, n_mfcc)`  
**Output**: per-frame saliency scalar `(B, T, 1)`

**How it works**:
1. Apply `torch.fft.rfft` along the time axis → complex frequency spectrum
2. Take magnitude of top-K frequency bins (tuned to legal prosodic event rates: 1–4 Hz for objection/emphasis patterns)
3. Apply `torch.fft.irfft` → band-passed time signal
4. Feed through a 2-layer MLP → scalar saliency score per frame

**Why it matters**: Standard MFCC features treat all frequencies equally. Legal proceedings have characteristic prosodic signatures at specific frequencies (raised voice for objections, rhythmic cross-examination). Isolating these frequencies gives a supervision-free acoustic saliency signal.

---

### NOVEL-2: CrossModalAttentionFusion

**File**: `models.py` ~line 243  
**Inputs**: visual `(B,T,D_v)`, acoustic `(B,T,D_a)`, semantic `(B,T,D_s)`  
**Output**: fused `(B,T,D_v)`

**Architecture**:
```
acoustic → SpectralLegalSaliencyEncoder → saliency_weight (B,T,1)
acoustic → Linear → acoustic_proj (B,T,D_v)
weighted_acoustic = acoustic_proj × saliency_weight

# Cross-attention 1: visual attends to acoustic
Q=visual, K=V=weighted_acoustic → cross_attn_a

# Cross-attention 2: visual attends to semantic
Q=visual, K=V=semantic_proj → cross_attn_s

# Sigmoid gate: blend two fused views
gate = sigmoid(Linear([cross_attn_a, cross_attn_s]))
output = gate × cross_attn_a + (1-gate) × cross_attn_s
```

**Why it matters**: Existing fusion methods (late fusion, concatenation) do not weight acoustic by domain-relevant frequency content. The spectral saliency weighting allows the model to attend more strongly to acoustically salient moments (raised voices, gavel sounds) without any supervision.

---

### NOVEL-3: GatedTemporalRouting

**File**: `models.py` ~line 335  
**Input**: RNN hidden `h_rnn (B,T,D)`, attention hidden `h_attn (B,T,D)`  
**Output**: fused `(B,T,D)`

**Architecture**:
```
gate = sigmoid(Linear([h_rnn, h_attn]))       # (B,T,D) multi-dim gate
h_routed = gate × h_rnn + (1-gate) × h_attn  # element-wise
h_out = LayerNorm(h_routed + AdaptiveResidualContext(h_routed))
```

**Why it improves on prior work**: Prior DSN used a single scalar gate per frame. GTR uses a *D-dimensional* gate, allowing different feature dimensions to independently prefer local (RNN) or global (attention) context. This is equivalent to learned channel-wise routing.

---

### NOVEL-4: TemporalSegmentGraph

**File**: `models.py` ~line 379  
**Input**: frame features `(B,T,D)`  
**Output**: graph-enhanced features `(B,T,D)`

**Architecture**:
```
# Build sparse k-NN graph (k=5) by cosine similarity along time
sim_matrix = cosine_sim(features, features)   # (T,T)
top_k_edges = argsort(sim_matrix, dim=-1)[:, :k]   # (T,k)

# 2-round message passing
for round in [1, 2]:
    edge_attn = softmax(Linear(concat[node_i, node_j]))   # learned edge weights
    msg = sum(edge_attn × node_j_features)                # aggregate
    node_features = LayerNorm(node_features + msg)
```

**Complexity**: O(T·k) vs O(T²) for full self-attention.  
**Why it matters**: Temporally distant frames that are semantically similar (e.g., same judge speaking at start and end) can exchange context through graph edges, something RNNs cannot do and attention does expensively.

---

### NOVEL-5: ConversationalHypergraphAttention

**File**: `models.py` ~line 520  
**Inputs**: frame features `(B,T,D)`, speaker_mask `(B,T)`, event_mask `(B,T,C)`  
**Output**: hypergraph-enhanced features `(B,T,D)`

**Architecture**:
```
# Construct hyperedges
speaker_hyperedges: one edge per unique speaker role
event_hyperedges: one edge per legal event category (C edges)

# Node-to-edge aggregation (attentive)
hyperedge_embedding = attentive_sum(node_features in hyperedge)

# Edge-to-node broadcast
node_update = sum(hyperedge_embedding for each hyperedge containing node)
output = LayerNorm(node_features + Linear(node_update))
```

**Why it matters**: Speaker turns and legal event phases (opening, testimony, cross-examination, closing) define natural semantic groups. Frames within the same speaker turn or event phase should share context regardless of temporal distance. Standard graph methods require explicit edges; hyperedges provide efficient set-level grouping.

---

### DSN — Main Backbone

**File**: `models.py`, class `DSN`  
**Default config**: `--model-type enhanced`

Full forward pass:
```python
def forward(x, acoustic=None, semantic=None, speaker_mask=None, event_mask=None):
    x = MultiScaleConv1D(x)                         # (1,T,512)
    if acoustic or semantic:
        x = CrossModalAttentionFusion(x, acoustic, semantic)
    h_rnn, _ = BiLSTM(x)                            # (1,T,512)
    h_attn = positional_encoding(h_rnn)
    h_attn = MHSA_block_1(h_attn)
    h_attn = MHSA_block_2(h_attn)
    h_fused = GatedTemporalRouting(h_rnn, h_attn)
    h_graph = TemporalSegmentGraph(h_fused)
    if speaker_mask or event_mask:
        h_graph = ConversationalHypergraphAttention(h_graph, speaker_mask, event_mask)
    p = sigmoid(FC(h_graph))                        # (1,T,1)
    return p
```

---

### DSN_Transformer

**File**: `models.py`, class `DSN_Transformer`  
**Config**: `--model-type transformer`

Replaces Bi-LSTM with sinusoidal positional encoding + 4 stacked MHSA blocks. Higher capacity, no sequential bottleneck. Best for long videos (T > 1000 frames).

---

### NOVEL-6: DualPathwayDSN

**File**: `models.py` ~line 814  
**Config**: `--model-type dual`

```
visual_branch = DSN(dropout=0.25)    # visual-heavy
audio_branch  = DSN(dropout=0.30)    # audio-heavy (higher dropout for regularisation)

p_v = visual_branch(x)              # (1,T,1)
p_a = audio_branch(x, acoustic)     # (1,T,1)

alpha = sigmoid(MLP(concat(p_v, p_a)))  # per-frame mixing weight (1,T,1)
p_out = alpha × p_v + (1-alpha) × p_a
```

Alpha near 1 = trust visual branch (evidence display frames).  
Alpha near 0 = trust audio branch (judicial pronouncements, objections).

---

## Rewards: `rewards.py`

### Core 6-Component Reward

**Function**: `compute_reward(seq, actions, epoch, warmup_epochs)`

```python
# Component computations
reward_div      = mean pairwise cosine dissimilarity of selected frames
reward_cov      = submodular facility-location: min_j max_i sim(selected_i, all_j)
reward_spread   = variance of selected frame positions / T²
reward_compact  = -mean gap between consecutive selected frames (penalises diffuse selection)
reward_narrative = mean cosine similarity of consecutive selected frames (coherence)
reward_legal_density = mean semantic_boost[pick_indices]  (legal keyword density)
```

**Weight schedule** (controlled by `progress = epoch / warmup_epochs`):

```
w_div       = 0.35 - 0.10 × progress   (0.35 at epoch 0 → 0.25 at full ramp)
w_cov       = 0.40 - 0.10 × progress   (0.40 → 0.30)
w_spread    = 0.15  (constant)
w_compact   = 0.10  (constant)
w_narrative = 0.10 × progress           (0.0 → 0.10)
w_legal     = 0.10 × progress           (0.0 → 0.10)
```

---

### NOVEL-R3: Reward Warm-Start Curriculum

**Why**: Narrative flow and legal density require a partially trained policy to be meaningful. At epoch 0, the policy selects randomly; computing narrative coherence over random selections produces a noisy signal that destabilises early training (observed: reward collapse by epoch 3 in baseline).

**Solution**: Ramp auxiliary weights from 0 → full over `--reward-warmup-epochs` epochs. Core diversity and coverage components carry full weight initially, providing a stable training signal while the policy bootstraps.

---

### NOVEL-R9: OT Temporal Diversity

**Function**: `compute_ot_temporal_diversity(actions, n)`

```python
pick_idxs = actions.nonzero()              # selected frame positions
norm_picks = sorted(pick_idxs / (n-1))    # normalise to [0,1]
uniform_q  = linspace(0, 1, k)            # ideal uniform distribution
W1 = mean |norm_picks - uniform_q|        # 1D Wasserstein distance
return 1.0 - W1                            # 1.0 = perfectly uniform
```

**Why 1D Wasserstein**: The closed-form solution for 1D OT is the mean absolute difference between sorted empirical CDF and target CDF. No expensive LP solver needed. This measures how evenly the summary samples the video timeline — a key quality criterion for legal summaries that must cover all proceeding phases.

---

### InfoNCE Contrastive Bonus

**Function**: `compute_contrastive_bonus(seq, actions, speaker_mask)`

```python
# Positive pairs: selected frames
pos_feats = seq[actions == 1]   # (k, D)
neg_feats = seq[actions == 0]   # (T-k, D)

# InfoNCE loss: selected frames should be similar to each other
# and dissimilar to unselected frames
# τ = 0.07 (SimCLR temperature)
bonus = InfoNCE(pos_feats, neg_feats, temperature=0.07)
```

Applied with weight `--contrastive-weight` (default 0.05).

---

### Counterfactual Per-Frame Attribution

**Function**: `compute_per_frame_attribution(seq, actions)`

For each selected frame `i`, compute the marginal reward contribution:
```
attribution_i = R(actions) - R(actions with frame_i removed)
```

This replaces the global scalar reward with a dense per-frame signal. The policy gradient then uses `attribution_i` as the advantage for frame `i`'s log-probability, giving a much lower-variance gradient than the scalar REINFORCE baseline.

---

## Training: `main.py`

### NOVEL-T0: Phase-0 Contrastive Pretrain

**Function**: `pretrain_contrastive(model, dataset, train_keys, num_epochs)`

**What is trained**: All parameters except the final FC (policy) head, which is frozen.

**Loss** (fully differentiable — no discrete sampling):
```python
probs = model(seq)                    # (1,T,1)
p_soft = probs.squeeze()             # (T,)

# Entropy maximisation: push policy toward 0.5 selection probability
L_entropy = -Bernoulli(p_soft).entropy().mean()

# Soft diversity: soft-weighted cosine distance from mean
p_norm = p_soft / p_soft.sum()
mean_feat = sum(p_norm × feats)
L_diversity = -sum(p_norm × cosine_distance(feats, mean_feat))

# OT temporal coverage: push soft mean position toward centre + maximise spread
soft_mean_pos = sum(p_norm × linspace(0,1,T))
L_ot = (soft_mean_pos - 0.5)² - sum(p_norm × (positions - soft_mean_pos)²)

L_pretrain = L_entropy + 0.5 × L_diversity + 0.3 × L_ot
```

**Why**: Random initialisation means the policy head starts near constant 0.5. The encoder has random features, making Phase 1 rewards noisy for the first ~15 epochs. Pre-training the encoder with differentiable diversity objectives gives the attention and graph modules meaningful input representations before RL begins.

---

### Phase-1 Exploration

- **Epochs**: `max_epoch - phase2_epochs` (default: 70)
- **LR scheduler**: `CosineAnnealingWarmRestarts(T_0 = phase1_epochs // 3)` — 3 warm-restart cycles
- **Entropy**: decays exponentially from `entropy_start=0.10` → `0.02`
- **Action-lock**: starts at 70th percentile, decays to 50th
- **Early stopping**: patience=10 on validation F1-max

---

### Phase-2 Exploitation

- **Epochs**: `phase2_epochs` (default: 30)
- **LR**: 10× smaller than Phase 1, cosine annealing to 1/100th
- **Entropy**: continues from 0.02 → `entropy_end=0.001`
- **Initialisation**: reload best Phase-1 checkpoint

---

### PPO-Clip Policy Gradient

Replaces vanilla REINFORCE in the RL loop:

```python
# Reference log-probs from old policy (frozen during inner steps)
log_probs_old = Bernoulli(probs).log_prob(actions).detach()

for inner_step in range(ppo_inner_steps):
    probs_new = model(seq)
    log_probs_new = Bernoulli(probs_new).log_prob(actions)
    ratio = exp(log_probs_new - log_probs_old)          # π_new / π_old
    ratio_clipped = clamp(ratio, 1-ε, 1+ε)

    # Per-frame PPO loss (counterfactual case)
    ppo_loss = -min(ratio × advantage,
                    ratio_clipped × advantage).mean()
    ppo_loss.backward()

# Entropy + length penalty added once outside inner loop
cost = β × (mean(p) - 0.15)² - entropy_coef × H(p)
cost.backward()
optimizer.step()
```

**Why PPO over REINFORCE**: Vanilla REINFORCE has unbounded gradient steps — a single lucky (or unlucky) episode can cause a destructive weight update. PPO-clip constrains how much the policy can change per update, making training monotonically more stable.

---

### Adaptive Action-Lock

A percentile-based mechanism to force selection of high-salience frames:

```python
lock_pct = action_lock_start - (start - end) × progress  # 0.70 → 0.50
threshold = percentile(probs, lock_pct × 100)
# Frames above threshold are forced to action=1 before reward computation
locked_actions = max(actions, (probs > threshold).float())
```

**Key fix from baseline**: Previous defaults were 95th/85th percentile. At 95th percentile on a 200-frame video, only 10 frames are ever forced — the policy was effectively locked to a tiny subset and couldn't explore. New defaults 70th/50th allow 30–50% of frames to be force-selected, encouraging exploration of the importance landscape.

---

### Self-Paced Curriculum

```python
# Easy videos (stable rewards, low variance) first
# Hard videos (volatile rewards) phased in after warmup epochs
difficulty = var(reward_history[-10:])
sorted_idxs = argsort(difficulty + N(0, 0.01))  # noisy sort
```

During the first `warmup=10` epochs, videos are in random order. Afterward, sorted ascending by reward variance — easy first, hard introduced gradually.

---

### TTA Multi-Scale Ensemble Inference

```python
tta_scales = [1.0, 0.8, 1.2]
tta_accum = []

for scale in tta_scales:
    seq_scaled = interpolate(seq, size=int(T×scale))     # (1, T', 1024)
    probs_mc = mean([model(seq_scaled) for _ in range(K//2)])  # MC-Dropout
    probs_orig = interp(probs_mc, T'→T)                  # back to original length
    tta_accum.append(probs_orig)

final_probs = mean(tta_accum)   # average across 3 scales
```

---

## Evaluation: `vsum_tools.py`

**Protocol** (standard ECCV16/SumMe/TVSum):

1. Importance scores `p_t` → **knapsack** select frames under 15% budget → binary `machine_summary`
2. Compare against `K` human annotator binary summaries
3. **TVSum**: `F1 = mean over K users of F1(machine, user_k)`
4. **SumMe**: `F1 = max over K users of F1(machine, user_k)`

Both `F1-mean` and `F1-max` are reported. Primary metric matches dataset convention.

---

## Data Flow End-to-End

```
video.mp4
    │
    │  (offline preprocessing)
    ▼
GoogLeNet pool5 → features (T, 1024)
    │
    │  stored in HDF5
    ▼
dataset[key] = {
    'features':       (T, 1024)   visual
    'acoustic':       (T, 128)    MFCC (optional)
    'semantic':       (T, D)      legal keyword vector (optional)
    'speaker_mask':   (T,)        speaker role ID (optional)
    'event_mask':     (T, C)      event category one-hot (optional)
    'change_points':  (S, 2)      segment boundaries
    'n_frames':       int         total frames in original video
    'n_frame_per_seg':(S,)        frames per segment
    'picks':          (T,)        sampled frame indices
    'user_summary':   (K, N)      K human binary summaries over N frames
    'gtscore':        (T,)        averaged human importance scores (optional)
}
    │
    ▼
DSN.forward(features, acoustic, semantic, speaker_mask, event_mask)
    → p_t (T,)
    │
    ▼
Bernoulli(p_t) → a_t ∈ {0,1}^T
    │
    ▼
compute_reward(features, actions, epoch)  +  InfoNCE  +  OT
    → scalar R
    │
    ▼
PPO-Clip gradient → Adam update
    │ (after training)
    ▼
evaluate_with_ensemble(K=10, tta_scales=[1.0, 0.8, 1.2])
    → p_t averaged
    │
    ▼
vsum_tools.generate_summary(p_t, change_points, budget=0.15)
    → machine_summary (N,)  binary
    │
    ▼
vsum_tools.evaluate_summary(machine_summary, user_summary)
    → F1-mean, F1-max, Precision, Recall
```
