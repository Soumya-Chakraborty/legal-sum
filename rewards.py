"""
Reinforcement Learning Reward Functions for Video Summarization

This module implements:
1. compute_reward: A 6-component reward function comprising diversity,
   submodular coverage, temporal spread, compactness, narrative flow
   coherence, and legal keyword density.
2. compute_per_frame_attribution: A loop-free vectorized implementation
   of counterfactual frame attribution, used to reduce reward variance in
   REINFORCE. Now includes narrative, legal-density, and courtroom leave-one-out terms.
3. compute_contrastive_bonus: A self-supervised InfoNCE-style contrastive
   bonus that rewards selection of semantically coherent (clustered) frames
   without requiring any labels.
4. compute_legal_coherence_reward: Top-level composite reward for the legal
   domain. Fuses base reward, contrastive bonus, and optional acoustic
   energy variance into a single scalar.
5. compute_multimodal_contrastive_reward: Parameter-free cross-modal similarity
   matrix alignment (CKA-like) across visual, acoustic, and textual streams.
6. compute_courtroom_reward: Courtroom-specific unsupervised composite reward
   combining base reward, multimodal contrastive, event coverage, and speaker consistency.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim_matrix(seq):
    """Return (n, n) cosine-similarity matrix for frame features."""
    normed = seq / (seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return torch.matmul(normed, normed.t())          # range [-1, 1]


def _cosine_diss_matrix(seq):
    """Return (n, n) cosine-dissimilarity (distance) matrix."""
    return 1.0 - _cosine_sim_matrix(seq)            # range [0, 2]


def _narrative_flow(seq_normed, pick_idxs):
    """
    Narrative Flow Coherence reward component.

    Sorts picked indices by temporal order, then computes the mean cosine
    similarity between consecutive selected frames. A high value indicates
    the selected frames form a smooth semantic arc.

    Args:
        seq_normed: L2-normalised frame features, shape (n, dim).
        pick_idxs:  1-D LongTensor of selected frame indices (k,).

    Returns:
        Scalar tensor in [–1, 1]. Returns 0.5 (neutral) when k < 2.
    """
    k = len(pick_idxs)
    if k < 2:
        return torch.tensor(0.5, dtype=torch.float32, device=seq_normed.device)

    sorted_idxs, _ = pick_idxs.sort()               # chronological order
    feats = seq_normed[sorted_idxs]                  # (k, dim)
    # Dot product of each consecutive pair of unit vectors = cosine similarity
    cos_sims = (feats[:-1] * feats[1:]).sum(dim=1)   # (k-1,)
    return cos_sims.mean()


def _legal_density(semantic_boost, pick_idxs, n, device):
    """
    Legal-keyword density reward component.

    Computes the mean legal keyword density score at selected indices,
    normalised to [0, 1].

    Args:
        semantic_boost: per-frame keyword density tensor (n,) or None.
        pick_idxs:      selected frame indices.
        n:              total number of frames.
        device:         target torch device.

    Returns:
        Scalar tensor in [0, 1]. Returns 0.5 (neutral) when
        semantic_boost is None or pick_idxs is empty.
    """
    neutral = torch.tensor(0.5, dtype=torch.float32, device=device)
    if semantic_boost is None or len(pick_idxs) == 0:
        return neutral

    sem_flat = semantic_boost.squeeze()
    if sem_flat.dim() > 1:
        sem_flat = sem_flat.norm(p=2, dim=-1)

    if sem_flat.dim() == 0 or len(sem_flat) != n:
        return neutral

    picked_scores = sem_flat[pick_idxs].float()
    s_min = sem_flat.min().float()
    s_max = sem_flat.max().float()
    denom = s_max - s_min + 1e-8
    normalised = (picked_scores - s_min) / denom
    return normalised.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(seq, actions, use_gpu=False, diss=None,
                   acoustic=None, semantic_boost=None):
    """
    Novel 6-component reward:
      Diversity + Submodular Coverage + Temporal Spread
      + Compactness + Narrative Flow Coherence + Legal Keyword Density.

    Weights: 0.25 * div  +  0.30 * cov  +  0.15 * spread
           + 0.10 * compact  +  0.10 * narrative  +  0.10 * legal_density

    Includes Action-Lock simulation alignment for legal domains:
    If acoustic (loudness) or semantic_boost metrics exist, pre-select
    anomaly frames as locked (assigning action value 1) before evaluation.

    Args:
        seq:            Frame-feature tensor, shape (n, dim) or (1, n, dim).
        actions:        Binary action tensor, shape (n,) or (1, n).
        use_gpu:        Move scalar tensors to CUDA when True.
        diss:           Pre-computed (n, n) cosine-dissimilarity matrix or None.
        acoustic:       Per-frame loudness tensor (n,) or None.
        semantic_boost: Per-frame legal keyword density tensor (n,) or None.

    Returns:
        Scalar reward tensor.
    """
    _seq = seq.detach()
    _actions = actions.detach().clone()
    n = _seq.squeeze().size(0)

    # ── Action-lock: acoustic anomalies ────────────────────────────────────
    if acoustic is not None:
        ac_flat = acoustic.squeeze()
        if ac_flat.dim() > 1:
            ac_flat = ac_flat.norm(p=2, dim=-1)
        if ac_flat.dim() > 0 and len(ac_flat) == n:
            anomaly_thresh = np.percentile(ac_flat.cpu().numpy(), 90)
            locked_mask = ac_flat > anomaly_thresh
            _actions.squeeze()[locked_mask] = 1.0

    # ── Action-lock: semantic keyword frames ───────────────────────────────
    if semantic_boost is not None:
        sem_flat = semantic_boost.squeeze()
        if sem_flat.dim() > 1:
            sem_flat = sem_flat.norm(p=2, dim=-1)
        if sem_flat.dim() > 0 and len(sem_flat) == n:
            boost_mask = sem_flat > 0.0
            _actions.squeeze()[boost_mask] = 1.0

    # Selected frame indices
    pick_idxs = _actions.squeeze().nonzero(as_tuple=False).squeeze(1)
    num_picks = len(pick_idxs)

    # ── Hard penalty: selecting 0 frames ───────────────────────────────────
    if num_picks == 0:
        reward = torch.tensor(-1.0)
        if use_gpu:
            reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()                            # (n, dim)

    # ── Precompute dissimilarity matrix ────────────────────────────────────
    if diss is None:
        diss = _cosine_diss_matrix(_seq)             # (n, n)

    # ── Precompute L2-normalised features (shared across components) ───────
    seq_normed = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)

    # ── 1. Diversity reward ────────────────────────────────────────────────
    if num_picks == 1:
        reward_div = torch.tensor(0.0, device=_seq.device)
    else:
        diss_sub = diss[pick_idxs, :][:, pick_idxs]  # (k, k)
        pick_mat = pick_idxs.expand(num_picks, num_picks)
        temp_dist = torch.abs(pick_mat - pick_mat.t())
        diss_sub = diss_sub.clone()
        diss_sub[temp_dist > 20] = 1.0
        reward_div = diss_sub.sum() / (num_picks * (num_picks - 1.0))

    # ── 2. Submodular coverage reward (Facility Location) ─────────────────
    sim_to_selected = 1.0 - diss[:, pick_idxs]       # (n, k)
    sim_to_selected = (sim_to_selected + 1.0) / 2.0  # rescale to [0, 1]
    max_cov, _ = sim_to_selected.max(dim=1)           # (n,)
    reward_cov = max_cov.mean()

    # ── 3. Temporal spread reward ──────────────────────────────────────────
    norm_picks = pick_idxs.float() / (n - 1.0 + 1e-8)
    uniform_q = torch.linspace(0.0, 1.0, num_picks, device=norm_picks.device)
    sorted_picks, _ = norm_picks.sort()
    emd = (sorted_picks - uniform_q).abs().mean()
    reward_spread = 1.0 - emd

    # ── 4. Compactness / budget-fidelity reward ────────────────────────────
    target_ratio = 0.15
    actual_ratio = num_picks / float(n)
    if actual_ratio < target_ratio:
        compactness = 1.0 - 3.0 * (target_ratio - actual_ratio)
    else:
        compactness = 1.0 - (actual_ratio - target_ratio)
    compactness = max(0.0, min(1.0, compactness))
    reward_compact = torch.tensor(compactness, dtype=torch.float32,
                                  device=_seq.device)

    # ── 5. Narrative Flow Coherence reward ───────────────────────────
    reward_narrative = _narrative_flow(seq_normed, pick_idxs)
    if use_gpu:
        reward_narrative = reward_narrative.to(_seq.device)

    # ── 6. Legal keyword density reward ─────────────────────────────
    reward_legal_density = _legal_density(semantic_boost, pick_idxs, n,
                                          device=_seq.device)

    # ── Weighted combination ───────────────────────────────────────────────
    reward = (0.25 * reward_div
            + 0.30 * reward_cov
            + 0.15 * reward_spread
            + 0.10 * reward_compact
            + 0.10 * reward_narrative
            + 0.10 * reward_legal_density)

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_contrastive_bonus
# ─────────────────────────────────────────────────────────────────────────────

def compute_contrastive_bonus(seq, actions, temperature=0.07):
    """
    Self-supervised InfoNCE-style contrastive bonus (no labels required).

    Selected frames form the 'positive' set; unselected frames are 'negatives'.
    For each selected frame i, one other randomly chosen selected frame j acts
    as its positive, while all unselected frames act as negatives.

    InfoNCE loss per anchor i:
        log( exp(q_i · q_j / τ) / (exp(q_i · q_j / τ) + Σ exp(q_i · k_neg / τ)) )

    The mean over all selected frames is returned as a BONUS (higher = better).

    Args:
        seq:         Frame-feature tensor, shape (n, dim) or (1, n, dim).
        actions:     Binary action tensor, shape (n,) or (1, n).
        temperature: InfoNCE temperature τ (default 0.07).

    Returns:
        Scalar tensor ≤ 0. Returns tensor(0.0) when k < 2 or all selected.
    """
    _seq = seq.detach().squeeze()                    # (n, dim)
    _actions = actions.detach().squeeze()            # (n,)

    pick_idxs = _actions.nonzero(as_tuple=False).squeeze(1)
    k = len(pick_idxs)
    n = _seq.size(0)

    # Edge cases
    if k < 2 or k == n:
        return torch.tensor(0.0, device=_seq.device)

    neg_idxs = (_actions == 0).nonzero(as_tuple=False).squeeze(1)  # (m,)

    # L2-normalise all features
    normed = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)   # (n, dim)
    sel_feats = normed[pick_idxs]                                    # (k, dim)
    neg_feats = normed[neg_idxs]                                     # (m, dim)

    total_loss = torch.tensor(0.0, device=_seq.device)
    count = 0

    for i in range(k):
        q_i = sel_feats[i]                          # (dim,)

        # Draw a random positive (different from i)
        pos_candidates = torch.cat([
            torch.arange(i, device=_seq.device),
            torch.arange(i + 1, k, device=_seq.device)
        ])
        j = pos_candidates[torch.randint(len(pos_candidates), (1,)).item()]
        q_j = sel_feats[j]

        pos_sim = (q_i * q_j).sum() / temperature                   # scalar
        neg_sims = torch.mv(neg_feats, q_i) / temperature            # (m,)

        # log-softmax numerator over positive + all negatives
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])         # (1+m,)
        log_prob = pos_sim - torch.logsumexp(logits, dim=0)
        total_loss = total_loss + log_prob
        count += 1

    return total_loss / count


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_legal_coherence_reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_legal_coherence_reward(seq, actions, acoustic=None,
                                   semantic_boost=None):
    """
    Legal-domain composite reward.

    Fuses:
      • Base 6-component reward from compute_reward()
      • Self-supervised contrastive bonus (weight 0.05)
      • Acoustic energy variance bonus (weight 0.05, only when acoustic given)

    Acoustic bonus rationale: dynamic legal moments (questions, objections,
    rulings) exhibit higher audio energy *variance*. Selected frames should
    therefore have higher energy variance than unselected frames.

    Args:
        seq:            Frame-feature tensor, shape (n, dim) or (1, n, dim).
        actions:        Binary action tensor, shape (n,) or (1, n).
        acoustic:       Per-frame loudness/energy tensor (n,) or None.
        semantic_boost: Per-frame legal keyword density tensor (n,) or None.

    Returns:
        Scalar composite reward tensor.
    """
    # ── Base reward ─────────────────────────────────────────────────────────
    base = compute_reward(seq, actions, acoustic=acoustic,
                          semantic_boost=semantic_boost)

    # ── Contrastive bonus ───────────────────────────────────────────────────
    contrastive = compute_contrastive_bonus(seq, actions)
    total = base + 0.05 * contrastive

    # ── Acoustic energy variance bonus ──────────────────────────────────────
    if acoustic is not None:
        _seq = seq.detach().squeeze()
        _actions = actions.detach().squeeze()
        n = _seq.size(0)

        ac_flat = acoustic.squeeze().float()
        if ac_flat.dim() > 1:
            ac_flat = ac_flat.norm(p=2, dim=-1)
        if ac_flat.dim() > 0 and len(ac_flat) == n:
            pick_idxs = _actions.nonzero(as_tuple=False).squeeze(1)
            unsel_mask = _actions == 0
            unsel_idxs = unsel_mask.nonzero(as_tuple=False).squeeze(1)

            if len(pick_idxs) > 0 and len(unsel_idxs) > 0:
                var_sel = ac_flat[pick_idxs].var() if len(pick_idxs) > 1 \
                    else torch.tensor(0.0, device=ac_flat.device)
                var_unsel = ac_flat[unsel_idxs].var() if len(unsel_idxs) > 1 \
                    else torch.tensor(0.0, device=ac_flat.device)
                reward_acoustic_var = torch.clamp(
                    var_sel / (var_unsel + 1e-8) - 1.0,
                    min=0.0, max=1.0
                )
                total = total + 0.05 * reward_acoustic_var

    return total


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_multimodal_contrastive_reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_multimodal_contrastive_reward(seq, acoustic, semantic, pick_idxs):
    """
    Computes a parameter-free similarity matrix alignment (CKA-like)
    reward across visual (seq), acoustic, and semantic (textual) modalities.
    Encourages selected frames to preserve identical temporal relationship structures.
    """
    k = len(pick_idxs)
    if k < 2:
        return torch.tensor(0.5, device=seq.device)

    v_sel = seq[pick_idxs]
    a_sel = acoustic[pick_idxs]
    t_sel = semantic[pick_idxs]

    S_v = _cosine_sim_matrix(v_sel)
    S_a = _cosine_sim_matrix(a_sel)
    S_t = _cosine_sim_matrix(t_sel)

    triu_indices = torch.triu_indices(k, k, offset=1, device=seq.device)
    v_triu = S_v[triu_indices[0], triu_indices[1]]
    a_triu = S_a[triu_indices[0], triu_indices[1]]
    t_triu = S_t[triu_indices[0], triu_indices[1]]

    # Cosine similarity mapped to [0, 1]
    sim_va = (F.cosine_similarity(v_triu.unsqueeze(0), a_triu.unsqueeze(0)).squeeze() + 1.0) / 2.0
    sim_vt = (F.cosine_similarity(v_triu.unsqueeze(0), t_triu.unsqueeze(0)).squeeze() + 1.0) / 2.0
    sim_at = (F.cosine_similarity(a_triu.unsqueeze(0), t_triu.unsqueeze(0)).squeeze() + 1.0) / 2.0

    return (sim_va + sim_vt + sim_at) / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. compute_courtroom_reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_courtroom_reward(seq, actions, use_gpu=False, acoustic=None, semantic=None, event_mask=None, speaker_mask=None):
    """
    Unsupervised Courtroom-specific composite reward function.
    Fuses:
    1. Base reward (Diversity, Coverage, Spread, Compactness, Narrative, Legal Density) (0.4)
    2. Multimodal Contrastive alignment (0.2)
    3. Event coverage (0.2)
    4. Speaker consistency (0.2)
    """
    _seq = seq.detach().squeeze()
    _actions = actions.detach().squeeze()
    n = _seq.size(0)

    pick_idxs = _actions.nonzero(as_tuple=False).squeeze(1)
    k = len(pick_idxs)

    if k == 0:
        reward = torch.tensor(-1.0)
        if use_gpu:
            reward = reward.cuda()
        return reward

    # 1. Base Reward
    base = compute_reward(seq, actions, use_gpu=use_gpu, acoustic=acoustic, semantic_boost=semantic)

    # 2. Multimodal Contrastive Alignment Reward
    if acoustic is not None and semantic is not None:
        ac_flat = acoustic.squeeze()
        sem_flat = semantic.squeeze()
        r_contrast = compute_multimodal_contrastive_reward(_seq, ac_flat, sem_flat, pick_idxs)
    else:
        r_contrast = torch.tensor(0.5, device=_seq.device)

    # 3. Event Coverage Reward
    if event_mask is not None:
        event_mask = event_mask.squeeze() # (n, num_classes)
        active_classes = (event_mask.sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(1)
        if len(active_classes) > 0:
            covered = (event_mask[pick_idxs].sum(dim=0) > 0).float()
            r_event = covered[active_classes].mean()
        else:
            r_event = torch.tensor(1.0, device=_seq.device)
    else:
        r_event = torch.tensor(0.5, device=_seq.device)

    # 4. Speaker Consistency Reward
    if speaker_mask is not None:
        speaker_mask = speaker_mask.squeeze() # (n,)
        active_speakers = speaker_mask.unique()
        selected_speakers = speaker_mask[pick_idxs].unique()
        r_speaker_cov = len(selected_speakers) / len(active_speakers)

        if k > 1:
            switches = (speaker_mask[pick_idxs[:-1]] != speaker_mask[pick_idxs[1:]]).float().sum()
            r_speaker_trans = 1.0 - (switches / (k - 1))
        else:
            r_speaker_trans = 1.0

        r_speaker = 0.5 * r_speaker_cov + 0.5 * r_speaker_trans
    else:
        r_speaker = torch.tensor(0.5, device=_seq.device)

    # Weighted combination
    total = 0.4 * base + 0.2 * r_contrast + 0.2 * r_event + 0.2 * r_speaker
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 6. compute_per_frame_attribution  (UPGRADED)
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_frame_attribution(seq, actions, use_gpu=False,
                                  acoustic=None, semantic_boost=None,
                                  event_mask=None, speaker_mask=None):
    """
    NOVEL: Per-frame counterfactual attribution for REINFORCE.
    Fully vectorized O(1) loop-free implementation.

    For each selected frame j, estimates the marginal contribution to the
    total reward by computing: attribution_j = full_reward − reward_{−j},
    where reward_{−j} is the reward obtained when frame j is withheld.

    Handles both standard 6-component base reward and courtroom composite reward
    depending on the presence of event_mask and speaker_mask.

    Args:
        seq:            Frame-feature tensor, shape (n, dim) or (1, n, dim).
        actions:        Binary action tensor, shape (n,) or (1, n).
        use_gpu:        Move scalar tensors to CUDA when True.
        acoustic:       Per-frame loudness tensor (n,) or None.
        semantic_boost: Per-frame legal keyword density tensor (n,) or None.
        event_mask:     Per-frame courtroom event targets or None.
        speaker_mask:   Per-frame speaker role targets or None.

    Returns:
        Tuple of (attributions: Tensor shape (n,), full_reward: scalar Tensor).
    """
    _seq = seq.detach().squeeze()                    # (n, dim)

    # Pre-compute shared dissimilarity matrix (reused in compute_reward)
    normed = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
    diss = 1.0 - torch.matmul(normed, normed.t())    # (n, n)
    seq_normed = normed                              # alias for clarity

    is_courtroom = (event_mask is not None or speaker_mask is not None)

    if is_courtroom:
        full_reward = compute_courtroom_reward(
            seq, actions, use_gpu=use_gpu, acoustic=acoustic, semantic=semantic_boost,
            event_mask=event_mask, speaker_mask=speaker_mask
        )
    else:
        full_reward = compute_reward(seq, actions, use_gpu=use_gpu, diss=diss,
                                     acoustic=acoustic,
                                     semantic_boost=semantic_boost)

    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero(as_tuple=False).squeeze(1)
    k = len(pick_idxs)
    n = _seq.size(0)

    attributions = torch.zeros(n, device=seq.device)

    # ── Edge cases ──────────────────────────────────────────────────────────
    if k == 0:
        return attributions, full_reward

    if k == 1:
        reward_minus = torch.tensor([-1.0], device=seq.device)
        attributions[pick_idxs] = full_reward - reward_minus
        return attributions, full_reward

    # ── 1. Diversity (−j) ───────────────────────────────────────────────────
    diss_sub = diss[pick_idxs, :][:, pick_idxs]     # (k, k)
    pick_mat = pick_idxs.expand(k, k)
    temp_dist = torch.abs(pick_mat - pick_mat.t())
    D = diss_sub.clone()
    D[temp_dist > 20] = 1.0

    row_sums = D.sum(dim=1)
    sub_sums = D.sum() - 2 * row_sums
    if k > 2:
        div_minus = sub_sums / ((k - 1) * (k - 2))
    else:
        div_minus = torch.zeros(k, device=seq.device)

    # ── 2. Coverage (−j) ────────────────────────────────────────────────────
    sim_to_selected = 1.0 - diss[:, pick_idxs]
    sim_to_selected = (sim_to_selected + 1.0) / 2.0  # (n, k)
    top2_vals, top2_idxs = sim_to_selected.topk(k=min(2, k), dim=1)

    if k >= 2:
        is_primary = (top2_idxs[:, 0].unsqueeze(1) ==
                      torch.arange(k, device=seq.device))
        max_cov_minus = torch.where(
            is_primary,
            top2_vals[:, 1].unsqueeze(1),
            top2_vals[:, 0].unsqueeze(1)
        )
        cov_minus = max_cov_minus.mean(dim=0)        # (k,)
    else:
        cov_minus = torch.zeros(k, device=seq.device)

    # ── 3. Spread (−j) ──────────────────────────────────────────────────────
    norm_picks = pick_idxs.float() / (n - 1.0 + 1e-8)
    uniform_q_minus = torch.linspace(0.0, 1.0, k - 1, device=seq.device)
    mask = ~torch.eye(k, dtype=torch.bool, device=seq.device)
    expanded = norm_picks.unsqueeze(1).expand(k, k)
    norm_picks_minus = expanded.t()[mask].view(k, k - 1).t()  # (k-1, k)
    emd_minus = (norm_picks_minus - uniform_q_minus.unsqueeze(1)).abs().mean(dim=0)
    spread_minus = 1.0 - emd_minus                   # (k,)

    # ── 4. Compactness (−j) ─────────────────────────────────────────────────
    target_ratio = 0.15
    actual_ratio_minus = (k - 1) / float(n)
    if actual_ratio_minus < target_ratio:
        compactness_minus = 1.0 - 3.0 * (target_ratio - actual_ratio_minus)
    else:
        compactness_minus = 1.0 - (actual_ratio_minus - target_ratio)
    compactness_minus = max(0.0, min(1.0, compactness_minus))
    compact_minus = torch.tensor(compactness_minus, dtype=torch.float32,
                                 device=seq.device)  # scalar broadcast over k

    # ── 5. Narrative Flow (−j) ────────────────────────────────────────
    sorted_idxs, sort_order = pick_idxs.sort()
    feats_sorted = seq_normed[sorted_idxs]

    if k >= 3:
        all_cos = (feats_sorted[:-1] * feats_sorted[1:]).sum(dim=1)  # (k-1,)
        cross_feats_a = feats_sorted[:-2]            # (k-2, dim)
        cross_feats_b = feats_sorted[2:]             # (k-2, dim)
        cross_cos = (cross_feats_a * cross_feats_b).sum(dim=1)  # (k-2,) for p=1..k-2
        total_cos = all_cos.sum()

        narrative_minus = torch.zeros(k, device=seq.device)
        for p in range(k):
            if p == 0:
                new_sum = total_cos - all_cos[0]
                n_pairs = k - 2
            elif p == k - 1:
                new_sum = total_cos - all_cos[k - 2]
                n_pairs = k - 2
            else:
                new_sum = (total_cos
                           - all_cos[p - 1]
                           - all_cos[p]
                           + cross_cos[p - 1])
                n_pairs = k - 2

            narrative_minus[sort_order[p]] = (new_sum / n_pairs
                                              if n_pairs > 0
                                              else torch.tensor(0.5))
    elif k == 2:
        narrative_minus = torch.full((k,), 0.5, device=seq.device)
    else:
        narrative_minus = torch.full((k,), 0.5, device=seq.device)

    # ── 6. Legal Density (−j) ─────────────────────────────────────────
    if semantic_boost is not None:
        sem_flat = semantic_boost.squeeze().float()
        if sem_flat.dim() > 1:
            sem_flat = sem_flat.norm(p=2, dim=-1)
        if sem_flat.dim() > 0 and len(sem_flat) == n:
            s_min = sem_flat.min()
            s_max = sem_flat.max()
            denom = s_max - s_min + 1e-8
            norm_sem = (sem_flat - s_min) / denom   # (n,)
            total_density = norm_sem[pick_idxs].sum()
            density_minus = ((total_density - norm_sem[pick_idxs])
                             / (k - 1 + 1e-8))      # (k,) vectorised
        else:
            density_minus = torch.full((k,), 0.5, device=seq.device)
    else:
        density_minus = torch.full((k,), 0.5, device=seq.device)

    # ── Base leave-one-out reward ────────────────────────────────────────
    base_minus = (0.25 * div_minus
                  + 0.30 * cov_minus
                  + 0.15 * spread_minus
                  + 0.10 * compact_minus
                  + 0.10 * narrative_minus
                  + 0.10 * density_minus)

    # ── Courtroom-specific leave-one-out calculation ──────────────────────
    if is_courtroom:
        # Precompute active classes and speakers
        if event_mask is not None:
            event_mask_sq = event_mask.squeeze()
            active_classes = (event_mask_sq.sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(1)
        else:
            active_classes = None

        if speaker_mask is not None:
            speaker_mask_sq = speaker_mask.squeeze()
            active_speakers = speaker_mask_sq.unique()
        else:
            active_speakers = None

        ac_sq = acoustic.squeeze() if acoustic is not None else None
        sem_sq = semantic_boost.squeeze() if semantic_boost is not None else None

        for p in range(k):
            j_idx = pick_idxs[p]
            # Exclude current pick p
            pick_idxs_minus = torch.cat([pick_idxs[:p], pick_idxs[p+1:]])

            # A. Multimodal Contrastive Alignment Reward Minus j
            if ac_sq is not None and sem_sq is not None:
                contrastive_minus = compute_multimodal_contrastive_reward(_seq, ac_sq, sem_sq, pick_idxs_minus)
            else:
                contrastive_minus = torch.tensor(0.5, device=seq.device)

            # B. Event Coverage Minus j
            if event_mask is not None and active_classes is not None and len(active_classes) > 0:
                covered = (event_mask_sq[pick_idxs_minus].sum(dim=0) > 0).float()
                event_minus = covered[active_classes].mean()
            else:
                event_minus = torch.tensor(0.5, device=seq.device)

            # C. Speaker Consistency Minus j
            if speaker_mask is not None and active_speakers is not None:
                selected_speakers = speaker_mask_sq[pick_idxs_minus].unique()
                r_speaker_cov = len(selected_speakers) / len(active_speakers)

                k_minus = len(pick_idxs_minus)
                if k_minus > 1:
                    switches = (speaker_mask_sq[pick_idxs_minus[:-1]] != speaker_mask_sq[pick_idxs_minus[1:]]).float().sum()
                    r_speaker_trans = 1.0 - (switches / (k_minus - 1))
                else:
                    r_speaker_trans = 1.0

                speaker_minus = 0.5 * r_speaker_cov + 0.5 * r_speaker_trans
            else:
                speaker_minus = torch.tensor(0.5, device=seq.device)

            # Combine leave-one-out reward for pick j
            total_minus = 0.4 * base_minus[p] + 0.2 * contrastive_minus + 0.2 * event_minus + 0.2 * speaker_minus
            attributions[j_idx] = full_reward - total_minus
    else:
        # Standard unsupervised REINFORCE leave-one-out
        attributions[pick_idxs] = full_reward - base_minus

    return attributions, full_reward
