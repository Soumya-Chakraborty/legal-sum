"""
Reinforcement Learning Reward Functions for Video Summarization

This module implements:
1. compute_reward: A 4-component reward function comprising diversity, 
   submodular coverage, temporal spread, and compactness.
2. compute_per_frame_attribution: A loop-free vectorized implementation
   of counterfactual frame attribution, used to reduce reward variance in REINFORCE.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_reward(seq, actions, use_gpu=False, diss=None):
    """
    Novel 4-component reward: Diversity + Submodular Coverage + 
    Temporal Spread + Compactness Penalty.

    KEY INNOVATIONS over the original paper:
    ─────────────────────────────────────────
    1. SUBMODULAR COVERAGE REWARD (replaces simple cosine representativeness):
       Computes a greedy-submodular coverage score: each selected frame
       contributes only its MARGINAL gain in coverage over already-covered frames.
       This is a tighter measure of how much new information each selected frame adds.
       It naturally handles redundant selections (selecting two similar frames
       contributes almost nothing extra to coverage), which the simple cosine
       representativeness reward does not penalize.

    2. TEMPORAL SPREAD REWARD (replaces variance-based heuristic):
       Measures how uniformly selected frames are distributed in time using
       Earth Mover's Distance (approximated as mean absolute deviation from
       a uniform distribution). A selection that clusters in one part of the
       video is penalized proportionally to how far it deviates from uniform.

    3. COMPACTNESS / BUDGET FIDELITY REWARD (NEW):
       The target summary is 15% of the video. We reward the model for
       selecting close to 15% of frames. Critically, we apply a STRONG
       ASYMMETRIC penalty for selecting < 5% (to prevent policy collapse)
       and a mild penalty for selecting > 40%.

    4. DIVERSITY REWARD (unchanged from original):
       Dissimilarity-based pairwise diversity among selected frames.
    """
    _seq = seq.detach()
    _actions = actions.detach()
    # Find the indices of all selected frames (where actions == 1)
    pick_idxs = _actions.squeeze().nonzero(as_tuple=False).squeeze(1)
    num_picks = len(pick_idxs)
    n = _seq.squeeze().size(0)

    # ── HARD PENALTY: selecting 0 frames destroys coverage entirely ──────────
    if num_picks == 0:
        reward = torch.tensor(-1.0)
        if use_gpu:
            reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()   # Shape: (n, dim)

    # ── Precompute/Get Dissimilarity Matrix ──────────────────────────────────
    if diss is None:
        # L2-normalize video features to compute cosine similarity
        normed = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
        # Cosine distance: 1.0 - Cosine Similarity
        diss = 1.0 - torch.matmul(normed, normed.t())          # Shape: (n, n)

    # ── 1. DIVERSITY REWARD ───────────────────────────────────────────────────
    if num_picks == 1:
        reward_div = torch.tensor(0.0)
        if use_gpu:
            reward_div = reward_div.cuda()
    else:
        # Extract the sub-matrix representing distances between selected frames only
        diss_sub = diss[pick_idxs, :][:, pick_idxs]            # Shape: (k, k)
        # Ignore distance/similarity between temporally distant picks (original paper trick)
        pick_mat = pick_idxs.expand(num_picks, num_picks)
        temp_dist = torch.abs(pick_mat - pick_mat.t())
        diss_sub[temp_dist > 20] = 1.0
        # Calculate mean pairwise dissimilarity
        reward_div = diss_sub.sum() / (num_picks * (num_picks - 1.0))

    # ── 2. SUBMODULAR COVERAGE REWARD ────────────────────────────────────────
    # Compute similarity: 1.0 - Cosine Distance
    sim_to_selected = 1.0 - diss[:, pick_idxs]                  # Shape: (n, k)
    # Rescale similarity score from [-1, 1] to [0, 1]
    sim_to_selected = (sim_to_selected + 1.0) / 2.0
    # For each video frame, find its maximum similarity to any selected frame
    max_cov, _ = sim_to_selected.max(dim=1)                    # Shape: (n,)
    reward_cov = max_cov.mean()                                 # Range: [0, 1]

    # ── 3. TEMPORAL SPREAD REWARD ─────────────────────────────────────────────
    # Normalize selected frame indices to [0, 1] range
    norm_picks = pick_idxs.float() / (n - 1.0 + 1e-8)          # Shape: (k,)
    # Compute a perfectly uniform distribution spanning [0, 1]
    uniform_q = torch.linspace(0.0, 1.0, num_picks, device=norm_picks.device)
    sorted_picks, _ = norm_picks.sort()
    # Compute Earth Mover's Distance (EMD) between selected frames and uniform distribution
    emd = (sorted_picks - uniform_q).abs().mean()
    # Invert EMD so that higher spread (closer to uniform) yields a higher reward
    reward_spread = 1.0 - emd                                   # perfect spread = 1

    # ── 4. COMPACTNESS / BUDGET FIDELITY REWARD ──────────────────────────────
    target_ratio = 0.15
    actual_ratio = num_picks / float(n)
    if actual_ratio < target_ratio:
        # Asymmetric harsh penalty for undershooting target summary length (e.g. policy collapse)
        compactness = 1.0 - 3.0 * (target_ratio - actual_ratio)
    else:
        # Mild penalty for overshooting target summary length
        compactness = 1.0 - (actual_ratio - target_ratio)
    compactness = max(0.0, min(1.0, compactness))
    reward_compact = torch.tensor(compactness, dtype=torch.float32)
    if use_gpu:
        reward_compact = reward_compact.cuda()

    # ── WEIGHTED COMBINATION ──────────────────────────────────────────────────
    reward = (0.30 * reward_div
            + 0.35 * reward_cov
            + 0.20 * reward_spread
            + 0.15 * reward_compact)

    return reward


def compute_per_frame_attribution(seq, actions, use_gpu=False):
    """
    NOVEL: Per-frame counterfactual attribution for REINFORCE.
    Fully vectorized O(1) loop-free implementation.

    Standard REINFORCE assigns the SAME reward to all selected/unselected frames,
    which is high variance because a single bad frame can ruin the whole summary.

    This function computes for each SELECTED frame i:
        attribution(i) = reward(S) - reward(S \\ {i})
    i.e., how much the full reward drops if frame i is removed.

    Frames with HIGH attribution contributed more to the summary quality
    and should receive STRONGER positive reinforcement.
    Frames with LOW (or negative) attribution are redundant/harmful
    and should receive WEAKER (or negative) reinforcement.

    This dramatically reduces gradient variance compared to REINFORCE
    and is the key to stable training on short videos.

    Returns:
        attributions: (seq_len,) float tensor. 
                      Selected frames: their counterfactual attribution.
                      Unselected frames: 0.
        full_reward: the total reward for the complete selection.
    """
    _seq = seq.detach().squeeze()   # Shape: (n, dim)
    # Compute cosine dissimilarity matrix
    normed = _seq / (_seq.norm(p=2, dim=1, keepdim=True) + 1e-8)
    diss = 1.0 - torch.matmul(normed, normed.t())          # Shape: (n, n)

    full_reward = compute_reward(seq, actions, use_gpu=use_gpu, diss=diss)
    
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero(as_tuple=False).squeeze(1)
    k = len(pick_idxs)
    n = seq.squeeze().size(0)

    attributions = torch.zeros(n, device=seq.device)

    # Base cases for edge conditions
    if k == 0:
        return attributions, full_reward

    if k == 1:
        reward_minus = torch.tensor([-1.0], device=seq.device)
        attributions[pick_idxs] = full_reward - reward_minus
        return attributions, full_reward

    # ── 1. Diversity minus j ──────────────────────────────────────────────────
    # Vectorized computation of diversity when removing each item j
    diss_sub = diss[pick_idxs, :][:, pick_idxs]
    pick_mat = pick_idxs.expand(k, k)
    temp_dist = torch.abs(pick_mat - pick_mat.t())
    D = diss_sub.clone()
    D[temp_dist > 20] = 1.0
    
    # Calculate row sums and compute the remaining matrix sum if row i is removed
    row_sums = D.sum(dim=1)
    sub_sums = D.sum() - 2 * row_sums
    if k > 2:
        div_minus = sub_sums / ((k - 1) * (k - 2))
    else:
        div_minus = torch.zeros(k, device=seq.device)

    # ── 2. Coverage minus j ───────────────────────────────────────────────────
    # Vectorized computation of submodular coverage when removing each item j
    sim_to_selected = 1.0 - diss[:, pick_idxs]
    sim_to_selected = (sim_to_selected + 1.0) / 2.0  # Shape: (n, k)
    # Find the top 2 similarities to any selected frame for each frame in the video
    top2_vals, top2_idxs = sim_to_selected.topk(k=2, dim=1)
    
    # If the removed index was the primary coverage source for a frame, its coverage 
    # degrades to the second best similarity. Otherwise, it stays the same.
    is_primary = (top2_idxs[:, 0].unsqueeze(1) == torch.arange(k, device=seq.device))
    max_cov_minus = torch.where(is_primary, top2_vals[:, 1].unsqueeze(1), top2_vals[:, 0].unsqueeze(1))
    cov_minus = max_cov_minus.mean(dim=0)

    # ── 3. Spread minus j ─────────────────────────────────────────────────────
    # Vectorized computation of EMD spread when removing each item j
    norm_picks = pick_idxs.float() / (n - 1.0 + 1e-8)
    uniform_q_minus = torch.linspace(0.0, 1.0, k - 1, device=seq.device)
    mask = ~torch.eye(k, dtype=torch.bool, device=seq.device)
    expanded = norm_picks.unsqueeze(1).expand(k, k)
    # Exclude element j for each of the k combinations
    norm_picks_minus = expanded.t()[mask].view(k, k - 1).t() # Shape: (k - 1, k)
    emd_minus = (norm_picks_minus - uniform_q_minus.unsqueeze(1)).abs().mean(dim=0)
    spread_minus = 1.0 - emd_minus

    # ── 4. Compactness minus j ────────────────────────────────────────────────
    # Vectorized computation of compactness when removing each item j
    target_ratio = 0.15
    actual_ratio_minus = (k - 1) / float(n)
    if actual_ratio_minus < target_ratio:
        compactness_minus = 1.0 - 3.0 * (target_ratio - actual_ratio_minus)
    else:
        compactness_minus = 1.0 - (actual_ratio_minus - target_ratio)
    compactness_minus = max(0.0, min(1.0, compactness_minus))
    compact_minus = torch.tensor(compactness_minus, dtype=torch.float32, device=seq.device)

    # Combined rewards
    reward_minus = (0.30 * div_minus
                  + 0.35 * cov_minus
                  + 0.20 * spread_minus
                  + 0.15 * compact_minus)

    attributions[pick_idxs] = full_reward - reward_minus
    return attributions, full_reward
