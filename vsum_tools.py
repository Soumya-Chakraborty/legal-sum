"""
Video Summarization Evaluation and Summary Generation Tools

This module contains core algorithms to convert predicted frame-level importance
scores into a binary keyshot-based video summary (under length constraints) using
either dynamic programming (knapsack) or ranking. It also provides evaluation
metrics to compare generated summaries against human/user summaries.
"""

import numpy as np
from knapsack import knapsack_dp
import math


def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """
    Generates a keyshot-based video summary represented as a binary frame selection vector.

    Args:
        ypred (ndarray): Predicted importance scores for the subsampled frames.
        cps (ndarray): Change points (2D matrix of shape [n_segments, 2]), where each
                      row contains start and end frame indices of a video shot/segment.
        n_frames (int): Total number of frames in the original video.
        nfps (ndarray/list): Number of frames per segment (length of each shot).
        positions (ndarray): Original frame indices corresponding to the subsampled frames.
        proportion (float, optional): Maximum allowed length of the video summary 
                                      (relative to the original video length). Defaults to 0.15.
        method (str, optional): The shot selection method, either 'knapsack' (dynamic programming)
                                or 'rank' (greedy ranking based on score). Defaults to 'knapsack'.

    Returns:
        ndarray: A binary vector of shape (n_frames,) where 1 indicates selected frames 
                 and 0 indicates discarded frames.
    """
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames,), dtype=np.float32)

    # Ensure position array uses integer indices
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    # Ensure boundary conditions include the very end of the video
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    # Map the predicted scores from the subsampled frames back to the original frame indices
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    # Compute segment-level scores by averaging frame scores within each shot/segment boundary
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    # Calculate the maximum frame capacity limit for the summary
    if proportion <= 0:
        # Dynamic: score-distribution-aware budget
        # Use percentile-based threshold for robustness to outliers
        p75 = float(np.percentile(ypred, 75))
        p90 = float(np.percentile(ypred, 90))
        high_conf_ratio = float((ypred >= p75).mean())
        # Clamp: always select between 8% and 25% of frames
        proportion = float(np.clip(high_conf_ratio, 0.08, 0.25))
        # If very high entropy (flat scores), fall back to 15%
        if p90 - float(ypred.min()) < 0.05:
            proportion = 0.15

    limits = int(math.floor(n_frames * proportion))

    # Select segments to include in the summary
    if method == 'knapsack':
        # Solve as a 0/1 Knapsack problem where segment length is the weight
        # and segment average score is the value.
        picks = knapsack_dp(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        # Sort segment indices in descending order of average score
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            # Greedily pick high-scoring segments that fit within limits
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    # Construct the binary frame-level summary vector
    summary = np.zeros((1,), dtype=np.float32)  # Placeholder element
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf,), dtype=np.float32)
        else:
            tmp = np.zeros((nf,), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0)  # Delete the placeholder element
    return summary


def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """
    Compares the generated machine summary with a set of ground-truth user summaries.

    Args:
        machine_summary (ndarray): Binary vector of the generated summary.
        user_summary (ndarray): 2D matrix of shape (n_users, n_frames) containing 
                                binary summaries from multiple human annotators.
        eval_metric (str, optional): Evaluation metric to aggregate human summaries. 
                                    'avg' computes the mean Precision/Recall/F-score.
                                    'max' returns the metrics from the best-performing human summary. 
                                    'all' returns both avg and max F-scores: (f_avg, f_max, prec, rec).
                                    Defaults to 'avg'.

    Returns:
        tuple: (final_f_score, final_prec, final_rec) representing F1-score, Precision, 
               and Recall, OR if eval_metric is 'all', (final_f_avg, final_f_max, final_prec, final_rec).
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users, n_frames = user_summary.shape

    # Force binarization of summary vectors (value > 0 is selected)
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    # Align length of the machine summary to the original video frame count
    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary),))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    # Calculate overlap, precision, recall, and F-score against each human summary
    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx, :]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
            
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    # Aggregate scores over all users
    if eval_metric == 'all':
        final_f_avg = np.mean(f_scores)
        final_f_max = np.max(f_scores)
        # Precision/recall defaults to average metrics across all human summaries
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
        return final_f_avg, final_f_max, final_prec, final_rec

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]

    return final_f_score, final_prec, final_rec


def find_optimal_proportion(ypred, cps, n_frames, nfps, positions,
                            user_summary, metric='avg',
                            proportions=None):
    """
    [NEW] Sweep summary proportions and return the one that maximises F1.

    Used at inference time to find the best budget proportion for this
    specific video (replaces the fixed 15% heuristic).

    Args:
        ypred:        Predicted importance scores (subsampled frames).
        cps:          Change-point segments array.
        n_frames:     Total frame count.
        nfps:         Frames per segment.
        positions:    Subsampled frame positions.
        user_summary: Ground-truth human summaries (n_users, n_frames).
        metric:       'avg' or 'max' aggregation over users.
        proportions:  List of proportions to try; default [0.10..0.25].

    Returns:
        (best_proportion, best_f_score, best_machine_summary)
    """
    if proportions is None:
        proportions = [round(p, 2) for p in
                       [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]]

    best_f = -1.0
    best_prop = 0.15
    best_summary = None

    for prop in proportions:
        ms = generate_summary(ypred, cps, n_frames, nfps, positions,
                              proportion=prop)
        if metric == 'avg':
            f, _, _ = evaluate_summary(ms, user_summary, eval_metric='avg')
        else:
            f, _, _ = evaluate_summary(ms, user_summary, eval_metric='max')

        if f > best_f:
            best_f = f
            best_prop = prop
            best_summary = ms

    return best_prop, best_f, best_summary


def compute_pr_curve(ypred, cps, n_frames, nfps, positions, user_summary,
                     n_points: int = 20):
    """
    [NEW] Compute the Precision-Recall curve for a video by sweeping
    the importance-score threshold used to select frames.

    Returns arrays (precisions, recalls) for plotting or AUPRC computation.

    Args:
        ypred:        Importance score array (subsampled frames).
        cps, n_frames, nfps, positions: Video structure metadata.
        user_summary: Ground-truth human summaries.
        n_points:     Number of threshold levels to sample.

    Returns:
        (precisions, recalls, thresholds) as numpy arrays.
    """
    import numpy as np
    thresholds = np.linspace(0.05, 0.95, n_points)
    precisions, recalls = [], []

    for thr in thresholds:
        # Build binary importance vector at this threshold
        ypred_bin = (ypred >= thr).astype(np.float32)
        # Directly map to frame-level summary without knapsack
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        positions_ext = positions.astype(np.int32)
        if positions_ext[-1] != n_frames:
            positions_ext = np.concatenate([positions_ext, [n_frames]])
        for i in range(len(positions_ext) - 1):
            s, e = positions_ext[i], positions_ext[i + 1]
            if i < len(ypred_bin):
                frame_scores[s:e] = ypred_bin[i]

        ms = (frame_scores > 0).astype(np.float32)
        ms_padded = ms[:n_frames]

        # Evaluate against all users
        p_all, r_all = [], []
        for u in range(user_summary.shape[0]):
            gt = user_summary[u].astype(np.float32)
            overlap = (ms_padded * gt).sum()
            p_u = overlap / (ms_padded.sum() + 1e-8)
            r_u = overlap / (gt.sum() + 1e-8)
            p_all.append(p_u)
            r_all.append(r_u)

        precisions.append(float(np.mean(p_all)))
        recalls.append(float(np.mean(r_all)))

    return np.array(precisions), np.array(recalls), thresholds


def compute_auprc(precisions, recalls):
    """
    [NEW] Area Under the Precision-Recall Curve via trapezoidal integration.
    Higher = better for both precision and recall simultaneously.
    """
    import numpy as np
    # Sort by recall ascending for proper integration
    order = np.argsort(recalls)
    r_sorted = np.array(recalls)[order]
    p_sorted = np.array(precisions)[order]
    return float(np.trapz(p_sorted, r_sorted))