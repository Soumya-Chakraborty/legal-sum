import os
import sys
import h5py
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import DSN
import vsum_tools
from knapsack import knapsack_dp

def test_benchmark_video():
    print("==========================================================")
    print("         BENCHMARK TEST OF LEGALSUM vs. SCSL-SGC          ")
    print("==========================================================\n")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(base_dir, 'datasets/eccv16_dataset_summe_google_pool5.h5')
    checkpoint_path = os.path.join(base_dir, 'log/summe-counterfactual-optimized/model_best.pth.tar')
    
    if not os.path.exists(dataset_path) or not os.path.exists(checkpoint_path):
        print("Error: Dataset or checkpoint not found.")
        return

    # Let's test on 'video_18'
    video_name = 'video_18'
    print(f"Loading benchmark test video: {video_name}")
    
    with h5py.File(dataset_path, 'r') as dataset:
        features = dataset[video_name]['features'][...]
        cps = dataset[video_name]['change_points'][...]
        num_frames = dataset[video_name]['n_frames'][()]
        nfps = dataset[video_name]['n_frame_per_seg'][...].tolist()
        positions = dataset[video_name]['picks'][...]
        user_summary = dataset[video_name]['user_summary'][...]

    # 1. Standard SCSL-SGC Model Inference
    model = DSN(in_dim=1024, hid_dim=256, num_layers=2, cell='lstm', num_heads=8, dropout=0.40)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    features_t = torch.from_numpy(features).unsqueeze(0).float()
    with torch.no_grad():
        probs = model(features_t).squeeze().numpy()

    # Standard SCSL-SGC summary compilation
    standard_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
    standard_f1, standard_prec, standard_rec = vsum_tools.evaluate_summary(standard_summary, user_summary, eval_metric='max')

    # 2. LegalSum (High-Motion Anomaly Lock) Summary Compilation
    # Compute frame-level motion scores from GoogLeNet feature variance as a proxy
    feature_diff = np.abs(features[1:] - features[:-1]).mean(axis=1)
    motion_scores = np.zeros((features.shape[0],), dtype=np.float32)
    motion_scores[1:] = feature_diff
    
    # Map to frame level
    frame_motion = np.zeros((num_frames,), dtype=np.float32)
    padded_positions = np.concatenate([positions, [num_frames]])
    for i in range(len(padded_positions) - 1):
        pos_left, pos_right = padded_positions[i], padded_positions[i + 1]
        if i < len(motion_scores):
            frame_motion[pos_left:pos_right] = motion_scores[i]

    # Segment calculations
    n_segs = len(nfps)
    motion_threshold = np.percentile(frame_motion, 85)
    locked_segs = []
    seg_scores = []

    # Map probability scores to frame-level
    frame_scores = np.zeros((num_frames,), dtype=np.float32)
    for i in range(len(padded_positions) - 1):
        pos_left, pos_right = padded_positions[i], padded_positions[i + 1]
        if i < len(probs):
            frame_scores[pos_left:pos_right] = probs[i]

    for i in range(n_segs):
        start, end = int(cps[i, 0]), int(cps[i, 1] + 1)
        seg_motion = frame_motion[start:end]
        max_motion = float(np.max(seg_motion)) if len(seg_motion) > 0 else 0.0
        
        if max_motion > motion_threshold:
            locked_segs.append(i)
            seg_scores.append(1.0)
        else:
            seg_scores.append(float(frame_scores[start:end].mean()))

    # Solve Knapsack with 20% limit
    limit = int(num_frames * 0.20)
    picks = knapsack_dp(seg_scores, nfps, n_segs, limit)
    for locked in locked_segs:
        if locked not in picks:
            picks.append(locked)

    legal_summary = np.zeros((num_frames,), dtype=np.int32)
    for seg_idx in range(n_segs):
        if seg_idx in picks:
            start, end = cps[seg_idx]
            legal_summary[start:end+1] = 1

    legal_f1, legal_prec, legal_rec = vsum_tools.evaluate_summary(legal_summary, user_summary, eval_metric='max')

    print("---------------------------------------------------------")
    print(f"Results for Video: {video_name}")
    print(f"Total Frames:      {num_frames}")
    print("---------------------------------------------------------")
    print("Metrics Comparison:")
    print(f"  SCSL-SGC (Standard):  F1: {standard_f1:.1%} | Precision: {standard_prec:.1%} | Recall: {standard_rec:.1%}")
    print(f"  LegalSum (Verifiable): F1: {legal_f1:.1%} | Precision: {legal_prec:.1%} | Recall: {legal_rec:.1%}")
    print("---------------------------------------------------------")
    print(f"Action-Lock successfully preserved {len(locked_segs)} key action segments in LegalSum.")
    print("==========================================================")

if __name__ == '__main__':
    test_benchmark_video()
