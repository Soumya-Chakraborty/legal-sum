import torch
import json
import numpy as np
import scipy.stats
from demo.legal_dataset import LegalCourtroomDataset
from models import DSN
from rewards import compute_courtroom_reward
import vsum_tools

def validate_and_compute_metrics(model_path, annotations_path, features_dir):
    # Load courtroom dataset
    dataset = LegalCourtroomDataset(
        annotations_path=annotations_path,
        features_dir=features_dir,
        num_classes=3,
        num_roles=3
    )
    
    # Initialize model
    model = DSN(in_dim=1024, hid_dim=256, num_layers=2, cell='lstm', num_heads=8, dropout=0.25)
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Handle state_dict if it has DataParallel wrapping
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    video_keys = dataset.keys()
    
    results = {}
    
    for key in video_keys:
        # Get data
        item = dataset[key]
        seq = torch.from_numpy(item['features']).unsqueeze(0).float()
        acoustic = torch.from_numpy(item['acoustic']).unsqueeze(0).float()
        semantic = torch.from_numpy(item['semantic']).unsqueeze(0).float()
        
        # Predict importance probabilities
        with torch.no_grad():
            probs = model(seq, acoustic, semantic).squeeze().numpy()
            
        # 1. Knapsack Summary & F-score
        cps = item['change_points']
        n_frames = int(item['n_frames'])
        nfps = item['n_frame_per_seg'].tolist()
        positions = item['picks']
        user_summary = item['user_summary']
        
        machine_summary = vsum_tools.generate_summary(probs, cps, n_frames, nfps, positions)
        
        # F-score (TVSum = avg, SumMe = max)
        f_avg, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'avg')
        f_max, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'max')
        
        # 2. Rank Correlation (Spearman & Kendall)
        gtscore = item['gtscore']
        spearman_corr, _ = scipy.stats.spearmanr(probs, gtscore)
        kendall_corr, _ = scipy.stats.kendalltau(probs, gtscore)
        
        # 3. Courtroom Specific Coverage
        event_mask = item['event_mask']
        speaker_mask = item['speaker_mask']
        
        # Selected frames
        pick_idxs = np.where(machine_summary == 1)[0]
        
        # Event Coverage: fraction of active classes represented in selected frames
        active_classes = np.where(event_mask.sum(axis=0) > 0)[0]
        if len(active_classes) > 0:
            covered_classes = np.where(event_mask[pick_idxs].sum(axis=0) > 0)[0]
            event_coverage = len(np.intersect1d(covered_classes, active_classes)) / len(active_classes)
        else:
            event_coverage = 1.0
            
        # Speaker turn consistency: switches count
        if len(pick_idxs) > 1:
            switches = (speaker_mask[pick_idxs[:-1]] != speaker_mask[pick_idxs[1:]]).sum()
            speaker_consistency = 1.0 - (switches / (len(pick_idxs) - 1))
        else:
            speaker_consistency = 1.0
            
        results[key] = {
            'f_score_avg': float(f_avg),
            'f_score_max': float(f_max),
            'spearman_corr': float(spearman_corr),
            'kendall_corr': float(kendall_corr),
            'event_coverage': float(event_coverage),
            'speaker_consistency': float(speaker_consistency)
        }
        
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python validate_metrics.py <annotations.json> <features_dir> [model_path]")
        sys.exit(1)
        
    anno = sys.argv[1]
    f_dir = sys.argv[2]
    m_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    res = validate_and_compute_metrics(m_path, anno, f_dir)
    print(json.dumps(res, indent=2))
