import torch
import pytest
from rewards import (
    compute_multimodal_contrastive_reward,
    compute_courtroom_reward,
    compute_per_frame_attribution
)

def test_courtroom_rewards():
    n_frames = 100
    visual_dim = 128
    acoustic_dim = 40
    textual_dim = 64
    
    seq = torch.randn(n_frames, visual_dim)
    acoustic = torch.randn(n_frames, acoustic_dim)
    semantic = torch.randn(n_frames, textual_dim)
    
    # 5 frames selected
    actions = torch.zeros(n_frames)
    pick_idxs = [10, 25, 40, 65, 80]
    actions[pick_idxs] = 1.0
    
    # Dummy courtroom masks
    event_mask = torch.zeros(n_frames, 3)
    event_mask[10:20, 0] = 1.0
    event_mask[25:35, 1] = 1.0
    
    speaker_mask = torch.randint(0, 3, (n_frames,))
    
    # Test multimodal contrastive reward directly
    r_contrast = compute_multimodal_contrastive_reward(seq, acoustic, semantic, torch.tensor(pick_idxs))
    assert r_contrast.item() >= 0.0
    assert r_contrast.item() <= 1.0
    
    # Test courtroom reward
    r_court = compute_courtroom_reward(
        seq, actions, use_gpu=False,
        acoustic=acoustic, semantic=semantic,
        event_mask=event_mask, speaker_mask=speaker_mask
    )
    assert r_court.item() > -1.0
    
    # Test counterfactual attribution with courtroom reward
    attributions, full_reward = compute_per_frame_attribution(
        seq, actions, use_gpu=False,
        acoustic=acoustic, semantic_boost=semantic,
        event_mask=event_mask, speaker_mask=speaker_mask
    )
    
    assert attributions.shape == (n_frames,)
    assert full_reward.item() == r_court.item()
    # Selected frames should have potentially non-zero attribution, unselected frames should have 0 attribution
    for idx in range(n_frames):
        if idx not in pick_idxs:
            assert attributions[idx].item() == 0.0
        else:
            # Selected frame attribution should be a real value
            assert not torch.isnan(attributions[idx])
