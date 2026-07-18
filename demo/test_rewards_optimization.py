import torch
import numpy as np
import pytest
from rewards import RunningRewardNormalizer, compute_contrastive_bonus, compute_courtroom_reward

def test_running_normalizer():
    normalizer = RunningRewardNormalizer(num_components=6)
    rewards = torch.randn(5, 6)
    norm_rewards = normalizer(rewards)
    assert norm_rewards.shape == (5, 6)

def test_hard_negative_contrastive():
    # 5 frames
    seq = torch.randn(5, 10)
    actions = torch.tensor([0, 1, 0, 1, 0])
    speaker_mask = torch.tensor([0, 0, 1, 1, 2])
    
    # Run contrastive bonus
    cb = compute_contrastive_bonus(seq, actions, speaker_mask=speaker_mask)
    assert cb is not None
