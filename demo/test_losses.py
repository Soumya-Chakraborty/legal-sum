import torch
import pytest

def test_multimodal_alignment_loss():
    from demo.losses import MultimodalAlignmentLoss
    
    loss_fn = MultimodalAlignmentLoss(
        visual_dim=128,
        acoustic_dim=40,
        textual_dim=64,
        projection_dim=32,
        num_classes=3,
        num_roles=3
    )
    
    T = 10
    visual = torch.randn(T, 128)
    acoustic = torch.randn(T, 40)
    textual = torch.randn(T, 64)
    
    pred_events = torch.randn(T, 3)
    pred_speaker = torch.randn(T, 3)
    pred_importance = torch.rand(T)
    
    target_events = torch.zeros(T, 3)
    target_events[2:5, 0] = 1.0
    target_speaker = torch.ones(T, dtype=torch.long)
    target_importance = torch.rand(T)
    
    losses = loss_fn(
        visual=visual,
        acoustic=acoustic,
        textual=textual,
        pred_events=pred_events,
        pred_speaker=pred_speaker,
        pred_importance=pred_importance,
        target_events=target_events,
        target_speaker=target_speaker,
        target_importance=target_importance
    )
    
    assert "total_loss" in losses
    assert "contrastive_loss" in losses
    assert "event_loss" in losses
    assert "speaker_loss" in losses
    assert "summarization_loss" in losses
    assert losses["total_loss"].item() > 0.0
