import torch
import pytest
from models import ConversationalHypergraphAttention, DSN

def test_conversational_hypergraph_attention_forward():
    # 1. Test basic shape consistency
    hid_dim = 128
    cha = ConversationalHypergraphAttention(hid_dim=hid_dim, dropout=0.1)
    
    B, T = 2, 10
    h = torch.randn(B, T, hid_dim)
    
    # Mock masks
    speaker_mask = torch.randint(0, 3, (B, T))
    event_mask = torch.randint(0, 2, (B, T, 3)).float()
    
    out = cha(h, speaker_mask, event_mask)
    assert out.shape == h.shape, f"Expected shape {h.shape}, got {out.shape}"
    
    # 2. Test fallback (when masks are None)
    out_none = cha(h, None, None)
    assert torch.equal(out_none, torch.zeros_like(h)), "Expected zero tensor when masks are None"

def test_dsn_with_hypergraph():
    # 1. Instantiate DSN
    model = DSN(in_dim=1024, hid_dim=64, num_layers=1, cell='lstm', num_heads=2)
    
    B, T = 2, 20
    x = torch.randn(B, T, 1024)
    acoustic = torch.randn(B, T, 40)
    
    speaker_mask = torch.randint(0, 3, (B, T))
    event_mask = torch.randint(0, 2, (B, T, 3)).float()
    
    # Forward with hypergraph attention active
    p = model(x, acoustic=acoustic, speaker_mask=speaker_mask, event_mask=event_mask)
    assert p.shape == (B, T, 1)
    assert p.min() >= 0.0 and p.max() <= 1.0
    
    # Forward falling back to standard TSG
    p_fallback = model(x, acoustic=acoustic)
    assert p_fallback.shape == (B, T, 1)
    assert p_fallback.min() >= 0.0 and p_fallback.max() <= 1.0
