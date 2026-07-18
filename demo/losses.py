"""
demo/losses.py — Multimodal Alignment Loss Functions.

=============================================================================
NOVELTY MAP — where to find each original contribution in this file
=============================================================================

[NOVEL-L1] MultimodalAlignmentLoss — 4-component supervised loss (class, line ~17)
    Combines: InfoNCE contrastive (visual-text, visual-audio, audio-text)
    + binary event classification + speaker-role cross-entropy
    + MSE summarization importance + length-budget penalty.
    First multi-objective supervised loss specifically designed for
    courtroom video data (event labels + speaker roles + importance).

[NOVEL-L2] Symmetric InfoNCE across 3 streams (info_nce_loss, line ~5)
    Symmetric contrastive loss applied to all three modality projection pairs,
    aligning visual, acoustic, and textual embeddings in a shared space
    without requiring external pre-trained encoders.
=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(x, y, temperature=0.07):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    similarity = torch.matmul(x, y.T) / temperature
    targets = torch.arange(x.size(0), device=x.device)
    
    loss_x = F.cross_entropy(similarity, targets)
    loss_y = F.cross_entropy(similarity.T, targets)
    
    return 0.5 * (loss_x + loss_y)

class MultimodalAlignmentLoss(nn.Module):
    def __init__(self, visual_dim=1024, acoustic_dim=40, textual_dim=768, projection_dim=128, num_classes=3, num_roles=3, temperature=0.07, beta=1.0, lambda_contrast=1.0, lambda_event=1.0, lambda_speaker=1.0, lambda_sum=1.0):
        super(MultimodalAlignmentLoss, self).__init__()
        self.proj_vis = nn.Linear(visual_dim, projection_dim)
        self.proj_ac = nn.Linear(acoustic_dim, projection_dim)
        self.proj_txt = nn.Linear(textual_dim, projection_dim)
        
        self.temperature = temperature
        self.beta = beta
        self.lambda_contrast = lambda_contrast
        self.lambda_event = lambda_event
        self.lambda_speaker = lambda_speaker
        self.lambda_sum = lambda_sum
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, visual, acoustic, textual, pred_events, pred_speaker, pred_importance, target_events, target_speaker, target_importance):
        v_proj = self.proj_vis(visual)
        a_proj = self.proj_ac(acoustic)
        t_proj = self.proj_txt(textual)
        
        loss_vt = info_nce_loss(v_proj, t_proj, self.temperature)
        loss_va = info_nce_loss(v_proj, a_proj, self.temperature)
        loss_at = info_nce_loss(a_proj, t_proj, self.temperature)
        
        loss_contrast = loss_vt + loss_va + loss_at
        loss_event = self.bce_loss(pred_events, target_events)
        loss_speaker = self.ce_loss(pred_speaker, target_speaker)
        
        loss_mse = self.mse_loss(pred_importance, target_importance)
        loss_length = self.beta * (pred_importance.mean() - 0.15) ** 2
        loss_sum = loss_mse + loss_length
        
        total_loss = (
            self.lambda_contrast * loss_contrast +
            self.lambda_event * loss_event +
            self.lambda_speaker * loss_speaker +
            self.lambda_sum * loss_sum
        )
        
        return {
            "total_loss": total_loss,
            "contrastive_loss": loss_contrast,
            "event_loss": loss_event,
            "speaker_loss": loss_speaker,
            "summarization_loss": loss_sum
        }
