"""
Deep Summarization Network (DSN) Model Architectures

This module defines the PyTorch network models used for predicting video frame-level
importance scores. It implements two main architectures:
1. DSN: Bi-LSTM/GRU coupled with Multi-Head Self-Attention (MHSA) and Feed-Forward (FFN) blocks.
2. DSN_Transformer: A fully attention-based network with sinusoidal Positional Encoding (PE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['DSN', 'DSN_Transformer']


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for capturing global temporal dependencies.
    Includes layer normalization and residual connections.
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        # PyTorch MultiheadAttention expects (batch, seq_len, embed_dim) if batch_first=True
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for Multi-Head Self-Attention using Pre-LN.
        """
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # Self-attention: query=x, key=x, value=x
        return x + self.dropout(attn_out)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation, residual connection, and LayerNorm.
    Similar to the position-wise feed-forward networks inside standard Transformers.
    """

    def __init__(self, embed_dim, ff_dim=1024, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass for Feed-Forward Network using Pre-LN.
        """
        return x + self.net(self.norm(x))


class MultiScaleConv1D(nn.Module):
    """
    Parallel 1D convolutions with different kernel sizes to extract multi-scale
    temporal receptive fields and local-neighbor context from frame features.
    """
    def __init__(self, in_dim, out_dim):
        super(MultiScaleConv1D, self).__init__()
        self.conv3 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=7, padding=3)
        self.proj = nn.Linear(in_dim, out_dim // 4)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        # Transpose input for Conv1d: (batch_size, in_dim, seq_len)
        x_t = x.transpose(1, 2)
        c3 = self.conv3(x_t).transpose(1, 2)
        c5 = self.conv5(x_t).transpose(1, 2)
        c7 = self.conv7(x_t).transpose(1, 2)
        c1 = self.proj(x)
        out = torch.cat([c1, c3, c5, c7], dim=-1)
        return self.norm(out)


class DSN(nn.Module):
    """
    State-of-the-Art Deep Summarization Network:
    MultiScaleConv1D + Bi-LSTM + 2x MHSA + 2x FFN + Dynamic Gate → importance score.
    """

    def __init__(self, in_dim=1024, hid_dim=256, num_layers=2, cell='lstm',
                 num_heads=8, dropout=0.25):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"

        # Multi-scale convolutional projection layer
        self.input_proj = MultiScaleConv1D(in_dim, hid_dim * 2)

        # Recurrent Neural Network (LSTM or GRU) to model local sequence dependencies
        if cell == 'lstm':
            self.rnn = nn.LSTM(
                hid_dim * 2, hid_dim, num_layers=num_layers,
                bidirectional=True, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            self.rnn = nn.GRU(
                hid_dim * 2, hid_dim, num_layers=num_layers,
                bidirectional=True, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

        # Since the RNN is bidirectional, the output dimension is hid_dim * 2
        rnn_out_dim = hid_dim * 2

        # Two stacked Transformer Encoder layers (Self-Attention + Feed-Forward blocks)
        self.attn1 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
        self.ff1 = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim * 4, dropout=dropout)
        self.attn2 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
        self.ff2 = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim * 4, dropout=dropout)

        # Final LayerNorm for the Pre-LN Transformer stack
        self.final_norm = nn.LayerNorm(rnn_out_dim)

        # Learnable gating module to fuse local and global context dynamically
        self.gate = nn.Sequential(
            nn.Linear(rnn_out_dim * 2, rnn_out_dim),
            nn.Sigmoid()
        )

        # Final regression head mapping hidden features to predicted importance scores
        self.fc = nn.Sequential(
            nn.Linear(rnn_out_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1)
        )

    def _positional_encoding(self, x):
        """
        Computes and adds sinusoidal positional encoding to input representations.
        """
        batch, seq_len, d_model = x.shape
        pe = torch.zeros(seq_len, d_model, device=x.device)
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return x + pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass for DSN.

        Args:
            x (Tensor): Input video features of shape (batch_size, seq_len, in_dim).

        Returns:
            Tensor: Frame importance probabilities in [0, 1] of shape (batch_size, seq_len, 1).
        """
        x = self.input_proj(x)
        h_rnn, _ = self.rnn(x)  # Bidirectional RNN output (local features)

        # Run attention stack with Pre-LN and Positional Encoding (global features)
        h_attn = self._positional_encoding(h_rnn)
        h_attn = self.ff1(self.attn1(h_attn))
        h_attn = self.ff2(self.attn2(h_attn))
        h_attn = self.final_norm(h_attn)

        # Dynamically fuse local and global context via gating mechanism
        gate_val = self.gate(torch.cat([h_rnn, h_attn], dim=-1))
        h_fused = gate_val * h_rnn + (1.0 - gate_val) * h_attn

        # Apply sigmoid to output the probability representing the importance score
        p = torch.sigmoid(self.fc(h_fused))
        return p


class DSN_Transformer(nn.Module):
    """
    Fully Transformer-based DSN with sinusoidal Positional Encoding (PE) 
    and Pre-LN (norm_first=True) encoder blocks.
    
    This architecture relies entirely on attention mechanisms to draw global
    context and dependencies across all video frames in parallel.
    """

    def __init__(self, in_dim=1024, hid_dim=512, num_layers=4,
                 num_heads=8, dropout=0.1):
        super(DSN_Transformer, self).__init__()
        self.input_proj = nn.Linear(in_dim, hid_dim)
        self.input_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

        # Define single standard transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, nhead=num_heads, dim_feedforward=hid_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True  # Enables Pre-LN style configuration for stable training
        )
        # Stack multiple layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(hid_dim),
            enable_nested_tensor=False
        )
        
        # Classification/regression head for frame-level selection
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, 1)
        )

    def _positional_encoding(self, x):
        """
        Computes and adds sinusoidal positional encoding to input representations.
        
        Since self-attention is permutation-invariant, positional encoding is
        necessary to give the model awareness of the temporal order of frames.
        """
        batch, seq_len, d_model = x.shape
        pe = torch.zeros(seq_len, d_model, device=x.device)
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Handle cases where d_model is odd
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return x + pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass for DSN_Transformer.

        Args:
            x (Tensor): Input video features of shape (batch_size, seq_len, in_dim).

        Returns:
            Tensor: Frame importance probabilities in [0, 1] of shape (batch_size, seq_len, 1).
        """
        x = self.input_norm(self.input_proj(x))
        x = self._positional_encoding(x)
        x = self.dropout(x)
        h = self.encoder(x)
        p = torch.sigmoid(self.fc(h))
        return p