import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['DSN', 'DSN_Transformer']


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module for capturing global temporal dependencies."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with residual + LayerNorm."""

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
        return self.norm(x + self.net(x))


class DSN(nn.Module):
    """
    Enhanced Deep Summarization Network:
    Bi-LSTM + 2x Multi-Head Self-Attention + 2x Feed-Forward → importance score.

    Key enhancements over original:
      - Multi-layer Bi-LSTM for richer sequential representation
      - Multi-Head Self-Attention to capture global frame dependencies
      - Transformer-style Feed-Forward blocks with residual + LayerNorm
      - Dropout for regularization
    """

    def __init__(self, in_dim=1024, hid_dim=256, num_layers=2, cell='lstm',
                 num_heads=8, dropout=0.25):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"

        self.input_proj = nn.Linear(in_dim, hid_dim * 2)
        self.input_norm = nn.LayerNorm(hid_dim * 2)

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

        rnn_out_dim = hid_dim * 2

        # Two Transformer encoder blocks
        self.attn1 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
        self.ff1 = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim * 4, dropout=dropout)
        self.attn2 = MultiHeadSelfAttention(rnn_out_dim, num_heads=num_heads, dropout=dropout)
        self.ff2 = FeedForward(rnn_out_dim, ff_dim=rnn_out_dim * 4, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(rnn_out_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        x = self.input_norm(self.input_proj(x))
        h, _ = self.rnn(x)
        h = self.ff1(self.attn1(h))
        h = self.ff2(self.attn2(h))
        p = torch.sigmoid(self.fc(h))
        return p


class DSN_Transformer(nn.Module):
    """
    Fully Transformer-based DSN with sinusoidal PE and Pre-LN encoder blocks.
    Lighter and fully parallelizable.
    """

    def __init__(self, in_dim=1024, hid_dim=512, num_layers=4,
                 num_heads=8, dropout=0.1):
        super(DSN_Transformer, self).__init__()
        self.input_proj = nn.Linear(in_dim, hid_dim)
        self.input_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, nhead=num_heads, dim_feedforward=hid_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(hid_dim),
            enable_nested_tensor=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, 1)
        )

    def _positional_encoding(self, x):
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
        x = self.input_norm(self.input_proj(x))
        x = self._positional_encoding(x)
        x = self.dropout(x)
        h = self.encoder(x)
        p = torch.sigmoid(self.fc(h))
        return p