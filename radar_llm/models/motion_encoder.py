"""
Motion Encoder: auxiliary training network.

Encodes paired motion data (SMPL-X pose parameters) into semantic features
that guide the radar tokenizer to learn motion-aware representations.
This encoder is ONLY used during training (not at inference).
"""

import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """Encodes SMPL motion sequences into semantic feature vectors.

    Takes a sequence of SMPL pose parameters and produces feature vectors
    aligned with the radar tokenizer's codebook space. Used during training
    to provide supervision signal via motion-radar alignment loss.

    Architecture:
        Input: (B, T, motion_dim)  e.g., (B, 16, 72) for SMPL-24
        -> Temporal Transformer Encoder
        -> Mean pooling over time
        -> Projection to codebook dimension
        -> Output: (B, num_anchors, codebook_dim)
    """

    def __init__(self, motion_dim: int = 72, hidden_dim: int = 256,
                 out_dim: int = 512, num_anchors: int = 16,
                 num_layers: int = 4, nhead: int = 8, seq_len: int = 16):
        super().__init__()
        self.num_anchors = num_anchors
        self.out_dim = out_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Positional encoding for temporal sequence
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project to anchor-wise features
        self.anchor_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_anchors * out_dim),
        )

    def forward(self, motion_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_seq: (B, T, motion_dim) SMPL pose parameter sequence
                motion_dim = 72 for SMPL-24 (24 joints * 3 axis-angle)

        Returns:
            motion_features: (B, num_anchors, out_dim) motion semantic features
        """
        B, T, _ = motion_seq.shape

        # Embed and add positional encoding
        x = self.input_proj(motion_seq)  # (B, T, hidden)
        x = x + self.pos_embed[:, :T, :]

        # Temporal encoding
        x = self.transformer(x)  # (B, T, hidden)

        # Pool across time
        x = x.mean(dim=1)  # (B, hidden)

        # Project to anchor features
        anchor_feats = self.anchor_proj(x)  # (B, num_anchors * out_dim)
        anchor_feats = anchor_feats.reshape(B, self.num_anchors, self.out_dim)

        return anchor_feats
