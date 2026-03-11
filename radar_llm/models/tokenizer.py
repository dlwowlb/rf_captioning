"""
Motion-guided Radar Tokenizer (Aggregate VQ-VAE).

Full tokenizer pipeline:
1. Template-prior Grouping Encoder (P4Conv) -> per-anchor features
2. Vector Quantization -> discrete tokens
3. Masked Context Aggregation Decoder -> reconstructed point cloud features
4. Motion alignment loss (with auxiliary Motion Encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .p4conv import P4ConvEncoder
from .vq import VectorQuantizer
from .motion_encoder import MotionEncoder


class MaskedContextAggregationDecoder(nn.Module):
    """Decoder that reconstructs masked anchor trajectories via cross-attention.

    During training, 50% of anchor trajectories are masked. The decoder uses
    cross-attention between masked queries and visible anchor features to
    reconstruct the full set of features, enforcing robust representations.

    Architecture:
        - Learnable mask tokens for masked anchors
        - Cross-attention: masked queries attend to visible keys/values
        - Self-attention within all (masked + visible) tokens
        - MLP head for point cloud reconstruction
    """

    def __init__(self, dim: int = 512, num_heads: int = 8,
                 num_layers: int = 4, num_anchors: int = 16,
                 num_points: int = 64, seq_len: int = 16,
                 point_dim: int = 5):
        super().__init__()
        self.dim = dim
        self.num_anchors = num_anchors
        self.num_points = num_points
        self.seq_len = seq_len
        self.point_dim = point_dim

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Anchor positional embeddings
        self.anchor_pos_embed = nn.Parameter(
            torch.randn(1, num_anchors, dim) * 0.02
        )

        # Cross-attention layers: masked tokens attend to visible tokens
        self.cross_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(dim, num_heads, dropout=0.1,
                                      batch_first=True)
            )
            self.self_attn_layers.append(
                nn.MultiheadAttention(dim, num_heads, dropout=0.1,
                                      batch_first=True)
            )
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim),
                nn.Dropout(0.1),
            ))
            self.norm_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))

        # Reconstruction head: predict point cloud from anchor features
        self.recon_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, (num_points // num_anchors) * seq_len * point_dim),
        )

    def forward(self, z_q: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_q: (B, A, D) quantized anchor features
            mask: (B, A) boolean mask, True = masked (to reconstruct)

        Returns:
            decoded: (B, A, D) reconstructed full anchor features
            recon_points: (B, T, N, point_dim) reconstructed point cloud
        """
        B, A, D = z_q.shape

        # Add anchor positional embeddings
        z_q = z_q + self.anchor_pos_embed[:, :A, :]

        if mask is not None and mask.any():
            # Separate visible and masked tokens
            visible_mask = ~mask  # (B, A)

            # Build per-sample visible/masked features
            # For simplicity, use masking with mask tokens
            masked_input = z_q.clone()
            mask_expanded = mask.unsqueeze(-1).expand_as(z_q)  # (B, A, D)
            masked_input = torch.where(
                mask_expanded, self.mask_token.expand(B, A, D), masked_input
            )

            x = masked_input
            visible_kv = z_q.clone()
            visible_kv = torch.where(
                mask_expanded, torch.zeros_like(visible_kv), visible_kv
            )

            # Key padding mask for cross-attention (True = ignore)
            key_mask = mask  # (B, A) — masked positions should be ignored as keys
        else:
            x = z_q
            visible_kv = z_q
            key_mask = None

        # Transformer decoder layers
        for cross_attn, self_attn, ffn, norms in zip(
            self.cross_attn_layers, self.self_attn_layers,
            self.ffn_layers, self.norm_layers
        ):
            # Cross-attention: all tokens query visible tokens
            residual = x
            x = norms[0](x)
            x_cross, _ = cross_attn(
                query=x, key=visible_kv, value=visible_kv,
                key_padding_mask=key_mask
            )
            x = residual + x_cross

            # Self-attention
            residual = x
            x = norms[1](x)
            x_self, _ = self_attn(query=x, key=x, value=x)
            x = residual + x_self

            # FFN
            residual = x
            x = norms[2](x)
            x = residual + ffn(x)

        decoded = x  # (B, A, D)

        # Reconstruct point cloud from decoded anchor features
        # Each anchor reconstructs its local region
        recon_flat = self.recon_head(decoded)  # (B, A, pts_per_anchor * T * D_pt)
        pts_per_anchor = self.num_points // self.num_anchors
        recon_flat = recon_flat.reshape(
            B, self.num_anchors, self.seq_len, pts_per_anchor, self.point_dim
        )
        # Merge anchors -> (B, T, N, point_dim)
        recon_points = recon_flat.permute(0, 2, 1, 3, 4).reshape(
            B, self.seq_len, self.num_points, self.point_dim
        )

        return decoded, recon_points


class RadarTokenizer(nn.Module):
    """Full Motion-guided Radar Tokenizer (Aggregate VQ-VAE).

    Combines:
    - P4Conv Encoder (Template-prior Grouping)
    - VQ Codebook (Aggregated Quantization)
    - Masked Context Aggregation Decoder
    - Motion Encoder (auxiliary, training only)
    """

    def __init__(self, point_dim: int = 5, seq_len: int = 16,
                 num_points: int = 64, num_anchors: int = 16,
                 encoder_dim: int = 128, encoder_out_dim: int = 512,
                 codebook_size: int = 512, codebook_dim: int = 512,
                 commitment_cost: float = 0.25, mask_ratio: float = 0.5,
                 decoder_heads: int = 8, decoder_layers: int = 4,
                 motion_dim: int = 72, motion_hidden: int = 256,
                 recon_weight: float = 1.0, vq_weight: float = 1.0,
                 motion_align_weight: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_anchors = num_anchors
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.motion_align_weight = motion_align_weight

        # Encoder
        self.encoder = P4ConvEncoder(
            point_dim=point_dim,
            encoder_dim=encoder_dim,
            encoder_out_dim=encoder_out_dim,
            num_anchors=num_anchors,
            seq_len=seq_len,
        )

        # Vector Quantization
        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder
        self.decoder = MaskedContextAggregationDecoder(
            dim=codebook_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            num_anchors=num_anchors,
            num_points=num_points,
            seq_len=seq_len,
            point_dim=point_dim,
        )

        # Motion Encoder (auxiliary, for training alignment)
        self.motion_encoder = MotionEncoder(
            motion_dim=motion_dim,
            hidden_dim=motion_hidden,
            out_dim=codebook_dim,
            num_anchors=num_anchors,
            seq_len=seq_len,
        )

    def generate_mask(self, B: int, A: int,
                      device: torch.device) -> torch.Tensor:
        """Generate random mask for anchor trajectories.

        Args:
            B: batch size
            A: number of anchors
            device: torch device

        Returns:
            mask: (B, A) boolean tensor, True = masked
        """
        num_masked = int(A * self.mask_ratio)
        mask = torch.zeros(B, A, dtype=torch.bool, device=device)
        for b in range(B):
            indices = torch.randperm(A, device=device)[:num_masked]
            mask[b, indices] = True
        return mask

    def encode(self, point_clouds: torch.Tensor) -> Tuple[torch.Tensor,
                                                            torch.Tensor]:
        """Encode point clouds to discrete tokens.

        Args:
            point_clouds: (B, T, N, D) radar point cloud sequence

        Returns:
            token_ids: (B, num_anchors) discrete token indices
            z_q: (B, num_anchors, codebook_dim) quantized features
        """
        anchor_feats, _ = self.encoder(point_clouds)
        z_q, _, token_ids, _ = self.vq(anchor_feats)
        return token_ids, z_q

    def forward(self, point_clouds: torch.Tensor,
                motion_seq: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """Full forward pass with losses.

        Args:
            point_clouds: (B, T, N, D) radar point cloud sequence
            motion_seq: (B, T, motion_dim) optional paired SMPL motion data

        Returns:
            dict with:
                - loss: total training loss
                - recon_loss: point cloud reconstruction loss
                - vq_loss: vector quantization loss
                - motion_loss: motion-radar alignment loss (if motion provided)
                - token_ids: (B, A) discrete token indices
                - perplexity: codebook utilization
                - recon_points: (B, T, N, D) reconstructed point cloud
        """
        B = point_clouds.shape[0]
        device = point_clouds.device

        # 1. Encode: P4Conv + Template-prior Grouping
        anchor_feats, point_feats = self.encoder(point_clouds)
        # anchor_feats: (B, A, C), point_feats: (B, T, N, C)

        # 2. Vector Quantization
        z_q, vq_loss, token_ids, perplexity = self.vq(anchor_feats)

        # 3. Masked Context Aggregation Decoding
        mask = self.generate_mask(B, self.num_anchors, device) if self.training else None
        decoded_feats, recon_points = self.decoder(z_q, mask=mask)

        # 4. Reconstruction loss
        recon_loss = F.mse_loss(recon_points, point_clouds)

        # 5. Motion alignment loss (if motion data is available)
        motion_loss = torch.tensor(0.0, device=device)
        if motion_seq is not None:
            motion_feats = self.motion_encoder(motion_seq)  # (B, A, C)
            # L2 alignment between radar and motion features
            motion_loss = F.mse_loss(anchor_feats, motion_feats.detach())

        # Total loss
        total_loss = (
            self.recon_weight * recon_loss
            + self.vq_weight * vq_loss
            + self.motion_align_weight * motion_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "motion_loss": motion_loss,
            "token_ids": token_ids,
            "perplexity": perplexity,
            "recon_points": recon_points,
        }
