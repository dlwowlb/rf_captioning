"""
Full Radar Q-Former Pipeline.

Combines Module 1 (ViT-Tiny), Module 2 (Tokenizer), Module 3 (Q-Former)
into a single end-to-end model.

Pipeline:
  Input: (B, T, H, W) × 3 radar heatmaps (RD, RA, DA)
  → Module 1: ViT-Tiny per view → (B*T, N_patches, 192) × 3
  → Module 2: Positional embedding fusion → (B*T, N_patches, 192) × 3
  → Module 3a: Spatial Q-Former → (B*T, 32, 768) per frame
  → Module 3b: Temporal Q-Former → (B, 32, 768) Frame-Query output
  Output: Z_Frame-Query (B, 32, 768) aligned with Z_text
"""

import torch
import torch.nn as nn
from einops import rearrange

from .vit_tiny import TriViewViTTiny
from .radar_tokenizer import RadarTokenizer
from .q_former import SpatialQFormer, TemporalQFormer


class RadarQFormer(nn.Module):
    """
    Complete Radar Q-Former: radar heatmaps → frame-query embeddings.

    This is the Student model that learns to align with HY-Motion's
    Text Encoder (Teacher) output Z_text.
    """

    def __init__(
        self,
        # Module 1: ViT-Tiny
        img_size=64,
        patch_size=8,
        vit_embed_dim=192,
        vit_depth=6,
        vit_num_heads=3,
        # Module 2: Tokenizer
        mask_ratio=0.15,
        # Module 3: Q-Former
        qformer_hidden_dim=768,
        num_spatial_queries=32,
        num_frame_queries=32,
        qformer_num_heads=8,
        num_layers_spatial=4,
        num_layers_temporal=4,
        max_frames=120,
        dropout=0.1,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.qformer_hidden_dim = qformer_hidden_dim
        self.num_frame_queries = num_frame_queries

        # Module 1: ViT-Tiny (3 independent encoders for RD, RA, DA)
        self.vit = TriViewViTTiny(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            dropout=dropout,
        )

        # Module 2: Radar Tokenizer (positional embeddings + view masking)
        grid_size = img_size // patch_size
        self.tokenizer = RadarTokenizer(
            embed_dim=vit_embed_dim,
            num_range_bins=grid_size,
            num_doppler_bins=grid_size,
            num_angle_bins=grid_size,
            mask_ratio=mask_ratio,
        )

        # Module 3a: Spatial Q-Former (per-frame aggregation)
        self.spatial_qformer = SpatialQFormer(
            hidden_dim=qformer_hidden_dim,
            input_dim=vit_embed_dim,
            num_queries=num_spatial_queries,
            num_heads=qformer_num_heads,
            num_layers=num_layers_spatial,
            dropout=dropout,
        )

        # Module 3b: Temporal Q-Former (sequence-level aggregation)
        self.temporal_qformer = TemporalQFormer(
            hidden_dim=qformer_hidden_dim,
            num_frame_queries=num_frame_queries,
            num_heads=qformer_num_heads,
            num_layers=num_layers_temporal,
            max_frames=max_frames,
            dropout=dropout,
        )

    def forward(self, rd, ra, da, return_spatial=False):
        """
        Full forward pass: radar heatmaps → Frame-Query embeddings.

        Args:
            rd: (B, T, H, W) Range-Doppler heatmap sequence
            ra: (B, T, H, W) Range-Angle heatmap sequence
            da: (B, T, H, W) Doppler-Angle heatmap sequence
            return_spatial: if True, also return per-frame spatial queries

        Returns:
            frame_queries: (B, num_frame_queries, qformer_hidden_dim)
                          = Z_Frame-Query aligned with Z_text
            spatial_queries: (optional) (B, T, num_spatial_queries, hidden_dim)
        """
        B, T, H, W = rd.shape

        # Flatten batch and time: (B*T, 1, H, W)
        rd_flat = rearrange(rd, "b t h w -> (b t) 1 h w")
        ra_flat = rearrange(ra, "b t h w -> (b t) 1 h w")
        da_flat = rearrange(da, "b t h w -> (b t) 1 h w")

        # Module 1: ViT-Tiny feature extraction
        feat_rd, feat_ra, feat_da = self.vit(rd_flat, ra_flat, da_flat)
        # Each: (B*T, N_patches, vit_embed_dim)

        # Module 2: Positional embedding fusion + view masking
        tok_rd, tok_ra, tok_da, mask_info = self.tokenizer(
            feat_rd, feat_ra, feat_da, training=self.training
        )

        # Module 3a: Spatial Q-Former (per-frame)
        spatial_out = self.spatial_qformer(tok_rd, tok_ra, tok_da)
        # (B*T, num_spatial_queries, qformer_hidden_dim)

        # Reshape for temporal: (B, T * num_spatial_queries, D)
        nq = spatial_out.shape[1]
        spatial_all = rearrange(spatial_out, "(b t) n d -> b (t n) d", b=B, t=T)

        # Module 3b: Temporal Q-Former (sequence-level)
        frame_queries = self.temporal_qformer(spatial_all, T)
        # (B, num_frame_queries, qformer_hidden_dim)

        if return_spatial:
            spatial_per_frame = rearrange(
                spatial_out, "(b t) n d -> b t n d", b=B, t=T
            )
            return frame_queries, spatial_per_frame

        return frame_queries

    def get_pooled_embedding(self, rd, ra, da):
        """
        Get a single pooled embedding vector for similarity comparison.
        Mean-pools the frame queries into a single vector.

        Returns:
            (B, qformer_hidden_dim)
        """
        frame_queries = self.forward(rd, ra, da)  # (B, Nq, D)
        return frame_queries.mean(dim=1)  # (B, D)
