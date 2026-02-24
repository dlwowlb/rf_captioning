"""
Module 1: ViT-Tiny Feature Extractor for Radar Heatmaps.

Extracts spatial features from each 2D radar heatmap (RD, RA, DA) independently.
Uses ViT-Tiny architecture (patch_size=8, embed_dim=192, depth=6, heads=3).

Input:  (B*T, 1, H, W) single-channel heatmap
Output: (B*T, N_patches, embed_dim) patch features
        where N_patches = (H/patch_size) * (W/patch_size) = 64 for H=W=64, patch=8
"""

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Convert 2D heatmap to patch embeddings via convolution."""

    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input heatmap
        Returns:
            (B, N_patches, embed_dim)
        """
        x = self.proj(x)                    # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, N_patches, embed_dim)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """
    ViT-Tiny feature extractor for a single radar heatmap view.

    Architecture:
      PatchEmbed -> [CLS] + Patches + PosEmbed -> N x TransformerBlock -> output patches
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=1,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable positional embedding (no CLS token; we use all patches)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) single-channel heatmap
        Returns:
            (B, N_patches, embed_dim) patch features
        """
        x = self.patch_embed(x)          # (B, N_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # (B, N_patches, embed_dim)


class TriViewViTTiny(nn.Module):
    """
    Three independent ViT-Tiny encoders for RD, RA, DA heatmaps.

    Each view gets its own ViT-Tiny with shared architecture but separate weights,
    allowing each to specialize in its view's characteristics.
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        kwargs = dict(
            img_size=img_size, patch_size=patch_size, in_channels=1,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.vit_rd = ViTTiny(**kwargs)
        self.vit_ra = ViTTiny(**kwargs)
        self.vit_da = ViTTiny(**kwargs)

    def forward(self, rd, ra, da):
        """
        Args:
            rd: (B*T, 1, H, W) Range-Doppler heatmap
            ra: (B*T, 1, H, W) Range-Angle heatmap
            da: (B*T, 1, H, W) Doppler-Angle heatmap
        Returns:
            feat_rd: (B*T, N, D) RD patch features
            feat_ra: (B*T, N, D) RA patch features
            feat_da: (B*T, N, D) DA patch features
        """
        feat_rd = self.vit_rd(rd)
        feat_ra = self.vit_ra(ra)
        feat_da = self.vit_da(da)
        return feat_rd, feat_ra, feat_da
