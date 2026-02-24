"""
Module 3: Q-Former for Spatial and Temporal Aggregation.

Two-stage Q-Former architecture:
  1. Spatial Q-Former: Concatenate RD+RA+DA tokens as K/V,
     use 32 learnable Spatial-Query tokens as Q.
     Output: per-frame spatial summary (32 tokens per frame).

  2. Temporal Q-Former: Add frame-level PE to spatial queries,
     concatenate across frames as K/V, use 32 Frame-Query tokens as Q.
     Output: final Frame-Query tokens for the entire sequence.
"""

import torch
import torch.nn as nn
from einops import rearrange


class CrossAttentionBlock(nn.Module):
    """Cross-attention block: Q attends to K/V from a different source."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, kv):
        """
        Args:
            query: (B, Nq, D) query tokens
            kv: (B, Nkv, D) key-value source
        Returns:
            (B, Nq, D) updated query tokens
        """
        q = self.norm_q(query)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, kv_normed, kv_normed)
        query = query + attn_out
        query = query + self.ffn(self.norm_ff(query))
        return query


class SelfAttentionBlock(nn.Module):
    """Self-attention block for query refinement."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class QFormerLayer(nn.Module):
    """Single Q-Former layer: cross-attention + self-attention."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(dim, num_heads, dropout)
        self.self_attn = SelfAttentionBlock(dim, num_heads, dropout)

    def forward(self, query, kv):
        query = self.cross_attn(query, kv)
        query = self.self_attn(query)
        return query


class SpatialQFormer(nn.Module):
    """
    Spatial Q-Former: aggregates per-frame radar tokens into spatial queries.

    Input: concatenated RD+RA+DA tokens for one frame (3*N_patches tokens)
    Output: 32 spatial query tokens per frame
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        input_dim: int = 192,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries

        # Project from ViT-Tiny dim to Q-Former dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable spatial query tokens
        self.spatial_queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )

        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens_rd, tokens_ra, tokens_da):
        """
        Args:
            tokens_rd: (B, N, D_in) RD tokens (D_in = ViT-Tiny embed_dim = 192)
            tokens_ra: (B, N, D_in) RA tokens
            tokens_da: (B, N, D_in) DA tokens
        Returns:
            spatial_out: (B, num_queries, hidden_dim) spatial query output
        """
        B = tokens_rd.shape[0]

        # Concatenate all three views: (B, 3*N, D_in)
        concat_tokens = torch.cat([tokens_rd, tokens_ra, tokens_da], dim=1)

        # Project to Q-Former dimension
        kv = self.input_proj(concat_tokens)  # (B, 3*N, hidden_dim)

        # Initialize queries
        queries = self.spatial_queries.expand(B, -1, -1)  # (B, num_queries, hidden_dim)

        # Apply Q-Former layers
        for layer in self.layers:
            queries = layer(queries, kv)

        queries = self.norm(queries)
        return queries  # (B, num_queries, hidden_dim)


class TemporalQFormer(nn.Module):
    """
    Temporal Q-Former: aggregates per-frame spatial queries into frame-level queries.

    Input: spatial queries from all frames (T * num_spatial_queries tokens)
    Output: 32 frame-query tokens summarizing the entire sequence
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_frame_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        max_frames: int = 120,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frame_queries = num_frame_queries

        # Frame-level positional embedding
        self.frame_pe = nn.Embedding(max_frames, hidden_dim)

        # Learnable frame query tokens
        self.frame_queries = nn.Parameter(
            torch.randn(1, num_frame_queries, hidden_dim) * 0.02
        )

        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, spatial_queries_all_frames, num_frames):
        """
        Args:
            spatial_queries_all_frames: (B, T * Nq_spatial, D) spatial queries from all frames
            num_frames: int, number of frames T
        Returns:
            frame_queries: (B, num_frame_queries, D)
        """
        B, total_tokens, D = spatial_queries_all_frames.shape
        nq_per_frame = total_tokens // num_frames
        device = spatial_queries_all_frames.device

        # Add frame-level positional embedding
        # Each frame's spatial queries get the same frame PE
        frame_indices = torch.arange(num_frames, device=device)
        frame_pe = self.frame_pe(frame_indices)  # (T, D)
        # Repeat PE for each spatial query within a frame
        frame_pe = frame_pe.unsqueeze(1).expand(-1, nq_per_frame, -1)  # (T, Nq, D)
        frame_pe = frame_pe.reshape(1, total_tokens, D).expand(B, -1, -1)

        kv = spatial_queries_all_frames + frame_pe

        # Initialize frame queries
        queries = self.frame_queries.expand(B, -1, -1)  # (B, num_frame_queries, D)

        # Apply Q-Former layers
        for layer in self.layers:
            queries = layer(queries, kv)

        queries = self.norm(queries)
        return queries  # (B, num_frame_queries, D)
