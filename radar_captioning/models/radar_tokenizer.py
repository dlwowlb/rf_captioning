"""
Module 2: Radar Tokenizer with Positional Embedding Fusion and View Masking.

Takes patch features from Module 1 (ViT-Tiny) and adds:
  - Shared dimension embeddings (Range is shared between RD and RA)
  - Unique dimension embeddings (Doppler, Angle, view type)
  - Random view masking during training

Token construction:
  Token_RD = Feature + PE_Range(r) + PE_Doppler(d) + PE_View(RD)
  Token_RA = Feature + PE_Range(r) + PE_Angle(a)  + PE_View(RA)
  Token_DA = Feature + PE_Doppler(d) + PE_Angle(a) + PE_View(DA)
"""

import torch
import torch.nn as nn


class RadarTokenizer(nn.Module):
    """
    Adds structured positional embeddings to radar features and
    optionally masks entire views during training.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_range_bins: int = 8,    # patch grid H
        num_doppler_bins: int = 8,  # patch grid W
        num_angle_bins: int = 8,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.num_angle_bins = num_angle_bins
        self.mask_ratio = mask_ratio

        # Shared positional embeddings for dimensions that appear in multiple views
        self.pe_range = nn.Embedding(num_range_bins, embed_dim)    # shared: RD, RA
        self.pe_doppler = nn.Embedding(num_doppler_bins, embed_dim)  # shared: RD, DA
        self.pe_angle = nn.Embedding(num_angle_bins, embed_dim)      # shared: RA, DA

        # View-type embeddings (which heatmap this token comes from)
        self.pe_view = nn.Embedding(3, embed_dim)  # 0=RD, 1=RA, 2=DA

        # [MASK] token for view masking
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def _get_grid_indices(self, num_patches, grid_h, grid_w):
        """Get row/col indices for a flattened patch grid."""
        rows = torch.arange(grid_h).unsqueeze(1).expand(grid_h, grid_w).reshape(-1)
        cols = torch.arange(grid_w).unsqueeze(0).expand(grid_h, grid_w).reshape(-1)
        return rows[:num_patches], cols[:num_patches]

    def forward(self, feat_rd, feat_ra, feat_da, training=True):
        """
        Add positional embeddings and optionally mask views.

        Args:
            feat_rd: (B, N, D) RD patch features from Module 1
            feat_ra: (B, N, D) RA patch features
            feat_da: (B, N, D) DA patch features
            training: whether to apply view masking

        Returns:
            tokens_rd: (B, N, D) tokenized RD features
            tokens_ra: (B, N, D) tokenized RA features
            tokens_da: (B, N, D) tokenized DA features
            mask_info: dict with masking information
        """
        B, N, D = feat_rd.shape
        device = feat_rd.device

        # Compute patch grid indices
        # RD: rows = range bins, cols = doppler bins
        rd_rows, rd_cols = self._get_grid_indices(N, self.num_range_bins, self.num_doppler_bins)
        rd_rows, rd_cols = rd_rows.to(device), rd_cols.to(device)

        # RA: rows = range bins, cols = angle bins
        ra_rows, ra_cols = self._get_grid_indices(N, self.num_range_bins, self.num_angle_bins)
        ra_rows, ra_cols = ra_rows.to(device), ra_cols.to(device)

        # DA: rows = doppler bins, cols = angle bins
        da_rows, da_cols = self._get_grid_indices(N, self.num_doppler_bins, self.num_angle_bins)
        da_rows, da_cols = da_rows.to(device), da_cols.to(device)

        # Token_RD = Feature + PE_Range(r) + PE_Doppler(d) + PE_View(RD)
        tokens_rd = (
            feat_rd
            + self.pe_range(rd_rows).unsqueeze(0).expand(B, -1, -1)
            + self.pe_doppler(rd_cols).unsqueeze(0).expand(B, -1, -1)
            + self.pe_view(torch.zeros(1, dtype=torch.long, device=device)).unsqueeze(0).expand(B, N, -1)
        )

        # Token_RA = Feature + PE_Range(r) + PE_Angle(a) + PE_View(RA)
        tokens_ra = (
            feat_ra
            + self.pe_range(ra_rows).unsqueeze(0).expand(B, -1, -1)
            + self.pe_angle(ra_cols).unsqueeze(0).expand(B, -1, -1)
            + self.pe_view(torch.ones(1, dtype=torch.long, device=device)).unsqueeze(0).expand(B, N, -1)
        )

        # Token_DA = Feature + PE_Doppler(d) + PE_Angle(a) + PE_View(DA)
        tokens_da = (
            feat_da
            + self.pe_doppler(da_rows).unsqueeze(0).expand(B, -1, -1)
            + self.pe_angle(da_cols).unsqueeze(0).expand(B, -1, -1)
            + self.pe_view(2 * torch.ones(1, dtype=torch.long, device=device)).unsqueeze(0).expand(B, N, -1)
        )

        # View masking during training: randomly mask entire views with [MASK] tokens
        mask_info = {"rd_masked": False, "ra_masked": False, "da_masked": False}
        if training and self.mask_ratio > 0:
            mask_token_expanded = self.mask_token.expand(B, N, -1)
            # Randomly mask each view independently
            if torch.rand(1).item() < self.mask_ratio:
                tokens_rd = mask_token_expanded.clone()
                mask_info["rd_masked"] = True
            if torch.rand(1).item() < self.mask_ratio:
                tokens_ra = mask_token_expanded.clone()
                mask_info["ra_masked"] = True
            if torch.rand(1).item() < self.mask_ratio:
                tokens_da = mask_token_expanded.clone()
                mask_info["da_masked"] = True

        return tokens_rd, tokens_ra, tokens_da, mask_info
