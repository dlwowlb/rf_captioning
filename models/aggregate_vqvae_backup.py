"""
Aggregate VQ-VAE for RadarLLM.

FIX 1: L_rec uses center-normalized GT (same coordinate space as encoder)
FIX 2: anchor_pool → attention pooling (replaces Linear(32768, 512))

Training losses:
  L_VQ = L_rec + L_emb + L_commit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from .point_encoder import RadarPointCloudEncoder


# ─────────────────────────────────────────────────────────
# Masked Context Aggregation
# ─────────────────────────────────────────────────────────

class MaskedContextAggregation(nn.Module):
    def __init__(self, feat_dim=512, num_anchors=64, num_layers=4,
                 num_heads=8, mask_ratio=0.5, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_anchors = num_anchors
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, feat_dim) * 0.02)
        self.anchor_pos_embed = nn.Parameter(torch.randn(1, 1, num_anchors, feat_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=num_heads,
            dim_feedforward=feat_dim * 4, dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.vis_proj = nn.Linear(feat_dim, feat_dim)

    def _generate_mask(self, B, L, Ng, device):
        num_mask = int(Ng * self.mask_ratio)
        mask = torch.zeros(B, Ng, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(Ng, device=device)[:num_mask]
            mask[b, idx] = True
        return mask

    def forward(self, F_group, training=True):
        B, L, Ng, C = F_group.shape
        device = F_group.device

        F_group = F_group + self.anchor_pos_embed[:, :, :Ng, :]

        if not training:
            return F_group, torch.zeros(B, Ng, dtype=torch.bool, device=device), F_group

        mask = self._generate_mask(B, L, Ng, device)
        vis_mask = ~mask

        num_vis = vis_mask.sum(dim=1)
        max_vis = num_vis.max().item()

        F_vis_list = []
        for b in range(B):
            vis_idx = torch.nonzero(vis_mask[b], as_tuple=True)[0]
            f_vis = F_group[b, :, vis_idx, :]
            if vis_idx.shape[0] < max_vis:
                pad = torch.zeros(L, max_vis - vis_idx.shape[0], C, device=device)
                f_vis = torch.cat([f_vis, pad], dim=1)
            F_vis_list.append(f_vis)
        F_vis = torch.stack(F_vis_list, dim=0)

        memory = self.vis_proj(F_vis.reshape(B * L, max_vis, C))

        mask_tokens = self.mask_token.expand(B, L, Ng, C)
        F_query = F_group.clone()
        mask_3d = mask.unsqueeze(1).unsqueeze(-1).expand(B, L, Ng, C)
        F_query = torch.where(mask_3d, mask_tokens, F_query)
        F_query_flat = F_query.reshape(B * L, Ng, C)

        F_msk = self.decoder(F_query_flat, memory)
        F_all = F_msk.reshape(B, L, Ng, C)

        return F_all, mask, F_vis


# ─────────────────────────────────────────────────────────
# ★ FIX 2: Attention-based Anchor Pooling
# ─────────────────────────────────────────────────────────

class AnchorAttentionPool(nn.Module):
    """
    Ng개 anchor feature를 attention으로 하나의 벡터로 집약.

    기존: Linear(Ng*C=32768, 512) → 1680만 파라미터
    변경: Attention query(학습 가능) → O(Ng*C) = O(32K) 파라미터

    learnable query가 Ng개 anchor에 attention하여
    가장 중요한 anchor 정보를 집약.
    """

    def __init__(self, feat_dim: int, num_anchors: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        # 학습 가능한 query token (1개)
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, batch_first=True
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, out_dim),
        )

    def forward(self, F_all: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_all: (B, L, Ng, C) anchor features
        Returns:
            z_e: (B, L, out_dim)
        """
        B, L, Ng, C = F_all.shape

        # (B*L, Ng, C) as key/value, (B*L, 1, C) as query
        kv = F_all.reshape(B * L, Ng, C)
        q = self.query.expand(B * L, -1, -1)                  # (B*L, 1, C)

        attn_out, _ = self.attn(q, kv, kv)                    # (B*L, 1, C)
        attn_out = attn_out.squeeze(1)                         # (B*L, C)

        z_e = self.proj(attn_out)                              # (B*L, out_dim)
        return z_e.reshape(B, L, -1)                           # (B, L, out_dim)


# ─────────────────────────────────────────────────────────
# Vector Quantizer
# ─────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=512, codebook_dim=512, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e):
        B, L, D = z_e.shape
        flat = z_e.reshape(-1, D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
                - 2 * flat @ self.codebook.weight.t())
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices).reshape(B, L, D)
        commit_loss = (F.mse_loss(z_e.detach(), z_q)
                       + self.commitment_cost * F.mse_loss(z_e, z_q.detach()))
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices.reshape(B, L), commit_loss


# ─────────────────────────────────────────────────────────
# Point Cloud Decoder (for L_rec)
# ─────────────────────────────────────────────────────────

class PointCloudDecoder(nn.Module):
    def __init__(self, input_dim=512, num_points=128, point_dim=3):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_points * point_dim),
        )

    def forward(self, z):
        B, L, D = z.shape
        return self.decoder(z).reshape(B, L, self.num_points, self.point_dim)


# ─────────────────────────────────────────────────────────
# Chamfer Distance
# ─────────────────────────────────────────────────────────

def chamfer_distance(pred, target):
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-1)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


# ─────────────────────────────────────────────────────────
# Full Aggregate VQ-VAE
# ─────────────────────────────────────────────────────────

class AggregateVQVAE(nn.Module):
    """
    Motion-Guided Radar Tokenizer.

    FIX 1: L_rec uses center-normalized GT from encoder
    FIX 2: anchor pooling → attention pooling (compact parameters)
    """

    def __init__(self, config: dict):
        super().__init__()
        tok_cfg = config.get("tokenizer", config)

        self.encoder = RadarPointCloudEncoder(tok_cfg)
        feat_dim = self.encoder.out_dim
        num_anchors = self.encoder.num_anchors
        codebook_dim = tok_cfg.get("codebook_dim", 512)

        self.masked_agg = MaskedContextAggregation(
            feat_dim=feat_dim, num_anchors=num_anchors,
            num_layers=tok_cfg.get("decoder_num_layers", 4),
            num_heads=tok_cfg.get("decoder_num_heads", 8),
            mask_ratio=tok_cfg.get("mask_ratio", 0.5),
        )

        # ★ FIX 2: attention pooling (기존 Linear(32768,512) 대체)
        self.anchor_pool = AnchorAttentionPool(
            feat_dim=feat_dim, num_anchors=num_anchors, out_dim=codebook_dim,
        )

        self.quantizer = VectorQuantizer(
            codebook_size=tok_cfg.get("codebook_size", 512),
            codebook_dim=codebook_dim,
            commitment_cost=tok_cfg.get("commitment_cost", 0.25),
        )

        self.pc_decoder = PointCloudDecoder(
            input_dim=codebook_dim,
            num_points=tok_cfg.get("points_per_frame", 128)
            if "points_per_frame" in tok_cfg
            else config.get("dataset", {}).get("points_per_frame", 128),
        )

        self.num_anchors = num_anchors
        self.feat_dim = feat_dim

    def encode(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out = self.encoder(point_cloud)
        F_group = enc_out["F_group"]
        F_all, _, _ = self.masked_agg(F_group, training=False)
        z_e = self.anchor_pool(F_all)
        z_q, indices, _ = self.quantizer(z_e)
        return indices, z_q

    def forward(
        self,
        point_cloud: torch.Tensor,
        motion_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # ── Encode (returns centered point cloud too) ──
        enc_out = self.encoder(point_cloud)
        F_group = enc_out["F_group"]                           # (B, L, Ng, C)
        xyz_centered = enc_out["xyz_centered"]                 # (B, T, N, 3)
        B, L, Ng, C = F_group.shape

        # ── Masked context aggregation ──
        F_all, mask, F_vis = self.masked_agg(F_group, training=self.training)

        # ── Anchor pooling (attention) ──
        z_e = self.anchor_pool(F_all)                          # (B, L, D)

        # ── Vector quantization ──
        z_q, indices, loss_commit = self.quantizer(z_e)

        # ── ★ FIX 1: L_rec — centered 좌표계 GT 사용 ──
        reconstructed = self.pc_decoder(z_q)                   # (B, L, N, 3) centered space

        T = xyz_centered.shape[1]
        stride = max(1, T // L)
        # ★ GT도 center-normalized된 xyz_centered 사용 (encoder와 같은 좌표계)
        gt_frames = xyz_centered[:, ::stride, :, :][:, :L]     # (B, L, N, 3) centered space

        loss_rec = chamfer_distance(
            reconstructed.reshape(-1, reconstructed.shape[2], 3),
            gt_frames.reshape(-1, gt_frames.shape[2], 3),
        )

# ── L_emb ──
        loss_emb = torch.tensor(0.0, device=point_cloud.device)
        if motion_features is not None:
            F_mot = motion_features
            if F_mot.shape[1] != L:
                F_mot = F_mot.transpose(1, 2)
                F_mot = F.adaptive_avg_pool1d(F_mot, L)
                F_mot = F_mot.transpose(1, 2)
            loss_emb = F.mse_loss(z_e, F_mot)

        # ── Total ──
        loss = loss_rec + loss_emb + loss_commit

        return {
            "indices": indices,
            "z_q": z_q,
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_emb": loss_emb,
            "loss_commit": loss_commit,
        }