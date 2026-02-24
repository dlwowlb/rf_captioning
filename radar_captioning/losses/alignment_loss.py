"""
Stage 1 Loss Functions: MSE + Contrastive (InfoNCE) Loss.

Aligns Radar Q-Former output (Z_Frame-Query) with HY-Motion text encoder
output (Z_text) in a shared embedding space.

MSE Loss: directly minimizes distance between paired embeddings.
Contrastive Loss: pushes matched pairs together, unmatched pairs apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    Combined MSE + InfoNCE contrastive loss for representation alignment.

    L = α * L_MSE + β * L_Contrastive
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()

    def forward(self, z_radar, z_text):
        """
        Compute alignment loss.

        Args:
            z_radar: (B, D) pooled radar embedding (from Q-Former)
            z_text: (B, D) text embedding (from HY-Motion text encoder, frozen)

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        # L2 normalize both embeddings to prevent collapse and operate
        # in cosine similarity space. MSE on normalized vectors = 2(1-cos_sim).
        z_radar_norm = F.normalize(z_radar, dim=-1)
        z_text_norm = F.normalize(z_text.detach(), dim=-1)

        # MSE Loss on normalized embeddings (equivalent to cosine distance)
        loss_mse = self.mse_loss(z_radar_norm, z_text_norm)

        # Contrastive Loss (symmetric InfoNCE)
        loss_contrastive = self._info_nce_loss(z_radar, z_text.detach())

        total_loss = self.mse_weight * loss_mse + self.contrastive_weight * loss_contrastive

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_contrastive": loss_contrastive.item(),
        }

    def _info_nce_loss(self, z_radar, z_text):
        """
        Symmetric InfoNCE contrastive loss.

        For each radar embedding, the matching text embedding is the positive,
        and all other text embeddings in the batch are negatives (and vice versa).
        """
        B = z_radar.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=z_radar.device)

        # L2 normalize
        z_radar_norm = F.normalize(z_radar, dim=-1)
        z_text_norm = F.normalize(z_text, dim=-1)

        # Cosine similarity matrix: (B, B)
        logits = torch.mm(z_radar_norm, z_text_norm.t()) / self.temperature

        # Labels: diagonal is positive (matching pairs)
        labels = torch.arange(B, device=z_radar.device)

        # Symmetric cross-entropy
        loss_r2t = F.cross_entropy(logits, labels)  # radar → text
        loss_t2r = F.cross_entropy(logits.t(), labels)  # text → radar

        return (loss_r2t + loss_t2r) / 2


class TokenAlignmentLoss(nn.Module):
    """
    Token-level alignment loss for Q-Former frame queries.

    Instead of pooling to a single vector, aligns the full set of
    frame query tokens with the text embedding via MSE + contrastive.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(self, frame_queries, z_text):
        """
        Args:
            frame_queries: (B, Nq, D) frame query tokens from Q-Former
            z_text: (B, D) text embedding target

        Returns:
            total_loss, loss_dict
        """
        B, Nq, D = frame_queries.shape

        # MSE: compare mean-pooled queries with text embedding
        z_radar_pooled = frame_queries.mean(dim=1)  # (B, D)
        loss_mse = F.mse_loss(z_radar_pooled, z_text)

        # Contrastive on pooled embeddings
        loss_contrastive = self._symmetric_info_nce(z_radar_pooled, z_text)

        total = self.mse_weight * loss_mse + self.contrastive_weight * loss_contrastive

        return total, {
            "loss_total": total.item(),
            "loss_mse": loss_mse.item(),
            "loss_contrastive": loss_contrastive.item(),
        }

    def _symmetric_info_nce(self, z1, z2):
        B = z1.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=z1.device)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(B, device=z1.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
