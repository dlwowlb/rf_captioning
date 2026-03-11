"""
Vector Quantization layer for the Aggregate VQ-VAE.

Implements the codebook of size 512 x 512 that maps continuous feature vectors
to discrete tokens. Uses EMA updates for codebook learning with straight-through
gradient estimator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """Vector Quantization with EMA codebook updates.

    Maps continuous feature vectors to nearest codebook entries,
    producing discrete tokens for language model consumption.

    Args:
        codebook_size: Number of codebook entries (K=512)
        codebook_dim: Dimension of each code vector (D=512)
        commitment_cost: Weight for commitment loss (beta)
        ema_decay: Decay rate for EMA codebook updates
    """

    def __init__(self, codebook_size: int = 512, codebook_dim: int = 512,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size,
                         1.0 / codebook_size)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum",
                             self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
        """Quantize input features.

        Args:
            z: (B, M, D) continuous feature vectors (M = num_anchors or tokens)

        Returns:
            z_q: (B, M, D) quantized features (straight-through)
            loss: scalar VQ loss (codebook + commitment)
            token_ids: (B, M) discrete token indices
            perplexity: scalar codebook utilization metric
        """
        B, M, D = z.shape
        assert D == self.codebook_dim, \
            f"Input dim {D} != codebook dim {self.codebook_dim}"

        # Flatten for distance computation
        z_flat = z.reshape(-1, D)  # (B*M, D)

        # Compute distances to all codebook entries: ||z - e||^2
        # = ||z||^2 - 2*z*e^T + ||e||^2
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )  # (B*M, K)

        # Find nearest codebook entry
        token_ids = dist.argmin(dim=-1)  # (B*M,)
        z_q = self.embedding(token_ids)  # (B*M, D)

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                # One-hot encoding of assignments
                encodings = F.one_hot(token_ids, self.codebook_size).float()

                # Update cluster sizes
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    encodings.sum(0), alpha=1 - self.ema_decay
                )

                # Update embedding sums
                embed_sum = encodings.t() @ z_flat  # (K, D)
                self.ema_embed_sum.mul_(self.ema_decay).add_(
                    embed_sum, alpha=1 - self.ema_decay
                )

                # Laplace smoothing to avoid empty clusters
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.codebook_size * 1e-5) * n
                )

                # Update codebook
                self.embedding.weight.data.copy_(
                    self.ema_embed_sum / cluster_size.unsqueeze(1)
                )

        # VQ losses
        # Codebook loss: move codebook entries toward encoder output
        codebook_loss = F.mse_loss(z_q, z_flat.detach())
        # Commitment loss: encourage encoder to commit to codebook entries
        commitment_loss = F.mse_loss(z_flat, z_q.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_st = z_flat + (z_q - z_flat).detach()
        z_q_st = z_q_st.reshape(B, M, D)

        # Codebook utilization metric
        token_ids_reshaped = token_ids.reshape(B, M)
        avg_probs = F.one_hot(token_ids, self.codebook_size).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, loss, token_ids_reshaped, perplexity

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries by token IDs.

        Args:
            token_ids: (B, M) discrete token indices

        Returns:
            z_q: (B, M, D) codebook vectors
        """
        return self.embedding(token_ids)
