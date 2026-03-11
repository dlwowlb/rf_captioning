"""
P4Conv (Point 4D Convolution) Encoder for spatio-temporal radar point cloud features.

Implements the P4Transformer-style convolution that processes 4D (x,y,z,t)
point cloud sequences to extract spatial and temporal features.
Reference: Fan et al., "Point 4D Transformer Networks for Spatio-Temporal Modeling
in Point Cloud Videos", CVPR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Find k-nearest neighbors.

    Args:
        x: (B, N, C) point features
        k: number of neighbors

    Returns:
        idx: (B, N, k) neighbor indices
    """
    # x: (B, N, C)
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
    xx = torch.sum(x ** 2, dim=-1, keepdim=True)      # (B, N, 1)
    dist = xx + inner + xx.transpose(2, 1)             # (B, N, N)
    _, idx = dist.topk(k=k, dim=-1, largest=False)     # (B, N, k)
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C) or (B, T, N, C)
        idx: (B, M) or (B, M, k)

    Returns:
        gathered: (B, M, C) or (B, M, k, C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    expand_shape = list(idx.shape)

    batch_idx = torch.arange(B, dtype=torch.long, device=points.device)
    batch_idx = batch_idx.view(view_shape).expand(expand_shape)

    if points.dim() == 3:
        return points[batch_idx, idx, :]
    elif points.dim() == 4:
        # (B, T, N, C) -> flatten T*N, index, reshape
        T, N, C = points.shape[1], points.shape[2], points.shape[3]
        points_flat = points.reshape(B, T * N, C)
        return points_flat[batch_idx, idx, :]
    return points[batch_idx, idx]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling.

    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of points to sample

    Returns:
        idx: (B, npoint) sampled point indices
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)          # (B, N)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)

    return centroids


class PointConv4D(nn.Module):
    """4D Point Convolution block: spatial kNN conv + temporal conv.

    Processes point cloud sequences (B, T, N, C) by:
    1. Spatial kNN grouping and MLP per frame
    2. Temporal 1D convolution across frames
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 16,
                 temporal_kernel: int = 3):
        super().__init__()
        self.k = k

        # Spatial branch: process kNN neighborhoods
        self.spatial_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal branch: 1D conv along time axis
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, T, N, 3) point coordinates
            features: (B, T, N, C_in) point features

        Returns:
            out: (B, T, N, C_out) updated features
        """
        B, T, N, C = features.shape

        # --- Spatial: per-frame kNN convolution ---
        all_frame_feats = []
        for t in range(T):
            xyz_t = xyz[:, t]        # (B, N, 3)
            feat_t = features[:, t]  # (B, N, C)

            # kNN in spatial coordinates
            idx = knn(xyz_t, self.k)  # (B, N, k)
            neighbors = index_points(feat_t, idx)  # (B, N, k, C)

            # Relative features: concat center with neighbor diff
            center = feat_t.unsqueeze(2).expand_as(neighbors)  # (B, N, k, C)
            edge = torch.cat([center, neighbors - center], dim=-1)  # (B, N, k, 2C)

            # MLP on edges
            edge_flat = edge.reshape(B * N * self.k, -1)
            edge_flat = self.spatial_mlp(edge_flat)
            edge = edge_flat.reshape(B, N, self.k, -1)

            # Max-pool over neighbors
            frame_feat = edge.max(dim=2)[0]  # (B, N, C_out)
            all_frame_feats.append(frame_feat)

        spatial_out = torch.stack(all_frame_feats, dim=1)  # (B, T, N, C_out)
        C_out = spatial_out.shape[-1]

        # --- Temporal: 1D conv across time per point ---
        # Reshape to (B*N, C_out, T) for Conv1d
        temporal_in = spatial_out.permute(0, 2, 3, 1).reshape(B * N, C_out, T)
        temporal_out = self.temporal_conv(temporal_in)  # (B*N, C_out, T)
        temporal_out = temporal_out.reshape(B, N, C_out, T).permute(0, 3, 1, 2)
        # (B, T, N, C_out)

        out = self.out_proj(spatial_out + temporal_out)
        return out


class P4ConvEncoder(nn.Module):
    """P4Conv-based encoder for radar point cloud sequences.

    Takes a sequence of radar point clouds and produces per-anchor
    spatio-temporal feature vectors.

    Architecture:
        Input: (B, T, N, D_in) where D_in = point_dim
        -> Embedding MLP
        -> P4Conv Block 1
        -> P4Conv Block 2
        -> P4Conv Block 3
        -> Anchor aggregation (template-based grouping)
        -> Output: (B, num_anchors, C_out)
    """

    def __init__(self, point_dim: int = 5, encoder_dim: int = 128,
                 encoder_out_dim: int = 512, num_anchors: int = 16,
                 seq_len: int = 16, k: int = 16):
        super().__init__()
        self.num_anchors = num_anchors
        self.seq_len = seq_len
        self.encoder_out_dim = encoder_out_dim

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(point_dim, encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, encoder_dim),
        )

        # P4Conv blocks with increasing capacity
        self.p4conv1 = PointConv4D(encoder_dim, encoder_dim, k=k)
        self.p4conv2 = PointConv4D(encoder_dim, encoder_dim * 2, k=k)
        self.p4conv3 = PointConv4D(encoder_dim * 2, encoder_out_dim, k=k)

        # Learnable anchor templates (body-part prior positions)
        # Initialized as a rough human body layout in normalized coordinates
        self.anchor_templates = nn.Parameter(
            torch.randn(num_anchors, 3) * 0.3
        )

        # Anchor feature aggregation: maps grouped point features to anchor features
        self.anchor_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_out_dim, encoder_out_dim),
        )

    def group_by_anchors(self, xyz: torch.Tensor,
                         features: torch.Tensor) -> torch.Tensor:
        """Template-prior grouping: assign points to nearest anchor.

        Args:
            xyz: (B, T, N, 3) point coordinates
            features: (B, T, N, C) point features

        Returns:
            anchor_feats: (B, num_anchors, C) aggregated anchor features
        """
        B, T, N, C = features.shape

        # Average point positions across time for stable grouping
        xyz_mean = xyz.mean(dim=1)  # (B, N, 3)

        # Compute distance from each point to each anchor
        anchors = self.anchor_templates.unsqueeze(0).expand(B, -1, -1)  # (B, A, 3)
        # (B, N, 1, 3) - (B, 1, A, 3) -> (B, N, A)
        dist = torch.sum(
            (xyz_mean.unsqueeze(2) - anchors.unsqueeze(1)) ** 2, dim=-1
        )

        # Soft assignment via softmax (differentiable)
        weights = F.softmax(-dist / 0.1, dim=-1)  # (B, N, A)

        # Average features across time
        feat_mean = features.mean(dim=1)  # (B, N, C)

        # Weighted aggregation: (B, A, N) @ (B, N, C) -> (B, A, C)
        anchor_feats = torch.bmm(weights.permute(0, 2, 1), feat_mean)

        return self.anchor_proj(anchor_feats)

    def forward(self, point_clouds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_clouds: (B, T, N, D) raw radar point cloud sequence
                D = point_dim (x, y, z, velocity, intensity)

        Returns:
            anchor_features: (B, num_anchors, encoder_out_dim) per-anchor features
            point_features: (B, T, N, encoder_out_dim) per-point features (for recon)
        """
        B, T, N, D = point_clouds.shape
        xyz = point_clouds[..., :3]  # (B, T, N, 3)

        # Embed input features
        feat = self.input_embed(point_clouds)  # (B, T, N, encoder_dim)

        # P4Conv feature extraction
        feat = self.p4conv1(xyz, feat)   # (B, T, N, encoder_dim)
        feat = self.p4conv2(xyz, feat)   # (B, T, N, encoder_dim*2)
        feat = self.p4conv3(xyz, feat)   # (B, T, N, encoder_out_dim)

        # Template-prior grouping
        anchor_features = self.group_by_anchors(xyz, feat)  # (B, A, C)

        return anchor_features, feat
