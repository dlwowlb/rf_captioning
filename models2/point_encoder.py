"""
Point cloud encoder for RadarLLM.

centering 없이 world 좌표 그대로 사용.
데이터 범위에 맞는 큰 고정 grid로 anchor 배치.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TemplateAnchorGrouping(nn.Module):
    def __init__(self, grid_size=(4, 4, 4),
                 bbox_min=(-0.6, -0.4, -1.0),    # z가 height
                 bbox_max=(0.6, 0.4, 1.0),
                 radius=0.5, max_neighbors=32):
        super().__init__()
        self.grid_size = grid_size
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.num_anchors = grid_size[0] * grid_size[1] * grid_size[2]

        gx = torch.linspace(bbox_min[0], bbox_max[0], grid_size[0])
        gy = torch.linspace(bbox_min[1], bbox_max[1], grid_size[1])
        gz = torch.linspace(bbox_min[2], bbox_max[2], grid_size[2])
        grid = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), dim=-1)
        self.register_buffer("anchors", grid.reshape(-1, 3))

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, _ = xyz.shape
        Ng = self.num_anchors
        K = min(self.max_neighbors, N)

        anchors = self.anchors[None, None].expand(B, T, -1, -1)
        diff = anchors.unsqueeze(3) - xyz.unsqueeze(2)
        dist = torch.norm(diff, dim=-1)

        dist_masked = dist.clone()
        dist_masked[dist >= self.radius] = float("inf")

        topk_dist, topk_idx = torch.topk(dist_masked, K, dim=-1, largest=False)
        topk_idx = topk_idx.clamp(0, N - 1)

        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        xyz_exp = xyz.unsqueeze(2).expand(-1, -1, Ng, -1, -1)
        grouped_xyz = torch.gather(xyz_exp, 3, idx_exp) - anchors.unsqueeze(3)
        grouped_mask = topk_dist < self.radius

        if K < self.max_neighbors:
            grouped_xyz = F.pad(grouped_xyz, (0, 0, 0, self.max_neighbors - K), value=0.)
            grouped_mask = F.pad(grouped_mask, (0, self.max_neighbors - K), value=False)

        return grouped_xyz, grouped_mask


class PointConv4D(nn.Module):
    def __init__(self, in_dim=3, hidden_dims=None, out_dim=512,
                 temporal_stride=2, token_unit=2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        self.temporal_stride = temporal_stride
        self.token_unit = token_unit

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        self.spatial_mlp = nn.Sequential(*layers)
        self.temporal_conv = nn.Conv1d(prev, out_dim, kernel_size=token_unit,
                                       stride=temporal_stride, padding=0)
        self.out_dim = out_dim

    def forward(self, grouped_xyz, grouped_mask):
        B, T, Ng, K, D = grouped_xyz.shape
        x = self.spatial_mlp(grouped_xyz)
        H = x.shape[-1]

        mask_exp = grouped_mask.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(~mask_exp, float("-inf"))
        x = x.max(dim=3).values
        all_empty = ~grouped_mask.any(dim=3)
        x[all_empty] = 0.0

        x = x.permute(0, 2, 3, 1).reshape(B * Ng, H, T)
        x = self.temporal_conv(x)
        L = x.shape[2]
        return x.reshape(B, Ng, self.out_dim, L).permute(0, 3, 1, 2)


class RadarPointCloudEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        grid_size = tuple(config.get("grid_size", [4, 4, 4]))
        channels = config.get("encoder_channels", [64, 128, 256, 512])
        self.temporal_stride = config.get("temporal_stride", 2)
        self.token_unit = config.get("token_unit", 2)

        self.grouping = TemplateAnchorGrouping(
            grid_size=grid_size,
            radius=config.get("neighborhood_radius", 1.5),
        )
        self.p4conv = PointConv4D(
            in_dim=3, hidden_dims=channels[:-1], out_dim=channels[-1],
            temporal_stride=self.temporal_stride, token_unit=self.token_unit,
        )
        self.num_anchors = self.grouping.num_anchors
        self.out_dim = channels[-1]

    def get_output_length(self, T: int) -> int:
        return (T - self.token_unit) // self.temporal_stride + 1

    def forward(self, point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        xyz = point_cloud[..., :3]  # (B, T, N, 3)

        # ── per-frame centering ──
        valid = (xyz.norm(dim=-1) > 1e-6)
        valid_count = valid.float().sum(dim=-1, keepdim=True).clamp(min=1)
        masked_xyz = xyz * valid.unsqueeze(-1).float()
        centroid = masked_xyz.sum(dim=2) / valid_count       # (B, T, 3)
        xyz_centered = xyz - centroid.unsqueeze(2)            # (B, T, N, 3)

        grouped_xyz, grouped_mask = self.grouping(xyz_centered)  # centered 좌표 입력
        F_group = self.p4conv(grouped_xyz, grouped_mask)

        return {
            "F_group": F_group,
            "xyz_centered": xyz_centered,
        }