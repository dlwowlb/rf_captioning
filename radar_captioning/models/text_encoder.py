"""
Text Encoder: Teacher Model for Stage 1 Alignment.

In the full pipeline, this uses HY-Motion's frozen text encoder (Qwen3 + CLIP).
For toy experiments without HY-Motion weights, we provide:
  1. SurrogateTextEncoder: Deterministic hash-based encoder that produces
     diverse embeddings guaranteed to separate different action types.
  2. (Future) CLIPTextEncoder: wraps HY-Motion's real text encoder.

The key output is Z_text: a 768-dimensional embedding that captures the
semantic content of the motion description.
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# Known action keywords for semantic hashing
ACTION_KEYWORDS = [
    "walk", "forward", "backward", "sit", "stand", "still",
    "wave", "phone", "jump", "stretch", "run", "turn",
    "slow", "fast", "close", "far", "left", "right",
]


class SurrogateTextEncoder(nn.Module):
    """
    Deterministic text encoder for toy experiments.

    Produces diverse embeddings using:
    1. Keyword detection → one-hot action features
    2. Numerical extraction → physical parameter features
    3. Deterministic hash → unique text fingerprint
    4. Linear projection → embed_dim output

    This guarantees that:
    - Different actions produce well-separated embeddings
    - Same text always produces the same embedding (deterministic)
    - Physical parameters (distance, speed, angle) are captured
    """

    def __init__(self, embed_dim: int = 768, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature dimensions:
        # - keyword features: len(ACTION_KEYWORDS) = 18
        # - physical params: 3 (normalized distance, speed, angle)
        # - hash features: 32 (SHA-256 = 32 bytes)
        feat_dim = len(ACTION_KEYWORDS) + 3 + 32

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Initialize with fixed random weights (deterministic)
        torch.manual_seed(12345)
        for p in self.proj.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p)
            else:
                nn.init.normal_(p, std=0.01)

        # Freeze projection to ensure determinism
        for p in self.proj.parameters():
            p.requires_grad = False

    def _extract_features(self, text: str) -> torch.Tensor:
        """Extract deterministic features from text."""
        text_lower = text.lower()
        features = []

        # Keyword features (binary)
        for kw in ACTION_KEYWORDS:
            features.append(1.0 if kw in text_lower else 0.0)

        # Extract numerical values (distance, speed, angle)
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', text_lower)
        nums = [float(n) for n in numbers[:3]]
        while len(nums) < 3:
            nums.append(0.0)
        # Normalize roughly
        features.append(nums[0] / 10.0)   # distance (assume max ~10m)
        features.append(nums[1] / 5.0)    # speed (assume max ~5 m/s)
        features.append(nums[2] / 90.0)   # angle (assume max ~90 deg)

        # Deterministic hash features (32-dim, SHA-256 = 32 bytes)
        h = hashlib.sha256(text_lower.encode()).digest()
        hash_features = [(b / 255.0 - 0.5) * 2.0 for b in h[:32]]
        features.extend(hash_features)

        return torch.tensor(features, dtype=torch.float32)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text strings into Z_text embeddings.

        Args:
            texts: list of B text strings
        Returns:
            z_text: (B, embed_dim) text embedding (L2-normalized)
        """
        device = next(self.proj.parameters()).device
        features = torch.stack([self._extract_features(t) for t in texts]).to(device)
        z = self.proj(features)
        # L2 normalize to unit sphere
        z = F.normalize(z, dim=-1)
        return z
