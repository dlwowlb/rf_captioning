"""
Motion Encoder for RadarLLM L_emb.

HY-Motion latent(201D) 위에 가벼운 Transformer Encoder를 얹어서
프레임 간 temporal 관계까지 학습.

기존 Linear만:
  frame 1: latent(201) → Linear → F_mot[1]  ← 독립
  frame 2: latent(201) → Linear → F_mot[2]  ← 독립
  → 각 프레임이 서로 모름

Transformer 추가:
  [frame 1, frame 2, ..., frame T] → Linear → Transformer
  → F_mot[1]은 주변 프레임 context를 봄
  → "걷다가 멈추는" 같은 temporal 패턴 포착 가능

별도 사전학습 불필요 — VQ-VAE와 함께 학습됨.
HY-Motion latent가 이미 rich하므로 Transformer는 2 layer로 충분.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MotionEncoder(nn.Module):
    """
    HY-Motion latent (201D) → Projection → Transformer → F_mot (512D)

    VQ-VAE와 함께 학습됨 (별도 사전학습 불필요).
    """

    def __init__(
        self,
        input_dim: int = 201,
        feat_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # Input projection: 201D → 512D
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, feat_dim) * 0.02
        )

        # Transformer encoder (lightweight: 2 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 2,   # 작게: 4x → 2x
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(feat_dim, feat_dim)

    def forward(
        self,
        motion: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            motion: (B, T, 201) HY-Motion latent
                    또는 (B, T, J, D) joints/rot6d → 자동 flatten
            mask: (B, T) temporal mask
            target_length: VQ-VAE 출력 토큰 수 L_radar

        Returns:
            F_mot: (B, target_length, 512)
        """
        # Flatten if (B, T, J, D) → (B, T, J*D)
        if motion.ndim == 4:
            B, T = motion.shape[:2]
            motion = motion.reshape(B, T, -1)
        else:
            B, T = motion.shape[:2]

        # Projection: (B, T, input_dim) → (B, T, 512)
        x = self.input_proj(motion)

        # Positional encoding
        x = x + self.pos_encoding[:, :T, :]

        # Transformer: temporal dependency 학습
        padding_mask = (mask == 0) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Output projection
        x = self.output_proj(x)                                # (B, T, 512)

        # Temporal 정렬: 모션 프레임 → VQ-VAE 토큰 수
        if target_length is not None and target_length > 0:
            x = x.transpose(1, 2)                              # (B, 512, T)
            x = F.adaptive_avg_pool1d(x, target_length)        # (B, 512, L)
            x = x.transpose(1, 2)                              # (B, L, 512)

        return x


def build_motion_encoder(config: dict) -> MotionEncoder:
    """Config에서 MotionEncoder 생성."""
    me_cfg = config.get("motion_encoder", {})
    tok_cfg = config.get("tokenizer", {})

    input_type = me_cfg.get("input_type", "latent")

    # 입력 차원
    if "input_dim" in me_cfg:
        input_dim = me_cfg["input_dim"]
    elif input_type == "latent":
        input_dim = 201
    elif input_type == "joints":
        input_dim = me_cfg.get("num_joints", 22) * 3
    elif input_type == "rot6d":
        input_dim = me_cfg.get("num_joints", 22) * 6
    else:
        input_dim = 201

    feat_dim = me_cfg.get("feat_dim", tok_cfg.get("codebook_dim", 512))
    num_layers = me_cfg.get("num_layers", 2)
    num_heads = me_cfg.get("num_heads", 4)

    encoder = MotionEncoder(
        input_dim=input_dim,
        feat_dim=feat_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    params = sum(p.numel() for p in encoder.parameters())
    print(f"[MotionEncoder] {input_type}: {input_dim}D → Transformer({num_layers}L) → {feat_dim}D "
          f"(params: {params:,})")
    return encoder
