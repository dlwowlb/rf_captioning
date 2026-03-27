"""
Motion Encoder for Contrastive Alignment (L_rm).

HumanML3D 263D → temporal encoder → per-frame features → g_motion.
T2M-GPT VQ-VAE encoder를 기반으로 하되, 학습 가능 여부는 config에서 설정.

사용:
  motion_enc = build_motion_encoder(config)
  F_mot = motion_enc(motion_padded)   # (B, T, 263) → (B, T', feat_dim)
  g_motion = F_mot.mean(dim=1)        # (B, feat_dim)
"""

import torch
import torch.nn as nn

class MotionEncoder(nn.Module):
    def __init__(self, input_dim=201, feat_dim=512, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim))

    def forward(self, x):
        # x: (B, T, 201) → (B, T, 512)
        return self.proj(x)


'''
class MotionEncoder(nn.Module):
    """
    1D Conv temporal encoder.
    HumanML3D 263D input → downsampled feature sequence.
    """

    def __init__(self, input_dim=263, feat_dim=512, num_down=2,
                 stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width), nn.GELU())

        # Temporal convolution blocks with dilation
        layers = []
        for i in range(depth):
            dilation = dilation_growth_rate ** i
            layers.append(nn.Conv1d(
                width, width, kernel_size=3,
                padding=dilation, dilation=dilation))
            layers.append(nn.GELU())

        # Downsampling
        for _ in range(num_down):
            layers.append(nn.Conv1d(width, width, kernel_size=stride_t * 2,
                                     stride=stride_t, padding=stride_t // 2))
            layers.append(nn.GELU())

        self.temporal = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(width, feat_dim)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        Returns: (B, T', feat_dim)
        """
        h = self.input_proj(x)              # (B, T, width)
        h = self.temporal(h.transpose(1, 2))  # (B, width, T')
        h = h.transpose(1, 2)                # (B, T', width)
        return self.output_proj(h)            # (B, T', feat_dim)
'''

def build_motion_encoder(config):
    """
    Config에서 motion encoder를 생성.
    Pretrained checkpoint가 있으면 로드.
    """
    me_cfg = config.get("motion_encoder", {})

    encoder = MotionEncoder(
        input_dim=me_cfg.get("input_dim", 263),
        feat_dim=me_cfg.get("feat_dim", 512),
        num_down=me_cfg.get("num_down", 2),
        stride_t=me_cfg.get("stride_t", 2),
        width=me_cfg.get("width", 512),
        depth=me_cfg.get("depth", 3),
        dilation_growth_rate=me_cfg.get("dilation_growth_rate", 3),
    )

    # Load pretrained weights if available
    pretrained = me_cfg.get("pretrained")
    if pretrained:
        import os
        if os.path.exists(pretrained):
            ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
            # T2M-GPT checkpoint: extract encoder weights
            if "net" in ckpt:
                state = {k.replace("encoder.", ""): v
                         for k, v in ckpt["net"].items()
                         if k.startswith("encoder.")}
                encoder.load_state_dict(state, strict=False)
                print(f"[MotionEncoder] Loaded pretrained: {pretrained}")
            else:
                encoder.load_state_dict(ckpt, strict=False)
                print(f"[MotionEncoder] Loaded: {pretrained}")

    # Freeze if configured
    if me_cfg.get("freeze", False):
        for p in encoder.parameters():
            p.requires_grad = False
        print("[MotionEncoder] Frozen")

    return encoder