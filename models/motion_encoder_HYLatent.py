"""
Motion Encoder for RadarLLM — T2M-GPT / MotionGPT VQ-VAE Encoder.

Checkpoint key 구조 (실제 확인됨):
  vqvae.encoder.model.0.weight              (512, 263, 3)  — input conv
  vqvae.encoder.model.2.0.weight            (512, 512, 4)  — downsample conv 1
  vqvae.encoder.model.2.1.model.0.conv1.*   (512, 512, 3)  — ResBlock conv1
  vqvae.encoder.model.2.1.model.0.conv2.*   (512, 512, 1)  — ResBlock conv2
  vqvae.encoder.model.3.0.weight            (512, 512, 4)  — downsample conv 2
  vqvae.encoder.model.3.1.model.X.conv1/2.* — ResBlocks
  vqvae.encoder.model.4.weight              (512, 512, 3)  — output conv

따라서 ResConv1DBlock은 self.conv1 / self.conv2 명명을 사용해야 함.
(Sequential이 아닌 named attribute)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ═════════════════════════════════════════════════════════════
# T2M-GPT Exact Architecture (checkpoint key 호환)
# ═════════════════════════════════════════════════════════════

def get_activation(act_type: str = "relu") -> nn.Module:
    if act_type == "relu":
        return nn.ReLU()
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {act_type}")


class ResConv1DBlock(nn.Module):
    """
    T2M-GPT ResConv1DBlock — checkpoint key 호환.

    Checkpoint keys:
      *.conv1.weight, *.conv1.bias   (dilated conv)
      *.conv2.weight, *.conv2.bias   (1×1 conv)

    따라서 Sequential 대신 named attributes 사용:
      self.act1, self.conv1, self.act2, self.conv2
    """
    def __init__(self, n_in, n_hid, dilation=1, zero_out=False,
                 res_scale=1.0, activation="relu", norm=None):
        super().__init__()
        padding = dilation
        self.res_scale = res_scale

        self.act1 = get_activation(activation)
        self.conv1 = nn.Conv1d(n_in, n_hid, 3, 1, padding, dilation=dilation)
        self.act2 = get_activation(activation)
        self.conv2 = nn.Conv1d(n_hid, n_in, 1, 1, 0)

        # Optional GroupNorm
        self.norm1 = nn.GroupNorm(min(32, n_hid), n_hid) if norm == "group" else None
        self.norm2 = nn.GroupNorm(min(32, n_in), n_in) if norm == "group" else None

        if zero_out:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.act1(x)
        h = self.conv1(h)
        if self.norm1 is not None:
            h = self.norm1(h)
        h = self.act2(h)
        h = self.conv2(h)
        if self.norm2 is not None:
            h = self.norm2(h)
        return x + self.res_scale * h


class Resnet1D(nn.Module):
    """
    T2M-GPT Resnet1D — depth개 ResConv1DBlock 스택.
    reverse_dilation=True → dilations = [9, 3, 1] (depth=3, growth=3)

    Checkpoint keys: *.model.0.conv1/2, *.model.1.conv1/2, *.model.2.conv1/2
    """
    def __init__(self, n_in, n_depth, dilation_growth_rate=3,
                 reverse_dilation=True, activation="relu", norm=None):
        super().__init__()
        blocks = []
        for i in range(n_depth):
            dilation = dilation_growth_rate ** (n_depth - 1 - i if reverse_dilation else i)
            blocks.append(ResConv1DBlock(n_in, n_in, dilation=dilation,
                                         activation=activation, norm=norm))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class T2MGPTEncoder(nn.Module):
    """
    T2M-GPT VQ-VAE Encoder — checkpoint 구조 정확 매칭.

    Checkpoint key 구조:
      model.0: Conv1d(263→512, k=3, s=1, p=1)          — input
      model.1: ReLU                                      — (no params)
      model.2: Sequential(Conv1d(k=4,s=2,p=1), Resnet1D) — down 1
      model.3: Sequential(Conv1d(k=4,s=2,p=1), Resnet1D) — down 2
      model.4: Conv1d(512→512, k=3, s=1, p=1)           — output

    Input:  (B, C_in, T)
    Output: (B, C_out, T//4)
    """
    def __init__(self, input_emb_width=263, output_emb_width=512,
                 down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation="relu", norm=None):
        super().__init__()
        self.down_t = down_t
        self.stride_t = stride_t
        filter_t = stride_t * 2   # kernel=4
        pad_t = stride_t // 2     # pad=1

        blocks: List[nn.Module] = []
        # model.0: input conv
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        # model.1: activation
        blocks.append(get_activation(activation))
        # model.2, model.3: downsample stages
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate,
                         reverse_dilation=True, activation=activation, norm=norm),
            ))
        # model.4: output conv
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

    def get_output_length(self, T):
        L = T
        for _ in range(self.down_t):
            L = (L + 2 * (self.stride_t // 2) - self.stride_t * 2) // self.stride_t + 1
        return L


# ═════════════════════════════════════════════════════════════
# MotionEncoder Wrapper
# ═════════════════════════════════════════════════════════════

class MotionEncoder(nn.Module):
    """
    T2M-GPT encoder wrapper for RadarLLM L_emb.
    """
    def __init__(self, input_dim=263, feat_dim=512,
                 down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation="relu", norm=None,
                 # compat args (ignored)
                 num_heads=4, dropout=0.1, num_layers=2,
                 max_seq_len=1000, num_down=2):
        super().__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        actual_down_t = down_t if down_t != 2 else num_down

        self.encoder = T2MGPTEncoder(
            input_emb_width=input_dim, output_emb_width=feat_dim,
            down_t=actual_down_t, stride_t=stride_t, width=width,
            depth=depth, dilation_growth_rate=dilation_growth_rate,
            activation=activation, norm=norm,
        )
        self._input_adapter = None
        self._input_adapter_dim = None

    def forward(self, motion, mask=None, target_length=None):
        if motion.ndim == 4:
            B, T = motion.shape[:2]
            motion = motion.reshape(B, T, -1)

        actual_dim = motion.shape[-1]
        if actual_dim != self.input_dim:
            if self._input_adapter is None or self._input_adapter_dim != actual_dim:
                self._input_adapter_dim = actual_dim
                self._input_adapter = nn.Linear(actual_dim, self.input_dim).to(motion.device)
                print(f"[MotionEncoder] Created adapter: {actual_dim}D → {self.input_dim}D")
            motion = self._input_adapter(motion)

        x = motion.transpose(1, 2)       # (B, D, T)
        z = self.encoder(x)              # (B, feat_dim, L)
        F_mot = z.transpose(1, 2)        # (B, L, feat_dim)

        if target_length is not None and target_length > 0:
            if F_mot.shape[1] != target_length:
                F_mot = F_mot.transpose(1, 2)
                F_mot = F.adaptive_avg_pool1d(F_mot, target_length)
                F_mot = F_mot.transpose(1, 2)
        return F_mot

    def load_pretrained(self, ckpt_path, strict=False):
        """
        Checkpoint key mapping:
          vqvae.encoder.model.* → encoder.model.*
          net.encoder.model.*   → encoder.model.*
          vq_model.encoder.*    → encoder.*
        """
        if not os.path.exists(ckpt_path):
            print(f"[MotionEncoder] ✗ Not found: {ckpt_path}")
            return False

        print(f"[MotionEncoder] Loading pretrained from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Unwrap checkpoint
        if isinstance(ckpt, dict):
            for key in ["model_state_dict", "state_dict", "model", "net"]:
                if key in ckpt:
                    ckpt = ckpt[key]
                    break

        # ★ Prefix patterns (vqvae.encoder. 추가)
        prefix_patterns = [
            ("vqvae.encoder.", "encoder."),     # ★ 이 checkpoint 형식
            ("net.encoder.", "encoder."),        # T2M-GPT original
            ("vq_model.encoder.", "encoder."),   # MotionGPT
            ("motiongpt.vq_model.encoder.", "encoder."),
            ("encoder.", "encoder."),            # already encoder-only
        ]

        encoder_state = None
        used_prefix = None
        for src, dst in prefix_patterns:
            matched = {dst + k[len(src):]: v
                       for k, v in ckpt.items() if k.startswith(src)}
            if matched:
                encoder_state = matched
                used_prefix = src
                break

        if encoder_state is None:
            encoder_state = ckpt
            used_prefix = "(direct)"

        print(f"[MotionEncoder] Found {len(encoder_state)} params, prefix='{used_prefix}'")

        # Shape compatibility
        my_state = self.state_dict()
        compatible = 0
        shape_mismatch = []
        for k, v in list(encoder_state.items()):
            if k in my_state:
                if my_state[k].shape == v.shape:
                    compatible += 1
                else:
                    shape_mismatch.append(k)
                    if not strict:
                        del encoder_state[k]

        if shape_mismatch:
            print(f"[MotionEncoder] Shape mismatches: {len(shape_mismatch)}")
            for k in shape_mismatch[:5]:
                print(f"  {k}: expected {my_state[k].shape}")

        missing, unexpected = self.load_state_dict(encoder_state, strict=strict)
        if missing:
            print(f"[MotionEncoder] Missing: {len(missing)} keys")
            for k in missing[:5]:
                print(f"  {k}")
        if unexpected:
            print(f"[MotionEncoder] Unexpected: {len(unexpected)} keys")
            for k in unexpected[:5]:
                print(f"  {k}")
        print(f"[MotionEncoder] ✓ Loaded {compatible} params")
        return compatible > 0

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        print("[MotionEncoder] Frozen")

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_output_length(self, T):
        return self.encoder.get_output_length(T)


# ═════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════

def build_motion_encoder(config: dict) -> MotionEncoder:
    me_cfg = config.get("motion_encoder", {})
    tok_cfg = config.get("tokenizer", {})

    input_type = me_cfg.get("input_type", "humanml3d")
    if "input_dim" in me_cfg:
        input_dim = me_cfg["input_dim"]
    elif input_type == "humanml3d":
        input_dim = 263
    elif input_type == "latent":
        input_dim = 201
    elif input_type == "joints":
        input_dim = me_cfg.get("num_joints", 22) * 3
    elif input_type == "rot6d":
        input_dim = me_cfg.get("num_joints", 22) * 6
    else:
        input_dim = 263

    feat_dim = me_cfg.get("feat_dim", tok_cfg.get("codebook_dim", 512))

    encoder = MotionEncoder(
        input_dim=input_dim,
        feat_dim=feat_dim,
        down_t=me_cfg.get("down_t", me_cfg.get("num_down", 2)),
        stride_t=me_cfg.get("stride_t", 2),
        width=me_cfg.get("width", 512),
        depth=me_cfg.get("depth", 3),
        dilation_growth_rate=me_cfg.get("dilation_growth_rate", 3),
        activation=me_cfg.get("activation", "relu"),
        norm=me_cfg.get("norm", None),
    )

    # Pretrained loading
    pretrained_path = me_cfg.get("pretrained", None)
    if pretrained_path and os.path.exists(pretrained_path):
        success = encoder.load_pretrained(pretrained_path)
        if success and me_cfg.get("freeze", False):
            encoder.freeze()
    elif pretrained_path:
        print(f"[MotionEncoder] ⚠ Not found: {pretrained_path}, training from scratch")

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[MotionEncoder] T2M-GPT: {input_type}({input_dim}D) "
          f"→ {feat_dim}D (total: {total:,}, trainable: {trainable:,})")
    return encoder