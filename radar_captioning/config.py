"""
Configuration for Radar Captioning Pipeline.

Radar heatmap sizes follow TI AWR1843 config:
  - Range bins = ADC samples = 256
  - Doppler bins = chirps per frame = 128
  - Angle bins = 64

For toy experiments we use smaller 64x64 heatmaps.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RadarConfig:
    """Radar hardware parameters (TI AWR1843 defaults)."""
    heatmap_size: int = 64          # H=W for each heatmap
    num_range_bins: int = 64
    num_doppler_bins: int = 64
    num_angle_bins: int = 64
    frame_rate: int = 30            # radar frames per second
    max_range_m: float = 10.0       # max range in meters
    max_doppler_ms: float = 5.0     # max Doppler velocity m/s
    max_angle_deg: float = 60.0     # max angle in degrees


@dataclass
class ViTTinyConfig:
    """Module 1: ViT-Tiny per-heatmap feature extractor."""
    img_size: int = 64
    patch_size: int = 8             # 64/8 = 8x8 = 64 patches
    in_channels: int = 1            # single-channel heatmap
    embed_dim: int = 192            # ViT-Tiny standard
    depth: int = 6                  # 6 transformer layers (Tiny)
    num_heads: int = 3              # 192/3 = 64 dim per head
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TokenizerConfig:
    """Module 2: Positional embedding fusion + masking."""
    embed_dim: int = 192            # matches ViT-Tiny output
    num_range_bins: int = 8         # patch grid size
    num_doppler_bins: int = 8
    num_angle_bins: int = 8
    mask_ratio: float = 0.15        # probability of masking a view


@dataclass
class QFormerConfig:
    """Module 3: Q-Former for spatial and frame-level queries."""
    hidden_dim: int = 768           # Q-Former internal dim (matches CLIP)
    input_dim: int = 192            # from ViT-Tiny
    num_spatial_queries: int = 32   # per-frame spatial query tokens
    num_frame_queries: int = 32     # final frame-level query tokens
    num_heads: int = 8              # 768/8 = 96 dim per head
    num_layers_spatial: int = 4     # spatial Q-Former depth
    num_layers_temporal: int = 4    # temporal Q-Former depth
    dropout: float = 0.1


@dataclass
class Stage1Config:
    """Stage 1: Representation Alignment training."""
    z_text_dim: int = 768           # CLIP embedding dimension
    mse_weight: float = 1.0
    contrastive_weight: float = 1.0   # equal weight with MSE for effective alignment
    temperature: float = 0.07      # InfoNCE temperature
    lr: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    batch_size: int = 16
    warmup_steps: int = 100


@dataclass
class Stage2Config:
    """Stage 2: LLM sentence generation with Projector + LoRA."""
    llm_name: str = "Qwen/Qwen2.5-0.5B"  # small LLM for toy experiments
    llm_hidden_dim: int = 896       # Qwen2.5-0.5B hidden size
    projector_hidden_dim: int = 768  # projector intermediate
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr_projector: float = 1e-4
    lr_lora: float = 2e-5
    num_epochs: int = 30
    batch_size: int = 8
    max_gen_length: int = 256


@dataclass
class CurriculumConfig:
    """Coarse-to-fine Curriculum Learning."""
    # Phase 1: Basic actions (walk, sit, stand)
    basic_lr: float = 1e-3
    basic_epochs: int = 20
    # Phase 2: Composition actions (walk + wave, talk + jump)
    composition_lr: float = 5e-4
    composition_epochs: int = 20
    # Phase 3: Occlusion (70% occluded + 30% clean)
    occlusion_lr: float = 1e-4
    occlusion_epochs: int = 20
    occlusion_ratio: float = 0.7   # fraction of occluded data


@dataclass
class DataConfig:
    """Dataset configuration."""
    num_samples: int = 300          # toy dataset size
    num_frames: int = 30            # frames per sample (1 sec at 30fps)
    heatmap_size: int = 64
    train_ratio: float = 0.8
    # SMPL-X joints (compatible with SMPL-H layout)
    num_smplx_joints: int = 52      # 22 body + 15 left hand + 15 right hand
    num_smplh_joints: int = 52      # backward-compatible alias
    num_body_joints: int = 22
    num_hand_joints: int = 15


@dataclass
class FullConfig:
    """Full pipeline config."""
    radar: RadarConfig = field(default_factory=RadarConfig)
    vit: ViTTinyConfig = field(default_factory=ViTTinyConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    qformer: QFormerConfig = field(default_factory=QFormerConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "output/radar_captioning"
