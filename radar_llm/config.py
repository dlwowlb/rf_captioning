"""Configuration for RadarLLM training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenizerConfig:
    """Configuration for Motion-guided Radar Tokenizer (Aggregate VQ-VAE)."""

    # Point cloud input
    num_points: int = 64          # number of points per frame (sparse radar)
    point_dim: int = 5            # (x, y, z, velocity, intensity)
    seq_len: int = 16             # temporal sequence length (frames)

    # Template-prior Grouping
    num_anchors: int = 16         # number of template anchors (body parts)

    # P4Conv Encoder
    encoder_dim: int = 128        # intermediate encoder dimension
    encoder_out_dim: int = 512    # final encoder output dimension

    # Masked Context Aggregation (Decoder)
    mask_ratio: float = 0.5       # 50% anchor trajectory masking
    decoder_heads: int = 8        # cross-attention heads
    decoder_layers: int = 4       # number of transformer decoder layers
    decoder_dim: int = 512        # decoder hidden dimension

    # Vector Quantization codebook
    codebook_size: int = 512      # number of codes
    codebook_dim: int = 512       # dimension of each code vector
    commitment_cost: float = 0.25 # VQ commitment loss weight

    # Motion Encoder (auxiliary)
    motion_input_dim: int = 72    # SMPL pose parameters (24 joints * 3)
    motion_hidden_dim: int = 256
    motion_out_dim: int = 512     # must match codebook_dim

    # Training
    recon_weight: float = 1.0
    vq_weight: float = 1.0
    motion_align_weight: float = 0.5  # motion-radar alignment loss


@dataclass
class LanguageModelConfig:
    """Configuration for Radar-aware Language Model."""

    # Shared embeddings
    embed_dim: int = 512                         # shared embedding dimension
    radar_vocab_size: int = 512                  # = codebook_size
    text_model_name: str = "google/flan-t5-small"  # FLAN-T5-small backbone

    # Training
    max_text_len: int = 128       # max caption token length
    max_radar_tokens: int = 16    # max number of radar tokens per sample
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    """Shared training configuration."""

    # Data
    data_root: str = "./data"
    batch_size: int = 32
    num_workers: int = 4

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_epochs: int = 100
    grad_clip: float = 1.0

    # Logging
    log_every: int = 50
    eval_every: int = 1           # evaluate every N epochs
    save_every: int = 5           # save checkpoint every N epochs
    output_dir: str = "./outputs"

    # Device
    device: str = "cuda"
    seed: int = 42

    # Stage selection
    stage: int = 1                # 1=tokenizer, 2=language model

    # Checkpoint
    tokenizer_ckpt: Optional[str] = None  # path to trained tokenizer (for stage 2)
