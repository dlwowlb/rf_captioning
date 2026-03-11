"""
Dataset for RadarLLM training.

Loads radar point cloud sequences, paired SMPL motion data, and text captions
for both tokenizer (Stage 1) and language model (Stage 2) training.

Expected data directory structure:
    data_root/
    ├── radar/                      # Radar point cloud sequences
    │   ├── sample_000.npy          # (T, N, D) arrays
    │   ├── sample_001.npy
    │   └── ...
    ├── motion/                     # SMPL motion parameters
    │   ├── sample_000.npz          # {pose: (T, 72), shape: (10,), root_translation: (T, 3)}
    │   ├── sample_001.npz
    │   └── ...
    ├── captions/                   # Text captions (one per sample)
    │   ├── sample_000.txt
    │   ├── sample_001.txt
    │   └── ...
    └── splits/
        ├── train.txt               # Sample IDs for training
        ├── val.txt                 # Sample IDs for validation
        └── test.txt                # Sample IDs for testing
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


class RadarCaptionDataset(Dataset):
    """Dataset for radar point cloud captioning.

    Each sample contains:
    - Radar point cloud sequence: (T, N, D)
    - SMPL motion parameters: (T, 72)  [optional, for Stage 1]
    - Text caption: str  [for Stage 2]
    """

    def __init__(self, data_root: str, split: str = "train",
                 seq_len: int = 16, num_points: int = 64,
                 point_dim: int = 5, motion_dim: int = 72,
                 load_motion: bool = True, load_captions: bool = True):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.seq_len = seq_len
        self.num_points = num_points
        self.point_dim = point_dim
        self.motion_dim = motion_dim
        self.load_motion = load_motion
        self.load_captions = load_captions

        # Load sample IDs
        split_file = os.path.join(data_root, "splits", f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover from radar directory
            radar_dir = os.path.join(data_root, "radar")
            if os.path.exists(radar_dir):
                self.sample_ids = sorted([
                    os.path.splitext(f)[0]
                    for f in os.listdir(radar_dir) if f.endswith(".npy")
                ])
            else:
                self.sample_ids = []

        print(f"[{split}] Loaded {len(self.sample_ids)} samples from {data_root}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _pad_or_crop_temporal(self, data: np.ndarray,
                              target_len: int) -> np.ndarray:
        """Pad or crop temporal dimension to target length."""
        T = data.shape[0]
        if T >= target_len:
            # Uniformly sample frames
            indices = np.linspace(0, T - 1, target_len, dtype=int)
            return data[indices]
        else:
            # Repeat-pad
            pad_len = target_len - T
            padding = np.repeat(data[-1:], pad_len, axis=0)
            return np.concatenate([data, padding], axis=0)

    def _pad_or_sample_points(self, points: np.ndarray,
                               target_n: int) -> np.ndarray:
        """Pad or subsample points to target count."""
        N = points.shape[0]
        if N >= target_n:
            indices = np.random.choice(N, target_n, replace=False)
            return points[indices]
        elif N > 0:
            # Pad by repeating with noise
            pad_n = target_n - N
            pad_indices = np.random.choice(N, pad_n, replace=True)
            padded = points[pad_indices]
            noise = np.random.randn(*padded.shape) * 0.01
            return np.concatenate([points, padded + noise], axis=0)
        else:
            return np.zeros((target_n, points.shape[-1]), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        result = {}

        # Load radar point cloud sequence
        radar_path = os.path.join(self.data_root, "radar", f"{sample_id}.npy")
        if os.path.exists(radar_path):
            radar_data = np.load(radar_path)  # Expected: (T, N, D)
            if radar_data.ndim == 2:
                # Single frame: (N, D) -> (1, N, D)
                radar_data = radar_data[np.newaxis]

            # Normalize temporal
            radar_data = self._pad_or_crop_temporal(radar_data, self.seq_len)

            # Normalize spatial (per-frame)
            processed_frames = []
            for t in range(self.seq_len):
                frame = radar_data[t]  # (N', D)
                # Ensure point dimension
                if frame.shape[-1] < self.point_dim:
                    pad = np.zeros((frame.shape[0],
                                    self.point_dim - frame.shape[-1]))
                    frame = np.concatenate([frame, pad], axis=-1)
                elif frame.shape[-1] > self.point_dim:
                    frame = frame[:, :self.point_dim]
                frame = self._pad_or_sample_points(frame, self.num_points)
                processed_frames.append(frame)

            radar_data = np.stack(processed_frames, axis=0)  # (T, N, D)
        else:
            # Generate dummy data for testing
            radar_data = np.random.randn(
                self.seq_len, self.num_points, self.point_dim
            ).astype(np.float32)

        result["point_clouds"] = torch.from_numpy(radar_data).float()

        # Load motion data (SMPL parameters)
        if self.load_motion:
            motion_path = os.path.join(self.data_root, "motion",
                                       f"{sample_id}.npz")
            if os.path.exists(motion_path):
                motion_data = np.load(motion_path)
                pose = motion_data["pose"]  # (T', 72)
                pose = self._pad_or_crop_temporal(pose, self.seq_len)
            else:
                pose = np.zeros((self.seq_len, self.motion_dim),
                                dtype=np.float32)
            result["motion_seq"] = torch.from_numpy(pose).float()

        # Load text caption
        if self.load_captions:
            caption_path = os.path.join(self.data_root, "captions",
                                        f"{sample_id}.txt")
            if os.path.exists(caption_path):
                with open(caption_path, "r") as f:
                    caption = f.read().strip()
            else:
                caption = "A person performs an action."
            result["caption"] = caption

        result["sample_id"] = sample_id
        return result


class DummyRadarDataset(Dataset):
    """Dummy dataset for testing and debugging.

    Generates random radar point clouds, motion sequences, and fixed captions.
    Useful for verifying the training pipeline before real data is available.
    """

    def __init__(self, num_samples: int = 1000, seq_len: int = 16,
                 num_points: int = 64, point_dim: int = 5,
                 motion_dim: int = 72):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_points = num_points
        self.point_dim = point_dim
        self.motion_dim = motion_dim

        self.captions = [
            "A person is walking forward slowly.",
            "A person raises both arms above their head.",
            "A person is sitting down on a chair.",
            "A person waves their right hand.",
            "A person is jogging in place.",
            "A person bends down to pick something up.",
            "A person is standing still.",
            "A person turns around in a circle.",
            "A person claps their hands together.",
            "A person stretches their arms out to the sides.",
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate random point cloud centered around a body-like distribution
        # Rough body shape: x in [-0.3, 0.3], y in [0, 1.8], z in [-0.2, 0.2]
        base_points = torch.randn(self.num_points, 3) * 0.3
        base_points[:, 1] = base_points[:, 1].abs() + 0.5  # height

        point_clouds = []
        for t in range(self.seq_len):
            # Add small temporal variation (motion)
            motion_noise = torch.randn(self.num_points, 3) * 0.05
            xyz = base_points + motion_noise

            # Add velocity and intensity channels
            velocity = torch.randn(self.num_points, 1) * 0.5
            intensity = torch.rand(self.num_points, 1)

            if self.point_dim == 5:
                frame = torch.cat([xyz, velocity, intensity], dim=-1)
            else:
                extra = torch.randn(self.num_points,
                                    self.point_dim - 3) * 0.1
                frame = torch.cat([xyz, extra], dim=-1)
            point_clouds.append(frame)

        point_clouds = torch.stack(point_clouds, dim=0)  # (T, N, D)

        # Random SMPL-like motion parameters
        motion_seq = torch.randn(self.seq_len, self.motion_dim) * 0.1

        caption = self.captions[idx % len(self.captions)]

        return {
            "point_clouds": point_clouds,
            "motion_seq": motion_seq,
            "caption": caption,
            "sample_id": f"dummy_{idx:06d}",
        }


def build_dataloader(data_root: str, split: str, batch_size: int,
                     num_workers: int = 4, seq_len: int = 16,
                     num_points: int = 64, point_dim: int = 5,
                     load_motion: bool = True, load_captions: bool = True,
                     use_dummy: bool = False, dummy_size: int = 1000
                     ) -> DataLoader:
    """Build DataLoader for training/evaluation.

    Args:
        data_root: path to data directory
        split: "train", "val", or "test"
        batch_size: batch size
        num_workers: dataloader workers
        use_dummy: use dummy dataset for debugging

    Returns:
        DataLoader instance
    """
    if use_dummy:
        dataset = DummyRadarDataset(
            num_samples=dummy_size,
            seq_len=seq_len,
            num_points=num_points,
            point_dim=point_dim,
        )
    else:
        dataset = RadarCaptionDataset(
            data_root=data_root,
            split=split,
            seq_len=seq_len,
            num_points=num_points,
            point_dim=point_dim,
            load_motion=load_motion,
            load_captions=load_captions,
        )

    # Custom collate to handle string captions
    def collate_fn(batch):
        result = {}
        result["point_clouds"] = torch.stack(
            [b["point_clouds"] for b in batch])
        if "motion_seq" in batch[0]:
            result["motion_seq"] = torch.stack(
                [b["motion_seq"] for b in batch])
        if "caption" in batch[0]:
            result["caption"] = [b["caption"] for b in batch]
        result["sample_id"] = [b["sample_id"] for b in batch]
        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=collate_fn,
    )
