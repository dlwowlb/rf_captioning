#!/usr/bin/env python3
"""
RadarLLM Data Generation Pipeline
==================================

End-to-end pipeline:
  Text Prompt → HY-Motion → SMPL motion → RF-Genesis radar simulation
  → Point Cloud extraction → Training data (NPY/NPZ/TXT)

This script generates training data by:
1. Taking text prompts describing human actions
2. Generating SMPL body motion via HY-Motion
3. Simulating mmWave radar signals via RF-Genesis
4. Extracting radar point clouds from signal frames
5. Saving everything in RadarLLM training format

Output directory structure:
    output_dir/
    ├── radar/          # (T, N, 6) point cloud sequences per sample
    ├── motion/         # SMPL params {pose, shape, root_translation}
    ├── captions/       # Text descriptions
    └── splits/         # train.txt, val.txt
"""

import os
import sys
import argparse
import numpy as np
import json
import time
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HY_MOTION_DIR = PROJECT_ROOT / "HY-Motion-1.0"
RF_GENESIS_DIR = PROJECT_ROOT / "RF-Genesis"

if HY_MOTION_DIR.exists():
    sys.path.insert(0, str(HY_MOTION_DIR))
if RF_GENESIS_DIR.exists():
    sys.path.insert(0, str(RF_GENESIS_DIR))


# ===========================================================================
# Text prompts for training data generation
# ===========================================================================
DEFAULT_PROMPTS = [
    # Walking
    "a person walks forward slowly",
    "a person walks forward quickly",
    "a person walks backward",
    "a person walks in a circle",
    "a person walks to the left",
    "a person walks to the right",
    "a person walks diagonally",
    "a person walks forward then stops",

    # Running/Jogging
    "a person jogs in place",
    "a person runs forward",
    "a person runs and then stops suddenly",

    # Arm movements
    "a person raises both arms above their head",
    "a person waves their right hand",
    "a person waves their left hand",
    "a person waves both hands",
    "a person claps their hands together",
    "a person stretches their arms out to the sides",
    "a person puts their hands on their hips",
    "a person reaches forward with both arms",

    # Full body
    "a person jumps in place",
    "a person jumps forward",
    "a person does a jumping jack",
    "a person bends down to pick something up",
    "a person squats down and stands back up",
    "a person sits down on a chair",
    "a person stands up from a sitting position",
    "a person turns around",
    "a person spins in a circle",
    "a person stretches their body",

    # Specific actions
    "a person kicks with their right leg",
    "a person kicks with their left leg",
    "a person bows forward",
    "a person leans to the right",
    "a person leans to the left",
    "a person steps to the side",
    "a person marches in place",
    "a person dances",
    "a person punches forward with their right hand",
    "a person throws an object",

    # Standing/Idle
    "a person stands still",
    "a person stands and looks around",
    "a person shifts their weight from one foot to the other",
    "a person crosses their arms",
    "a person scratches their head",

    # Complex actions
    "a person walks forward and waves",
    "a person walks forward and turns around",
    "a person picks something up and puts it down",
    "a person walks in a zigzag pattern",
    "a person stumbles while walking",
]


# ===========================================================================
# Motion conversion (from motion_to_doppler.py)
# ===========================================================================
def _rot6d_to_rotation_matrix_np(rot6d):
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def _rotation_matrix_to_axis_angle_np(rot_mat):
    from scipy.spatial.transform import Rotation
    orig_shape = rot_mat.shape[:-2]
    flat = rot_mat.reshape(-1, 3, 3)
    r = Rotation.from_matrix(flat)
    aa = r.as_rotvec()
    return aa.reshape(*orig_shape, 3)


def convert_hymotion_to_smpl(model_output):
    """Convert HY-Motion output to SMPL-24 format.

    Returns: (pose_params: (T, 72), translation: (T, 3), shape: (10,))
    """
    import torch

    if isinstance(model_output, dict) and 'rot6d' in model_output:
        rot6d = model_output['rot6d']
        transl = model_output['transl']
        if isinstance(rot6d, torch.Tensor):
            rot6d = rot6d.cpu().numpy()
        if isinstance(transl, torch.Tensor):
            transl = transl.cpu().numpy()
        if rot6d.ndim == 4:
            rot6d = rot6d[0]
        if transl.ndim == 3:
            transl = transl[0]

        body_rot6d = rot6d[:, :22, :]
        rot_matrices = _rot6d_to_rotation_matrix_np(body_rot6d)
        body_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)
        hand_aa = np.zeros((rot6d.shape[0], 2, 3), dtype=body_aa.dtype)
        smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)
        pose_params = smpl_aa.reshape(rot6d.shape[0], -1)
        return pose_params, transl, np.zeros(10)

    elif isinstance(model_output, dict) and 'latent_denorm' in model_output:
        smpl_data = model_output['latent_denorm']
        if isinstance(smpl_data, torch.Tensor):
            smpl_data = smpl_data.cpu().numpy()
        if smpl_data.ndim == 3:
            smpl_data = smpl_data[0]
        num_frames = smpl_data.shape[0]
        translation = smpl_data[:, :3]
        global_orient_6d = smpl_data[:, 3:9].reshape(num_frames, 1, 6)
        body_pose_6d = smpl_data[:, 9:135].reshape(num_frames, 21, 6)
        all_rot6d = np.concatenate([global_orient_6d, body_pose_6d], axis=1)
        rot_matrices = _rot6d_to_rotation_matrix_np(all_rot6d)
        body_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)
        hand_aa = np.zeros((num_frames, 2, 3), dtype=body_aa.dtype)
        smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)
        pose_params = smpl_aa.reshape(num_frames, -1)
        return pose_params, translation, np.zeros(10)

    else:
        raise ValueError("Unsupported HY-Motion output format")


# ===========================================================================
# Point cloud extraction from radar frames
# ===========================================================================
def radar_frames_to_pointclouds(radar_frames, radar_config_path, num_points=64):
    """Extract point clouds from raw radar signal frames.

    Uses RF-Genesis's pointcloud processing pipeline:
    radar_frames → Range FFT → Clutter removal → Doppler FFT → AoA → (x,y,z,v,snr,range)

    Args:
        radar_frames: (num_frames, num_tx, num_rx, chirps, adc_samples) complex
        radar_config_path: path to TI1843_config.json
        num_points: target number of points per frame

    Returns:
        point_clouds: (num_frames, num_points, 6) array [x,y,z,velocity,SNR,range]
    """
    from genesis.raytracing.radar import Radar
    from genesis.visualization.pointcloud import (
        PointCloudProcessCFG, frame2pointcloud, reg_data
    )

    radar = Radar(radar_config_path)
    pc_cfg = PointCloudProcessCFG(radar)

    all_pcs = []
    num_frames = radar_frames.shape[0]

    for i in range(num_frames):
        frame = radar_frames[i]  # (num_tx, num_rx, chirps, adc_samples)

        # Convert to complex if not already
        if not np.iscomplexobj(frame):
            frame = frame.astype(np.complex128)

        # frame2pointcloud expects (num_tx, num_rx, chirps, adc_samples)
        pc = frame2pointcloud(frame, pc_cfg)  # (6, N) or (6, 0)

        if pc.shape[1] == 0:
            # No detections - use zeros
            pc_reg = np.zeros((num_points, 6), dtype=np.float32)
        else:
            pc = np.transpose(pc, (1, 0))  # (N, 6)
            # Filter out static objects (velocity > 0)
            pc = pc[pc[:, 3] > 0]
            if len(pc) == 0:
                pc_reg = np.zeros((num_points, 6), dtype=np.float32)
            else:
                pc_reg = reg_data(pc, num_points)  # (num_points, 6)

        all_pcs.append(pc_reg)

    return np.stack(all_pcs, axis=0)  # (num_frames, num_points, 6)


# ===========================================================================
# Single sample generation
# ===========================================================================
def generate_single_sample(
    prompt: str,
    sample_id: str,
    output_dir: str,
    hy_motion_runtime=None,
    duration: float = 3.0,
    cfg_scale: float = 5.0,
    seed: int = 42,
    num_points: int = 64,
    radar_config_path: str = None,
    skip_visualization: bool = True,
):
    """Generate one training sample: motion + radar simulation + point clouds.

    Args:
        prompt: text description of the action
        sample_id: unique ID for this sample
        output_dir: root output directory
        hy_motion_runtime: pre-loaded T2MRuntime instance
        duration: motion duration in seconds
        num_points: points per frame for point cloud
    """
    import torch

    os.makedirs(os.path.join(output_dir, "radar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "motion"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "captions"), exist_ok=True)

    if radar_config_path is None:
        radar_config_path = str(RF_GENESIS_DIR / "models" / "TI1843_config.json")

    # Check if already generated
    radar_path = os.path.join(output_dir, "radar", f"{sample_id}.npy")
    if os.path.exists(radar_path):
        print(f"  [{sample_id}] Already exists, skipping.")
        return True

    # --- Step 1: Generate motion ---
    print(f"  [{sample_id}] Generating motion for: '{prompt}'")
    try:
        _, _, model_output = hy_motion_runtime.generate_motion(
            text=prompt,
            seeds_csv=str(seed),
            duration=duration,
            cfg_scale=cfg_scale,
            output_format="dict",
        )
    except Exception as e:
        print(f"  [{sample_id}] Motion generation failed: {e}")
        return False

    # Convert to SMPL format
    try:
        pose_params, translation, shape = convert_hymotion_to_smpl(model_output)
    except Exception as e:
        print(f"  [{sample_id}] Motion conversion failed: {e}")
        return False

    # Save motion NPZ (temporary for RF-Genesis)
    tmp_motion_path = os.path.join(output_dir, "_tmp_motion.npz")
    np.savez(tmp_motion_path,
             pose=pose_params, shape=shape,
             root_translation=translation, gender="neutral")

    # Also save final motion data
    motion_out_path = os.path.join(output_dir, "motion", f"{sample_id}.npz")
    np.savez(motion_out_path,
             pose=pose_params, shape=shape,
             root_translation=translation, gender="neutral")

    # --- Step 2: RF-Genesis radar simulation ---
    print(f"  [{sample_id}] Running radar simulation...")
    try:
        from genesis.raytracing import pathtracer, signal_generator

        # Calculate sensor position
        traj_center = translation.mean(axis=0)
        sensor_origin = [
            float(traj_center[0]),
            float(traj_center[1] + 1.0),
            float(traj_center[2] + 3.0),
        ]
        sensor_target = [
            float(traj_center[0]),
            float(traj_center[1] + 1.0),
            float(traj_center[2]),
        ]

        # Ray tracing (needs to run from genesis/ directory)
        original_dir = os.getcwd()
        os.chdir(str(RF_GENESIS_DIR / "genesis"))
        try:
            body_pir, body_aux = pathtracer.trace(
                os.path.abspath(tmp_motion_path)
            )
        finally:
            os.chdir(original_dir)

        # Generate radar signal frames
        radar_frames = signal_generator.generate_signal_frames(
            body_pir, body_aux, None,
            radar_config=radar_config_path,
            sensor_origin=sensor_origin,
            sensor_target=sensor_target,
        )

    except Exception as e:
        print(f"  [{sample_id}] Radar simulation failed: {e}")
        # Clean up
        if os.path.exists(tmp_motion_path):
            os.remove(tmp_motion_path)
        return False

    # --- Step 3: Extract point clouds ---
    print(f"  [{sample_id}] Extracting point clouds from {radar_frames.shape[0]} frames...")
    try:
        point_clouds = radar_frames_to_pointclouds(
            radar_frames, radar_config_path, num_points=num_points
        )
        # Save point clouds: (T, N, 6) = [x, y, z, velocity, SNR, range]
        np.save(radar_path, point_clouds.astype(np.float32))
        print(f"  [{sample_id}] Point cloud shape: {point_clouds.shape}")
    except Exception as e:
        print(f"  [{sample_id}] Point cloud extraction failed: {e}")
        if os.path.exists(tmp_motion_path):
            os.remove(tmp_motion_path)
        return False

    # --- Step 4: Save caption ---
    caption_path = os.path.join(output_dir, "captions", f"{sample_id}.txt")
    with open(caption_path, "w") as f:
        f.write(prompt)

    # Clean up temp
    if os.path.exists(tmp_motion_path):
        os.remove(tmp_motion_path)

    print(f"  [{sample_id}] Done! PC: {point_clouds.shape}, "
          f"Motion: {pose_params.shape}")
    return True


# ===========================================================================
# Batch generation
# ===========================================================================
def generate_dataset(
    prompts: list,
    output_dir: str,
    model_path: str = None,
    duration: float = 3.0,
    num_points: int = 64,
    num_augments: int = 3,
    train_ratio: float = 0.8,
    device: str = "cuda",
):
    """Generate full training dataset from text prompts.

    Each prompt generates multiple samples with different random seeds
    for data augmentation (varying motions for the same description).

    Args:
        prompts: list of text descriptions
        output_dir: root output directory
        model_path: path to HY-Motion checkpoint
        duration: motion duration per sample
        num_points: points per radar frame
        num_augments: number of augmented samples per prompt (different seeds)
        train_ratio: fraction for training split
    """
    import torch

    if model_path is None:
        model_path = str(HY_MOTION_DIR / "ckpts" / "tencent" / "HY-Motion-1.0")

    # Load HY-Motion once
    print("=" * 60)
    print("[DataGen] Loading HY-Motion model...")
    print("=" * 60)
    from hymotion.utils.t2m_runtime import T2MRuntime

    config_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")

    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        device_ids=[0] if device == "cuda" and torch.cuda.is_available() else None,
        disable_prompt_engineering=True,
    )

    # Generate samples
    os.makedirs(output_dir, exist_ok=True)
    all_sample_ids = []
    success_count = 0
    fail_count = 0

    total = len(prompts) * num_augments
    print(f"[DataGen] Generating {total} samples "
          f"({len(prompts)} prompts x {num_augments} augments)...")

    for p_idx, prompt in enumerate(prompts):
        for aug_idx in range(num_augments):
            sample_id = f"sample_{p_idx:04d}_aug{aug_idx:02d}"
            seed = p_idx * 1000 + aug_idx * 7 + 42

            print(f"\n[{success_count + fail_count + 1}/{total}] {sample_id}")
            ok = generate_single_sample(
                prompt=prompt,
                sample_id=sample_id,
                output_dir=output_dir,
                hy_motion_runtime=runtime,
                duration=duration,
                seed=seed,
                num_points=num_points,
            )

            if ok:
                all_sample_ids.append(sample_id)
                success_count += 1
            else:
                fail_count += 1

    # Create train/val splits
    print(f"\n[DataGen] Creating splits: {success_count} success, {fail_count} failed")
    np.random.seed(42)
    np.random.shuffle(all_sample_ids)
    split_idx = int(len(all_sample_ids) * train_ratio)
    train_ids = all_sample_ids[:split_idx]
    val_ids = all_sample_ids[split_idx:]

    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    with open(os.path.join(splits_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_ids))

    print(f"[DataGen] Split: {len(train_ids)} train, {len(val_ids)} val")
    print(f"[DataGen] Dataset saved to: {output_dir}")

    # Save generation metadata
    metadata = {
        "num_prompts": len(prompts),
        "num_augments": num_augments,
        "total_generated": success_count,
        "total_failed": fail_count,
        "duration": duration,
        "num_points": num_points,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "prompts": prompts,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate RadarLLM training data from text prompts"
    )
    parser.add_argument("--output-dir", type=str, default="./data/generated",
                        help="Output directory for generated data")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to HY-Motion model checkpoint")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Motion duration in seconds (default: 3.0)")
    parser.add_argument("--num-points", type=int, default=64,
                        help="Points per radar frame (default: 64)")
    parser.add_argument("--num-augments", type=int, default=3,
                        help="Augmented samples per prompt (default: 3)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="JSON file with custom prompts (list of strings)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")

    args = parser.parse_args()

    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    print(f"[DataGen] {len(prompts)} prompts, {args.num_augments} augments each")
    print(f"[DataGen] Output: {args.output_dir}")

    generate_dataset(
        prompts=prompts,
        output_dir=args.output_dir,
        model_path=args.model_path,
        duration=args.duration,
        num_points=args.num_points,
        num_augments=args.num_augments,
        train_ratio=args.train_ratio,
        device=args.device,
    )


if __name__ == "__main__":
    main()
