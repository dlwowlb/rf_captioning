#!/usr/bin/env python3
"""
HY-Motion + RF-Genesis Integration Example (SMPL-X 52-joint)
=============================================================

This example demonstrates:
1. Generating motion from text using HY-Motion (SMPL-H 52 joints)
2. Converting motion to SMPL-X 52-joint format for RF-Genesis
3. Running RF-Genesis Pathtracer physics simulation
4. Generating Doppler heatmaps via proper FMCW radar signal processing

Usage:
    python examples/motion_to_doppler.py \
        --prompt "a person walking forward" \
        --duration 3.0 \
        --output-dir output/test

Requirements:
    - HY-Motion-1.0 model weights in ckpts/tencent/HY-Motion-1.0
    - RF-Genesis dependencies (Mitsuba, drjit, etc.)
    - smplx library: pip install smplx
    - SMPL-X model files in RF-Genesis/models/smplx_models/
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ============================================================================
# Path Setup
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

HY_MOTION_DIR = PROJECT_ROOT / "HY-Motion-1.0"
if HY_MOTION_DIR.exists():
    sys.path.insert(0, str(HY_MOTION_DIR))

RF_GENESIS_DIR = PROJECT_ROOT / "RF-Genesis"
if RF_GENESIS_DIR.exists():
    sys.path.insert(0, str(RF_GENESIS_DIR))


def check_dependencies():
    """Check if required modules are available."""
    missing = []

    try:
        from hymotion.utils.t2m_runtime import T2MRuntime
        print("[OK] HY-Motion module found")
    except ImportError as e:
        missing.append(f"HY-Motion: {e}")

    try:
        from genesis.raytracing import pathtracer, signal_generator
        from genesis.visualization import visualize
        print("[OK] RF-Genesis module found")
    except ImportError as e:
        missing.append(f"RF-Genesis: {e}")

    try:
        import smplx
        print("[OK] smplx library found")
    except ImportError as e:
        missing.append(f"smplx: {e} (install with: pip install smplx)")

    if missing:
        print("\n[ERROR] Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        return False

    return True


def generate_motion_hymotion(
    prompt: str,
    duration: float,
    model_path: str,
    device: str = "cuda",
    cfg_scale: float = 5.0,
    seed: int = 42,
) -> dict:
    """Generate motion from text using HY-Motion."""
    from hymotion.utils.t2m_runtime import T2MRuntime

    config_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[HY-Motion] Loading model from {model_path}")

    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        device_ids=[0] if device == "cuda" and torch.cuda.is_available() else None,
        disable_prompt_engineering=True,
    )

    print(f"[HY-Motion] Generating motion for: '{prompt}'")
    print(f"[HY-Motion] Duration: {duration}s, CFG Scale: {cfg_scale}")

    _, _, model_output = runtime.generate_motion(
        text=prompt,
        seeds_csv=str(seed),
        duration=duration,
        cfg_scale=cfg_scale,
        output_format="dict",
    )

    print(f"[HY-Motion] Motion generated successfully!")
    return model_output


# ============================================================================
# Rotation Conversion Utilities
# ============================================================================

def _rot6d_to_rotation_matrix_np(rot6d):
    """Convert 6D rotation to rotation matrix (Gram-Schmidt).

    The 6D representation stores two 3D vectors consecutively:
    [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z] (first two columns of the rotation matrix).
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def _rotation_matrix_to_axis_angle_np(rot_mat):
    """Convert rotation matrix to axis-angle via scipy.

    Projects each matrix to the nearest valid rotation (SO(3)) via SVD
    to handle numerical imprecision from the Gram-Schmidt process.
    """
    from scipy.spatial.transform import Rotation
    orig_shape = rot_mat.shape[:-2]
    flat = rot_mat.reshape(-1, 3, 3)
    # SVD projection: R_valid = U @ Vt, with det correction to ensure SO(3)
    U, _, Vt = np.linalg.svd(flat)
    dets = np.linalg.det(U @ Vt)
    # Flip last column of U where determinant is negative
    sign = np.ones_like(U)
    sign[dets < 0, :, -1] = -1
    flat = (U * sign) @ Vt
    r = Rotation.from_matrix(flat)
    aa = r.as_rotvec()
    return aa.reshape(*orig_shape, 3)


# ============================================================================
# SMPL-X 52-Joint Conversion (preserves finger joints)
# ============================================================================

def _convert_rot6d_to_smplx52_pose(rot6d, transl):
    """
    Convert HY-Motion rot6d output to SMPL-X 52-joint axis-angle format.

    HY-Motion outputs 52 joints in SMPL-H format:
      22 body joints + 15 left hand + 15 right hand = 52 joints

    This is directly compatible with SMPL-X body model layout:
      global_orient (1*3) + body_pose (21*3) + left_hand (15*3) + right_hand (15*3)
      = 3 + 63 + 45 + 45 = 156D

    Args:
        rot6d: (num_frames, 52, 6) or (B, num_frames, 52, 6)
        transl: (num_frames, 3) or (B, num_frames, 3)

    Returns:
        pose_params: (num_frames, 156) - SMPL-X 52-joint axis-angle
        translation: (num_frames, 3)
    """
    if isinstance(rot6d, torch.Tensor):
        rot6d = rot6d.cpu().numpy()
    if isinstance(transl, torch.Tensor):
        transl = transl.cpu().numpy()

    if rot6d.ndim == 4:
        rot6d = rot6d[0]
    if transl.ndim == 3:
        transl = transl[0]

    num_frames = rot6d.shape[0]
    num_input_joints = rot6d.shape[1]

    if num_input_joints >= 52:
        # Full SMPL-H/SMPL-X: take all 52 joints
        all_rot6d = rot6d[:, :52, :]
    elif num_input_joints >= 22:
        # Only body joints: pad hand joints with identity (flat hand)
        body_rot6d = rot6d[:, :22, :]
        # Identity rotation in 6D: [1,0,0, 0,1,0]
        identity_6d = np.zeros((num_frames, 30, 6), dtype=rot6d.dtype)
        identity_6d[..., 0] = 1.0  # first column of identity
        identity_6d[..., 4] = 1.0  # second column of identity
        all_rot6d = np.concatenate([body_rot6d, identity_6d], axis=1)
    else:
        raise ValueError(f"Expected >= 22 joints, got {num_input_joints}")

    # Convert all 52 joints: 6D -> rotation matrix -> axis-angle
    rot_matrices = _rot6d_to_rotation_matrix_np(all_rot6d)  # (N, 52, 3, 3)
    all_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)  # (N, 52, 3)

    pose_params = all_aa.reshape(num_frames, -1)  # (N, 156)

    print(f"[Convert] rot6d input: {num_input_joints} joints -> "
          f"SMPL-X 52 joints (156D) preserving finger articulation")

    return pose_params, transl


def _convert_latent_to_smplx52_pose(smpl_data):
    """
    Convert 201D latent format to SMPL-X 52-joint format.

    Latent layout: [translation(3), global_orient_6d(6), body_pose_6d(21*6=126)]
    Only body joints are available; hands are set to identity (flat).

    Args:
        smpl_data: (num_frames, 135+) latent array

    Returns:
        pose_params: (num_frames, 156)
        translation: (num_frames, 3)
    """
    num_frames = smpl_data.shape[0]

    translation = smpl_data[:, :3]
    global_orient_6d = smpl_data[:, 3:9].reshape(num_frames, 1, 6)
    body_pose_6d = smpl_data[:, 9:135].reshape(num_frames, 21, 6)

    # global orient (1) + 21 body joints = 22 body joints
    body_rot6d = np.concatenate([global_orient_6d, body_pose_6d], axis=1)

    # Add identity rotation for 30 hand joints (flat hand)
    identity_6d = np.zeros((num_frames, 30, 6), dtype=smpl_data.dtype)
    identity_6d[..., 0] = 1.0
    identity_6d[..., 4] = 1.0

    all_rot6d = np.concatenate([body_rot6d, identity_6d], axis=1)  # (N, 52, 6)

    rot_matrices = _rot6d_to_rotation_matrix_np(all_rot6d)
    all_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)
    pose_params = all_aa.reshape(num_frames, -1)  # (N, 156)

    print(f"[Convert] Latent path: 22 body + 30 identity hand = 52 SMPL-X joints (156D)")

    return pose_params, translation


def convert_hymotion_to_rfgenesis_format(
    model_output: dict,
    output_path: str,
) -> str:
    """
    Convert HY-Motion output to RF-Genesis compatible SMPL-X format.

    Now saves full 52-joint (156D) SMPL-X pose parameters,
    preserving finger joint articulation when available.

    Args:
        model_output: Output from HY-Motion generate_motion
        output_path: Path to save the .npz file

    Returns:
        Path to the saved .npz file
    """
    if isinstance(model_output, dict) and 'rot6d' in model_output and 'transl' in model_output:
        rot6d = model_output['rot6d']
        transl = model_output['transl']
        print(f"[Convert] Using rot6d: {_shape_str(rot6d)}, transl: {_shape_str(transl)}")
        pose_params, translation = _convert_rot6d_to_smplx52_pose(rot6d, transl)

    elif isinstance(model_output, dict) and 'latent_denorm' in model_output:
        smpl_data = model_output['latent_denorm']
        if isinstance(smpl_data, torch.Tensor):
            smpl_data = smpl_data.cpu().numpy()
        if smpl_data.ndim == 3:
            smpl_data = smpl_data[0]
        print(f"[Convert] Using latent_denorm: {smpl_data.shape}")
        pose_params, translation = _convert_latent_to_smplx52_pose(smpl_data)

    else:
        # Generic fallback
        if isinstance(model_output, dict):
            smpl_data = None
            for key in ['smpl_params', 'motion', 'output']:
                if key in model_output:
                    smpl_data = model_output[key]
                    break
            if smpl_data is None:
                for key, value in model_output.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        smpl_data = value
                        print(f"[Convert] Warning: using generic key '{key}'")
                        break
                else:
                    raise ValueError(
                        f"Cannot find motion data in output. Keys: {model_output.keys()}")
        else:
            smpl_data = model_output

        if isinstance(smpl_data, torch.Tensor):
            smpl_data = smpl_data.cpu().numpy()
        if smpl_data.ndim == 3:
            smpl_data = smpl_data[0]

        print(f"[Convert] Fallback path, data shape: {smpl_data.shape}")
        num_frames = smpl_data.shape[0]
        num_values = smpl_data.shape[1]

        if num_values >= 135:
            pose_params, translation = _convert_latent_to_smplx52_pose(smpl_data)
        elif num_values >= 159:
            # Already 156D pose + 3 translation
            translation = smpl_data[:, :3]
            pose_params = smpl_data[:, 3:159]
        else:
            # Minimal fallback: try to extract what we can
            translation = smpl_data[:, :3] if num_values >= 3 else np.zeros((num_frames, 3))

            if num_values >= 69:
                body_aa = smpl_data[:, 3:69].reshape(num_frames, 22, 3)
            elif num_values >= 66:
                body_aa = smpl_data[:, :66].reshape(num_frames, 22, 3)
            else:
                n_joints = (num_values - 3) // 3 if num_values > 3 else num_values // 3
                body_aa = smpl_data[:, 3:3 + n_joints * 3].reshape(num_frames, n_joints, 3)
                if n_joints < 22:
                    pad = np.zeros((num_frames, 22 - n_joints, 3), dtype=body_aa.dtype)
                    body_aa = np.concatenate([body_aa, pad], axis=1)

            # Pad with 30 identity hand joints to reach 52 joints
            hand_aa = np.zeros((num_frames, 30, 3), dtype=body_aa.dtype)
            smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)  # (N, 52, 3)
            pose_params = smpl_aa.reshape(num_frames, -1)  # (N, 156)

    # Save in SMPL-X format for RF-Genesis
    shape_params = np.zeros(10)

    np.savez(
        output_path,
        pose=pose_params,
        shape=shape_params,
        root_translation=translation,
        gender="neutral",
    )

    print(f"[Convert] Saved to {output_path}")
    print(f"[Convert] Pose shape: {pose_params.shape} "
          f"({pose_params.shape[-1] // 3} joints), "
          f"Translation shape: {translation.shape}")

    return output_path


def _shape_str(x):
    """Helper to get shape string from tensor or array."""
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    elif isinstance(x, np.ndarray):
        return str(x.shape)
    return str(type(x))


def run_rf_simulation(
    motion_npz_path: str,
    output_dir: str,
    radar_config_path: str = None,
    skip_environment: bool = True,
    visualize_output: bool = True,
) -> np.ndarray:
    """
    Run the RF-Genesis Pathtracer physics simulation pipeline.

    Uses the Mitsuba-based ray tracer for physically accurate PIR generation,
    followed by FMCW radar signal synthesis.
    """
    from genesis.raytracing import pathtracer, signal_generator
    from genesis.visualization import visualize

    motion_npz_path = os.path.abspath(motion_npz_path)
    output_dir = os.path.abspath(output_dir)

    if radar_config_path is None:
        radar_config_path = str(RF_GENESIS_DIR / "models" / "TI1843_config.json")
    radar_config_path = os.path.abspath(radar_config_path)

    print(f"[RF-Genesis] Starting Pathtracer physics simulation")

    # Compute sensor position from trajectory
    smpl_data = np.load(motion_npz_path, allow_pickle=True)
    root_translation = smpl_data['root_translation']
    traj_center = root_translation.mean(axis=0)
    sensor_distance = 3.0
    sensor_origin = [
        traj_center[0],
        traj_center[1],
        traj_center[2] + sensor_distance,
    ]
    sensor_target = [
        traj_center[0],
        traj_center[1],
        traj_center[2],
    ]
    print(f"[RF-Genesis] Sensor origin={sensor_origin}, target={sensor_target}")

    # Step 1: Ray tracing with Mitsuba Pathtracer
    print("[RF-Genesis] Step 1/3: Ray tracing body PIRs (SMPL-X 52 joints)...")
    original_dir = os.getcwd()
    os.chdir(str(RF_GENESIS_DIR / "genesis"))
    try:
        body_pir, body_aux = pathtracer.trace(motion_npz_path)
    finally:
        os.chdir(original_dir)

    # Step 2: FMCW Signal generation
    print("[RF-Genesis] Step 2/3: Generating FMCW radar signal frames...")
    env_pir = None

    radar_frames = signal_generator.generate_signal_frames(
        body_pir,
        body_aux,
        env_pir,
        radar_config=radar_config_path,
        sensor_origin=sensor_origin,
        sensor_target=sensor_target,
    )

    print(f"[RF-Genesis] Radar frames shape: {radar_frames.shape}")

    radar_output_path = os.path.join(output_dir, "radar_frames.npy")
    np.save(radar_output_path, radar_frames)

    # Step 3: Visualization
    if visualize_output:
        print("[RF-Genesis] Step 3/3: Generating visualization...")
        torch.set_default_device('cpu')
        video_path = os.path.join(output_dir, "output.mp4")
        visualize.save_video(
            radar_config_path,
            radar_output_path,
            motion_npz_path,
            video_path,
        )
        print(f"[RF-Genesis] Saved video to {video_path}")

    return radar_frames


def generate_doppler_visualization(
    radar_frames: np.ndarray,
    output_dir: str,
    duration: float,
    prompt: str,
):
    """
    Generate Doppler visualizations from FMCW radar signal frames.

    Performs proper FMCW signal processing:
    1. Range FFT along ADC samples (fast-time axis)
    2. Doppler FFT along chirps (slow-time axis)
    3. dB conversion and normalization

    Args:
        radar_frames: (num_frames, num_tx, num_rx, chirps, adc_samples) complex
        output_dir: Output directory
        duration: Motion duration in seconds
        prompt: Original text prompt
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("[Viz] Generating Doppler visualizations (FMCW processing)...")

    num_frames = radar_frames.shape[0]

    rd_maps = []
    for i in range(num_frames):
        frame = radar_frames[i]  # (num_tx, num_rx, chirps, adc_samples), complex

        # Proper FMCW radar processing:
        # 1. Range FFT along ADC samples (last axis) - extracts range info
        # 2. Doppler FFT along chirps (second-to-last axis) - extracts velocity info
        # CRITICAL: Do NOT take abs() before FFT - phase carries Doppler info
        if frame.ndim >= 2 and frame.shape[-1] > 1 and frame.shape[-2] > 1:
            range_fft = np.fft.fft(frame, axis=-1)
            rd = np.fft.fftshift(np.fft.fft(range_fft, axis=-2), axes=-2)
            rd = 20 * np.log10(np.abs(rd) + 1e-12)
        else:
            rd = 20 * np.log10(np.abs(frame) + 1e-12)

        # Average across TX/RX antennas: (tx, rx, chirps, adc) -> (chirps, adc)
        if rd.ndim > 2:
            dims_to_reduce = tuple(range(rd.ndim - 2))
            rd = np.mean(rd, axis=dims_to_reduce)

        rd_maps.append(rd)

    rd_maps = np.array(rd_maps)
    rd_maps = rd_maps - np.max(rd_maps)  # Normalize to 0 dB peak

    time_axis = np.linspace(0, duration, num_frames)

    # 1. Sample Range-Doppler frames
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    sample_indices = np.linspace(0, num_frames - 1, 5, dtype=int)

    for i, (ax, idx) in enumerate(zip(axes.flat[:5], sample_indices)):
        if rd_maps[idx].ndim >= 2:
            im = ax.imshow(rd_maps[idx], aspect='auto', cmap='jet', vmin=-60, vmax=0)
            ax.set_title(f'Frame {idx} (t={time_axis[idx]:.2f}s)')
            ax.set_xlabel('Range bin')
            ax.set_ylabel('Doppler bin')
            plt.colorbar(im, ax=ax, label='dB')

    axes.flat[-1].axis('off')
    plt.suptitle(f"Range-Doppler Maps (SMPL-X 52-joint)\n{prompt}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "range_doppler_samples.png"), dpi=150)
    plt.close()

    # 2. Micro-Doppler Spectrogram (time vs Doppler)
    if rd_maps[0].ndim >= 2:
        # Sum along range axis to get Doppler profile per frame
        doppler_spec = np.array([np.mean(rd, axis=-1) for rd in rd_maps]).T

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(
            doppler_spec,
            aspect='auto',
            cmap='jet',
            origin='lower',
            extent=[0, duration, 0, doppler_spec.shape[0]],
            vmin=-60,
            vmax=0,
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Doppler bin')
        ax.set_title(f"Micro-Doppler Spectrogram (SMPL-X 52-joint)\n{prompt}")
        plt.colorbar(im, label='Power (dB)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doppler_spectrogram.png"), dpi=150)
        plt.close()

    print(f"[Viz] Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate motion and RF Doppler simulation (SMPL-X 52-joint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-p", "--prompt", required=True,
                        help="Text prompt for motion generation")
    parser.add_argument("-d", "--duration", type=float, default=3.0,
                        help="Duration in seconds (default: 3.0)")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--model-path", default=None,
                        help="Path to HY-Motion model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cfg-scale", type=float, default=5.0,
                        help="CFG scale for motion generation (default: 5.0)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--skip-motion-gen", action="store_true",
                        help="Skip motion generation, use existing obj_diff.npz")

    args = parser.parse_args()

    print("=" * 60)
    print("HY-Motion + RF-Genesis Pipeline (SMPL-X 52-joint)")
    print("=" * 60)

    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)

    if args.model_path is None:
        args.model_path = str(HY_MOTION_DIR / "ckpts" / "tencent" / "HY-Motion-1.0")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    motion_npz_path = os.path.join(output_dir, "obj_diff.npz")

    print(f"\nConfiguration:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Duration: {args.duration}s")
    print(f"  Output: {output_dir}")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Body Model: SMPL-X (52 joints with finger articulation)")
    print("-" * 60)

    if not args.skip_motion_gen:
        print("\n[Step 1/3] Generating motion with HY-Motion...")
        print("-" * 40)

        model_output = generate_motion_hymotion(
            prompt=args.prompt,
            duration=args.duration,
            model_path=args.model_path,
            device=args.device,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )

        print("\n[Step 2/3] Converting to SMPL-X 52-joint format...")
        print("-" * 40)

        convert_hymotion_to_rfgenesis_format(
            model_output=model_output,
            output_path=motion_npz_path,
        )
    else:
        print("\n[Step 1-2/3] Skipping motion generation...")
        if not os.path.exists(motion_npz_path):
            print(f"[ERROR] Motion file not found: {motion_npz_path}")
            sys.exit(1)

    # Step 3: Run RF simulation with Pathtracer
    print("\n[Step 3/3] Running RF-Genesis Pathtracer simulation...")
    print("-" * 40)

    radar_frames = run_rf_simulation(
        motion_npz_path=motion_npz_path,
        output_dir=output_dir,
        visualize_output=not args.no_visualize,
    )

    # Generate Doppler visualizations
    if not args.no_visualize:
        print("\n[Extra] Generating Doppler visualizations...")
        print("-" * 40)

        generate_doppler_visualization(
            radar_frames=radar_frames,
            output_dir=output_dir,
            duration=args.duration,
            prompt=args.prompt,
        )

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Body model: SMPL-X (52 joints, including finger articulation)")
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  - {f} ({size / 1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
