#!/usr/bin/env python3
"""
HY-Motion + RF-Genesis Integration Example
===========================================

This example demonstrates:
1. Generating motion from text using HY-Motion
2. Converting motion to RF signal simulation using RF-Genesis
3. Visualizing Doppler images

Usage:
    python examples/motion_to_doppler.py \
        --prompt "a person walking forward" \
        --duration 3.0 \
        --output-dir output/test

Requirements:
    - HY-Motion-1.0 model weights in ckpts/tencent/HY-Motion-1.0
    - RF-Genesis dependencies (Mitsuba, MDM, etc.)
    - SMPL model files
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ============================================================================
# Path Setup - Resolve paths relative to this script's location
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add HY-Motion to path
HY_MOTION_DIR = PROJECT_ROOT / "HY-Motion-1.0"
if HY_MOTION_DIR.exists():
    sys.path.insert(0, str(HY_MOTION_DIR))

# Add RF-Genesis to path
RF_GENESIS_DIR = PROJECT_ROOT / "RF-Genesis"
if RF_GENESIS_DIR.exists():
    sys.path.insert(0, str(RF_GENESIS_DIR))


def check_dependencies():
    """Check if required modules are available."""
    missing = []

    # Check HY-Motion
    try:
        from hymotion.utils.t2m_runtime import T2MRuntime
        print("[OK] HY-Motion module found")
    except ImportError as e:
        missing.append(f"HY-Motion: {e}")

    # Check RF-Genesis
    try:
        from genesis.raytracing import pathtracer, signal_generator
        from genesis.visualization import visualize
        print("[OK] RF-Genesis module found")
    except ImportError as e:
        missing.append(f"RF-Genesis: {e}")

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
    """
    Generate motion from text using HY-Motion.

    Args:
        prompt: Text description of the motion
        duration: Duration in seconds
        model_path: Path to HY-Motion model
        device: Device to use (cuda/cpu)
        cfg_scale: Classifier-free guidance scale
        seed: Random seed

    Returns:
        Dictionary containing SMPL parameters
    """
    from hymotion.utils.t2m_runtime import T2MRuntime

    config_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[HY-Motion] Loading model from {model_path}")

    # Initialize runtime
    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        device_ids=[0] if device == "cuda" and torch.cuda.is_available() else None,
        disable_prompt_engineering=True,  # Use original prompt without rewriting
    )

    print(f"[HY-Motion] Generating motion for: '{prompt}'")
    print(f"[HY-Motion] Duration: {duration}s, CFG Scale: {cfg_scale}")

    # Generate motion
    _, _, model_output = runtime.generate_motion(
        text=prompt,
        seeds_csv=str(seed),
        duration=duration,
        cfg_scale=cfg_scale,
        output_format="dict",
    )

    print(f"[HY-Motion] Motion generated successfully!")

    return model_output


def convert_hymotion_to_rfgenesis_format(
    model_output: dict,
    output_path: str,
) -> str:
    """
    Convert HY-Motion output to RF-Genesis compatible format.

    HY-Motion outputs SMPL-H parameters that need to be saved in
    the format expected by RF-Genesis pathtracer.

    Args:
        model_output: Output from HY-Motion generate_motion
        output_path: Path to save the .npz file

    Returns:
        Path to the saved .npz file
    """
    from scipy.spatial.transform import Rotation as R

    # Extract SMPL parameters from model output
    # HY-Motion returns: rot6d (B, L, J, 6), transl (B, L, 3)

    if isinstance(model_output, dict):
        # HY-Motion returns rot6d and transl separately
        if 'rot6d' in model_output and 'transl' in model_output:
            rot6d = model_output['rot6d']
            transl = model_output['transl']

            if isinstance(rot6d, torch.Tensor):
                rot6d = rot6d.cpu().numpy()
            if isinstance(transl, torch.Tensor):
                transl = transl.cpu().numpy()

            # Remove batch dimension if present
            if rot6d.ndim == 4:
                rot6d = rot6d[0]  # (L, J, 6)
            if transl.ndim == 3:
                transl = transl[0]  # (L, 3)

            num_frames = rot6d.shape[0]
            num_joints = rot6d.shape[1]

            print(f"[Convert] rot6d shape: {rot6d.shape}, transl shape: {transl.shape}")

            # Convert rot6d to axis-angle using same method as RF-Genesis
            # RF-Genesis uses: rot6d -> matrix -> euler (XYZ) -> axis-angle

            def rot6d_to_matrix(rot_6d):
                """Convert 6D rotation to 3x3 rotation matrix (same as PyTorch3D)."""
                a1, a2 = rot_6d[..., :3], rot_6d[..., 3:6]
                b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
                b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
                b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
                b3 = np.cross(b1, b2, axis=-1)
                return np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)

            def matrix_to_euler_xyz(rot_mat):
                """Convert rotation matrix to Euler angles (XYZ order)."""
                r = R.from_matrix(rot_mat)
                return r.as_euler('xyz')

            def euler_to_axis_angle(euler_xyz):
                """Convert Euler angles (XYZ) to axis-angle."""
                r = R.from_euler('xyz', euler_xyz)
                return r.as_rotvec()

            # Convert all rotations: rot6d -> matrix -> euler -> axis-angle
            pose_params = np.zeros((num_frames, 72))  # 24 joints * 3

            for frame_idx in range(num_frames):
                for joint_idx in range(min(num_joints, 24)):
                    rot_6d = rot6d[frame_idx, joint_idx]
                    rot_mat = rot6d_to_matrix(rot_6d)
                    euler = matrix_to_euler_xyz(rot_mat)
                    axis_angle = euler_to_axis_angle(euler)
                    pose_params[frame_idx, joint_idx*3:(joint_idx+1)*3] = axis_angle

            # Apply coordinate transformation if needed
            # HY-Motion: Y-up, Z-forward (looking at -Z)
            # RF-Genesis: Y-up, Z-forward (sensor at origin looking at -Z, body at Z=3)
            #
            # The body in RF-Genesis is placed at position - body_offset where body_offset = [0, 1, 3]
            # So if root_translation is [0, 0, 0], body is at [0, -1, -3]
            # Sensor looks from [0,0,0] toward [0,0,-5]

            root_translation = transl.copy()

            # Scale translation if needed (HY-Motion might use different scale)
            # RF-Genesis expects meters, adjust if HY-Motion uses different units

            print(f"[Convert] Translation range: X[{root_translation[:,0].min():.3f}, {root_translation[:,0].max():.3f}], "
                  f"Y[{root_translation[:,1].min():.3f}, {root_translation[:,1].max():.3f}], "
                  f"Z[{root_translation[:,2].min():.3f}, {root_translation[:,2].max():.3f}]")

        else:
            # Fallback: try to find motion data in latent_denorm format
            if 'latent_denorm' in model_output:
                smpl_data = model_output['latent_denorm']
            elif 'motion' in model_output:
                smpl_data = model_output['motion']
            else:
                for key, value in model_output.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        smpl_data = value
                        break
                else:
                    raise ValueError(f"Cannot find motion data. Keys: {model_output.keys()}")

            if isinstance(smpl_data, torch.Tensor):
                smpl_data = smpl_data.cpu().numpy()
            if smpl_data.ndim == 3:
                smpl_data = smpl_data[0]

            num_frames = smpl_data.shape[0]
            print(f"[Convert] SMPL data shape: {smpl_data.shape}")

            # Parse 201D format: [trans(3), root_rot6d(6), body_rot6d(21*6=126), ...]
            root_translation = smpl_data[:, :3].copy()

            if smpl_data.shape[1] >= 135:
                rot6d_flat = smpl_data[:, 3:135]  # (L, 132) = 22 joints * 6
                rot6d = rot6d_flat.reshape(num_frames, 22, 6)

                def rot6d_to_matrix(rot_6d):
                    a1, a2 = rot_6d[..., :3], rot_6d[..., 3:6]
                    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
                    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
                    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
                    b3 = np.cross(b1, b2, axis=-1)
                    return np.stack([b1, b2, b3], axis=-1)

                pose_params = np.zeros((num_frames, 72))

                for frame_idx in range(num_frames):
                    for joint_idx in range(22):
                        rot_6d = rot6d[frame_idx, joint_idx]
                        rot_mat = rot6d_to_matrix(rot_6d)
                        r = R.from_matrix(rot_mat)
                        euler = r.as_euler('xyz')
                        axis_angle = R.from_euler('xyz', euler).as_rotvec()
                        pose_params[frame_idx, joint_idx*3:(joint_idx+1)*3] = axis_angle
            else:
                pose_params = smpl_data[:, 3:75] if smpl_data.shape[1] >= 75 else np.zeros((num_frames, 72))
    else:
        raise ValueError(f"Expected dict output, got {type(model_output)}")

    # Save in RF-Genesis format
    shape_params = np.zeros(10)

    np.savez(
        output_path,
        pose=pose_params,
        shape=shape_params,
        root_translation=root_translation,
        gender="male"
    )

    print(f"[Convert] Saved to {output_path}")
    print(f"[Convert] Pose shape: {pose_params.shape}, Translation shape: {root_translation.shape}")

    return output_path


def run_rf_simulation(
    motion_npz_path: str,
    output_dir: str,
    radar_config_path: str = None,
    skip_environment: bool = True,
    visualize_output: bool = True,
) -> np.ndarray:
    """
    Run RF-Genesis simulation on motion data.

    Args:
        motion_npz_path: Path to SMPL parameters .npz file
        output_dir: Output directory
        radar_config_path: Path to radar config JSON
        skip_environment: Whether to skip environment PIR generation
        visualize_output: Whether to generate visualization video

    Returns:
        Radar frames array
    """
    from genesis.raytracing import pathtracer, signal_generator
    from genesis.visualization import visualize

    # Convert all paths to absolute to avoid issues with os.chdir()
    motion_npz_path = os.path.abspath(motion_npz_path)
    output_dir = os.path.abspath(output_dir)

    if radar_config_path is None:
        radar_config_path = str(RF_GENESIS_DIR / "models" / "TI1843_config.json")
    radar_config_path = os.path.abspath(radar_config_path)

    print(f"[RF-Genesis] Starting simulation")
    print(f"[RF-Genesis] Motion file: {motion_npz_path}")
    print(f"[RF-Genesis] Radar config: {radar_config_path}")

    # Step 1: Ray tracing to get body PIRs
    print("[RF-Genesis] Step 1/3: Ray tracing body PIRs...")

    # RF-Genesis pathtracer expects to run from genesis/ directory
    # (because it loads ../models/male.ply with relative path)
    original_dir = os.getcwd()
    os.chdir(str(RF_GENESIS_DIR / "genesis"))

    try:
        # Use absolute path - no need for relpath conversion
        body_pir, body_aux = pathtracer.trace(motion_npz_path)
    finally:
        os.chdir(original_dir)

    print(f"[RF-Genesis] Body PIR frames: {len(body_pir)}")

    # Step 2: Generate radar signal frames
    print("[RF-Genesis] Step 2/3: Generating radar signal frames...")

    # Environment PIR is optional
    env_pir = None

    radar_frames = signal_generator.generate_signal_frames(
        body_pir,
        body_aux,
        env_pir,
        radar_config=radar_config_path
    )

    print(f"[RF-Genesis] Radar frames shape: {radar_frames.shape}")

    # Save radar frames
    radar_output_path = os.path.join(output_dir, "radar_frames.npy")
    np.save(radar_output_path, radar_frames)
    print(f"[RF-Genesis] Saved radar frames to {radar_output_path}")

    # Step 3: Visualization
    if visualize_output:
        print("[RF-Genesis] Step 3/3: Generating visualization...")
        torch.set_default_device('cpu')  # Avoid OOM during visualization

        video_path = os.path.join(output_dir, "output.mp4")
        visualize.save_video(
            radar_config_path,
            radar_output_path,
            motion_npz_path,
            video_path
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
    Generate additional Doppler visualizations.

    Args:
        radar_frames: Radar signal frames from RF-Genesis
        output_dir: Output directory
        duration: Motion duration in seconds
        prompt: Original text prompt
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("[Viz] Generating Doppler visualizations...")

    num_frames = radar_frames.shape[0]

    # Process radar frames to get Range-Doppler maps
    # Assuming radar_frames is already in IF signal format
    rd_maps = []

    for i in range(num_frames):
        frame = radar_frames[i]

        # If complex, take magnitude
        if np.iscomplexobj(frame):
            frame = np.abs(frame)

        # Simple Range-Doppler processing
        if frame.ndim >= 2:
            # Apply 2D FFT if needed
            if frame.shape[-1] > 1 and frame.shape[-2] > 1:
                rd = np.fft.fftshift(np.fft.fft2(frame))
                rd = 20 * np.log10(np.abs(rd) + 1e-12)
            else:
                rd = 20 * np.log10(frame + 1e-12)
        else:
            rd = frame

        rd_maps.append(rd)

    rd_maps = np.array(rd_maps)

    # Normalize
    rd_maps = rd_maps - np.max(rd_maps)

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
    plt.suptitle(f"Range-Doppler Maps\n{prompt}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "range_doppler_samples.png"), dpi=150)
    plt.close()

    # 2. Micro-Doppler Spectrogram
    if rd_maps[0].ndim >= 2:
        doppler_spec = np.array([np.mean(rd, axis=1) for rd in rd_maps]).T

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(
            doppler_spec,
            aspect='auto',
            cmap='jet',
            origin='lower',
            extent=[0, duration, 0, doppler_spec.shape[0]],
            vmin=-60,
            vmax=0
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Doppler bin')
        ax.set_title(f"Micro-Doppler Spectrogram\n{prompt}")
        plt.colorbar(im, label='Power (dB)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doppler_spectrogram.png"), dpi=150)
        plt.close()

    print(f"[Viz] Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate motion and RF Doppler simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Text prompt for motion generation"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=3.0,
        help="Duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to HY-Motion model (default: HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=5.0,
        help="CFG scale for motion generation (default: 5.0)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--skip-motion-gen",
        action="store_true",
        help="Skip motion generation, use existing obj_diff.npz"
    )

    args = parser.parse_args()

    # Check dependencies
    print("=" * 60)
    print("HY-Motion + RF-Genesis Integration Pipeline")
    print("=" * 60)

    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)

    # Setup paths
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
    print("-" * 60)

    if not args.skip_motion_gen:
        # Step 1: Generate motion with HY-Motion
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

        # Step 2: Convert to RF-Genesis format
        print("\n[Step 2/3] Converting to RF-Genesis format...")
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

    # Step 3: Run RF simulation
    print("\n[Step 3/3] Running RF-Genesis simulation...")
    print("-" * 40)

    radar_frames = run_rf_simulation(
        motion_npz_path=motion_npz_path,
        output_dir=output_dir,
        visualize_output=not args.no_visualize,
    )

    # Optional: Generate additional visualizations
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
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  - {f} ({size / 1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
