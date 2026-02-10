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

# ============================================================================
# SMPL-H to SMPL-24 Conversion Utilities
# ============================================================================
#
# SMPL-H has 52 joints: 22 body + 30 hand finger joints.
# Standard SMPL has 24 joints: 22 body + L_Hand (joint 22) + R_Hand (joint 23).
#
# BUG (fixed): Naively taking the first 72 values (24*3) from 156D SMPL-H
# axis-angle causes indices 66-71 to contain L_Index1/L_Index2/L_Index3
# (finger rotations) instead of L_Hand/R_Hand (hand root rotations).
# SMPL_Layer interprets these as hand root joints, distorting body pose.
#
# FIX: Extract only 22 body joints, then append identity rotations for
# L_Hand and R_Hand to produce correct 24-joint (72D) SMPL format.
# ============================================================================
 
 
def _rot6d_to_rotation_matrix_np(rot6d):
    """Convert 6D rotation representation to rotation matrix (numpy).
 
    Uses Gram-Schmidt orthogonalization (Zhou et al., CVPR 2019).
 
    Args:
        rot6d: (..., 6) array of 6D rotation representations
 
    Returns:
        (..., 3, 3) rotation matrices
    """
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
    """Convert rotation matrix to axis-angle (numpy, via scipy).
 
    Args:
        rot_mat: (..., 3, 3) rotation matrices
 
    Returns:
        (..., 3) axis-angle vectors
    """
    from scipy.spatial.transform import Rotation
    orig_shape = rot_mat.shape[:-2]
    flat = rot_mat.reshape(-1, 3, 3)
    r = Rotation.from_matrix(flat)
    aa = r.as_rotvec()
    return aa.reshape(*orig_shape, 3)
 
 
def _convert_rot6d_to_smpl24_pose(rot6d, transl):
    """Convert SMPL-H rot6d to standard SMPL 24-joint axis-angle.
 
    Takes only 22 body joints from SMPL-H and appends identity rotations
    for L_Hand/R_Hand to create proper 24-joint SMPL format.
 
    Args:
        rot6d: (num_frames, num_joints, 6) or (B, num_frames, num_joints, 6)
        transl: (num_frames, 3) or (B, num_frames, 3)
 
    Returns:
        pose_params: (num_frames, 72) - SMPL 24-joint axis-angle
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
 
    # CRITICAL: Take only first 22 body joints, discard finger joints (22-51)
    body_rot6d = rot6d[:, :22, :]  # (num_frames, 22, 6)
 
    # Convert 6D -> rotation matrix -> axis-angle (vectorized, no per-frame loop)
    rot_matrices = _rot6d_to_rotation_matrix_np(body_rot6d)  # (num_frames, 22, 3, 3)
    body_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)  # (num_frames, 22, 3)
 
    # Identity rotation (zero axis-angle) for L_Hand (joint 22) and R_Hand (joint 23)
    hand_aa = np.zeros((num_frames, 2, 3), dtype=body_aa.dtype)
 
    # 22 body + 2 hand root = 24 SMPL joints
    smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)  # (num_frames, 24, 3)
    pose_params = smpl_aa.reshape(num_frames, -1)  # (num_frames, 72)
 
    print(f"[Convert] rot6d input: {num_input_joints} joints -> "
          f"22 body + 2 identity hand = 24 SMPL joints (72D)")
 
    return pose_params, transl
 

 
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
    if isinstance(model_output, dict) and 'rot6d' in model_output and 'transl' in model_output:
        # Preferred: HY-Motion returns rot6d (B, L, J, 6) and transl (B, L, 3)
        rot6d = model_output['rot6d']
        transl = model_output['transl']
 
        print(f"[Convert] Using rot6d: {_shape_str(rot6d)}, transl: {_shape_str(transl)}")
        pose_params, translation = _convert_rot6d_to_smpl24_pose(rot6d, transl)
 
    elif isinstance(model_output, dict) and 'latent_denorm' in model_output:
        # Fallback: 201D latent [translation(3), global_orient_6d(6), body_pose_6d(21*6)]
        smpl_data = model_output['latent_denorm']
        if isinstance(smpl_data, torch.Tensor):
            smpl_data = smpl_data.cpu().numpy()
        if smpl_data.ndim == 3:
            smpl_data = smpl_data[0]
 
        print(f"[Convert] Using latent_denorm: {smpl_data.shape}")
 
        num_frames = smpl_data.shape[0]  






        translation = smpl_data[:, :3]
        

        global_orient_6d = smpl_data[:, 3:9].reshape(num_frames, 1, 6)
        body_pose_6d = smpl_data[:, 9:135].reshape(num_frames, 21, 6)
 
        # global orient (1) + 21 body joints = 22 body joints
        all_rot6d = np.concatenate([global_orient_6d, body_pose_6d], axis=1)
 
        rot_matrices = _rot6d_to_rotation_matrix_np(all_rot6d)
        body_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)
 
        # Add identity for L_Hand/R_Hand
        hand_aa = np.zeros((num_frames, 2, 3), dtype=body_aa.dtype)
        smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)  # (N, 24, 3)
        pose_params = smpl_aa.reshape(num_frames, -1)
 
        print(f"[Convert] Latent path: 22 body + 2 identity hand = 24 SMPL joints (72D)")



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
            # 201D latent or similar 6D-rotation format
            translation = smpl_data[:, :3]
            global_orient_6d = smpl_data[:, 3:9].reshape(num_frames, 1, 6)
            body_pose_6d = smpl_data[:, 9:135].reshape(num_frames, 21, 6)
 
            all_rot6d = np.concatenate([global_orient_6d, body_pose_6d], axis=1)
            rot_matrices = _rot6d_to_rotation_matrix_np(all_rot6d)
            body_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)
 
            hand_aa = np.zeros((num_frames, 2, 3), dtype=body_aa.dtype)
            smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)
            pose_params = smpl_aa.reshape(num_frames, -1)
        else:
            # Pre-converted axis-angle format
            # CRITICAL: Do NOT take first 72 values blindly from 156D SMPL-H data.
            # Indices 66-71 would be finger joints, not L_Hand/R_Hand.
            # Instead: take 22 body joints (indices 0-65) + add 2 identity hand joints.
            translation = smpl_data[:, :3] if num_values >= 3 else np.zeros((num_frames, 3))
 
            if num_values >= 69:
                # translation(3) + 22 body joints(66) = 69 values minimum
                body_aa = smpl_data[:, 3:69].reshape(num_frames, 22, 3)
            elif num_values >= 66:
                body_aa = smpl_data[:, :66].reshape(num_frames, 22, 3)
            else:
                n_joints = (num_values - 3) // 3 if num_values > 3 else num_values // 3
                body_aa = smpl_data[:, 3:3 + n_joints * 3].reshape(num_frames, n_joints, 3)
                if n_joints < 22:
                    pad = np.zeros((num_frames, 22 - n_joints, 3), dtype=body_aa.dtype)
                    body_aa = np.concatenate([body_aa, pad], axis=1)
 
            # Append identity for L_Hand/R_Hand (NEVER use 156D indices 66-71)
            hand_aa = np.zeros((num_frames, 2, 3), dtype=body_aa.dtype)
            smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)
            pose_params = smpl_aa.reshape(num_frames, -1)
    # Save in RF-Genesis format
    shape_params = np.zeros(10)  # Default shape parameters
 
    np.savez(
        output_path,
        pose=pose_params,
        shape=shape_params,
        root_translation=translation,
        gender="neutral"
    )
 
    print(f"[Convert] Saved to {output_path}")
    print(f"[Convert] Pose shape: {pose_params.shape}, Translation shape: {translation.shape}")
 
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
    from genesis.raytracing import pathtracer, signal_generator
    from genesis.visualization import visualize

    motion_npz_path = os.path.abspath(motion_npz_path)
    output_dir = os.path.abspath(output_dir)

    if radar_config_path is None:
        radar_config_path = str(RF_GENESIS_DIR / "models" / "TI1843_config.json")
    radar_config_path = os.path.abspath(radar_config_path)

    print(f"[RF-Genesis] Starting simulation")

    # ── 수정: 센서 위치를 미리 계산 ──
    smpl_data = np.load(motion_npz_path, allow_pickle=True)
    root_translation = smpl_data['root_translation']
    traj_center = root_translation.mean(axis=0)
    sensor_distance = 3.0
    sensor_origin = [
        traj_center[0],
        traj_center[1] + 1.0,
        traj_center[2] + sensor_distance,
    ]
    sensor_target = [
        traj_center[0],
        traj_center[1] + 1.0,
        traj_center[2],
    ]
    print(f"[RF-Genesis] Sensor origin={sensor_origin}, target={sensor_target}")

    # Step 1: Ray tracing
    print("[RF-Genesis] Step 1/3: Ray tracing body PIRs...")
    original_dir = os.getcwd()
    os.chdir(str(RF_GENESIS_DIR / "genesis"))
    try:
        body_pir, body_aux = pathtracer.trace(motion_npz_path)
    finally:
        os.chdir(original_dir)

    # Step 2: Signal generation (with coordinate transform)
    print("[RF-Genesis] Step 2/3: Generating radar signal frames...")
    env_pir = None

    radar_frames = signal_generator.generate_signal_frames(
        body_pir,
        body_aux,
        env_pir,
        radar_config=radar_config_path,
        sensor_origin=sensor_origin,      # ← 추가
        sensor_target=sensor_target,      # ← 추가
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
 
        if rd.ndim > 2:
            # (3, 4, 128, 256) -> (128, 256)으로 평균내기
            dims_to_reduce = tuple(range(rd.ndim - 2))
            rd = np.mean(rd, axis=dims_to_reduce)

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