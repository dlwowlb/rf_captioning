#!/usr/bin/env python3
"""
HY-Motion + RF-Genesis Integration Example
===========================================

This example demonstrates:
1. Generating motion from text using HY-Motion
2. Converting motion to RF signal simulation using RF-Genesis
3. Extracting point clouds (identical to generate_pt.py pipeline)
4. Visualizing point cloud-based Doppler/Range/Angle spectrograms

Usage:
    python examples/visualization_generated_pt.py \
        --prompt "A person turns a doorknob and opens a door" \
        --duration 4.0 \
        --output-dir output/test
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

# Add HY-Motion to path
HY_MOTION_DIR = PROJECT_ROOT / "HY-Motion-1.0"
if HY_MOTION_DIR.exists():
    sys.path.insert(0, str(HY_MOTION_DIR))

# Add RF-Genesis to path
RF_GENESIS_DIR = PROJECT_ROOT / "RF-Genesis"
if RF_GENESIS_DIR.exists():
    sys.path.insert(0, str(RF_GENESIS_DIR))


def check_dependencies():
    missing = []
    try:
        from hymotion.utils.t2m_runtime import T2MRuntime
        print("[OK] HY-Motion module found")
    except ImportError as e:
        missing.append(f"HY-Motion: {e}")

    try:
        from genesis.raytracing import pathtracer, signal_generator
        from genesis.visualization import visualize
        from genesis.visualization.pointcloud import (
            PointCloudProcessCFG, process_pc, Radar
        )
        print("[OK] RF-Genesis module found")
    except ImportError as e:
        missing.append(f"RF-Genesis: {e}")

    if missing:
        print("\n[ERROR] Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        return False
    return True


# ============================================================================
# HY-Motion: Text → Motion
# ============================================================================

def generate_motion_hymotion(
    prompt: str,
    duration: float,
    model_path: str,
    device: str = "cuda",
    cfg_scale: float = 5.0,
    seed: int = 42,
) -> dict:
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
# SMPL Conversion Utilities
# ============================================================================

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


def _convert_rot6d_to_smpl24_pose(rot6d, transl):
    if isinstance(rot6d, torch.Tensor):
        rot6d = rot6d.cpu().numpy()
    if isinstance(transl, torch.Tensor):
        transl = transl.cpu().numpy()
    if rot6d.ndim == 4:
        rot6d = rot6d[0]
    if transl.ndim == 3:
        transl = transl[0]

    num_frames = rot6d.shape[0]
    body_rot6d = rot6d[:, :22, :]

    rot_matrices = _rot6d_to_rotation_matrix_np(body_rot6d)
    body_aa = _rotation_matrix_to_axis_angle_np(rot_matrices)
    hand_aa = np.zeros((num_frames, 2, 3), dtype=body_aa.dtype)

    smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)
    pose_params = smpl_aa.reshape(num_frames, -1)
    return pose_params, transl


def convert_hymotion_to_rfgenesis_format(model_output: dict, output_path: str) -> str:
    if isinstance(model_output, dict) and 'rot6d' in model_output and 'transl' in model_output:
        rot6d = model_output['rot6d']
        transl = model_output['transl']
        pose_params, translation = _convert_rot6d_to_smpl24_pose(rot6d, transl)

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
    else:
        raise ValueError("Unsupported model output format")

    shape_params = np.zeros(10)
    np.savez(
        output_path,
        pose=pose_params,
        shape=shape_params,
        root_translation=translation,
        gender="neutral",
    )
    return output_path


def _shape_str(x):
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    elif isinstance(x, np.ndarray):
        return str(x.shape)
    return str(type(x))


# ============================================================================
# RF-Genesis Simulation + Point Cloud Extraction
# ============================================================================

def run_rf_simulation(
    motion_npz_path: str,
    output_dir: str,
    radar_config_path: str = None,
    skip_environment: bool = True,
    visualize_output: bool = True,
    points_per_frame: int = 128,
) -> tuple:
    """
    Run RF-Genesis simulation and extract point clouds.

    Returns:
        radar_frames: (T, num_tx, num_rx, chirps, adc_samples) raw radar
        point_clouds:  (T, points_per_frame, 6) — identical to generate_pt.py
                       columns: x, y, z, velocity, energy, range
    """
    from genesis.raytracing import pathtracer, signal_generator
    from genesis.visualization import visualize
    from genesis.visualization.pointcloud import (
        PointCloudProcessCFG, process_pc, Radar
    )

    motion_npz_path = os.path.abspath(motion_npz_path)
    output_dir = os.path.abspath(output_dir)

    if radar_config_path is None:
        radar_config_path = str(RF_GENESIS_DIR / "models" / "TI1843_config.json")
    radar_config_path = os.path.abspath(radar_config_path)

    # ── Sensor placement ──
    smpl_data = np.load(motion_npz_path, allow_pickle=True)
    root_translation = smpl_data['root_translation']
    traj_center = root_translation.mean(axis=0)
    sensor_distance = 3.0
    sensor_origin = [traj_center[0], traj_center[1], traj_center[2] + sensor_distance]
    sensor_target = [traj_center[0], traj_center[1], traj_center[2]]

    # ── Step 1: Ray tracing ──
    print("[RF-Genesis] Step 1/4: Ray tracing body PIRs...")
    original_dir = os.getcwd()
    os.chdir(str(RF_GENESIS_DIR / "genesis"))
    try:
        body_pir, body_aux = pathtracer.trace(motion_npz_path)
    finally:
        os.chdir(original_dir)

    # ── Step 2: Signal generation ──
    print("[RF-Genesis] Step 2/4: Generating radar signal frames...")
    env_pir = None
    radar_frames = signal_generator.generate_signal_frames(
        body_pir, body_aux, env_pir,
        radar_config=radar_config_path,
        sensor_origin=sensor_origin,
        sensor_target=sensor_target,
    )
    radar_output_path = os.path.join(output_dir, "radar_frames.npy")
    np.save(radar_output_path, radar_frames)
    print(f"[RF-Genesis] Radar frames: {radar_frames.shape}")

    # ── Step 3: Point cloud extraction (identical to generate_pt.py) ──
    print("[RF-Genesis] Step 3/4: Extracting point clouds...")
    radar = Radar(radar_config_path)
    pc_cfg = PointCloudProcessCFG(radar)

    point_clouds = []
    for frame_idx in range(len(radar_frames)):
        frame = radar_frames[frame_idx]
        pc = process_pc(pc_cfg, frame)  # (N, 6): x, y, z, velocity, energy, range

        # 128점으로 정규화 (generate_pt.py와 동일)
        if len(pc) == 0:
            pc_padded = np.zeros((points_per_frame, 6), dtype=np.float32)
        elif len(pc) >= points_per_frame:
            # Energy(column 4) 기준 top-128
            top_idx = np.argsort(-pc[:, 4])[:points_per_frame]
            pc_padded = pc[top_idx]
        else:
            # 부족하면 zero padding (랜덤 복제 안 함)
            pc_padded = np.zeros((points_per_frame, 6), dtype=np.float32)
            pc_padded[:len(pc)] = pc

        point_clouds.append(pc_padded)

    point_clouds = np.array(point_clouds, dtype=np.float32)
    # shape: (num_radar_frames, 128, 6)
    print(f"[RF-Genesis] Point clouds: {point_clouds.shape}")

    # 포인트 클라우드도 저장
    pc_output_path = os.path.join(output_dir, "point_clouds.npy")
    np.save(pc_output_path, point_clouds)

    # ── Step 4: Original video visualization (optional) ──
    if visualize_output:
        print("[RF-Genesis] Step 4/4: Generating video visualization...")
        torch.set_default_device('cpu')
        video_path = os.path.join(output_dir, "output.mp4")
        visualize.save_video(radar_config_path, radar_output_path, motion_npz_path, video_path)

    return radar_frames, point_clouds


# ============================================================================
# Point Cloud Visualization (generate_pt와 동일한 데이터 기반)
# ============================================================================

def generate_doppler_visualization(
    radar_frames: np.ndarray,
    point_clouds: np.ndarray,
    output_dir: str,
    duration: float,
    prompt: str,
):
    """
    Visualize extracted point clouds.

    Args:
        radar_frames:  (T, tx, rx, chirps, adc) — raw radar (for reference)
        point_clouds:  (T, 128, 6) — x, y, z, velocity, energy, range
        output_dir:    save path
        duration:      motion duration in seconds
        prompt:        text prompt for title
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("[Viz] Generating point cloud visualizations...")
    T, N, D = point_clouds.shape
    time_axis = np.linspace(0, duration, T)

    # Valid mask: zero-padded 포인트 제외
    valid = np.linalg.norm(point_clouds[:, :, :3], axis=-1) > 1e-6
    pts_per_frame = valid.sum(axis=1)

    print(f"[Viz] Frames: {T}, Points per frame: "
          f"min={pts_per_frame.min()}, max={pts_per_frame.max()}, "
          f"mean={pts_per_frame.mean():.1f}")

    # ==========================================
    # Plot 1: Time Spectrograms (Doppler, Range, Azimuth)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # --- Micro-Doppler (Velocity vs Time) ---
    for t in range(T):
        v = valid[t]
        if v.any():
            vel = point_clouds[t, v, 3]
            energy = point_clouds[t, v, 4]
            energy_norm = energy / (energy.max() + 1e-12)
            axes[0].scatter(
                np.full_like(vel, time_axis[t]), vel,
                c=energy_norm, cmap='hot', s=3, alpha=0.7,
                vmin=0, vmax=1,
            )
    axes[0].set_ylabel('Doppler Velocity (m/s)')
    axes[0].set_title('Micro-Doppler Spectrogram (Point Cloud)')
    axes[0].grid(True, alpha=0.3)

    # --- Range vs Time ---
    for t in range(T):
        v = valid[t]
        if v.any():
            rng = point_clouds[t, v, 5]
            energy = point_clouds[t, v, 4]
            energy_norm = energy / (energy.max() + 1e-12)
            axes[1].scatter(
                np.full_like(rng, time_axis[t]), rng,
                c=energy_norm, cmap='hot', s=3, alpha=0.7,
                vmin=0, vmax=1,
            )
    axes[1].set_ylabel('Range (m)')
    axes[1].set_title('Range-Time Spectrogram (Point Cloud)')
    axes[1].grid(True, alpha=0.3)

    # --- Azimuth (X position) vs Time ---
    for t in range(T):
        v = valid[t]
        if v.any():
            x = point_clouds[t, v, 0]
            energy = point_clouds[t, v, 4]
            energy_norm = energy / (energy.max() + 1e-12)
            axes[2].scatter(
                np.full_like(x, time_axis[t]), x,
                c=energy_norm, cmap='hot', s=3, alpha=0.7,
                vmin=0, vmax=1,
            )
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('X Position (m)')
    axes[2].set_title('Azimuth-Time Spectrogram (Point Cloud)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Point Cloud Spectrograms\n\"{prompt}\"", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pc_spectrograms.png"), dpi=150)
    plt.close()
    print("[Viz] Saved: pc_spectrograms.png")

    # ==========================================
    # Plot 2: 3D Point Cloud (중간 프레임)
    # ==========================================
    mid = T // 2
    v_mid = valid[mid]
    if v_mid.any():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        pc_mid = point_clouds[mid, v_mid]

        sc = ax.scatter(
            pc_mid[:, 0], pc_mid[:, 2], pc_mid[:, 1],
            c=pc_mid[:, 3], cmap='coolwarm', s=20, edgecolors='k', linewidths=0.3,
        )
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Y (m)')
        ax.set_title(f'3D Point Cloud (frame {mid}/{T}, {v_mid.sum()} valid pts)')
        plt.colorbar(sc, label='Velocity (m/s)', shrink=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pc_3d_midframe.png"), dpi=150)
        plt.close()
        print("[Viz] Saved: pc_3d_midframe.png")

    # ==========================================
    # Plot 3: 3D Point Cloud 다중 프레임 (시간 경과)
    # ==========================================
    num_snapshots = min(6, T)
    snapshot_indices = np.linspace(0, T - 1, num_snapshots, dtype=int)

    fig, axes_list = plt.subplots(
        2, 3, figsize=(18, 10),
        subplot_kw={'projection': '3d'},
    )
    axes_flat = axes_list.flatten()

    for ax_idx, frame_idx in enumerate(snapshot_indices):
        ax = axes_flat[ax_idx]
        v = valid[frame_idx]
        if v.any():
            pc_f = point_clouds[frame_idx, v]
            sc = ax.scatter(
                pc_f[:, 0], pc_f[:, 2], pc_f[:, 1],
                c=pc_f[:, 3], cmap='coolwarm', s=10,
            )
            ax.set_title(f't={time_axis[frame_idx]:.2f}s ({v.sum()} pts)', fontsize=9)
        else:
            ax.set_title(f't={time_axis[frame_idx]:.2f}s (0 pts)', fontsize=9)
        ax.set_xlabel('X', fontsize=7)
        ax.set_ylabel('Z', fontsize=7)
        ax.set_zlabel('Y', fontsize=7)
        ax.tick_params(labelsize=6)

    plt.suptitle(f"Point Cloud Over Time\n\"{prompt}\"", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pc_3d_timeline.png"), dpi=150)
    plt.close()
    print("[Viz] Saved: pc_3d_timeline.png")

    # ==========================================
    # Plot 4: Velocity-Range scatter (중간 프레임)
    # ==========================================
    if v_mid.any():
        pc_mid = point_clouds[mid, v_mid]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Velocity vs Range
        sc0 = axes[0].scatter(
            pc_mid[:, 5], pc_mid[:, 3],
            c=pc_mid[:, 4], cmap='hot', s=20, edgecolors='k', linewidths=0.3,
        )
        axes[0].set_xlabel('Range (m)')
        axes[0].set_ylabel('Velocity (m/s)')
        axes[0].set_title(f'Velocity-Range (frame {mid})')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(sc0, ax=axes[0], label='Energy')

        # X vs Y (top-down view)
        sc1 = axes[1].scatter(
            pc_mid[:, 0], pc_mid[:, 2],
            c=pc_mid[:, 4], cmap='hot', s=20, edgecolors='k', linewidths=0.3,
        )
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        axes[1].set_title(f'Top-Down View (frame {mid})')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(sc1, ax=axes[1], label='Energy')

        plt.suptitle(f"Point Cloud Analysis\n\"{prompt}\"", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pc_analysis.png"), dpi=150)
        plt.close()
        print("[Viz] Saved: pc_analysis.png")

    # ==========================================
    # Plot 5: 프레임별 포인트 수 + 에너지 통계
    # ==========================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].bar(time_axis, pts_per_frame, width=duration / T * 0.8, color='steelblue')
    axes[0].axhline(y=pts_per_frame.mean(), color='red', linestyle='--', label=f'mean={pts_per_frame.mean():.1f}')
    axes[0].set_ylabel('Valid Points')
    axes[0].set_title('Points per Frame')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    mean_energy = []
    for t in range(T):
        v = valid[t]
        if v.any():
            mean_energy.append(point_clouds[t, v, 4].mean())
        else:
            mean_energy.append(0.0)
    axes[1].plot(time_axis, mean_energy, color='darkorange', linewidth=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Mean Energy')
    axes[1].set_title('Mean Point Energy per Frame')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Point Cloud Statistics\n\"{prompt}\"", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pc_statistics.png"), dpi=150)
    plt.close()
    print("[Viz] Saved: pc_statistics.png")

    print(f"[Viz] All visualizations saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate motion and RF Doppler simulation with point cloud visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt for motion generation")
    parser.add_argument("-d", "--duration", type=float, default=3.0, help="Duration in seconds (default: 3.0)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--model-path", default=None, help="Path to HY-Motion model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--points-per-frame", type=int, default=128, help="Points per frame (default: 128)")
    parser.add_argument("--no-visualize", action="store_true", help="Skip video visualization generation")
    parser.add_argument("--skip-motion-gen", action="store_true", help="Skip motion generation (use existing npz)")

    args = parser.parse_args()

    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)

    if args.model_path is None:
        args.model_path = str(HY_MOTION_DIR / "ckpts" / "tencent" / "HY-Motion-1.0")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    motion_npz_path = os.path.join(output_dir, "obj_diff.npz")

    # ── Step 1: Motion generation ──
    if not args.skip_motion_gen:
        model_output = generate_motion_hymotion(
            prompt=args.prompt,
            duration=args.duration,
            model_path=args.model_path,
            device=args.device,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
        convert_hymotion_to_rfgenesis_format(
            model_output=model_output,
            output_path=motion_npz_path,
        )
    else:
        if not os.path.exists(motion_npz_path):
            print(f"[ERROR] Motion file not found: {motion_npz_path}")
            sys.exit(1)

    # ── Step 2: RF simulation + point cloud extraction ──
    radar_frames, point_clouds = run_rf_simulation(
        motion_npz_path=motion_npz_path,
        output_dir=output_dir,
        visualize_output=not args.no_visualize,
        points_per_frame=args.points_per_frame,
    )

    # ── Step 3: Point cloud visualization ──
    generate_doppler_visualization(
        radar_frames=radar_frames,
        point_clouds=point_clouds,
        output_dir=output_dir,
        duration=args.duration,
        prompt=args.prompt,
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"  Prompt:       \"{args.prompt}\"")
    print(f"  Duration:     {args.duration}s")
    print(f"  Radar frames: {radar_frames.shape}")
    print(f"  Point clouds: {point_clouds.shape}")
    valid = np.linalg.norm(point_clouds[:, :, :3], axis=-1) > 1e-6
    pts = valid.sum(axis=1)
    print(f"  Pts/frame:    min={pts.min()}, max={pts.max()}, mean={pts.mean():.1f}")
    print(f"  Output dir:   {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()