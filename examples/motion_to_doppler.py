#!/usr/bin/env python3
"""
HY-Motion + RF-Genesis Integration Example
===========================================
 
This example demonstrates:
1. Generating motion from text using HY-Motion
2. Converting motion to RF signal simulation using RF-Genesis
3. Visualizing RD, RA, DA and Time-Spectrograms (Upgraded)
 
Usage:
    python examples/motion_to_doppler.py \
        --prompt "A person leaps forward and lands in a squat." \
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
    num_input_joints = rot6d.shape[1]
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
        # Fallback omitted for brevity but keeping standard path
        raise ValueError("Unsupported model output format")

    shape_params = np.zeros(10)
    np.savez(
        output_path,
        pose=pose_params,
        shape=shape_params,
        root_translation=translation,
        gender="neutral"
    )
    return output_path
 

def _shape_str(x):
    if isinstance(x, torch.Tensor): return str(tuple(x.shape))
    elif isinstance(x, np.ndarray): return str(x.shape)
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

    smpl_data = np.load(motion_npz_path, allow_pickle=True)
    root_translation = smpl_data['root_translation']
    traj_center = root_translation.mean(axis=0)
    sensor_distance = 3.0
    sensor_origin = [traj_center[0], traj_center[1], traj_center[2] + sensor_distance]
    sensor_target = [traj_center[0], traj_center[1], traj_center[2]]

    print("[RF-Genesis] Step 1/3: Ray tracing body PIRs...")
    original_dir = os.getcwd()
    os.chdir(str(RF_GENESIS_DIR / "genesis"))
    try:
        body_pir, body_aux = pathtracer.trace(motion_npz_path)
    finally:
        os.chdir(original_dir)

    print("[RF-Genesis] Step 2/3: Generating radar signal frames...")
    env_pir = None
    radar_frames = signal_generator.generate_signal_frames(
        body_pir, body_aux, env_pir, radar_config=radar_config_path,
        sensor_origin=sensor_origin, sensor_target=sensor_target,
    )
    radar_output_path = os.path.join(output_dir, "radar_frames.npy")
    np.save(radar_output_path, radar_frames)

    if visualize_output:
        print("[RF-Genesis] Step 3/3: Generating visualization...")
        torch.set_default_device('cpu')
        video_path = os.path.join(output_dir, "output.mp4")
        visualize.save_video(radar_config_path, radar_output_path, motion_npz_path, video_path)

    return radar_frames
 
 
# ============================================================================
# NEW: 3D Radar Cube Generation & Visualization (RD, RA, DA + Spectrograms)
# ============================================================================
def generate_doppler_visualization(
    radar_frames: np.ndarray,
    output_dir: str,
    duration: float,
    prompt: str,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
 
    print("[Viz] Generating RD, RA, DA and Time-Spectrogram visualizations...")
    num_frames = radar_frames.shape[0]
 
    # Lists to hold spectrogram data over time
    doppler_time = []
    range_time = []
    angle_time = []
 
    for i in range(num_frames):
        # frame: (num_tx, num_rx, chirp_per_frame, adc_samples), complex
        frame = radar_frames[i]  
        
        if frame.ndim >= 2 and frame.shape[-1] > 1 and frame.shape[-2] > 1:
            num_tx, num_rx, chirps, adcs = frame.shape
            
            # MIMO 배열 병합: (Virtual Antennas, Chirps, ADCs)
            num_va = num_tx * num_rx
            frame_va = frame.reshape(num_va, chirps, adcs)
            
            # 1. Range FFT
            range_fft = np.fft.fft(frame_va, axis=-1)
            
            # 2. Clutter Removal (MTI) - 시간(Chirp) 축 기준 배경 제거
            mean_clutter = np.mean(range_fft, axis=-2, keepdims=True)
            range_fft_clean = range_fft - mean_clutter
            
            # 3. Doppler FFT
            doppler_fft = np.fft.fftshift(np.fft.fft(range_fft_clean, axis=-2), axes=-2)
            
            # 4. Angle FFT (가상 안테나 축 기준) - Zero padding으로 부드러운 각도 맵 생성
            num_angle_bins = 64
            angle_fft = np.fft.fftshift(np.fft.fft(doppler_fft, n=num_angle_bins, axis=0), axes=0)
            
            # 3D Radar Cube (Angle, Doppler, Range)의 Magnitude
            cube_abs = np.abs(angle_fft)
        else:
            # 예외 처리: 데이터가 올바르지 않은 경우
            cube_abs = np.zeros((64, frame.shape[-2], frame.shape[-1]))
 
        # Extract features for Time-Spectrograms (Max pooling을 통해 peak 보존)
        # Doppler-Time: Angle, Range에 대해 Max
        doppler_time.append(np.max(cube_abs, axis=(0, 2)))
        # Range-Time: Angle, Doppler에 대해 Max
        range_time.append(np.max(cube_abs, axis=(0, 1)))
        # Angle-Time: Doppler, Range에 대해 Max
        angle_time.append(np.max(cube_abs, axis=(1, 2)))
        
        # 중간 프레임의 단일 RD, RA, DA 추출 (시각화용)
        if i == num_frames // 2:
            mid_rd = np.max(cube_abs, axis=0) # Shape: (Doppler, Range)
            mid_ra = np.max(cube_abs, axis=1) # Shape: (Angle, Range)
            mid_da = np.max(cube_abs, axis=2) # Shape: (Angle, Doppler)

    # 리스트를 배열로 변환 및 dB 스케일링
    def to_db(arr):
        arr = np.array(arr)
        arr_db = 20 * np.log10(arr + 1e-12)
        return arr_db - np.max(arr_db) # Normalize to 0 dB max

    doppler_time_db = to_db(doppler_time).T  # Shape: (Doppler, Time)
    range_time_db = to_db(range_time).T      # Shape: (Range, Time)
    angle_time_db = to_db(angle_time).T      # Shape: (Angle, Time)
    
    mid_rd_db = 20 * np.log10(mid_rd + 1e-12)
    mid_ra_db = 20 * np.log10(mid_ra + 1e-12)
    mid_da_db = 20 * np.log10(mid_da + 1e-12)
 
    # ==========================================
    # Plot 1: Middle Frame RD, RA, DA Maps
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(mid_rd_db, aspect='auto', cmap='jet', origin='lower', vmin=np.max(mid_rd_db)-60)
    axes[0].set_title(f"Range-Doppler (Mid Frame)")
    axes[0].set_xlabel('Range Bin')
    axes[0].set_ylabel('Doppler Bin')
    plt.colorbar(im0, ax=axes[0], label='dB')

    im1 = axes[1].imshow(mid_ra_db, aspect='auto', cmap='jet', origin='lower', vmin=np.max(mid_ra_db)-60)
    axes[1].set_title(f"Range-Angle (Mid Frame)")
    axes[1].set_xlabel('Range Bin')
    axes[1].set_ylabel('Angle Bin')
    plt.colorbar(im1, ax=axes[1], label='dB')

    im2 = axes[2].imshow(mid_da_db, aspect='auto', cmap='jet', origin='lower', vmin=np.max(mid_da_db)-60)
    axes[2].set_title(f"Doppler-Angle (Mid Frame)")
    axes[2].set_xlabel('Doppler Bin')
    axes[2].set_ylabel('Angle Bin')
    plt.colorbar(im2, ax=axes[2], label='dB')

    plt.suptitle(f"Single Frame Features\nPrompt: {prompt}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frame_rd_ra_da_maps.png"), dpi=150)
    plt.close()

    # ==========================================
    # Plot 2: Time Spectrograms (Micro-Doppler, Range-Time, Angle-Time)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    im0 = axes[0].imshow(doppler_time_db, aspect='auto', cmap='jet', origin='lower', extent=[0, duration, 0, doppler_time_db.shape[0]], vmin=-60, vmax=0)
    axes[0].set_title(f"Micro-Doppler Spectrogram (Time vs Doppler)")
    axes[0].set_ylabel('Doppler Bin')
    
    im1 = axes[1].imshow(range_time_db, aspect='auto', cmap='jet', origin='lower', extent=[0, duration, 0, range_time_db.shape[0]], vmin=-60, vmax=0)
    axes[1].set_title(f"Range-Time Spectrogram")
    axes[1].set_ylabel('Range Bin')
    
    im2 = axes[2].imshow(angle_time_db, aspect='auto', cmap='jet', origin='lower', extent=[0, duration, 0, angle_time_db.shape[0]], vmin=-60, vmax=0)
    axes[2].set_title(f"Angle-Time Spectrogram")
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angle Bin')

    plt.suptitle(f"Time-Evolution Spectrograms\nPrompt: {prompt}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_spectrograms.png"), dpi=150)
    plt.close()
 
    print(f"[Viz] Saved visualizations to {output_dir}")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Generate motion and RF Doppler simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt for motion generation")
    parser.add_argument("-d", "--duration", type=float, default=3.0, help="Duration in seconds (default: 3.0)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--model-path", default=None, help="Path to HY-Motion model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization generation")
    parser.add_argument("--skip-motion-gen", action="store_true", help="Skip motion generation")
 
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
 
    if not args.skip_motion_gen:
        model_output = generate_motion_hymotion(
            prompt=args.prompt, duration=args.duration, model_path=args.model_path,
            device=args.device, cfg_scale=args.cfg_scale, seed=args.seed,
        )
        convert_hymotion_to_rfgenesis_format(model_output=model_output, output_path=motion_npz_path)
    else:
        if not os.path.exists(motion_npz_path):
            sys.exit(1)
 
    radar_frames = run_rf_simulation(
        motion_npz_path=motion_npz_path, output_dir=output_dir,
        visualize_output=not args.no_visualize,
    )
 
    if not args.no_visualize:
        generate_doppler_visualization(
            radar_frames=radar_frames, output_dir=output_dir,
            duration=args.duration, prompt=args.prompt,
        )
 
if __name__ == "__main__":
    main()