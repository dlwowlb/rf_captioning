#!/usr/bin/env python3
"""
RadarLLM 데이터셋 생성 (모션 데이터 포함 버전)
================================================

논문 Figure 4의 학습 파이프라인:

  Radar Point Cloud ──→ Encoder ──→ F_group ──→ Mask ──→ F_all
                                                           │
  Paired SMPL Motion ──→ Motion Encoder ──────────→ F_mot  │
                                                     │     │
                                              L_emb = ||F_all - F_mot||²

따라서 각 샘플에 저장해야 하는 것:
  1. point_cloud:    (T_radar, 128, 6) — 레이더 포인트 클라우드
  2. text:           str               — 텍스트 설명
  3. motion_joints:  (T_motion, 22, 3) — SMPL joint positions (모션 인코더 입력)
  4. motion_rot6d:   (T_motion, 22, 6) — 6D rotations (모션 인코더 입력 대안)
  5. motion_transl:  (T_motion, 3)     — root translation

사용법:
  python scripts/generate_pt.py \
      --mode hymotion \
      --prompts_json HY-Motion-1.0/examples/objgpt3000.json \
      --hymotion_config HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0/config.yml --hymotion_ckpt HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0/latest.ckpt \
      --radar_config RF-Genesis/models/TI1843_config.json \
      --output_dir data/radar_text_dataset/New_preprocessing
      


"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

def find_project_root():
    d = SCRIPT_DIR
    for _ in range(5):
        if (d / "HY-Motion-1.0").exists():
            return d
        d = d.parent
    return SCRIPT_DIR.parent

PROJECT_ROOT = find_project_root()
HY_MOTION_DIR = PROJECT_ROOT / "HY-Motion-1.0"
RF_GENESIS_DIR = PROJECT_ROOT / "RF-Genesis"

for p in [HY_MOTION_DIR, RF_GENESIS_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ============================================================
# SMPL-24 변환 함수 (기존과 동일)
# ============================================================

def rot6d_to_rotation_matrix(rot6d):
    import torch.nn.functional as F
    if isinstance(rot6d, np.ndarray):
        rot6d = torch.from_numpy(rot6d).float()
    x = rot6d.view(*rot6d.shape[:-1], 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("...i,...i->...", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_matrix_to_axis_angle(rot_mat):
    from scipy.spatial.transform import Rotation
    orig_shape = rot_mat.shape[:-2]
    flat = rot_mat.reshape(-1, 3, 3)
    if isinstance(flat, torch.Tensor):
        flat = flat.cpu().numpy()
    return Rotation.from_matrix(flat).as_rotvec().reshape(*orig_shape, 3)


def convert_hymotion_output_to_smpl24(rot6d, transl):
    if isinstance(rot6d, torch.Tensor):
        rot6d = rot6d.cpu()
    if isinstance(transl, torch.Tensor):
        transl = transl.cpu().numpy()
    if rot6d.ndim == 4:
        rot6d = rot6d[0]
    if isinstance(transl, np.ndarray) and transl.ndim == 3:
        transl = transl[0]
    if isinstance(rot6d, torch.Tensor):
        rot6d_tensor = rot6d.float()
    else:
        rot6d_tensor = torch.from_numpy(rot6d).float()

    num_frames = rot6d_tensor.shape[0]
    body_rot6d = rot6d_tensor[:, :22, :]
    rot_matrices = rot6d_to_rotation_matrix(body_rot6d)
    body_aa = rotation_matrix_to_axis_angle(rot_matrices)
    hand_aa = np.zeros((num_frames, 2, 3), dtype=np.float64)
    smpl_aa = np.concatenate([body_aa, hand_aa], axis=1)
    pose = smpl_aa.reshape(num_frames, -1).astype(np.float32)

    return {
        "pose": pose,
        "shape": np.zeros(10, dtype=np.float32),
        "root_translation": transl.astype(np.float32) if isinstance(transl, np.ndarray) else transl,
        "gender": "neutral",
    }


# ============================================================
# ★ 핵심 추가: HY-Motion 출력에서 모션 데이터 추출
# ============================================================

def extract_motion_data(model_output: dict) -> dict:
    """
    HY-Motion 출력에서 VQ-VAE L_emb 학습에 필요한 모션 데이터를 추출.

    논문 Section 4.2:
      "motion semantic features extracted from the paired motion
       by a motion encoder, enriching semantics learned efficiently"

    모션 인코더의 입력으로 사용할 수 있는 데이터:
      1. rot6d:      (L, 22, 6) — joint rotations in 6D representation
      2. transl:     (L, 3)     — root translation
      3. keypoints3d:(L, J, 3)  — 3D joint positions (forward kinematics 결과)
      4. latent:     (L, 201)   — HY-Motion의 denormalized latent (rot6d + transl 합친 것)

    latent_denorm(201D) 구성:
      [0:3]     root translation
      [3:9]     root rotation 6D
      [9:135]   21 body joints × 6D rotation = 126D
      [135:201] 기타 features
    """
    motion_data = {}

    # ── rot6d: (B, L, J, 6) → (L, J, 6) ──
    rot6d = model_output.get("rot6d")
    if rot6d is not None:
        if isinstance(rot6d, torch.Tensor):
            rot6d = rot6d.cpu().numpy()
        if rot6d.ndim == 4:
            rot6d = rot6d[0]  # batch dim 제거
        motion_data["motion_rot6d"] = rot6d.astype(np.float32)
        # shape: (L, 22, 6) 또는 (L, 52, 6)

    # ── transl: (B, L, 3) → (L, 3) ──
    transl = model_output.get("transl")
    if transl is not None:
        if isinstance(transl, torch.Tensor):
            transl = transl.cpu().numpy()
        if transl.ndim == 3:
            transl = transl[0]
        motion_data["motion_transl"] = transl.astype(np.float32)

    # ── keypoints3d: (B, L, J, 3) → (L, J, 3) ──
    # Forward kinematics 결과 — 실제 3D 공간에서의 joint 위치
    # 모션 인코더 입력으로 가장 직관적
    k3d = model_output.get("keypoints3d")
    if k3d is not None:
        if isinstance(k3d, torch.Tensor):
            k3d = k3d.cpu().numpy()
        if k3d.ndim == 4:
            k3d = k3d[0]
        motion_data["motion_joints"] = k3d.astype(np.float32)
        # shape: (L, 22, 3) 또는 (L, 52, 3)

    # ── latent_denorm: (B, L, 201) → (L, 201) ──
    # HY-Motion의 내부 표현 — rot6d + transl이 합쳐진 형태
    # 이것도 모션 인코더 입력으로 사용 가능
    latent = model_output.get("latent_denorm")
    if latent is not None:
        if isinstance(latent, torch.Tensor):
            latent = latent.cpu().numpy()
        if latent.ndim == 3:
            latent = latent[0]
        motion_data["motion_latent"] = latent.astype(np.float32)
        # shape: (L, 201)

    return motion_data


# ============================================================
# RF-Genesis 시뮬레이션 (기존과 동일, 절대 경로 수정 포함)
# ============================================================

def run_rfgenesis_simulation(smpl_npz_path, radar_config_path, points_per_frame=128):
    smpl_npz_path = os.path.abspath(smpl_npz_path)
    radar_config_path = os.path.abspath(radar_config_path)

    from genesis.raytracing import pathtracer, signal_generator
    from genesis.raytracing.radar import Radar
    from genesis.visualization.pointcloud import PointCloudProcessCFG, process_pc

    smpl_data = np.load(smpl_npz_path, allow_pickle=True)
    root_translation = smpl_data["root_translation"]
    traj_center = root_translation.mean(axis=0)
    sensor_distance = 3.0
    sensor_origin = [float(traj_center[0]), float(traj_center[1]),
                     float(traj_center[2] + sensor_distance)]
    sensor_target = [float(traj_center[0]), float(traj_center[1]),
                     float(traj_center[2])]

    original_dir = os.getcwd()
    os.chdir(str(RF_GENESIS_DIR / "genesis"))
    try:
        body_pirs, body_auxs = pathtracer.trace(smpl_npz_path)
    finally:
        os.chdir(original_dir)

    radar_frames = signal_generator.generate_signal_frames(
        body_pirs, body_auxs, None,
        radar_config=radar_config_path,
        sensor_origin=sensor_origin, sensor_target=sensor_target,
    )

    radar = Radar(radar_config_path)
    pc_cfg = PointCloudProcessCFG(radar)

    point_clouds = []
    for frame in radar_frames:
        pc = process_pc(pc_cfg, frame)
        if len(pc) == 0:
            pc_padded = np.zeros((points_per_frame, 6), dtype=np.float32)
        elif len(pc) >= points_per_frame:
            top_idx = np.argsort(-pc[:, 4])[:points_per_frame]
            pc_padded = pc[top_idx]
        else:
            pc_padded = np.zeros((points_per_frame, 6), dtype=np.float32)
            pc_padded[:len(pc)] = pc
            ########제로 패딩 대신 랜덤으로 채워넣기##########
            #dup_idx = np.random.choice(len(pc), points_per_frame - len(pc), replace=True)
            #pc_padded[len(pc):] = pc[dup_idx]
        point_clouds.append(pc_padded)

    return np.array(point_clouds, dtype=np.float32), radar_frames


# ============================================================
# 데이터셋 생성: HY-Motion 모드
# ============================================================

def generate_from_hymotion(args):
    from hymotion.utils.t2m_runtime import T2MRuntime

    with open(args.prompts_json, "r") as f:
        prompts_data = json.load(f)

    prompts = []
    for entry in prompts_data["test_prompts_subset"]:
        parts = entry.split("#")
        prompts.append({
            "text": parts[0].strip(),
            "duration": int(parts[1]) / 30.0 if len(parts) > 1 else 3.0,
            "num_frames": int(parts[1]) if len(parts) > 1 else 90,
            "id": parts[2] if len(parts) > 2 else "000",
        })

    print(f"프롬프트 {len(prompts)}개 로드")

    runtime = T2MRuntime(
        config_path=args.hymotion_config,
        ckpt_name=args.hymotion_ckpt,
        device_ids=[0] if torch.cuda.is_available() else None,
        disable_prompt_engineering=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.abspath(os.path.join(args.output_dir, "_temp"))
    os.makedirs(temp_dir, exist_ok=True)
    index = []

    for i, prompt in enumerate(tqdm(prompts, desc="Generating dataset")):
        try:
            print(f"\n[{i+1}/{len(prompts)}] '{prompt['text']}'")

            # ── Step 1: HY-Motion 모션 생성 ──
            _, _, model_output = runtime.generate_motion(
                text=prompt["text"],
                seeds_csv="42",
                duration=prompt["duration"],
                cfg_scale=5.0,
                output_format="dict",
            )
            # model_output keys:
            #   rot6d:         (B, L, 22, 6)
            #   transl:        (B, L, 3)
            #   keypoints3d:   (B, L, J, 3)
            #   latent_denorm: (B, L, 201)

            # ── Step 2: SMPL-24 변환 (RF-Genesis 입력용) ──
            smpl_data = convert_hymotion_output_to_smpl24(
                model_output["rot6d"], model_output["transl"]
            )

            sample_id = int(prompt["id"]) if str(prompt["id"]).isdigit() else i
            temp_npz = os.path.join(temp_dir, f"motion_{sample_id:06d}.npz")

            #temp_npz = os.path.join(temp_dir, f"motion_{i:04d}.npz")
            np.savez(temp_npz, **smpl_data)

            # ── Step 3-5: RF-Genesis 레이더 시뮬레이션 ──
            point_clouds, _ = run_rfgenesis_simulation(
                smpl_npz_path=temp_npz,
                radar_config_path=args.radar_config,
            )

            # ── ★ 모션 데이터 추출 (L_emb용) ──
            motion_data = extract_motion_data(model_output)

            # ── Step 6: 저장 ──
            save_dict = {
                # 레이더 데이터 (tokenizer 입력)
                "point_cloud": point_clouds,                   # (T_radar, 128, 6)
                # 텍스트 (LM 타겟)
                "text": prompt["text"],
                # ★ 모션 데이터 (L_emb용 — motion encoder 입력)
                **motion_data,
                # 메타데이터
                "num_radar_frames": point_clouds.shape[0],
                "num_motion_frames": prompt["num_frames"],
            }

            #sample_id = int(prompt["id"]) if str(prompt["id"]).isdigit() else i
            #sample_path = os.path.join(args.output_dir, f"sample_{sample_id:06d}.npz")
            sample_path = os.path.join(args.output_dir, f"sample_{sample_id:06d}.npz")
            
            #sample_path = os.path.join(args.output_dir, f"sample_{i:06d}.npz")
            np.savez_compressed(sample_path, **save_dict)

            # 저장 내용 확인 출력
            print(f"  Saved: point_cloud={point_clouds.shape}", end="")
            for k, v in motion_data.items():
                print(f", {k}={v.shape}", end="")
            print()

            index.append({
                "path": os.path.abspath(sample_path),
                "text": prompt["text"],
                "id": prompt["id"],
                "num_radar_frames": int(point_clouds.shape[0]),
                "has_motion_joints": "motion_joints" in motion_data,
                "has_motion_rot6d": "motion_rot6d" in motion_data,
                "has_motion_latent": "motion_latent" in motion_data,
            })

            os.remove(temp_npz)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback; traceback.print_exc()
            continue

    # 인덱스 저장
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"\n완료: {len(index)}개 샘플 → {args.output_dir}")
    print(f"인덱스: {index_path}")

    # 검증
    if index:
        verify_dataset(args.output_dir)


# ============================================================
# 데이터셋 검증
# ============================================================

def verify_dataset(output_dir, num_samples=3):
    print(f"\n{'='*60}")
    print("데이터셋 검증")
    print(f"{'='*60}")

    npz_files = sorted(Path(output_dir).glob("sample_*.npz"))
    print(f"총 샘플 수: {len(npz_files)}")

    for f in npz_files[:num_samples]:
        data = np.load(f, allow_pickle=True)
        print(f"\n  {f.name}:")
        print(f"    text: \"{str(data['text'])[:80]}\"")

        # 레이더 데이터
        pc = data["point_cloud"]
        print(f"    point_cloud:    {pc.shape}  (T_radar, 128, 6)")

        # 모션 데이터 확인
        for key in ["motion_joints", "motion_rot6d", "motion_transl", "motion_latent"]:
            if key in data:
                print(f"    {key:18s}: {data[key].shape}")
            else:
                print(f"    {key:18s}: ✗ 없음")

    # 레이더/모션 프레임 수 비교
    if len(npz_files) > 0:
        data = np.load(npz_files[0], allow_pickle=True)
        T_radar = data["point_cloud"].shape[0]
        T_motion = data["motion_joints"].shape[0] if "motion_joints" in data else "?"
        print(f"\n  프레임 비율: motion({T_motion}) → radar({T_radar})")
        print(f"  (모션 30fps → 레이더 10fps, 약 3:1 비율)")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RadarLLM 데이터셋 생성 (모션 포함)")
    parser.add_argument("--mode", default="hymotion", choices=["hymotion", "humanml3d"])
    parser.add_argument("--output_dir", default="data/radar_text_dataset")
    parser.add_argument("--radar_config", default=None)
    parser.add_argument("--prompts_json", default=None)
    parser.add_argument("--hymotion_config", default=None)
    parser.add_argument("--hymotion_ckpt", default=None)
    parser.add_argument("--humanml3d_root", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.radar_config is None:
        default = RF_GENESIS_DIR / "models" / "TI1843_config.json"
        if default.exists():
            args.radar_config = str(default)

    if args.mode == "hymotion":
        generate_from_hymotion(args)
