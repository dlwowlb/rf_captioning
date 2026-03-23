#!/usr/bin/env python3
"""
기존 데이터셋에 HumanML3D 263D representation 추가.

기존 npz에 이미 저장된 motion_joints, motion_rot6d, motion_transl을 이용해
motion_humanml3d (263D) 를 계산하여 같은 npz에 덮어쓰기.

사용법:
  python scripts/add_humanml3d.py --data_dir data/radar_text_dataset/gpt_act_simple_263d
  python scripts/add_humanml3d.py --data_dir data/radar_text_dataset/train
  python scripts/add_humanml3d.py --data_dir data/radar_text_dataset --verify
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.humanml3d_repr import convert_to_humanml3d


def convert_dataset(data_dir: str, verify: bool = False):
    data_dir = Path(data_dir)

    # train/val/test 하위 폴더가 있으면 전부 처리
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name in ("train", "val", "test")]
    if subdirs:
        all_files = []
        for sd in subdirs:
            all_files.extend(sorted(sd.glob("*.npz")))
    else:
        all_files = sorted(data_dir.glob("*.npz"))

    if not all_files:
        print(f"ERROR: {data_dir}에 *.npz 없음")
        return

    print(f"변환 대상: {len(all_files)}개 파일")

    # 첫 파일 확인
    first = np.load(all_files[0], allow_pickle=True)
    print(f"기존 키: {list(first.files)}")
    if "motion_humanml3d" in first:
        print("⚠ motion_humanml3d가 이미 존재합니다. 덮어쓰기합니다.")

    has_joints = "motion_joints" in first
    has_rot6d = "motion_rot6d" in first
    has_transl = "motion_transl" in first
    print(f"  motion_joints: {'✓' if has_joints else '✗'}")
    print(f"  motion_rot6d:  {'✓' if has_rot6d else '✗'}")
    print(f"  motion_transl: {'✓' if has_transl else '✗'}")

    if not has_joints:
        print("ERROR: motion_joints가 없으면 263D 변환 불가!")
        print("  generate_pt.py를 다시 실행하여 motion_joints를 포함해야 합니다.")
        return

    success = 0
    fail = 0

    for npz_path in tqdm(all_files, desc="Adding HumanML3D 263D"):
        try:
            data = dict(np.load(npz_path, allow_pickle=True))

            joints = data["motion_joints"]           # (T, 22, 3) or (T, 52, 3)
            rot6d = data.get("motion_rot6d", None)   # (T, 22, 6) or None
            transl = data.get("motion_transl", None) # (T, 3) or None

            # joints가 52 관절이면 22개만 사용
            if joints.ndim == 3 and joints.shape[1] > 22:
                joints = joints[:, :22, :]
            if rot6d is not None and rot6d.ndim == 3 and rot6d.shape[1] > 22:
                rot6d = rot6d[:, :22, :]

            humanml3d = convert_to_humanml3d(
                joints=joints.astype(np.float32),
                rot6d=rot6d.astype(np.float32) if rot6d is not None else None,
                transl=transl.astype(np.float32) if transl is not None else None,
            )

            # 기존 데이터에 추가
            data["motion_humanml3d"] = humanml3d.astype(np.float32)

            # 덮어쓰기 저장
            np.savez_compressed(str(npz_path), **data)
            success += 1

        except Exception as e:
            print(f"\n  ✗ {npz_path.name}: {e}")
            fail += 1
            continue

    print(f"\n완료: {success}개 성공, {fail}개 실패")

    # 검증
    if verify and success > 0:
        print(f"\n{'='*60}")
        print("검증")
        print(f"{'='*60}")
        for npz_path in all_files[:3]:
            data = np.load(npz_path, allow_pickle=True)
            if "motion_humanml3d" in data:
                h = data["motion_humanml3d"]
                print(f"  {npz_path.name}:")
                print(f"    motion_humanml3d: {h.shape}")
                print(f"    range: [{h.min():.3f}, {h.max():.3f}]")
                print(f"    NaN: {np.isnan(h).any()}, Inf: {np.isinf(h).any()}")
                # 각 feature 블록 확인
                print(f"    r_velocity  [0:1]:    std={h[:, 0].std():.4f}")
                print(f"    l_velocity  [1:3]:    std={h[:, 1:3].std():.4f}")
                print(f"    root_y      [3:4]:    mean={h[:, 3].mean():.3f}")
                print(f"    ric_data    [4:67]:   std={h[:, 4:67].std():.4f}")
                print(f"    local_vel   [67:133]: std={h[:, 67:133].std():.4f}")
                print(f"    rot_data    [133:259]:std={h[:, 133:259].std():.4f}")
                print(f"    foot_contact[259:]:   mean={h[:, 259:].mean():.3f}")
            else:
                print(f"  ✗ {npz_path.name}: motion_humanml3d 없음")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_dataset(args.data_dir, args.verify)