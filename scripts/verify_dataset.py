#!/usr/bin/env python3
"""
RadarLLM 데이터셋 검증 스크립트

체크하는 항목:
  1. 파일 무결성: npz 로드 가능, 필수 키 존재
  2. 포인트클라우드 품질: 좌표 범위, 빈 프레임, 값 분포
  3. 모션 데이터 품질: shape, 범위, NaN/Inf
  4. 텍스트 품질: 빈 문장, 중복, 장소/물체 포함 여부
  5. 레이더-모션 시간 일치: 프레임 비율 확인
  6. 시각화: 포인트클라우드 산점도 + 시간별 centroid 궤적

사용법:
  python scripts/verify_dataset.py --data_dir data/radar_text_dataset
  python scripts/verify_dataset.py --data_dir data/radar_text_dataset/train --visualize
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="데이터셋 검증")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_check", type=int, default=None,
                        help="최대 검사 샘플 수 (None=전체)")
    parser.add_argument("--visualize", action="store_true",
                        help="포인트클라우드 시각화 저장")
    parser.add_argument("--output", default="dataset_report.json")
    return parser.parse_args()


# ============================================================
# 1. 파일 무결성
# ============================================================

def check_file_integrity(files):
    print("\n[1] 파일 무결성 검사")
    errors = []
    required_keys = ["point_cloud", "text"]
    optional_keys = ["motion_latent", "motion_joints", "motion_rot6d", "motion_transl"]

    key_stats = Counter()

    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            for k in required_keys:
                if k not in data:
                    errors.append(f"  ✗ {f.name}: '{k}' 없음")
            for k in data.files:
                key_stats[k] += 1
        except Exception as e:
            errors.append(f"  ✗ {f.name}: 로드 실패 - {e}")

    if errors:
        for e in errors[:10]:
            print(e)
        print(f"  총 {len(errors)}개 에러")
    else:
        print(f"  ✓ {len(files)}개 파일 정상")

    print(f"  키 분포:")
    for k, v in sorted(key_stats.items()):
        pct = v / len(files) * 100
        required = "★" if k in required_keys else " "
        print(f"    {required} {k}: {v}/{len(files)} ({pct:.0f}%)")

    has_motion = any(k in key_stats for k in optional_keys)
    if not has_motion:
        print(f"  ⚠ 모션 데이터 없음! L_emb 학습 불가")

    return len(errors) == 0


# ============================================================
# 2. 포인트클라우드 품질
# ============================================================

def check_point_cloud(files):
    print("\n[2] 포인트클라우드 품질 검사")

    all_x, all_y, all_z = [], [], []
    empty_frames = 0
    total_frames = 0
    shapes = Counter()
    issues = []

    for f in files:
        data = np.load(f, allow_pickle=True)
        pc = data["point_cloud"]
        shapes[pc.shape[1:]] += 1

        for t in range(pc.shape[0]):
            total_frames += 1
            frame = pc[t]
            xyz = frame[:, :3]

            if np.abs(xyz).sum() < 1e-6:
                empty_frames += 1
                continue

            valid = np.linalg.norm(xyz, axis=1) > 1e-6
            if valid.sum() == 0:
                empty_frames += 1
                continue

            all_x.extend(xyz[valid, 0].tolist())
            all_y.extend(xyz[valid, 1].tolist())
            all_z.extend(xyz[valid, 2].tolist())

        if np.any(np.isnan(pc)) or np.any(np.isinf(pc)):
            issues.append(f"  ✗ {f.name}: NaN/Inf 포함!")

    all_x, all_y, all_z = np.array(all_x), np.array(all_y), np.array(all_z)

    print(f"  프레임 shape 분포: {dict(shapes)}")
    print(f"  총 프레임: {total_frames}, 빈 프레임: {empty_frames} ({empty_frames/max(total_frames,1)*100:.1f}%)")

    if len(all_x) > 0:
        print(f"  x 범위: [{all_x.min():.3f}, {all_x.max():.3f}], mean={all_x.mean():.3f}, std={all_x.std():.3f}")
        print(f"  y 범위: [{all_y.min():.3f}, {all_y.max():.3f}], mean={all_y.mean():.3f}, std={all_y.std():.3f}")
        print(f"  z 범위: [{all_z.min():.3f}, {all_z.max():.3f}], mean={all_z.mean():.3f}, std={all_z.std():.3f}")

        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min())
        if max_range > 20:
            print(f"  ⚠ 좌표 범위가 매우 넓음 ({max_range:.1f}m) — 센서 설정 확인 필요")
        if max_range < 0.1:
            print(f"  ⚠ 좌표 범위가 매우 좁음 ({max_range:.3f}m) — 데이터가 축소된 상태?")

        if empty_frames / max(total_frames, 1) > 0.3:
            print(f"  ⚠ 빈 프레임 비율이 30% 이상 — RF-Genesis 시뮬레이션 확인 필요")
    else:
        print(f"  ✗ 유효한 포인트가 없음!")

    for issue in issues:
        print(issue)

    return {
        "total_frames": total_frames,
        "empty_frames": empty_frames,
        "x_range": [float(all_x.min()), float(all_x.max())] if len(all_x) > 0 else None,
        "y_range": [float(all_y.min()), float(all_y.max())] if len(all_y) > 0 else None,
        "z_range": [float(all_z.min()), float(all_z.max())] if len(all_z) > 0 else None,
    }


# ============================================================
# 3. 모션 데이터 품질
# ============================================================

def check_motion_data(files):
    print("\n[3] 모션 데이터 품질 검사")

    for key in ["motion_latent", "motion_joints", "motion_rot6d", "motion_transl"]:
        shapes = []
        has_nan = 0
        has_inf = 0
        value_ranges = []

        for f in files:
            data = np.load(f, allow_pickle=True)
            if key not in data:
                continue
            m = data[key]
            shapes.append(m.shape)
            if np.any(np.isnan(m)):
                has_nan += 1
            if np.any(np.isinf(m)):
                has_inf += 1
            value_ranges.append((m.min(), m.max(), m.mean(), m.std()))

        if not shapes:
            print(f"  {key}: 없음")
            continue

        shape_counter = Counter([str(s) for s in shapes])
        most_common = shape_counter.most_common(1)[0]
        vr = np.array(value_ranges)

        print(f"  {key}:")
        print(f"    shape: {most_common[0]} ({most_common[1]}/{len(shapes)}개)")
        print(f"    값 범위: [{vr[:,0].min():.3f}, {vr[:,1].max():.3f}]")
        print(f"    mean: {vr[:,2].mean():.3f}, std: {vr[:,3].mean():.3f}")
        if has_nan:
            print(f"    ✗ NaN 포함: {has_nan}개 파일")
        if has_inf:
            print(f"    ✗ Inf 포함: {has_inf}개 파일")


# ============================================================
# 4. 텍스트 품질
# ============================================================

def check_text_quality(files):
    print("\n[4] 텍스트 품질 검사")

    texts = []
    empty = 0
    has_location = []
    has_object = []

    location_words = {"kitchen", "park", "gym", "beach", "office", "room",
                      "garage", "garden", "stage", "court", "store", "café",
                      "shelter", "gallery", "lake", "mountain", "trail"}
    object_words = {"basketball", "piano", "guitar", "book", "cup", "chair",
                    "table", "ball", "sword", "shield", "gun", "phone",
                    "bicycle", "car", "canvas", "furniture", "scarf",
                    "tray", "barbell", "tightrope", "wok"}

    for f in files:
        data = np.load(f, allow_pickle=True)
        text = str(data["text"])
        texts.append(text)

        if not text.strip():
            empty += 1

        words = set(text.lower().split())
        loc = words & location_words
        obj = words & object_words
        if loc:
            has_location.append((f.name, text, loc))
        if obj:
            has_object.append((f.name, text, obj))

    lengths = [len(t.split()) for t in texts]
    unique_texts = len(set(texts))
    print(f"  총 텍스트: {len(texts)}")
    print(f"  고유 텍스트: {unique_texts} ({unique_texts/max(len(texts),1)*100:.0f}%)")
    print(f"  빈 텍스트: {empty}")
    print(f"  단어 수: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")

    text_counts = Counter(texts)
    top_dup = text_counts.most_common(5)
    if top_dup[0][1] > 1:
        print(f"  가장 많은 중복:")
        for text, count in top_dup[:3]:
            if count > 1:
                print(f"    \"{text[:60]}\" × {count}")

    if has_location:
        print(f"\n  ⚠ 장소 언급 {len(has_location)}개 (레이더에 안 보임!):")
        for name, text, words in has_location[:5]:
            print(f"    {name}: \"{text[:60]}\" ← {words}")

    if has_object:
        print(f"\n  ⚠ 물체 언급 {len(has_object)}개 (레이더에 안 보임!):")
        for name, text, words in has_object[:5]:
            print(f"    {name}: \"{text[:60]}\" ← {words}")

    if not has_location and not has_object:
        print(f"  ✓ 장소/물체 언급 없음 — 순수 신체 동작 텍스트")


# ============================================================
# 5. 레이더-모션 시간 일치
# ============================================================

def check_temporal_alignment(files):
    print("\n[5] 레이더-모션 시간 일치 검사")

    ratios = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        T_radar = data["point_cloud"].shape[0]

        T_motion = None
        for key in ["motion_latent", "motion_joints", "motion_rot6d"]:
            if key in data:
                T_motion = data[key].shape[0]
                break

        if T_motion is not None:
            ratio = T_motion / max(T_radar, 1)
            ratios.append(ratio)

    if ratios:
        ratios = np.array(ratios)
        print(f"  모션/레이더 프레임 비율:")
        print(f"    mean={ratios.mean():.2f}, min={ratios.min():.2f}, max={ratios.max():.2f}")
        print(f"    기대값: ~3.0 (모션 30fps / 레이더 10fps)")
        if ratios.mean() < 2 or ratios.mean() > 5:
            print(f"    ⚠ 비율이 비정상적 — fps 설정 확인 필요")
        else:
            print(f"    ✓ 정상 범위")
    else:
        print(f"  모션 데이터 없어서 확인 불가")


# ============================================================
# 6. 시각화
# ============================================================

def visualize_samples(files, output_dir="dataset_vis"):
    print(f"\n[6] 시각화 → {output_dir}/")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  ✗ pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    for f in files[:5]:
        data = np.load(f, allow_pickle=True)
        pc = data["point_cloud"]
        text = str(data["text"])

        fig = plt.figure(figsize=(16, 5))

        # 포인트클라우드 3D 산점도 (중간 프레임)
        ax1 = fig.add_subplot(131, projection="3d")
        mid = pc.shape[0] // 2
        frame = pc[mid]
        valid = np.linalg.norm(frame[:, :3], axis=1) > 1e-6
        if valid.sum() > 0:
            pts = frame[valid]
            ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5,
                       c=pts[:, 3] if pts.shape[1] > 3 else "blue")
        ax1.set_title(f"Frame {mid}/{pc.shape[0]}")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        # Centroid 궤적
        ax2 = fig.add_subplot(132)
        centroids = []
        for t in range(pc.shape[0]):
            xyz = pc[t, :, :3]
            valid = np.linalg.norm(xyz, axis=1) > 1e-6
            if valid.sum() > 0:
                centroids.append(xyz[valid].mean(axis=0))
            else:
                centroids.append([0, 0, 0])
        centroids = np.array(centroids)
        ax2.plot(centroids[:, 0], label="x")
        ax2.plot(centroids[:, 1], label="y")
        ax2.plot(centroids[:, 2], label="z")
        ax2.legend()
        ax2.set_title("Centroid trajectory")
        ax2.set_xlabel("Frame")

        # 프레임별 유효 점 수
        ax3 = fig.add_subplot(133)
        counts = []
        for t in range(pc.shape[0]):
            valid = np.linalg.norm(pc[t, :, :3], axis=1) > 1e-6
            counts.append(valid.sum())
        ax3.plot(counts)
        ax3.set_title("Valid points per frame")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Count")
        ax3.axhline(y=128, color="r", linestyle="--", label="max=128")
        ax3.legend()

        plt.suptitle(f"{f.name}\n\"{text[:80]}\"", fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{f.stem}_vis.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.rglob("sample_*.npz"))
    if not files:
        print(f"ERROR: {data_dir}에 sample_*.npz 없음")
        sys.exit(1)

    if args.max_check:
        files = files[:args.max_check]

    print(f"{'='*60}")
    print(f"데이터셋 검증: {data_dir}")
    print(f"검사 대상: {len(files)}개 파일")
    print(f"{'='*60}")

    check_file_integrity(files)
    pc_stats = check_point_cloud(files)
    check_motion_data(files)
    check_text_quality(files)
    check_temporal_alignment(files)

    if args.visualize:
        visualize_samples(files)

    print(f"\n{'='*60}")
    print("검증 완료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()