#!/usr/bin/env python3
"""
v8.0 Inference: Radar → HY-Motion 201D → Full Motion Reconstruction

핵심 파이프라인:
  Radar PC → LatentGraph → g_radar → hymotion_head → pred_201D
  pred_201D → HY-Motion decode_motion_from_latent() → rot6d, transl, keypoints3d

사용법:
  # 단일 샘플
  python scripts/inference_motion.py \
      --sample data/radar_text_dataset/sample_000001.npz \
      --ckpt checkpoints/latent_graph/latent_graph_best.pt

  # 전체 테스트셋
  python scripts/inference_motion.py \
      --data_dir data/radar_text_dataset/test \
      --ckpt checkpoints/latent_graph/latent_graph_best.pt \
      --output results/motion_predictions/

  # HY-Motion 디코더로 full motion 복원 (GPU 필요)
  python scripts/inference_motion.py \
      --sample data/radar_text_dataset/sample_000001.npz \
      --ckpt checkpoints/latent_graph/latent_graph_best.pt \
      --decode_motion \
      --hymotion_config HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0/config.yml \
      --hymotion_ckpt HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0/latest.ckpt
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.latent_graph_dynamics import LatentGraphDynamicsModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="v8.0 Inference: Radar → 201D → Motion")
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/latent_graph.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/motion_predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    # HY-Motion 디코딩 옵션
    parser.add_argument("--decode_motion", action="store_true",
                        help="HY-Motion 디코더로 full motion 복원")
    parser.add_argument("--hymotion_config", type=str, default=None)
    parser.add_argument("--hymotion_ckpt", type=str, default=None)
    parser.add_argument("--motion_length", type=int, default=90,
                        help="복원할 모션 길이 (frames, default 90 = 3초)")
    return parser.parse_args()


def load_model(config, ckpt_path, device):
    model = LatentGraphDynamicsModel(config).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(
            ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Model loaded: {ckpt_path}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt_path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_hymotion_pipeline(config_path, ckpt_path, device):
    """HY-Motion pipeline 로드 (full motion 복원용)."""
    try:
        from hymotion.utils.t2m_runtime import T2MRuntime
        from hymotion.pipeline.motion_diffusion import MotionFlowMatching
        from hymotion.utils.loaders import load_object

        with open(config_path, "r") as f:
            hm_config = yaml.safe_load(f)

        pipeline = load_object(
            hm_config["train_pipeline"],
            hm_config["train_pipeline_args"],
            network_module=hm_config["network_module"],
            network_module_args=hm_config["network_module_args"],
        )
        pipeline.load_in_demo(ckpt_path, build_text_encoder=False,
                              allow_empty_ckpt=False)
        pipeline.to(device)
        pipeline.eval()
        print(f"HY-Motion pipeline loaded: {ckpt_path}")
        return pipeline
    except Exception as e:
        print(f"⚠ HY-Motion pipeline 로드 실패: {e}")
        return None


def load_sample(npz_path, config, device):
    """단일 샘플 로드."""
    data = np.load(npz_path, allow_pickle=True)
    lg_cfg = config.get("latent_graph", config)
    ds_cfg = config.get("dataset", {})

    pc = data["point_cloud"].astype(np.float32)
    max_T = ds_cfg.get("max_seq_len", 100)
    T_r = min(pc.shape[0], max_T)
    D = min(pc.shape[-1], lg_cfg.get("point_in_dim", 4))
    N = ds_cfg.get("points_per_frame", 128)

    pc_padded = np.zeros((max_T, N, lg_cfg.get("point_in_dim", 4)), dtype=np.float32)
    pc_padded[:T_r, :, :D] = pc[:T_r, :, :D]
    mask = np.zeros(max_T, dtype=np.bool_)
    mask[:T_r] = True

    pc_tensor = torch.from_numpy(pc_padded).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

    # GT 201D (있으면)
    gt_201 = None
    if "motion_latent" in data:
        gt_201 = data["motion_latent"].astype(np.float32)

    gt_text = str(data["text"]) if "text" in data else None

    return pc_tensor, mask_tensor, gt_201, gt_text


@torch.no_grad()
def predict_201d(model, pc, mask):
    """Radar PC → 201D prediction."""
    pred_201 = model.predict_hymotion_latent(pc, mask)  # (B, 201)
    return pred_201


@torch.no_grad()
def decode_motion(pipeline, pred_201, motion_length=90, device="cuda"):
    """
    201D prediction → full motion via HY-Motion decoder.

    pred_201: (B, 201) — single-frame latent
    motion_length: int — 복원할 프레임 수

    Returns:
        dict with rot6d, transl, keypoints3d
    """
    B = pred_201.shape[0]

    # 201D를 시퀀스로 확장 (static pose 반복)
    # 실제로는 temporal variation이 없지만, 디코더는 시퀀스를 기대함
    latent_seq = pred_201.unsqueeze(1).expand(-1, motion_length, -1)  # (B, L, 201)
    latent_seq = latent_seq.to(device)

    # HY-Motion 디코더 호출
    motion_output = pipeline.decode_motion_from_latent(
        latent_seq, should_apply_smooothing=True)

    return motion_output


def evaluate_prediction(pred_201, gt_201):
    """201D prediction vs GT 비교."""
    if gt_201 is None:
        return {}

    gt_mean = gt_201.mean(axis=0)  # (201,)
    pred_np = pred_201.cpu().numpy().squeeze()  # (201,)

    mse = np.mean((pred_np - gt_mean) ** 2)
    mae = np.mean(np.abs(pred_np - gt_mean))

    # Cosine similarity
    cos_sim = np.dot(pred_np, gt_mean) / (
        np.linalg.norm(pred_np) * np.linalg.norm(gt_mean) + 1e-8)

    # Per-component analysis (201D 구성: [0:3]=transl, [3:9]=root_rot6d, [9:135]=body_rot6d)
    transl_mse = np.mean((pred_np[:3] - gt_mean[:3]) ** 2)
    root_rot_mse = np.mean((pred_np[3:9] - gt_mean[3:9]) ** 2)
    body_rot_mse = np.mean((pred_np[9:135] - gt_mean[9:135]) ** 2)

    return {
        "mse": float(mse),
        "mae": float(mae),
        "cosine_similarity": float(cos_sim),
        "transl_mse": float(transl_mse),
        "root_rot_mse": float(root_rot_mse),
        "body_rot_mse": float(body_rot_mse),
    }


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    model = load_model(config, args.ckpt, device)

    # ── Load HY-Motion pipeline (optional) ──
    hm_pipeline = None
    if args.decode_motion:
        if args.hymotion_config and args.hymotion_ckpt:
            hm_pipeline = load_hymotion_pipeline(
                args.hymotion_config, args.hymotion_ckpt, device)
        else:
            print("⚠ --decode_motion requires --hymotion_config and --hymotion_ckpt")

    # ── Collect samples ──
    samples = []
    if args.sample:
        samples = [Path(args.sample)]
    elif args.data_dir:
        samples = sorted(Path(args.data_dir).glob("*.npz"))
    if args.max_samples:
        samples = samples[:args.max_samples]

    if not samples:
        print("ERROR: No samples found")
        return

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"v8.0 Inference: Radar → 201D → Motion")
    print(f"  Samples: {len(samples)}")
    print(f"  Decode motion: {args.decode_motion and hm_pipeline is not None}")
    print(f"{'='*60}\n")

    # ── Run inference ──
    all_results = []
    all_metrics = []

    for npz_path in tqdm(samples, desc="Inference"):
        try:
            pc, mask, gt_201, gt_text = load_sample(
                npz_path, config, device)

            # ★ Predict 201D
            pred_201 = predict_201d(model, pc, mask)  # (1, 201)

            # Evaluate against GT
            metrics = evaluate_prediction(pred_201, gt_201)
            if metrics:
                all_metrics.append(metrics)

            result = {
                "file": npz_path.name,
                "text": gt_text,
                "pred_201_norm": float(pred_201.norm().item()),
                **metrics,
            }

            # Save predicted 201D
            pred_save_path = os.path.join(
                args.output, f"{npz_path.stem}_pred201.npy")
            np.save(pred_save_path, pred_201.cpu().numpy())

            # ★ Decode full motion (optional)
            if hm_pipeline is not None:
                motion_output = decode_motion(
                    hm_pipeline, pred_201, args.motion_length, device)

                # Save motion
                motion_save_path = os.path.join(
                    args.output, f"{npz_path.stem}_motion.npz")
                np.savez_compressed(motion_save_path,
                    rot6d=motion_output["rot6d"].numpy(),
                    transl=motion_output["transl"].numpy(),
                    keypoints3d=motion_output["keypoints3d"].numpy(),
                )
                result["motion_saved"] = motion_save_path
                print(f"  Motion saved: {motion_save_path}")

            all_results.append(result)

            # Print per-sample
            if gt_text:
                print(f"  [{npz_path.name}] \"{gt_text[:60]}\"")
            if metrics:
                print(f"    MSE={metrics['mse']:.6f}, "
                      f"cos={metrics['cosine_similarity']:.4f}, "
                      f"transl_mse={metrics['transl_mse']:.6f}")

        except Exception as e:
            print(f"  ✗ {npz_path.name}: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Summary ({len(all_results)} samples)")
    print(f"{'='*60}")

    if all_metrics:
        avg_mse = np.mean([m["mse"] for m in all_metrics])
        avg_mae = np.mean([m["mae"] for m in all_metrics])
        avg_cos = np.mean([m["cosine_similarity"] for m in all_metrics])
        avg_transl = np.mean([m["transl_mse"] for m in all_metrics])
        avg_root = np.mean([m["root_rot_mse"] for m in all_metrics])
        avg_body = np.mean([m["body_rot_mse"] for m in all_metrics])

        print(f"  201D Prediction Quality:")
        print(f"    MSE:              {avg_mse:.6f}")
        print(f"    MAE:              {avg_mae:.6f}")
        print(f"    Cosine Similarity:{avg_cos:.4f}")
        print(f"    Translation MSE:  {avg_transl:.6f}")
        print(f"    Root Rotation MSE:{avg_root:.6f}")
        print(f"    Body Rotation MSE:{avg_body:.6f}")

    # Save results
    results_path = os.path.join(args.output, "inference_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
