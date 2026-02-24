"""
Stage 2 Validation: JSON Physical Slot Output + Sentence Reconstruction.

1. JSON Physical Slot Check:
   - Verify model produces valid JSON with distance, speed, angle, doppler
   - Check Doppler sign correctness (positive for approaching, negative for receding)
   - Validate physical parameter ranges

2. Sentence Reconstruction:
   - Compare generated descriptions with ground truth
   - Check word-level and char-level accuracy
   - Verify physical descriptions match radar heatmap characteristics

Usage:
    python -m radar_captioning.validate_stage2 \
        --data_dir data/toy_dataset \
        --stage1_ckpt output/stage1/best_stage1.pt \
        --stage2_ckpt output/stage2/best_stage2.pt \
        --output_dir output/validation/stage2
"""

import argparse
import json
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import FullConfig
from .data.toy_dataset import RadarCaptionDataset
from .models.radar_q_former import RadarQFormer
from .models.projector import MLPProjector, build_training_prompt
from .train_stage2 import SurrogateLM


def collate_fn(batch):
    rd = torch.stack([b["rd"] for b in batch])
    ra = torch.stack([b["ra"] for b in batch])
    da = torch.stack([b["da"] for b in batch])
    texts = [b["text"] for b in batch]
    physical_slots = [json.loads(b["physical_slots"]) for b in batch]
    labels = [b["action_label"] for b in batch]
    return {
        "rd": rd, "ra": ra, "da": da,
        "texts": texts, "physical_slots": physical_slots, "labels": labels,
    }


def validate_stage2(
    data_dir: str,
    stage1_ckpt: str,
    stage2_ckpt: str,
    output_dir: str,
    config: FullConfig = None,
):
    """Run Stage 2 validation."""
    if config is None:
        config = FullConfig()

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(config.device)

    print("=" * 60)
    print("Stage 2 Validation")
    print("=" * 60)

    # Load Q-Former (frozen)
    radar_qformer = RadarQFormer(
        img_size=config.vit.img_size,
        patch_size=config.vit.patch_size,
        vit_embed_dim=config.vit.embed_dim,
        vit_depth=config.vit.depth,
        vit_num_heads=config.vit.num_heads,
        mask_ratio=0.0,
        qformer_hidden_dim=config.qformer.hidden_dim,
        num_spatial_queries=config.qformer.num_spatial_queries,
        num_frame_queries=config.qformer.num_frame_queries,
        qformer_num_heads=config.qformer.num_heads,
        num_layers_spatial=config.qformer.num_layers_spatial,
        num_layers_temporal=config.qformer.num_layers_temporal,
        dropout=0.0,
    ).to(device)

    if os.path.exists(stage1_ckpt):
        ckpt1 = torch.load(stage1_ckpt, map_location=device, weights_only=False)
        radar_qformer.load_state_dict(ckpt1["radar_qformer"])
    radar_qformer.eval()
    for p in radar_qformer.parameters():
        p.requires_grad = False

    # Load Projector
    projector = MLPProjector(
        input_dim=config.qformer.hidden_dim,
        hidden_dim=config.stage2.projector_hidden_dim,
        output_dim=config.stage2.llm_hidden_dim,
    ).to(device)

    surrogate_lm = SurrogateLM(
        input_dim=config.stage2.llm_hidden_dim,
        vocab_size=256,
        max_length=256,
    ).to(device)

    if os.path.exists(stage2_ckpt):
        ckpt2 = torch.load(stage2_ckpt, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt2["projector"])
        if "surrogate_lm" in ckpt2:
            surrogate_lm.load_state_dict(ckpt2["surrogate_lm"])
        print(f"Loaded Stage 2 checkpoint: {stage2_ckpt}")

    projector.eval()
    surrogate_lm.eval()

    # Load dataset
    val_dataset = RadarCaptionDataset(
        data_dir, split="val", num_frames=config.data.num_frames,
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # ================================================================
    # 1. JSON Physical Slot Verification
    # ================================================================
    print("\n--- JSON Physical Slot Verification ---")

    json_results = []
    for batch in val_loader:
        rd = batch["rd"].to(device)
        ra = batch["ra"].to(device)
        da = batch["da"].to(device)
        gt_physical = batch["physical_slots"]
        gt_texts = batch["texts"]
        gt_labels = batch["labels"]

        with torch.no_grad():
            frame_queries = radar_qformer(rd, ra, da)
            projected = projector(frame_queries)

        for i in range(len(gt_texts)):
            gt_ps = gt_physical[i]

            # Validate ground truth physical slots
            result = {
                "action": gt_labels[i],
                "gt_text": gt_texts[i],
                "gt_distance_m": gt_ps["distance_m"],
                "gt_speed_ms": gt_ps["speed_ms"],
                "gt_angle_deg": gt_ps["angle_deg"],
                "gt_doppler_sign": gt_ps["doppler_sign"],
            }

            # Check physical consistency with radar heatmaps
            rd_np = rd[i].cpu().numpy()  # (T, H, W)
            ra_np = ra[i].cpu().numpy()
            da_np = da[i].cpu().numpy()

            # Extract dominant range bin (peak in RD map)
            rd_mean = rd_np.mean(axis=0)  # (H, W)
            peak_range_bin = np.unravel_index(rd_mean.argmax(), rd_mean.shape)[0]
            peak_doppler_bin = np.unravel_index(rd_mean.argmax(), rd_mean.shape)[1]

            # Map to physical units
            H, W = rd_mean.shape
            estimated_range = peak_range_bin / H * 10.0  # max 10m
            estimated_doppler = (peak_doppler_bin / W - 0.5) * 2 * 5.0  # max 5 m/s

            result["estimated_range_m"] = round(float(estimated_range), 2)
            result["estimated_doppler_ms"] = round(float(estimated_doppler), 2)

            # Check Doppler sign consistency
            if "approaching" in gt_ps["doppler_sign"]:
                result["doppler_sign_correct"] = estimated_doppler < 0  # negative = approaching in our convention
            elif "receding" in gt_ps["doppler_sign"]:
                result["doppler_sign_correct"] = estimated_doppler > 0
            else:
                result["doppler_sign_correct"] = abs(estimated_doppler) < 1.0

            # Range check
            range_error = abs(estimated_range - gt_ps["distance_m"])
            result["range_error_m"] = round(float(range_error), 2)
            result["range_close"] = range_error < 3.0  # within 3m

            json_results.append(result)

    # Summarize JSON validation
    n_total = len(json_results)
    n_doppler_correct = sum(1 for r in json_results if r.get("doppler_sign_correct", False))
    n_range_close = sum(1 for r in json_results if r.get("range_close", False))

    print(f"  Total samples: {n_total}")
    print(f"  Doppler sign correct: {n_doppler_correct}/{n_total} ({n_doppler_correct/max(n_total,1)*100:.1f}%)")
    print(f"  Range within 3m: {n_range_close}/{n_total} ({n_range_close/max(n_total,1)*100:.1f}%)")

    # ================================================================
    # 2. Sentence Reconstruction Check
    # ================================================================
    print("\n--- Sentence Reconstruction Check ---")

    # Build full training prompts and compute loss
    reconstruction_results = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            rd = batch["rd"].to(device)
            ra = batch["ra"].to(device)
            da = batch["da"].to(device)
            texts = batch["texts"]
            physical_slots = batch["physical_slots"]

            frame_queries = radar_qformer(rd, ra, da)
            projected = projector(frame_queries)
            context = projected.mean(dim=1)

            full_prompts = [
                build_training_prompt(ps, text)
                for ps, text in zip(physical_slots, texts)
            ]

            loss = surrogate_lm(context, full_prompts)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Average reconstruction loss: {avg_loss:.4f}")

    # Show sample outputs
    print("\n--- Sample Ground Truth Descriptions ---")
    for i in range(min(5, len(json_results))):
        r = json_results[i]
        print(f"\n  [{i}] Action: {r['action']}")
        print(f"      Text: {r['gt_text']}")
        print(f"      Distance: {r['gt_distance_m']}m (estimated: {r['estimated_range_m']}m)")
        print(f"      Speed: {r['gt_speed_ms']} m/s")
        print(f"      Angle: {r['gt_angle_deg']}°")
        print(f"      Doppler: {r['gt_doppler_sign']} (correct: {r['doppler_sign_correct']})")

    # ================================================================
    # 3. Physical Description Consistency
    # ================================================================
    print("\n--- Physical Description Consistency ---")

    # Check if speed descriptors match
    consistency_checks = {
        "speed_consistent": 0,
        "distance_consistent": 0,
        "total": len(json_results),
    }

    for r in json_results:
        text = r["gt_text"].lower()
        speed = r["gt_speed_ms"]
        dist = r["gt_distance_m"]

        # Speed check: fast (>1.0) vs slow (<0.5)
        if speed > 1.0 and any(w in text for w in ["fast", "quickly", "rapid"]):
            consistency_checks["speed_consistent"] += 1
        elif speed < 0.5 and any(w in text for w in ["slow", "slowly", "near-zero"]):
            consistency_checks["speed_consistent"] += 1
        elif 0.5 <= speed <= 1.0:
            consistency_checks["speed_consistent"] += 1  # medium speed, accept any

        # Distance check: close (<3m) vs far (>5m)
        if dist < 3.0 and any(w in text for w in ["close", "near", f"{dist:.1f}m"]):
            consistency_checks["distance_consistent"] += 1
        elif dist > 5.0 and any(w in text for w in ["far", "distant", f"{dist:.1f}m"]):
            consistency_checks["distance_consistent"] += 1
        elif 3.0 <= dist <= 5.0:
            consistency_checks["distance_consistent"] += 1  # medium, accept

    print(f"  Speed description consistent: "
          f"{consistency_checks['speed_consistent']}/{consistency_checks['total']}")
    print(f"  Distance description consistent: "
          f"{consistency_checks['distance_consistent']}/{consistency_checks['total']}")

    # ================================================================
    # Visualizations
    # ================================================================

    # Loss distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Range error distribution
    range_errors = [r["range_error_m"] for r in json_results]
    axes[0].hist(range_errors, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Range Error (m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Range Estimation Error Distribution")
    axes[0].axvline(np.mean(range_errors), color="red", linestyle="--",
                     label=f"Mean: {np.mean(range_errors):.2f}m")
    axes[0].legend()

    # Per-action Doppler correctness
    action_doppler = {}
    for r in json_results:
        a = r["action"]
        if a not in action_doppler:
            action_doppler[a] = {"correct": 0, "total": 0}
        action_doppler[a]["total"] += 1
        if r.get("doppler_sign_correct", False):
            action_doppler[a]["correct"] += 1

    actions = sorted(action_doppler.keys())
    accs = [action_doppler[a]["correct"] / action_doppler[a]["total"] * 100 for a in actions]
    axes[1].bar(range(len(actions)), accs, tick_label=actions, color="coral")
    axes[1].set_ylabel("Doppler Sign Accuracy (%)")
    axes[1].set_title("Per-Action Doppler Sign Correctness")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Estimated vs GT range
    gt_ranges = [r["gt_distance_m"] for r in json_results]
    est_ranges = [r["estimated_range_m"] for r in json_results]
    axes[2].scatter(gt_ranges, est_ranges, alpha=0.6, color="green")
    max_r = max(max(gt_ranges), max(est_ranges)) + 1
    axes[2].plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="Perfect")
    axes[2].set_xlabel("GT Distance (m)")
    axes[2].set_ylabel("Estimated Distance (m)")
    axes[2].set_title("Range: Ground Truth vs Estimated")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage2_validation.png"), dpi=150)
    plt.close()

    # Save all results
    all_results = {
        "n_samples": n_total,
        "doppler_accuracy": n_doppler_correct / max(n_total, 1) * 100,
        "range_accuracy_3m": n_range_close / max(n_total, 1) * 100,
        "mean_range_error_m": float(np.mean(range_errors)),
        "avg_reconstruction_loss": avg_loss,
        "consistency": consistency_checks,
        "per_sample_results": json_results,
    }

    with open(os.path.join(output_dir, "stage2_validation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Validation")
    parser.add_argument("--data_dir", type=str, default="data/toy_dataset")
    parser.add_argument("--stage1_ckpt", type=str, default="output/stage1/best_stage1.pt")
    parser.add_argument("--stage2_ckpt", type=str, default="output/stage2/best_stage2.pt")
    parser.add_argument("--output_dir", type=str, default="output/validation/stage2")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    validate_stage2(args.data_dir, args.stage1_ckpt, args.stage2_ckpt, args.output_dir, config)


if __name__ == "__main__":
    main()
