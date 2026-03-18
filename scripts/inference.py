#!/usr/bin/env python3
"""
RadarLLM Inference 스크립트

사용법:
  python scripts/inference.py --sample data/radar_text_dataset/sample_110002.npz --tokenizer_ckpt checkpoints/tokenizer_final.pt
  python scripts/inference.py --data_dir data/radar_text_dataset/test
  python scripts/inference.py --data_dir data/radar_text_dataset/test --max_samples 10
  python scripts/inference.py --data_dir data/radar_text_dataset/test \
      --lm_ckpt checkpoints/lm_pretrain_best.pt --no_prompt
  python scripts/inference.py --data_dir data/radar_text_dataset/test \
      --output results/test_results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.aggregate_vqvae import AggregateVQVAE
from models.motion_encoder import build_motion_encoder
from models.language_model import RadarAwareLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="RadarLLM Inference")
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None)
    parser.add_argument("--lm_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--no_prompt", action="store_true")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_models(config, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from models.aggregate_vqvae import AggregateVQVAE
    vqvae = AggregateVQVAE(config).to(device)

    tok_ckpt = args.tokenizer_ckpt
    if tok_ckpt is None:
        for p in ["checkpoints/tokenizer_best.pt", "checkpoints/tokenizer_final.pt"]:
            if os.path.exists(p):
                tok_ckpt = p
                break

    if tok_ckpt and os.path.exists(tok_ckpt):
        ckpt = torch.load(tok_ckpt, map_location=device, weights_only=False)
        vqvae.load_state_dict(ckpt.get("vqvae_state_dict", ckpt))
        print(f"VQ-VAE loaded: {tok_ckpt}")
    else:
        print("⚠ VQ-VAE 체크포인트 없음!")

    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    from models.language_model import RadarAwareLanguageModel
    lm = RadarAwareLanguageModel(config).to(device)

    lm_ckpt = args.lm_ckpt
    if lm_ckpt is None:
        for p in ["checkpoints/lm_finetune_best.pt",
                   "checkpoints/lm_finetune_final.pt",
                   "checkpoints/lm_pretrain_best.pt",
                   "checkpoints/lm_pretrain_final.pt"]:
            if os.path.exists(p):
                lm_ckpt = p
                break

    if lm_ckpt and os.path.exists(lm_ckpt):
        ckpt = torch.load(lm_ckpt, map_location=device, weights_only=False)
        lm.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"LM loaded: {lm_ckpt}")
    else:
        print("⚠ LM 체크포인트 없음!")

    lm.eval()
    return vqvae, lm, device


def run_inference_single(vqvae, lm, pc_tensor, device, prompt=None,
                         num_beams=5, max_length=128):
    with torch.no_grad():
        indices, _ = vqvae.encode(pc_tensor)
        unique = indices.unique()
        n_unique = len(unique)
        captions = lm.generate_text(
            indices,
            max_length=max_length,
            num_beams=num_beams,
            prompt_template=prompt,
        )
    return captions, n_unique


def load_sample(npz_path, device):
    data = np.load(npz_path, allow_pickle=True)
    pc = torch.from_numpy(data["point_cloud"]).float().unsqueeze(0).to(device)
    gt_text = str(data["text"]) if "text" in data else None
    return pc, gt_text, data


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.no_prompt:
        prompt = None
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = config.get("language_model", {}).get("finetune", {}).get(
            "prompt_template", "Describe the motion <Motion Placeholder>.")

    vqvae, lm, device = load_models(config, args)

    samples = []
    if args.sample:
        samples = [Path(args.sample)]
    elif args.data_dir:
        samples = sorted(Path(args.data_dir).glob("*.npz"))
    else:
        test_dir = config.get("dataset", {}).get("test_dir", "data/radar_text_dataset/test")
        if os.path.isdir(test_dir):
            samples = sorted(Path(test_dir).glob("*.npz"))

    if not samples:
        print("ERROR: 테스트할 샘플이 없습니다.")
        print("  --sample 또는 --data_dir을 지정하세요.")
        return

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"\n{'='*60}")
    print(f"RadarLLM Inference")
    print(f"  샘플: {len(samples)}개")
    print(f"  프롬프트: {prompt}")
    print(f"  beams: {args.num_beams}")
    print(f"{'='*60}\n")

    results = []
    for npz_path in tqdm(samples, desc="Inference"):
        try:
            pc, gt_text, data = load_sample(npz_path, device)
            captions, n_unique = run_inference_single(
                vqvae, lm, pc, device,
                prompt=prompt,
                num_beams=args.num_beams,
                max_length=args.max_length,
            )
            caption = captions[0] if captions else ""

            result = {
                "file": npz_path.name,
                "ground_truth": gt_text,
                "prediction": caption,
                "unique_tokens": n_unique,
                "num_frames": int(pc.shape[1]),
            }
            results.append(result)

            print(f"\n  [{npz_path.name}] (unique={n_unique})")
            if gt_text:
                print(f"    GT:   \"{gt_text}\"")
            print(f"    Pred: \"{caption}\"")

        except Exception as e:
            print(f"  ✗ {npz_path.name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"결과 요약")
    print(f"{'='*60}")
    print(f"  총 샘플: {len(results)}")

    if results:
        uniques = [r["unique_tokens"] for r in results]
        print(f"  unique tokens: min={min(uniques)}, max={max(uniques)}, avg={np.mean(uniques):.1f}")
        predictions = [r["prediction"] for r in results]
        unique_preds = len(set(predictions))
        print(f"  고유 예측 문장: {unique_preds}/{len(results)}")

    if args.output:
        output_path = args.output
    else:
        output_path = "checkpoints/inference_results.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {output_path}")


if __name__ == "__main__":
    main()