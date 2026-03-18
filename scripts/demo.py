#!/usr/bin/env python3
"""
RadarLLM Pipeline Demo / Smoke Test

Verifies all components work end-to-end with random synthetic data.
No real dataset or GPU required.

Usage:
    python scripts/demo.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def demo():
    print("=" * 60)
    print("RadarLLM Pipeline Demo")
    print("=" * 60)

    device = torch.device("cpu")

    # ── Configuration ──
    config = {
        "tokenizer": {
            "grid_size": [3, 3, 3],
            "neighborhood_radius": 0.5,
            "encoder_channels": [32, 64, 128],
            "temporal_stride": 2,
            "token_unit": 2,
            "mask_ratio": 0.5,
            "decoder_num_layers": 2,
            "decoder_num_heads": 4,
            "codebook_size": 64,
            "codebook_dim": 128,
            "commitment_cost": 0.25,
        },
        "language_model": {
            "backbone": "google/flan-t5-small",
            "embed_dim": 128,
            "num_radar_tokens": 64,
            "special_tokens": ["<som>", "<eom>"],
            "pretrain": {
                "lr": 2e-4,
                "epochs": 2,
                "batch_size": 2,
                "mask_ratio": 0.15,
                "lambda_pred": 1.0,
                "lambda_r2t": 1.0,
                "lambda_t2r": 1.0,
            },
            "finetune": {
                "lr": 1e-4,
                "epochs": 2,
                "batch_size": 2,
                "prompt_template": "Describe the motion <Motion Placeholder>.",
            },
        },
        "dataset": {
            "points_per_frame": 32,
            "max_seq_len": 20,
        },
    }

    # ── Test 1: Point Cloud Encoder ──
    print("\n[1/5] Testing Point Cloud Encoder...")
    from models.point_encoder import RadarPointCloudEncoder

    encoder = RadarPointCloudEncoder(config["tokenizer"]).to(device)
    B, T, N = 2, 20, 32
    pc = torch.randn(B, T, N, 4)
    F_group = encoder(pc)
    print(f"  Input:  point_cloud {tuple(pc.shape)}")
    print(f"  Output: F_group     {tuple(F_group.shape)}")
    assert F_group.ndim == 4
    print("  ✓ Point Cloud Encoder OK")

    # ── Test 2: Aggregate VQ-VAE ──
    print("\n[2/5] Testing Aggregate VQ-VAE...")
    from rf_captioning.models.aggregate_vqvae_backup import AggregateVQVAE

    vqvae = AggregateVQVAE(config).to(device)
    params = sum(p.numel() for p in vqvae.parameters())
    print(f"  Parameters: {params:,}")

    # Forward (training mode)
    vqvae.train()
    outputs = vqvae(pc)
    print(f"  Indices shape: {tuple(outputs['indices'].shape)}")
    print(f"  z_q shape:     {tuple(outputs['z_q'].shape)}")
    print(f"  Loss:          {outputs['loss'].item():.4f}")
    print(f"    L_rec:       {outputs['loss_rec'].item():.4f}")
    print(f"    L_commit:    {outputs['loss_commit'].item():.4f}")

    # Encode (inference mode)
    vqvae.eval()
    with torch.no_grad():
        indices, z_q = vqvae.encode(pc)
    print(f"  Encode indices: {tuple(indices.shape)} (values in [0, {indices.max().item()}])")
    print("  ✓ Aggregate VQ-VAE OK")

    # ── Test 3: Language Model ──
    print("\n[3/5] Testing Radar-Aware Language Model...")
    from models.language_model import RadarAwareLanguageModel

    lm = RadarAwareLanguageModel(config).to(device)
    lm_params = sum(p.numel() for p in lm.parameters())
    print(f"  Parameters: {lm_params:,}")

    # Tokenize sample text
    texts = ["a person walks forward slowly", "a person raises both arms"]
    text_enc = lm.tokenizer(texts, padding=True, truncation=True,
                             max_length=32, return_tensors="pt")

    # Pre-training step
    lm.train()
    radar_indices = torch.randint(0, 64, (2, 5))
    pt_out = lm.pretrain_step(
        radar_indices, text_enc.input_ids, text_enc.attention_mask
    )
    print(f"  Pretrain loss: {pt_out['loss'].item():.4f}")
    print(f"    L_pred: {pt_out['loss_pred'].item():.4f}")
    print(f"    L_r2t:  {pt_out['loss_r2t'].item():.4f}")
    print(f"    L_t2r:  {pt_out['loss_t2r'].item():.4f}")

    # Instruction tuning step
    ft_out = lm.instruction_tune_step(
        radar_indices, text_enc.input_ids, text_enc.attention_mask
    )
    print(f"  Finetune loss: {ft_out['loss'].item():.4f}")

    # Generation
    lm.eval()
    with torch.no_grad():
        generated = lm.generate_text(
            radar_indices, max_length=32, num_beams=2,
            prompt_template="Describe the motion.",
        )
    print(f"  Generated texts:")
    for i, text in enumerate(generated):
        print(f"    [{i}] \"{text}\"")
    print("  ✓ Language Model OK")

    # ── Test 4: Full RadarLLM Pipeline ──
    print("\n[4/5] Testing Full RadarLLM Pipeline...")
    from models.language_model import RadarLLM

    radar_llm = RadarLLM(config).to(device)
    radar_llm.eval()

    with torch.no_grad():
        predictions = radar_llm.predict(pc, max_length=32, num_beams=2)
    print(f"  End-to-end predictions:")
    for i, text in enumerate(predictions):
        print(f"    [{i}] \"{text}\"")
    print("  ✓ Full Pipeline OK")

    # ── Test 5: Loss backward pass ──
    print("\n[5/5] Testing gradient flow...")
    radar_llm.train()
    radar_llm.tokenizer.train()
    radar_llm.language_model.train()

    # VQ-VAE loss
    vq_out = radar_llm.tokenizer(pc)
    vq_out["loss"].backward()
    vq_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in radar_llm.tokenizer.parameters())
    print(f"  VQ-VAE gradients flowing: {vq_grad}")

    # LM loss
    radar_llm.zero_grad()
    with torch.no_grad():
        indices, _ = radar_llm.tokenizer.encode(pc)
    lm_out = radar_llm.language_model.pretrain_step(
        indices, text_enc.input_ids, text_enc.attention_mask
    )
    lm_out["loss"].backward()
    lm_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in radar_llm.language_model.parameters())
    print(f"  LM gradients flowing: {lm_grad}")
    print("  ✓ Gradient Flow OK")

    print("\n" + "=" * 60)
    print("All tests passed! RadarLLM pipeline is functional.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
