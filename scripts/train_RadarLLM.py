#!/usr/bin/env python3
"""
RadarLLM 학습 스크립트 — MotionGPT Pretrained 통합
===================================================

파이프라인 (3단계):
  Stage 1: VQ-VAE + MotionEncoder(T2M-GPT) 학습 (100 epochs)
           MotionEncoder: HumanML3D(263D) → T2M-GPT Conv1D+ResBlock → F_mot(512D)
           Pretrained weights 로드 후 fine-tune 또는 freeze
           L_VQ = L_rec + L_emb + L_commit

  Stage 2: T5 Pre-training (300 epochs)
           L_pretrain = λ1·L_pred + λ2·L_r2t + λ3·L_t2r

  Stage 3: Instruction Tuning (100 epochs)

사용법:
  python scripts/train_RadarLLM.py --stage all --config configs/default.yaml
  python scripts/train_RadarLLM.py --stage tokenizer --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.aggregate_vqvae import AggregateVQVAE
from models.motion_encoder import build_motion_encoder
from models.language_model import RadarAwareLanguageModel


# ============================================================
# Dataset
# ============================================================

class RadarMotionTextDataset(Dataset):
    """
    generate_pt.py + add_humanml3d.py가 생성한 데이터 로드.

    각 .npz:
      - point_cloud:       (T_radar, 128, 6)
      - text:              str
      - motion_humanml3d:  (T_motion, 263)  ★ MotionGPT용
      - motion_latent:     (T_motion, 201)  (fallback)
      - motion_joints:     (T_motion, 22, 3)
      - motion_rot6d:      (T_motion, 22, 6)
      - motion_transl:     (T_motion, 3)
    """

    def __init__(self, data_dir, max_radar_frames=100, max_motion_frames=300,
                 points_per_frame=128, motion_input_type="humanml3d"):
        super().__init__()
        self.data_dir = data_dir
        self.max_radar_frames = max_radar_frames
        self.max_motion_frames = max_motion_frames
        self.points_per_frame = points_per_frame
        self.motion_input_type = motion_input_type

        self.samples = sorted(Path(data_dir).glob("*.npz"))
        print(f"[Dataset] {len(self.samples)} samples from {data_dir}, "
              f"motion_type={motion_input_type}")

        # 첫 샘플로 motion_humanml3d 존재 여부 확인
        if len(self.samples) > 0:
            first = np.load(self.samples[0], allow_pickle=True)
            available = [k for k in first.files if k.startswith("motion_")]
            print(f"[Dataset] Available motion keys: {available}")
            if motion_input_type == "humanml3d" and "motion_humanml3d" not in first:
                print(f"[Dataset] ⚠ motion_humanml3d not found! "
                      f"Run: python scripts/add_humanml3d.py --data_dir {data_dir}")
                print(f"[Dataset]   Falling back to available keys")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)

        # ── 레이더 포인트클라우드 ──
        pc = data["point_cloud"].astype(np.float32)
        T_r = min(pc.shape[0], self.max_radar_frames)
        pc = pc[:T_r]
        pc_4d = np.zeros((self.max_radar_frames, self.points_per_frame, 4),
                         dtype=np.float32)
        pc_4d[:T_r, :, :3] = pc[:, :, :3]
        pc_4d[:T_r, :, 3] = pc[:, :, 3]
        radar_mask = np.zeros(self.max_radar_frames, dtype=np.float32)
        radar_mask[:T_r] = 1.0

        # ── 모션 데이터 (우선순위: config → humanml3d → latent → joints → rot6d) ──
        motion = None

        # key 매핑: input_type → npz key
        type_to_key = {
            "humanml3d": "motion_humanml3d",
            "latent":    "motion_latent",
            "joints":    "motion_joints",
            "rot6d":     "motion_rot6d",
        }

        # 1. config 지정 타입 먼저
        preferred_key = type_to_key.get(self.motion_input_type)
        if preferred_key and preferred_key in data:
            motion = data[preferred_key].astype(np.float32)

        # 2. fallback 순서
        if motion is None:
            for key in ["motion_humanml3d", "motion_latent",
                        "motion_joints", "motion_rot6d"]:
                if key in data:
                    motion = data[key].astype(np.float32)
                    break

        # 3. 빈 데이터 fallback
        if motion is None:
            # input_dim에 맞게 빈 데이터 생성
            dim_map = {"humanml3d": 263, "latent": 201, "joints": 66, "rot6d": 132}
            dim = dim_map.get(self.motion_input_type, 263)
            motion = np.zeros((1, dim), dtype=np.float32)

        # (T, J, D) → flatten to (T, J*D)
        if motion.ndim == 3:
            T_m, J, D = motion.shape
            if J > 22:
                motion = motion[:, :22, :]
            motion = motion.reshape(T_m, -1)

        T_m = min(motion.shape[0], self.max_motion_frames)
        motion = motion[:T_m]

        # 패딩
        motion_dim = motion.shape[-1]
        motion_padded = np.zeros((self.max_motion_frames, motion_dim),
                                 dtype=np.float32)
        motion_padded[:T_m] = motion
        motion_mask = np.zeros(self.max_motion_frames, dtype=np.float32)
        motion_mask[:T_m] = 1.0

        text = str(data["text"])

        return {
            "point_cloud": torch.from_numpy(pc_4d),
            "radar_mask": torch.from_numpy(radar_mask),
            "motion": torch.from_numpy(motion_padded),
            "motion_mask": torch.from_numpy(motion_mask),
            "text": text,
            "num_radar_frames": T_r,
            "num_motion_frames": T_m,
        }


def collate_fn(batch):
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "radar_mask": torch.stack([b["radar_mask"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch]),
        "motion_mask": torch.stack([b["motion_mask"] for b in batch]),
        "texts": [b["text"] for b in batch],
        "num_radar_frames": [b["num_radar_frames"] for b in batch],
        "num_motion_frames": [b["num_motion_frames"] for b in batch],
    }


# ============================================================
# Test Captioning (학습 중 호출)
# ============================================================

@torch.no_grad()
def run_test_captioning(vqvae, lm, test_loader, device, epoch,
                        max_samples=5, prompt="Describe the motion."):
    vqvae.eval()
    lm.eval()

    print(f"\n{'─'*60}")
    print(f"  Epoch {epoch} — Test Captioning ({max_samples} samples)")
    print(f"{'─'*60}")

    count = 0
    for batch in test_loader:
        if count >= max_samples:
            break
        pc = batch["point_cloud"].to(device)
        texts_gt = batch["texts"]

        indices, _ = vqvae.encode(pc)
        predictions = lm.generate_text(indices, max_length=128, num_beams=5,
                                        prompt_template=prompt)
        for j in range(len(texts_gt)):
            if count >= max_samples:
                break
            print(f"  [{count+1}] GT:   \"{texts_gt[j]}\"")
            print(f"       Pred: \"{predictions[j]}\"")
            print()
            count += 1

    print(f"{'─'*60}\n")


@torch.no_grad()
def save_test_results(vqvae, lm, test_loader, device, output_dir, prompt):
    vqvae.eval()
    lm.eval()
    results = []
    for batch in tqdm(test_loader, desc="Final evaluation"):
        pc = batch["point_cloud"].to(device)
        indices, _ = vqvae.encode(pc)
        predictions = lm.generate_text(indices, max_length=128, num_beams=5,
                                        prompt_template=prompt)
        for gt, pred in zip(batch["texts"], predictions):
            results.append({"ground_truth": gt, "prediction": pred})

    out_path = os.path.join(output_dir, "test_captions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n캡셔닝 결과 저장: {out_path} ({len(results)}개)")
    for i, r in enumerate(results[:5]):
        print(f"  [{i+1}] GT:   \"{r['ground_truth']}\"")
        print(f"       Pred: \"{r['prediction']}\"")


# ============================================================
# Stage 1: VQ-VAE + MotionEncoder(T2M-GPT) 학습
# ============================================================

def train_tokenizer(config, args):
    """
    Aggregate VQ-VAE + T2M-GPT MotionEncoder 학습.

    MotionEncoder에 pretrained weight가 로드된 경우:
      - freeze=true:  MotionEncoder 고정, VQ-VAE만 학습
      - freeze=false: 둘 다 fine-tune
    """
    print("\n" + "=" * 60)
    print("Stage 1: VQ-VAE + MotionEncoder(T2M-GPT) 학습")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok_cfg = config["tokenizer"]
    me_cfg = config.get("motion_encoder", {})

    # ── VQ-VAE ──
    vqvae = AggregateVQVAE(config).to(device)
    print(f"VQ-VAE params: {sum(p.numel() for p in vqvae.parameters()):,}")

    # ── MotionEncoder (T2M-GPT, pretrained 자동 로드) ──
    motion_enc = build_motion_encoder(config).to(device)

    # ── Optimizer: trainable params만 ──
    trainable_params = [p for p in vqvae.parameters() if p.requires_grad]
    trainable_params += [p for p in motion_enc.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=tok_cfg["lr"],
                            weight_decay=tok_cfg.get("weight_decay", 1e-2))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tok_cfg["epochs"])

    # ── Data ──
    mi_type = me_cfg.get("input_type", "humanml3d")
    ds_cfg = config.get("dataset", {})

    train_ds = RadarMotionTextDataset(
        ds_cfg["train_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        motion_input_type=mi_type,
    )
    val_ds = RadarMotionTextDataset(
        ds_cfg["val_dir"],
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        points_per_frame=ds_cfg.get("points_per_frame", 128),
        motion_input_type=mi_type,
    )
    train_loader = DataLoader(train_ds, batch_size=tok_cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=tok_cfg["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    # ── Codebook 초기화 (data-driven) ──
    with torch.no_grad():
        all_z_e = []
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break
            pc_init = batch["point_cloud"].to(device)
            enc_out = vqvae.encoder(pc_init)
            F_group = enc_out["F_group"]
            F_all, _, _ = vqvae.masked_agg(F_group, training=False)
            z_e = vqvae.anchor_pool(F_all)
            all_z_e.append(z_e.reshape(-1, z_e.shape[-1]))
        flat = torch.cat(all_z_e, dim=0)
        n_codes = vqvae.quantizer.codebook_size
        if flat.shape[0] >= n_codes:
            idx = torch.randperm(flat.shape[0])[:n_codes]
        else:
            idx = torch.randint(0, flat.shape[0], (n_codes,))
        vqvae.quantizer.codebook.weight.data.copy_(flat[idx])
        vqvae.quantizer.codebook.weight.data += torch.randn_like(vqvae.quantizer.codebook.weight.data) * 0.1
        print(f"Codebook initialized with noise perturbation")


    # ── Training loop ──
    for epoch in range(tok_cfg["epochs"]):
        vqvae.train()
        motion_enc.train()
        losses = {"total": 0, "rec": 0, "emb": 0, "commit": 0}

        pbar = tqdm(train_loader, desc=f"VQ {epoch+1}/{tok_cfg['epochs']}")
        for batch in pbar:
            pc = batch["point_cloud"].to(device)
            motion = batch["motion"].to(device)

            T_radar = pc.shape[1]
            L_radar = vqvae.encoder.get_output_length(T_radar)

            # MotionEncoder: (B, T_m, 263) → (B, L_radar, 512)
            F_mot = motion_enc(motion, target_length=L_radar)

            outputs = vqvae(pc, motion_features=F_mot)

            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            for k in ["total", "rec", "emb", "commit"]:
                key = "loss" if k == "total" else f"loss_{k}"
                losses[k] += outputs[key].item()

            pbar.set_postfix(loss=f"{outputs['loss'].item():.4f}",
                             emb=f"{outputs['loss_emb'].item():.4f}")

        scheduler.step()

        # ── Diagnostics ──
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                indices, _ = vqvae.encode(pc)
                unique = indices.unique()
                print(f"  [Diag] epoch {epoch+1}: "
                      f"unique_tokens={len(unique)}/{vqvae.quantizer.codebook_size}, "
                      f"z_e_std={vqvae.anchor_pool(vqvae.masked_agg(vqvae.encoder(pc)['F_group'], training=False)[0]).std().item():.4f}")

        # ── Validation ──
        vqvae.eval()
        motion_enc.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pc = batch["point_cloud"].to(device)
                motion = batch["motion"].to(device)
                L_r = vqvae.encoder.get_output_length(pc.shape[1])
                F_mot = motion_enc(motion, target_length=L_r)
                val_loss += vqvae(pc, motion_features=F_mot)["loss"].item()
        val_loss /= max(len(val_loader), 1)

        n = max(len(train_loader), 1)
        print(f"Epoch {epoch+1}: loss={losses['total']/n:.4f} "
              f"(rec={losses['rec']/n:.4f} emb={losses['emb']/n:.4f} "
              f"commit={losses['commit']/n:.4f}) val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "vqvae_state_dict": vqvae.state_dict(),
                "motion_enc_state_dict": motion_enc.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(args.output_dir, "tokenizer_best.pt"))

    torch.save({
        "vqvae_state_dict": vqvae.state_dict(),
        "motion_enc_state_dict": motion_enc.state_dict(),
    }, os.path.join(args.output_dir, "tokenizer_final.pt"))
    print(f"Saved: tokenizer_best.pt, tokenizer_final.pt")
    return vqvae


# ============================================================
# Stage 2: LM Pre-training
# ============================================================

def pretrain_lm(config, args, vqvae=None):
    print("\n" + "=" * 60)
    print("Stage 2: LM Pre-training")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lm_cfg = config["language_model"]
    pt_cfg = lm_cfg["pretrain"]

    # ── Frozen VQ-VAE ──
    if vqvae is None:
        vqvae = AggregateVQVAE(config).to(device)
        ckpt_path = args.tokenizer_ckpt or os.path.join(
            args.output_dir, "tokenizer_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            vqvae.load_state_dict(
                ckpt.get("vqvae_state_dict",
                         ckpt.get("model_state_dict", ckpt)))
            print(f"VQ-VAE loaded from {ckpt_path}")
        else:
            print(f"⚠ VQ-VAE checkpoint not found: {ckpt_path}")
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # ── LM ──
    lm = RadarAwareLanguageModel(config).to(device)
    print(f"LM params: {sum(p.numel() for p in lm.parameters()):,}")

    # ── Data ──
    me_cfg = config.get("motion_encoder", {})
    mi_type = me_cfg.get("input_type", "humanml3d")
    ds_cfg = config.get("dataset", {})

    train_ds = RadarMotionTextDataset(
        ds_cfg["train_dir"], motion_input_type=mi_type,
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
    )
    train_loader = DataLoader(train_ds, batch_size=pt_cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)

    # Test loader
    test_dir = ds_cfg.get("test_dir", ds_cfg.get("val_dir"))
    test_loader = None
    if test_dir and os.path.isdir(test_dir):
        test_ds = RadarMotionTextDataset(
            test_dir, motion_input_type=mi_type,
            max_radar_frames=ds_cfg.get("max_seq_len", 100),
            max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        )
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2)

    optimizer = optim.AdamW(lm.parameters(), lr=float(pt_cfg["lr"]))
    best_loss = float("inf")
    caption_every = args.caption_every

    for epoch in range(pt_cfg["epochs"]):
        lm.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"PT {epoch+1}/{pt_cfg['epochs']}"):
            pc = batch["point_cloud"].to(device)
            texts = batch["texts"]
            text_enc = lm.tokenizer(texts, padding=True, truncation=True,
                                     max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                indices, _ = vqvae.encode(pc)

            outputs = lm.pretrain_step(
                indices, text_enc.input_ids, text_enc.attention_mask)

            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += outputs["loss"].item()

        avg = epoch_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch+1}: loss={avg:.4f}")

        if test_loader and (epoch + 1) % caption_every == 0:
            run_test_captioning(vqvae, lm, test_loader, device, epoch + 1)

        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model_state_dict": lm.state_dict()},
                       os.path.join(args.output_dir, "lm_pretrain_best.pt"))

    torch.save({"model_state_dict": lm.state_dict()},
               os.path.join(args.output_dir, "lm_pretrain_final.pt"))
    return lm


# ============================================================
# Stage 3: Instruction Tuning
# ============================================================

def finetune_lm(config, args, vqvae=None, lm=None):
    print("\n" + "=" * 60)
    print("Stage 3: Instruction Tuning")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ft_cfg = config["language_model"]["finetune"]

    # ── Frozen VQ-VAE ──
    if vqvae is None:
        vqvae = AggregateVQVAE(config).to(device)
        ckpt_path = args.tokenizer_ckpt or os.path.join(
            args.output_dir, "tokenizer_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            vqvae.load_state_dict(
                ckpt.get("vqvae_state_dict",
                         ckpt.get("model_state_dict", ckpt)))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # ── LM ──
    if lm is None:
        lm = RadarAwareLanguageModel(config).to(device)
        ckpt_path = args.lm_ckpt or os.path.join(
            args.output_dir, "lm_pretrain_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            lm.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"LM loaded from {ckpt_path}")

    # ── Data ──
    me_cfg = config.get("motion_encoder", {})
    mi_type = me_cfg.get("input_type", "humanml3d")
    ds_cfg = config.get("dataset", {})

    train_ds = RadarMotionTextDataset(
        ds_cfg["train_dir"], motion_input_type=mi_type,
        max_radar_frames=ds_cfg.get("max_seq_len", 100),
        max_motion_frames=ds_cfg.get("max_motion_frames", 300),
    )
    train_loader = DataLoader(train_ds, batch_size=ft_cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)

    # Test loader
    test_dir = ds_cfg.get("test_dir", ds_cfg.get("val_dir"))
    test_loader = None
    if test_dir and os.path.isdir(test_dir):
        test_ds = RadarMotionTextDataset(
            test_dir, motion_input_type=mi_type,
            max_radar_frames=ds_cfg.get("max_seq_len", 100),
            max_motion_frames=ds_cfg.get("max_motion_frames", 300),
        )
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2)

    optimizer = optim.AdamW(lm.parameters(), lr=float(ft_cfg["lr"]))
    prompt = ft_cfg.get("prompt_template",
                        "Describe the motion <Motion Placeholder>.")
    caption_every = args.caption_every

    # 학습 전 baseline 캡셔닝
    if test_loader:
        run_test_captioning(vqvae, lm, test_loader, device, epoch=0,
                            max_samples=3, prompt=prompt)

    best_loss = float("inf")
    for epoch in range(ft_cfg["epochs"]):
        lm.train()
        epoch_loss = 0

        for batch in tqdm(train_loader,
                          desc=f"FT {epoch+1}/{ft_cfg['epochs']}"):
            pc = batch["point_cloud"].to(device)
            texts = batch["texts"]
            text_enc = lm.tokenizer(texts, padding=True, truncation=True,
                                     max_length=128,
                                     return_tensors="pt").to(device)

            with torch.no_grad():
                indices, _ = vqvae.encode(pc)

            outputs = lm.instruction_tune_step(
                indices, text_enc.input_ids, text_enc.attention_mask, prompt)

            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += outputs["loss"].item()

        avg = epoch_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch+1}: loss={avg:.4f}")

        if test_loader and (epoch + 1) % caption_every == 0:
            run_test_captioning(vqvae, lm, test_loader, device, epoch + 1,
                                max_samples=5, prompt=prompt)

        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model_state_dict": lm.state_dict()},
                       os.path.join(args.output_dir, "lm_finetune_best.pt"))

    # 최종 캡셔닝 + 저장
    if test_loader:
        run_test_captioning(vqvae, lm, test_loader, device,
                            epoch=ft_cfg["epochs"], max_samples=10,
                            prompt=prompt)
        save_test_results(vqvae, lm, test_loader, device,
                         args.output_dir, prompt)

    torch.save({"model_state_dict": lm.state_dict()},
               os.path.join(args.output_dir, "lm_finetune_final.pt"))
    return lm


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["tokenizer", "pretrain", "finetune", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer_ckpt", default=None)
    parser.add_argument("--lm_ckpt", default=None)
    parser.add_argument("--caption_every", type=int, default=10,
                        help="테스트 캡셔닝 주기 (epoch)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Motion encoder: {config.get('motion_encoder', {}).get('input_type', '?')} "
          f"({config.get('motion_encoder', {}).get('input_dim', '?')}D)")
    print(f"Pretrained: {config.get('motion_encoder', {}).get('pretrained', 'None')}")

    if args.stage == "tokenizer":
        train_tokenizer(config, args)
    elif args.stage == "pretrain":
        pretrain_lm(config, args)
    elif args.stage == "finetune":
        finetune_lm(config, args)
    elif args.stage == "all":
        vqvae = train_tokenizer(config, args)
        lm = pretrain_lm(config, args, vqvae)
        finetune_lm(config, args, vqvae, lm)

    print("\n학습 완료!")


if __name__ == "__main__":
    main()