#!/usr/bin/env python3
"""
RadarLLM 학습 스크립트
====================

파이프라인 (3단계):
  Stage 1: VQ-VAE + MotionEncoder 학습 (100 epochs)
           MotionEncoder: HY-Motion latent(201D) → Transformer(2L) → F_mot(512D)
           VQ-VAE와 MotionEncoder가 함께 학습됨
           L_VQ = L_rec + L_emb + L_commit

  Stage 2: T5 Pre-training (300 epochs)
           L_pretrain = λ1·L_pred + λ2·L_r2t + λ3·L_t2r

  Stage 3: Instruction Tuning (100 epochs)
           + 테스트 캡셔닝

사용법:
  python scripts/train_v2.py --stage all --config configs/default.yaml
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
# 데이터셋
# ============================================================

class RadarMotionTextDataset(Dataset):
    """
    generate_pt_v2.py가 생성한 데이터 로드.

    각 .npz:
      - point_cloud:    (T_radar, 128, 6)
      - text:           str
      - motion_latent:  (T_motion, 201) ← HY-Motion latent (★ 핵심)
      - motion_joints:  (T_motion, 22, 3)  (fallback)
      - motion_rot6d:   (T_motion, 22, 6)  (fallback)
    """

    def __init__(self, data_dir, max_radar_frames=100, max_motion_frames=300,
                 points_per_frame=128, motion_input_type="latent"):
        super().__init__()
        self.data_dir = data_dir
        self.max_radar_frames = max_radar_frames
        self.max_motion_frames = max_motion_frames
        self.points_per_frame = points_per_frame
        self.motion_input_type = motion_input_type
        self.samples = sorted(Path(data_dir).glob("*.npz"))
        print(f"[Dataset] {len(self.samples)} samples from {data_dir}, "
              f"motion_type={motion_input_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)

        # ── 레이더 포인트클라우드 ──
        pc = data["point_cloud"].astype(np.float32)
        T_r = min(pc.shape[0], self.max_radar_frames)
        pc = pc[:T_r]
        pc_4d = np.zeros((self.max_radar_frames, self.points_per_frame, 4), dtype=np.float32)
        pc_4d[:T_r, :, :3] = pc[:, :, :3]
        pc_4d[:T_r, :, 3] = pc[:, :, 3]
        radar_mask = np.zeros(self.max_radar_frames, dtype=np.float32)
        radar_mask[:T_r] = 1.0

        # ── 모션 데이터 (우선순위: latent > joints > rot6d) ──
        motion_key = f"motion_{self.motion_input_type}"
        motion = None
        for key in [motion_key, "motion_latent", "motion_joints", "motion_rot6d"]:
            if key in data:
                motion = data[key].astype(np.float32)
                break

        if motion is None:
            motion = np.zeros((1, 201), dtype=np.float32)

        # joints/rot6d가 (T, J, D) 형태면 flatten → (T, J*D)
        if motion.ndim == 3:
            T_m, J, D = motion.shape
            if J > 22:  # 52 joints → 22 body only
                motion = motion[:, :22, :]
            motion = motion.reshape(T_m, -1)

        T_m = min(motion.shape[0], self.max_motion_frames)
        motion = motion[:T_m]

        # 패딩
        motion_dim = motion.shape[-1]
        motion_padded = np.zeros((self.max_motion_frames, motion_dim), dtype=np.float32)
        motion_padded[:T_m] = motion
        motion_mask = np.zeros(self.max_motion_frames, dtype=np.float32)
        motion_mask[:T_m] = 1.0

        text = str(data["text"])

        return {
            "point_cloud": torch.from_numpy(pc_4d),
            "radar_mask": torch.from_numpy(radar_mask),
            "motion": torch.from_numpy(motion_padded),       # (max_T_m, D)
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
# 테스트 캡셔닝 (학습 중 호출)
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
# Stage 1: VQ-VAE + MotionEncoder 동시 학습
# ============================================================

def train_tokenizer(config, args):
    """
    Aggregate VQ-VAE + MotionEncoder 동시 학습.

    MotionEncoder: HY-Motion latent(201D) → Transformer(2L) → F_mot(512D)
    VQ-VAE와 함께 학습됨.

    L_VQ = L_rec + L_emb + L_commit
    """
    print("\n" + "="*60) 
    print("Stage 1: VQ-VAE + MotionEncoder 학습")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok_cfg = config["tokenizer"]

    # VQ-VAE
    vqvae = AggregateVQVAE(config).to(device)
    print(f"VQ-VAE params: {sum(p.numel() for p in vqvae.parameters()):,}")

    # MotionEncoder (VQ-VAE와 함께 학습)
    motion_enc = build_motion_encoder(config).to(device)

    # 둘 다 학습
    all_params = list(vqvae.parameters()) + list(motion_enc.parameters())
    optimizer = optim.AdamW(all_params, lr=tok_cfg["lr"],
                            weight_decay=tok_cfg.get("weight_decay", 1e-2))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tok_cfg["epochs"])

    # Data
    mi_type = config.get("motion_encoder", {}).get("input_type", "latent")
    train_ds = RadarMotionTextDataset(config["dataset"]["train_dir"],
                                       motion_input_type=mi_type)
    val_ds = RadarMotionTextDataset(config["dataset"]["val_dir"],
                                     motion_input_type=mi_type)
    train_loader = DataLoader(train_ds, batch_size=tok_cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=tok_cfg["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    '''
    with torch.no_grad():
        first_batch = next(iter(train_loader))
        pc_init = first_batch["point_cloud"].to(device)
        enc_out = vqvae.encoder(pc_init)
        F_group = enc_out["F_group"]
        F_all, _, _ = vqvae.masked_agg(F_group, training=False)
        z_e = vqvae.anchor_pool(F_all)
        flat = z_e.reshape(-1, z_e.shape[-1])
        n_codes = vqvae.quantizer.codebook_size
        if flat.shape[0] >= n_codes:
            idx = torch.randperm(flat.shape[0])[:n_codes]
        else:
            idx = torch.randint(0, flat.shape[0], (n_codes,))
        vqvae.quantizer.codebook.weight.data.copy_(flat[idx])
        print(f"Codebook 초기화 완료: z_e {flat.shape[0]}개에서 {n_codes}개 코드 설정")
    '''
    with torch.no_grad():
        all_z_e = []
        for i, batch in enumerate(train_loader):
            if i >= 10:  # 10배치 = ~40개 샘플
                break
            pc_init = batch["point_cloud"].to(device)
            enc_out = vqvae.encoder(pc_init)
            F_group = enc_out["F_group"]
            F_all, _, _ = vqvae.masked_agg(F_group, training=False)
            z_e = vqvae.anchor_pool(F_all)
            all_z_e.append(z_e.reshape(-1, z_e.shape[-1]))
        
        flat = torch.cat(all_z_e, dim=0)
        n_codes = vqvae.quantizer.codebook_size
        idx = torch.randperm(flat.shape[0])[:n_codes]
        vqvae.quantizer.codebook.weight.data.copy_(flat[idx])
        #print(f"Codebook 초기화 완료: z_e {flat.shape[0]}개에서 {n_codes}개 코드 설정")



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

            # MotionEncoder: latent(201D) → Transformer → F_mot(512D)
            F_mot = motion_enc(motion, target_length=L_radar)

            outputs = vqvae(pc, motion_features=F_mot)

            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            for k in ["total", "rec", "emb", "commit"]:
                key = "loss" if k == "total" else f"loss_{k}"
                losses[k] += outputs[key].item()

            pbar.set_postfix(loss=f"{outputs['loss'].item():.4f}",
                             emb=f"{outputs['loss_emb'].item():.4f}")

        scheduler.step()

        if (epoch + 1) % 2 == 0:
                 with torch.no_grad():
                    # 진단 코드 전체
                    indices, _ = vqvae.encode(pc)
                    unique = indices.unique()
                    print(f"  [VQ 상태] epoch {epoch+1}")
                    print(f"  unique tokens: {len(unique)}/{vqvae.quantizer.codebook_size}")

                    F_mot = motion_enc(motion, target_length=L_radar)
                    print(f"  F_mot std: {F_mot.std().item():.4f}")
                    print(f"  F_mot 샘플간 차이: {(F_mot[0] - F_mot[-1]).abs().mean().item():.4f}")



                    # 2. z_e 다양성
                    enc_out = vqvae.encoder(pc)
                    F_group = enc_out["F_group"]
                    F_all, _, _ = vqvae.masked_agg(F_group, training=False)
                    z_e = vqvae.anchor_pool(F_all)
                    
                    print(f"  z_e std: {z_e.std().item():.4f}")
                    print(f"  z_e 샘플간 차이: {(z_e[0] - z_e[-1]).abs().mean().item():.4f}")
                    print(f"  F_group 샘플간 차이: {(F_group[0] - F_group[-1]).abs().mean().item():.4f}")

                    # 3. centered 좌표 + anchor 활성화
                    xyz_c = enc_out["xyz_centered"]
                    anchors = vqvae.encoder.grouping.anchors
                    radius = vqvae.encoder.grouping.radius

                    pts = xyz_c[0, 0]
                    valid = pts.norm(dim=-1) > 1e-6
                    pts_v = pts[valid]
                    if pts_v.shape[0] > 0:
                        print(f"  centered 범위: x[{pts_v[:,0].min():.2f},{pts_v[:,0].max():.2f}] "
                            f"y[{pts_v[:,1].min():.2f},{pts_v[:,1].max():.2f}] "
                            f"z[{pts_v[:,2].min():.2f},{pts_v[:,2].max():.2f}]")
                    else:
                        print(f"  centered 범위: 유효점 없음 (빈 프레임)")

                    dist = torch.cdist(anchors.unsqueeze(0), pts_v.unsqueeze(0))[0]
                    hits = (dist < radius).sum(dim=1)
                    print(f"  활성 anchor: {(hits > 0).sum().item()}/{anchors.shape[0]}")

                    # 4. 샘플별 활성 anchor 비교
                    for b in range(min(4, pc.shape[0])):
                        pts = xyz_c[b, 0]
                        valid = pts.norm(dim=-1) > 1e-6
                        pts_v = pts[valid]
                        dist = torch.cdist(anchors.unsqueeze(0), pts_v.unsqueeze(0))[0]
                        hits = (dist < radius).sum(dim=1)
                        print(f"  샘플{b}: 활성 {(hits > 0).sum().item()}/64, 유효점 {valid.sum().item()}")


        # Validation
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
            }, os.path.join(args.output_dir, "tokenizer_final.pt"))

    torch.save({
        "vqvae_state_dict": vqvae.state_dict(),
        "motion_enc_state_dict": motion_enc.state_dict(),
    }, os.path.join(args.output_dir, "tokenizer_final.pt"))
    return vqvae


# ============================================================
# Stage 2: LM Pre-training + 테스트 캡셔닝
# ============================================================

def pretrain_lm(config, args, vqvae=None):
    print("\n" + "="*60)
    print("Stage 2: LM Pre-training")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lm_cfg = config["language_model"]
    pt_cfg = lm_cfg["pretrain"]

    # Frozen VQ-VAE
    if vqvae is None:
        vqvae = AggregateVQVAE(config).to(device)
        ckpt_path = args.tokenizer_ckpt or os.path.join(args.output_dir, "tokenizer_best.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        vqvae.load_state_dict(ckpt.get("vqvae_state_dict", ckpt.get("model_state_dict", ckpt)))
        print(f"VQ-VAE loaded from {ckpt_path}")
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # LM
    lm = RadarAwareLanguageModel(config).to(device)
    print(f"LM params: {sum(p.numel() for p in lm.parameters()):,}")

    mi_type = config.get("motion_encoder", {}).get("input_type", "latent")
    train_ds = RadarMotionTextDataset(config["dataset"]["train_dir"],
                                       motion_input_type=mi_type)
    train_loader = DataLoader(train_ds, batch_size=pt_cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)

    # Test loader
    test_dir = config["dataset"].get("test_dir", config["dataset"].get("val_dir"))
    test_loader = None
    if test_dir and os.path.isdir(test_dir):
        test_ds = RadarMotionTextDataset(test_dir, motion_input_type=mi_type)
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

            outputs = lm.pretrain_step(indices, text_enc.input_ids, text_enc.attention_mask)
            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += outputs["loss"].item()

        avg = epoch_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch+1}: loss={avg:.4f}")

        # 테스트 캡셔닝
        if test_loader and (epoch + 1) % caption_every == 0:
            run_test_captioning(vqvae, lm, test_loader, device, epoch + 1)

        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model_state_dict": lm.state_dict()},
                       os.path.join(args.output_dir, "lm_pretrain_best.pt"))

    return lm


# ============================================================
# Stage 3: Instruction Tuning + 테스트 캡셔닝
# ============================================================

def finetune_lm(config, args, vqvae=None, lm=None):
    print("\n" + "="*60)
    print("Stage 3: Instruction Tuning")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ft_cfg = config["language_model"]["finetune"]

    # Frozen VQ-VAE
    if vqvae is None:
        vqvae = AggregateVQVAE(config).to(device)
        ckpt_path = args.tokenizer_ckpt or os.path.join(args.output_dir, "tokenizer_best.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        vqvae.load_state_dict(ckpt.get("vqvae_state_dict", ckpt.get("model_state_dict", ckpt)))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # LM
    if lm is None:
        lm = RadarAwareLanguageModel(config).to(device)
        ckpt_path = args.lm_ckpt or os.path.join(args.output_dir, "lm_pretrain_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            lm.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"LM loaded from {ckpt_path}")

    mi_type = config.get("motion_encoder", {}).get("input_type", "latent")
    train_ds = RadarMotionTextDataset(config["dataset"]["train_dir"],
                                       motion_input_type=mi_type)
    train_loader = DataLoader(train_ds, batch_size=ft_cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)

    # Test loader
    test_dir = config["dataset"].get("test_dir", config["dataset"].get("val_dir"))
    test_loader = None
    if test_dir and os.path.isdir(test_dir):
        test_ds = RadarMotionTextDataset(test_dir, motion_input_type=mi_type)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2)

    optimizer = optim.AdamW(lm.parameters(), lr=float(ft_cfg["lr"]))
    prompt = ft_cfg.get("prompt_template", "Describe the motion <Motion Placeholder>.")
    caption_every = args.caption_every

    # 학습 전 캡셔닝 (baseline)
    if test_loader:
        run_test_captioning(vqvae, lm, test_loader, device, epoch=0, max_samples=3,
                            prompt=prompt)

    best_loss = float("inf")
    for epoch in range(ft_cfg["epochs"]):
        lm.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"FT {epoch+1}/{ft_cfg['epochs']}"):
            pc = batch["point_cloud"].to(device)
            texts = batch["texts"]
            text_enc = lm.tokenizer(texts, padding=True, truncation=True,
                                     max_length=128, return_tensors="pt").to(device)
            
            with torch.no_grad():
                    indices, _ = vqvae.encode(pc)

            '''
            if (epoch + 1) % 2 == 0:
                 with torch.no_grad():
                    # 진단 코드 전체
                    indices, _ = vqvae.encode(pc)
                    unique = indices.unique()
                    print(f"  [VQ 상태] epoch {epoch+1}")
                    print(f"  unique tokens: {len(unique)}/{vqvae.quantizer.codebook_size}")


                    # 2. z_e 다양성
                    enc_out = vqvae.encoder(pc)
                    F_group = enc_out["F_group"]
                    F_all, _, _ = vqvae.masked_agg(F_group, training=False)
                    z_e = vqvae.anchor_pool(F_all)
                    print(f"  z_e std: {z_e.std().item():.4f}")
                    print(f"  z_e 샘플간 차이: {(z_e[0] - z_e[-1]).abs().mean().item():.4f}")
                    print(f"  F_group 샘플간 차이: {(F_group[0] - F_group[-1]).abs().mean().item():.4f}")

                    # 3. centered 좌표 + anchor 활성화
                    xyz_c = enc_out["xyz_centered"]
                    anchors = vqvae.encoder.grouping.anchors
                    radius = vqvae.encoder.grouping.radius

                    pts = xyz_c[0, 0]
                    valid = pts.norm(dim=-1) > 1e-6
                    pts_v = pts[valid]
                    print(f"  centered 범위: x[{pts_v[:,0].min():.2f},{pts_v[:,0].max():.2f}] "
                        f"y[{pts_v[:,1].min():.2f},{pts_v[:,1].max():.2f}] "
                        f"z[{pts_v[:,2].min():.2f},{pts_v[:,2].max():.2f}]")

                    dist = torch.cdist(anchors.unsqueeze(0), pts_v.unsqueeze(0))[0]
                    hits = (dist < radius).sum(dim=1)
                    print(f"  활성 anchor: {(hits > 0).sum().item()}/{anchors.shape[0]}")

                    # 4. 샘플별 활성 anchor 비교
                    for b in range(min(4, pc.shape[0])):
                        pts = xyz_c[b, 0]
                        valid = pts.norm(dim=-1) > 1e-6
                        pts_v = pts[valid]
                        dist = torch.cdist(anchors.unsqueeze(0), pts_v.unsqueeze(0))[0]
                        hits = (dist < radius).sum(dim=1)
                        print(f"  샘플{b}: 활성 {(hits > 0).sum().item()}/64, 유효점 {valid.sum().item()}")
                    '''
            outputs = lm.instruction_tune_step(indices, text_enc.input_ids,
                                                text_enc.attention_mask, prompt)
            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += outputs["loss"].item()

        avg = epoch_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch+1}: loss={avg:.4f}")

        # 테스트 캡셔닝
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
                            epoch=ft_cfg["epochs"], max_samples=10, prompt=prompt)
        save_test_results(vqvae, lm, test_loader, device, args.output_dir, prompt)

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
                        help="테스트 캡셔닝 주기 (epoch 단위)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

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
