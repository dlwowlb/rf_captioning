#!/usr/bin/env python3
"""
Stage 1 Evaluation — Filtered Version

변경사항 (원본 대비):
  --exclude_top N : 가장 빈도 높은 N개 클래스 제외 (default=1, "turn" 제거)
  --keep_top K    : 제외 후 상위 K개 클래스만 평가 (default=5)
  
  즉 기본값이면: turn 제외 → bend, wave, jump, stretch, balance 5개로 평가

사용법:
  python scripts/evaluate.py \
      --config configs/latent_graph.yaml \
      --ckpt checkpoints/latent_graph/latent_graph_best.pt \
      --data_dir data/radar_text_dataset/test \
      --exclude_top 1 --keep_top 5
"""

import os, sys, json, argparse, yaml, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.latent_graph_dynamics import LatentGraphDynamicsModel
from models.motion_encoder import build_motion_encoder


# ═══════════════════════════════════════════════════════════
# Action Labeling (원본과 동일)
# ═══════════════════════════════════════════════════════════

ACTION_CATEGORIES = [
    "walking or strolling", "running or jogging", "jumping or hopping",
    "sitting down or seated", "standing up or standing still", "kicking",
    "punching or striking", "throwing or tossing", "waving hands or arms",
    "picking up or grabbing an object", "pushing or pulling",
    "bending or bowing", "crouching or squatting", "turning or spinning",
    "dancing or rhythmic movement", "climbing or stepping up",
    "balancing on one leg", "stretching or reaching",
    "arm circles or arm exercises", "full body exercise or workout",
]
ACTION_SHORT_LABELS = [
    "walk", "run", "jump", "sit", "stand", "kick", "punch", "throw",
    "wave", "pick_up", "push", "bend", "crouch", "turn", "dance",
    "climb", "balance", "stretch", "arm_exercise", "exercise",
]


class ActionLabeler:
    def __init__(self, method="auto", cache_path=None):
        self.method = method
        self.cache = {}
        self.cache_path = cache_path
        self._classifier = None

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self.cache = json.load(f)
            print(f"[ActionLabeler] Loaded {len(self.cache)} cached labels")

        if method == "auto":
            self.method = self._detect_best_method()
        print(f"[ActionLabeler] Using method: {self.method}")

    def _detect_best_method(self):
        try:
            from transformers import pipeline as hf_pipeline
            self._classifier = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1,
            )
            print("[ActionLabeler] ✓ zero-shot classifier loaded")
            return "zero_shot"
        except:
            pass
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._cat_embeddings = self._embedder.encode(
                ACTION_CATEGORIES, normalize_embeddings=True)
            return "sentence_similarity"
        except:
            pass
        return "keyword"

    def label_batch(self, texts):
        results = {}
        uncached = []
        uncached_idx = []
        for i, t in enumerate(texts):
            if t in self.cache:
                results[i] = self.cache[t]
            else:
                uncached.append(t)
                uncached_idx.append(i)
        if uncached:
            if self.method == "zero_shot":
                labels = self._zero_shot_batch(uncached)
            elif self.method == "sentence_similarity":
                labels = self._sentence_similarity_batch(uncached)
            else:
                labels = [self._keyword_label(t) for t in uncached]
            for idx, text, label in zip(uncached_idx, uncached, labels):
                results[idx] = label
                self.cache[text] = label
        return [results[i] for i in range(len(texts))]

    def save_cache(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            print(f"[ActionLabeler] Saved {len(self.cache)} labels to {self.cache_path}")

    def _zero_shot_batch(self, texts):
        if self._classifier is None:
            from transformers import pipeline as hf_pipeline
            self._classifier = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1,
            )
        labels = []
        for text in tqdm(texts, desc="Zero-shot labeling"):
            result = self._classifier(text, ACTION_CATEGORIES, multi_label=False)
            best_idx = ACTION_CATEGORIES.index(result["labels"][0])
            labels.append(ACTION_SHORT_LABELS[best_idx])
        return labels

    def _sentence_similarity_batch(self, texts):
        if not hasattr(self, '_embedder') or self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._cat_embeddings = self._embedder.encode(
                ACTION_CATEGORIES, normalize_embeddings=True)
        text_embs = self._embedder.encode(texts, normalize_embeddings=True)
        sims = text_embs @ self._cat_embeddings.T
        return [ACTION_SHORT_LABELS[idx] for idx in sims.argmax(axis=1)]

    def _keyword_label(self, text):
        text_lower = text.lower()
        words = set(text_lower.split())
        KW = {
            "jump": ["jump","jumping","hop","leap","jacks"],
            "kick": ["kick","kicking"],
            "punch": ["punch","punching","strike","boxing","uppercut"],
            "wave": ["wave","waving","beckon"],
            "pick_up": ["pick","grab","grasp","lift"],
            "push": ["push","pull"],
            "dance": ["dance","dancing","sway","salsa","waltz"],
            "climb": ["climb","climbing","step up"],
            "balance": ["balance","balancing","one leg"],
            "stretch": ["stretch","stretching","reach","extending"],
            "bend": ["bend","bending","bow","lean","stoop"],
            "crouch": ["crouch","squat","squatting","kneel","lunge"],
            "turn": ["turn","turning","rotate","spin","pivot"],
            "run": ["run","running","jog","sprint","dash"],
            "walk": ["walk","walking","stroll","march"],
            "sit": ["sit","sitting","seated"],
            "stand": ["stand","standing"],
            "exercise": ["exercise","workout","pushup","burpee","plank"],
        }
        for action, keywords in KW.items():
            for kw in keywords:
                if " " in kw:
                    if kw in text_lower: return action
                else:
                    if kw in words: return action
        return "other"


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class EvalDataset(Dataset):
    def __init__(self, data_dir, max_T=100, max_N=128, D=4,
                 max_motion_frames=300, motion_type="humanml3d",
                 action_labeler=None):
        self.max_T, self.max_N, self.D = max_T, max_N, D
        self.max_motion_frames = max_motion_frames
        self.mi = motion_type
        self.samples = sorted(Path(data_dir).rglob("*.npz"))
        self.action_labeler = action_labeler
        print(f"[EvalDataset] {len(self.samples)} samples, motion={motion_type}")

        if action_labeler is not None:
            print("[EvalDataset] Pre-computing action labels...")
            all_texts = []
            for s in tqdm(self.samples, desc="Loading texts"):
                data = np.load(s, allow_pickle=True)
                all_texts.append(str(data["text"]))
            self._action_labels = action_labeler.label_batch(all_texts)
            action_labeler.save_cache()
        else:
            self._action_labels = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        text = str(data["text"])
        action = self._action_labels[idx] if self._action_labels else "other"

        pc = data["point_cloud"].astype(np.float32)
        Tr = min(pc.shape[0], self.max_T)
        Dc = min(pc.shape[-1], self.D)
        pc_out = np.zeros((self.max_T, self.max_N, self.D), np.float32)
        pc_out[:Tr, :, :Dc] = pc[:Tr, :, :Dc]
        mask = np.zeros(self.max_T, dtype=np.bool_)
        mask[:Tr] = True

        motion, motion_mask = self._load_motion_with_mask(data)

        return {
            "point_cloud": torch.from_numpy(pc_out),
            "temporal_mask": torch.from_numpy(mask),
            "motion": torch.from_numpy(motion),
            "motion_mask": torch.from_numpy(motion_mask),
            "text": text, "action": action,
            "path": str(self.samples[idx].name),
        }

    def _load_motion_with_mask(self, data):
        key_map = {"humanml3d": "motion_humanml3d", "latent": "motion_latent",
                    "joints": "motion_joints"}
        dim_map = {"humanml3d": 263, "latent": 201, "joints": 66}
        motion = None
        preferred = key_map.get(self.mi, "")
        if preferred and preferred in data:
            m = data[preferred].astype(np.float32)
            motion = m.reshape(m.shape[0], -1) if m.ndim == 3 else m
        if motion is None:
            for k in ["motion_humanml3d", "motion_latent", "motion_joints"]:
                if k in data:
                    m = data[k].astype(np.float32)
                    motion = m.reshape(m.shape[0], -1) if m.ndim == 3 else m
                    break
        if motion is None:
            dim = dim_map.get(self.mi, 263)
            motion = np.zeros((1, dim), np.float32)
        T_m = min(motion.shape[0], self.max_motion_frames)
        motion = motion[:T_m]
        motion_dim = motion.shape[-1]
        motion_padded = np.zeros((self.max_motion_frames, motion_dim), np.float32)
        motion_padded[:T_m] = motion
        motion_mask = np.zeros(self.max_motion_frames, dtype=np.bool_)
        motion_mask[:T_m] = True
        return motion_padded, motion_mask


def collate(batch):
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "temporal_mask": torch.stack([b["temporal_mask"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch]),
        "motion_mask": torch.stack([b["motion_mask"] for b in batch]),
        "texts": [b["text"] for b in batch],
        "actions": [b["action"] for b in batch],
        "paths": [b["path"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════
# ★ Class Filtering Utilities
# ═══════════════════════════════════════════════════════════

def compute_keep_classes(dataset, exclude_top=1, keep_top=5):
    """
    전체 데이터셋의 라벨 분포를 보고,
    상위 exclude_top개를 제외한 뒤 다음 keep_top개 클래스를 반환.
    """
    all_actions = []
    for i in range(len(dataset)):
        item = dataset[i]
        all_actions.append(item["action"])

    dist = Counter(all_actions)
    sorted_classes = [cls for cls, _ in dist.most_common()]

    excluded = sorted_classes[:exclude_top]
    remaining = sorted_classes[exclude_top:]
    kept = remaining[:keep_top]

    print(f"\n{'='*60}")
    print(f"Class Filtering")
    print(f"{'='*60}")
    print(f"  Original distribution ({len(dist)} classes, {len(all_actions)} samples):")
    for cls, cnt in dist.most_common():
        tag = "  ✗ EXCLUDE" if cls in excluded else ("  ★ KEEP" if cls in kept else "  · skip")
        print(f"    {cls:20s}: {cnt:4d} ({cnt/len(all_actions)*100:.1f}%){tag}")

    kept_count = sum(dist[c] for c in kept)
    print(f"\n  Kept: {len(kept)} classes, {kept_count} samples")
    print(f"  Classes: {kept}")

    return kept, excluded


def filter_dataset_indices(dataset, keep_classes):
    """keep_classes에 속하는 샘플의 인덱스만 반환."""
    keep_set = set(keep_classes)
    indices = []
    for i in range(len(dataset)):
        item = dataset[i]
        if item["action"] in keep_set:
            indices.append(i)
    return indices


def filter_extracted(emb_dict, keep_classes):
    """extract_all 결과에서 keep_classes만 필터링."""
    keep_set = set(keep_classes)
    mask = [a in keep_set for a in emb_dict["actions"]]
    mask = np.array(mask)

    filtered = {
        "g_radar": emb_dict["g_radar"][mask],
        "actions": [a for a, m in zip(emb_dict["actions"], mask) if m],
        "texts": [t for t, m in zip(emb_dict["texts"], mask) if m],
    }
    if "g_motion" in emb_dict:
        filtered["g_motion"] = emb_dict["g_motion"][mask]
    return filtered


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def load_model(config, ckpt_path, device):
    model = LatentGraphDynamicsModel(config).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_all(model, loader, device, motion_enc=None):
    model.eval()
    G_r, G_m, actions, texts = [], [], [], []
    for batch in tqdm(loader, desc="Extracting"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        g = model.encode(pc, mask)
        G_r.append(g.cpu())
        actions.extend(batch["actions"])
        texts.extend(batch["texts"])
        if motion_enc is not None:
            motion = batch["motion"].to(device)
            F_m = motion_enc(motion)
            G_m.append(F_m.mean(dim=1).cpu())
    result = {"g_radar": torch.cat(G_r), "actions": actions, "texts": texts}
    if G_m:
        result["g_motion"] = torch.cat(G_m)
    return result


# ═══════════════════════════════════════════════════════════
# 4.2 Context Interpretability (filtered)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run_context_interpretability(model, loader, device, keep_classes):
    print(f"\n{'='*60}")
    print(f"Exp 4.2: Context Interpretability (filtered: {keep_classes})")
    print(f"{'='*60}")
    model.eval()
    keep_set = set(keep_classes)
    action_ctx_means = defaultdict(list)
    action_ctx_change = defaultdict(list)

    for batch in tqdm(loader, desc="Context analysis"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        out = model.forward_sequence(pc, mask)
        ctx = out["context_history"]
        B, T, C = ctx.shape
        for b in range(B):
            action = batch["actions"][b]
            if action not in keep_set:
                continue
            Tv = int(mask[b].sum().item())
            ct = ctx[b, :Tv].cpu().numpy()
            action_ctx_means[action].append(ct.mean(axis=0))
            if Tv > 1:
                deltas = np.linalg.norm(ct[1:] - ct[:-1], axis=-1)
                action_ctx_change[action].append(deltas.mean())

    valid_actions = [a for a in keep_classes if len(action_ctx_means.get(a, [])) >= 2]

    if len(valid_actions) < 2:
        print("  ⚠ < 2 valid classes with ≥2 samples")
        return {"separability_ratio": 0.0}

    centroids, intra_dists = {}, {}
    for a in valid_actions:
        vecs = np.stack(action_ctx_means[a])
        centroids[a] = vecs.mean(axis=0)
        intra_dists[a] = np.mean(np.linalg.norm(vecs - centroids[a], axis=-1))
    avg_intra = np.mean(list(intra_dists.values()))

    inter_pairs = []
    for i, a1 in enumerate(valid_actions):
        for a2 in valid_actions[i+1:]:
            inter_pairs.append(np.linalg.norm(centroids[a1] - centroids[a2]))
    avg_inter = np.mean(inter_pairs) if inter_pairs else 0
    separation = avg_inter / max(avg_intra, 1e-8)

    print(f"  Context separability: inter/intra = {separation:.4f}")
    print(f"    avg inter-class dist: {avg_inter:.4f}")
    print(f"    avg intra-class dist: {avg_intra:.4f}")
    print(f"    valid classes: {len(valid_actions)}/{len(keep_classes)}")

    print(f"\n  Context temporal change rate:")
    for a in sorted(action_ctx_change.keys()):
        if a in keep_set:
            avg = np.mean(action_ctx_change[a])
            print(f"    {a:20s}: {avg:.4f} (n={len(action_ctx_change[a])})")

    return {
        "separability_ratio": float(separation),
        "avg_inter_dist": float(avg_inter),
        "avg_intra_dist": float(avg_intra),
        "n_valid_actions": len(valid_actions),
    }


# ═══════════════════════════════════════════════════════════
# 4.3 Robustness (filtered)
# ═══════════════════════════════════════════════════════════

def mask_lower_body_points(pc, point_mask, percentile=30):
    pc_c = pc.clone()
    B, T, N, D = pc.shape
    for b in range(B):
        for t in range(T):
            z_vals = pc[b, t, :, 2]
            valid = pc[b, t, :, :3].norm(dim=-1) > 1e-6
            if valid.sum() < 2:
                continue
            z_valid = z_vals[valid]
            threshold = torch.quantile(z_valid, percentile / 100.0)
            lower = valid & (z_vals <= threshold)
            pc_c[b, t, lower] = 0
    return pc_c


@torch.no_grad()
def run_robustness(model, loader, device, keep_classes):
    print(f"\n{'='*60}")
    print(f"Exp 4.3: Robustness (filtered: {keep_classes})")
    print(f"{'='*60}")
    model.eval()
    keep_set = set(keep_classes)

    print("\n  Part A: Lower-body point masking")
    clean_alpha, masked_alpha = [], []
    clean_sigma, masked_sigma = [], []
    node_alpha_clean = defaultdict(list)
    node_alpha_masked = defaultdict(list)

    for batch in tqdm(loader, desc="Robustness (body mask)"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        out_c = model.forward_sequence(pc, mask)
        pc_m = mask_lower_body_points(pc, None, percentile=30)
        out_m = model.forward_sequence(pc_m, mask)
        B, T = pc.shape[:2]
        for b in range(B):
            if batch["actions"][b] not in keep_set:
                continue
            Tv = int(mask[b].sum().item())
            for t in range(Tv):
                ac = out_c["confidence"][b, t, :, 0].cpu().numpy()
                am = out_m["confidence"][b, t, :, 0].cpu().numpy()
                clean_alpha.append(ac.mean())
                masked_alpha.append(am.mean())
                for m in range(ac.shape[0]):
                    node_alpha_clean[m].append(ac[m])
                    node_alpha_masked[m].append(am[m])
                sc = out_c["post_logvar"][b, t].exp().mean().item()
                sm = out_m["post_logvar"][b, t].exp().mean().item()
                clean_sigma.append(sc)
                masked_sigma.append(sm)

    if not clean_alpha:
        print("  ⚠ No valid samples after filtering")
        return {}

    ca, ma = np.mean(clean_alpha), np.mean(masked_alpha)
    cs, ms = np.mean(clean_sigma), np.mean(masked_sigma)
    print(f"  Confidence (α):  clean={ca:.4f} → masked={ma:.4f} (Δ={ma-ca:+.4f})")
    print(f"  Uncertainty (Σ): clean={cs:.4f} → masked={ms:.4f} (Δ={ms-cs:+.4f})")

    print(f"\n  Per-node α change:")
    M = len(node_alpha_clean)
    for m in range(M):
        nc, nm = np.mean(node_alpha_clean[m]), np.mean(node_alpha_masked[m])
        delta = nm - nc
        label = "↓ sensitive" if delta < -0.05 else "≈ stable"
        print(f"    Node {m}: {nc:.4f} → {nm:.4f} (Δ={delta:+.4f}) {label}")

    # Part B: point drop — only on filtered samples
    print("\n  Part B: Point drop severity curve")
    severities = [0.0, 0.3, 0.5, 0.7, 0.9]
    drop_results = {}
    for sev in severities:
        all_g, all_actions = [], []
        for batch in loader:
            pc = batch["point_cloud"].to(device)
            mask_t = batch["temporal_mask"].to(device)
            if sev > 0:
                B, T, N, D = pc.shape
                pc_d = pc.clone()
                for b_i in range(B):
                    for t_i in range(T):
                        n_drop = int(N * sev)
                        idx = torch.randperm(N)[:n_drop]
                        pc_d[b_i, t_i, idx] = 0
                g = model.encode(pc_d, mask_t)
            else:
                g = model.encode(pc, mask_t)

            # Filter
            for i, a in enumerate(batch["actions"]):
                if a in keep_set:
                    all_g.append(g[i:i+1].cpu())
                    all_actions.append(a)

        if not all_g:
            drop_results[sev] = 0.0
            continue
        g_all = torch.cat(all_g).numpy()
        unique = sorted(set(all_actions))
        if len(unique) >= 2:
            amap = {a: i for i, a in enumerate(unique)}
            y = np.array([amap[a] for a in all_actions])
            perm = np.random.RandomState(42).permutation(len(y))
            sp = int(len(y) * 0.8)
            if sp > 1 and len(y) - sp > 0:
                knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
                knn.fit(g_all[perm[:sp]], y[perm[:sp]])
                acc = float(accuracy_score(y[perm[sp:]], knn.predict(g_all[perm[sp:]])))
            else:
                acc = 0.0
        else:
            acc = 0.0
        drop_results[sev] = acc
        print(f"    drop={sev:.0%}: kNN_acc={acc:.4f}")

    return {
        "body_mask": {
            "alpha_clean": float(ca), "alpha_masked": float(ma),
            "sigma_clean": float(cs), "sigma_masked": float(ms),
        },
        "point_drop_curve": drop_results,
    }


# ═══════════════════════════════════════════════════════════
# 4.5 Semantic Readiness (filtered) — 핵심 실험
# ═══════════════════════════════════════════════════════════

def run_linear_probe(embeddings, config, keep_classes):
    print(f"\n{'='*60}")
    print(f"Exp 4.5: Semantic Readiness — FILTERED")
    print(f"  Classes: {keep_classes}")
    print(f"{'='*60}")

    # Filter
    emb = filter_extracted(embeddings, keep_classes)
    g = emb["g_radar"].numpy()
    actions = emb["actions"]
    unique = sorted(set(actions))
    K = len(unique)

    if K < 2:
        print("  ✗ < 2 classes after filtering")
        return {"status": "skipped"}

    amap = {a: i for i, a in enumerate(unique)}
    y = np.array([amap[a] for a in actions])

    dist = Counter(actions)
    print(f"  Samples: {len(actions)}, Classes: {K}")
    print(f"  Distribution:")
    for a, c in dist.most_common():
        print(f"    {a:20s}: {c:4d} ({c/len(actions)*100:.1f}%)")

    majority_pct = max(dist.values()) / len(actions)
    random_pct = 1.0 / K
    print(f"  Majority baseline: {majority_pct:.4f}")
    print(f"  Random baseline:   {random_pct:.4f}")

    N = len(y)
    perm = np.random.RandomState(42).permutation(N)
    sp = int(N * 0.8)
    tr_idx, te_idx = perm[:sp], perm[sp:]
    if sp < 2 or len(te_idx) < 1:
        return {"status": "insufficient_data"}

    results = {
        "majority_baseline": float(majority_pct),
        "random_baseline": float(random_pct),
        "num_classes": K,
        "num_samples": len(actions),
        "classes": unique,
    }

    # ── k-NN ──
    knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
    knn.fit(g[tr_idx], y[tr_idx])
    knn_pred = knn.predict(g[te_idx])
    knn_acc = accuracy_score(y[te_idx], knn_pred)
    knn_f1 = f1_score(y[te_idx], knn_pred, average="weighted", zero_division=0)
    knn_lift = knn_acc - majority_pct
    print(f"\n  k-NN (k=5): acc={knn_acc:.4f}, F1={knn_f1:.4f}, "
          f"lift_vs_majority={knn_lift:+.4f}, lift_vs_random={knn_acc-random_pct:+.4f}")
    results.update(knn_accuracy=knn_acc, knn_f1=knn_f1,
                   knn_lift_majority=float(knn_lift),
                   knn_lift_random=float(knn_acc - random_pct))

    # ── Linear probe ──
    D = g.shape[1]
    probe = nn.Linear(D, K)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Xt = torch.from_numpy(g[tr_idx]).float()
    yt = torch.from_numpy(y[tr_idx]).long()
    Xe = torch.from_numpy(g[te_idx]).float()

    for ep in range(100):  # 50→100 epochs
        probe.train()
        loss = F.cross_entropy(probe(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(Xe).argmax(1).numpy()
    lin_acc = accuracy_score(y[te_idx], pred)
    lin_f1 = f1_score(y[te_idx], pred, average="weighted", zero_division=0)
    lin_lift = lin_acc - majority_pct
    print(f"  Linear (100ep): acc={lin_acc:.4f}, F1={lin_f1:.4f}, "
          f"lift_vs_majority={lin_lift:+.4f}, lift_vs_random={lin_acc-random_pct:+.4f}")
    results.update(linear_accuracy=lin_acc, linear_f1=lin_f1,
                   linear_lift_majority=float(lin_lift),
                   linear_lift_random=float(lin_acc - random_pct))

    # ── Per-class accuracy ──
    print(f"\n  Per-class accuracy:")
    print(f"    {'class':20s} {'kNN':>8s} {'Linear':>8s} {'n_test':>8s}")
    print(f"    {'─'*48}")
    per_class = {}
    for cls_idx, cls_name in enumerate(unique):
        cls_mask = y[te_idx] == cls_idx
        n = int(cls_mask.sum())
        if n > 0:
            knn_cls = float((knn_pred[cls_mask] == cls_idx).mean())
            lin_cls = float((pred[cls_mask] == cls_idx).mean())
        else:
            knn_cls, lin_cls = 0.0, 0.0
        per_class[cls_name] = {"knn": knn_cls, "linear": lin_cls, "n": n}
        print(f"    {cls_name:20s} {knn_cls:8.4f} {lin_cls:8.4f} {n:8d}")
    results["per_class"] = per_class

    # ── Confusion matrix ──
    print(f"\n  Confusion matrix (linear probe):")
    cm = confusion_matrix(y[te_idx], pred)
    print(f"    {'':20s}", end="")
    for c in unique:
        print(f" {c[:6]:>6s}", end="")
    print()
    for i, cls_name in enumerate(unique):
        print(f"    {cls_name:20s}", end="")
        for j in range(len(unique)):
            print(f" {cm[i,j]:6d}", end="")
        print()
    results["confusion_matrix"] = cm.tolist()

    # ── Cross-modal retrieval ──
    if "g_motion" in emb:
        print(f"\n  Cross-modal Retrieval (filtered):")
        gr = F.normalize(emb["g_radar"], dim=-1).numpy()
        gm = F.normalize(emb["g_motion"], dim=-1).numpy()
        N_all = gr.shape[0]
        for name, q, gal in [("R→M", gr, gm), ("M→R", gm, gr)]:
            sim = q @ gal.T
            ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] + 1
                              for i in range(N_all)])
            r1 = (ranks <= 1).mean()
            r5 = (ranks <= 5).mean()
            r10 = (ranks <= 10).mean()
            mrr = (1.0 / ranks).mean()
            print(f"    {name}: R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, MRR={mrr:.4f}")
            results[f"retrieval_{name}"] = {
                "R@1": float(r1), "R@5": float(r5),
                "R@10": float(r10), "MRR": float(mrr),
            }

    return results


# ═══════════════════════════════════════════════════════════
# 4.4 Ablation (simplified)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def _quick_chamfer(model, loader, device, keep_classes, max_batches=20):
    model.eval()
    keep_set = set(keep_classes)
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        out = model.forward_sequence(pc, mask)
        recon = out["recon_sequence"]
        gt = pc[..., :3]
        gt_valid = gt.norm(dim=-1) > 1e-6
        B, T = pc.shape[:2]
        for b in range(B):
            if batch["actions"][b] not in keep_set:
                continue
            Tv = int(mask[b].sum().item())
            for t in range(min(Tv, 5)):
                vm = gt_valid[b, t]
                if vm.sum() < 2:
                    continue
                g, p = gt[b, t, vm], recon[b, t]
                gn = g - g.mean(0, keepdim=True)
                pn = p - p.mean(0, keepdim=True)
                d = (pn.unsqueeze(1) - gn.unsqueeze(0)).pow(2).sum(-1)
                total += (d.min(1).values.mean() + d.min(0).values.mean()).item()
                count += 1
    return total / max(count, 1)


def run_ablation(base_ckpt, ablation_dir, loader, device, config, keep_classes, emb):
    print(f"\n{'='*60}")
    print(f"Exp 4.4: Ablation Studies (filtered)")
    print(f"{'='*60}")
    results = {}

    # Full model
    emb_f = filter_extracted(emb, keep_classes)
    g = emb_f["g_radar"].numpy()
    actions = emb_f["actions"]
    unique = sorted(set(actions))
    if len(unique) >= 2:
        amap = {a: i for i, a in enumerate(unique)}
        y = np.array([amap[a] for a in actions])
        perm = np.random.RandomState(42).permutation(len(y))
        sp = int(len(y) * 0.8)
        knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
        knn.fit(g[perm[:sp]], y[perm[:sp]])
        full_acc = float(accuracy_score(y[perm[sp:]], knn.predict(g[perm[sp:]])))
    else:
        full_acc = 0.0

    model = load_model(config, base_ckpt, device)
    full_chamfer = _quick_chamfer(model, loader, device, keep_classes)
    results["full"] = {"knn_acc": full_acc, "chamfer": full_chamfer}
    print(f"  Full model: kNN={full_acc:.4f}, Chamfer={full_chamfer:.6f}")

    ABLATION_CONFIGS = {
        "w/o_context": {"disable_context": True},
        "w/o_self_attn": {"disable_self_attn": True},
        "w/o_confidence": {"disable_confidence": True},
        "w/o_contrastive": {"lambda_motion": 0.0},
    }
    for abl_name in ABLATION_CONFIGS:
        ckpt = os.path.join(ablation_dir, f"{abl_name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  {abl_name:20s}: ✗ not found")
            results[abl_name] = {"status": "missing"}
            continue
        print(f"  {abl_name:20s}: found, evaluating...")

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Filtered Stage 1 Evaluation")
    p.add_argument("--config", default="configs/latent_graph.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--experiment", default="all",
                   choices=["all", "context", "robustness", "ablation", "linear_probe"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", default="results/stage1_eval_filtered.json")
    p.add_argument("--ablation_dir", default="checkpoints/ablations")
    p.add_argument("--label_method", default="auto")
    p.add_argument("--label_cache", default="results/action_labels_cache.json")
    # ★ 핵심 추가 인자
    p.add_argument("--exclude_top", type=int, default=1,
                   help="가장 빈도 높은 N개 클래스 제외 (default=1)")
    p.add_argument("--keep_top", type=int, default=5,
                   help="제외 후 상위 K개 클래스만 사용 (default=5)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device set to use {device}")

    model = load_model(config, args.ckpt, device)

    motion_enc = None
    try:
        motion_enc = build_motion_encoder(config).to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        if "motion_enc_state_dict" in ckpt:
            motion_enc.load_state_dict(ckpt["motion_enc_state_dict"], strict=False)
        motion_enc.eval()
        for p in motion_enc.parameters():
            p.requires_grad = False
    except Exception as e:
        print(f"Motion encoder not available: {e}")

    action_labeler = ActionLabeler(method=args.label_method, cache_path=args.label_cache)

    lg = config["latent_graph"]
    me = config.get("motion_encoder", {})
    ds = EvalDataset(
        args.data_dir,
        max_T=config.get("dataset", {}).get("max_seq_len", 100),
        max_N=config.get("dataset", {}).get("points_per_frame", 128),
        D=lg.get("point_in_dim", 4),
        max_motion_frames=config.get("dataset", {}).get("max_motion_frames", 300),
        motion_type=me.get("input_type", "humanml3d"),
        action_labeler=action_labeler,
    )

    # ★ 클래스 필터링 결정
    keep_classes, excluded_classes = compute_keep_classes(
        ds, exclude_top=args.exclude_top, keep_top=args.keep_top)

    # ★ 필터링된 인덱스로 Subset 생성
    filtered_indices = filter_dataset_indices(ds, keep_classes)
    filtered_ds = Subset(ds, filtered_indices)
    print(f"\n  Filtered dataset: {len(filtered_ds)} samples "
          f"(from {len(ds)} total)")

    # 전체 데이터 로더 (배치 내 필터링은 각 함수에서)
    full_loader = DataLoader(ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate, num_workers=2)
    # 필터링된 데이터 로더 (linear probe용)
    filtered_loader = DataLoader(filtered_ds, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate, num_workers=2)

    R = {
        "config": args.config,
        "ckpt": args.ckpt,
        "label_method": action_labeler.method,
        "exclude_top": args.exclude_top,
        "keep_top": args.keep_top,
        "kept_classes": keep_classes,
        "excluded_classes": excluded_classes,
        "num_filtered_samples": len(filtered_ds),
    }
    run_all = args.experiment == "all"

    # Extract embeddings (전체 데이터에서 → 나중에 필터)
    emb = None
    if run_all or args.experiment == "linear_probe":
        emb = extract_all(model, full_loader, device, motion_enc)

    if run_all or args.experiment == "context":
        R["context"] = run_context_interpretability(
            model, full_loader, device, keep_classes)

    if run_all or args.experiment == "robustness":
        R["robustness"] = run_robustness(
            model, full_loader, device, keep_classes)

    if run_all or args.experiment == "ablation":
        R["ablation"] = run_ablation(
            args.ckpt, args.ablation_dir, full_loader, device,
            config, keep_classes, emb)

    if run_all or args.experiment == "linear_probe":
        R["linear_probe"] = run_linear_probe(emb, config, keep_classes)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    class Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(args.output, "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, cls=Enc)
    print(f"\nSaved: {args.output}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Summary (filtered: {keep_classes})")
    print(f"{'='*60}")
    print(f"  Excluded: {excluded_classes}")
    print(f"  Samples:  {len(filtered_ds)}/{len(ds)}")

    if "context" in R:
        print(f"  Context separability: {R['context'].get('separability_ratio', '?'):.4f}")

    if "robustness" in R and "body_mask" in R["robustness"]:
        bm = R["robustness"]["body_mask"]
        print(f"  α change: {bm['alpha_clean']:.4f} → {bm['alpha_masked']:.4f}")

    if "linear_probe" in R and "linear_accuracy" in R["linear_probe"]:
        lp = R["linear_probe"]
        print(f"  Majority baseline: {lp['majority_baseline']:.4f}")
        print(f"  Random baseline:   {lp['random_baseline']:.4f}")
        print(f"  k-NN:    {lp['knn_accuracy']:.4f} "
              f"(lift vs majority: {lp['knn_lift_majority']:+.4f}, "
              f"vs random: {lp['knn_lift_random']:+.4f})")
        print(f"  Linear:  {lp['linear_accuracy']:.4f} "
              f"(lift vs majority: {lp['linear_lift_majority']:+.4f}, "
              f"vs random: {lp['linear_lift_random']:+.4f})")
        if "retrieval_R→M" in lp:
            rm = lp["retrieval_R→M"]
            print(f"  Retrieval R→M: R@1={rm['R@1']:.4f}, R@5={rm['R@5']:.4f}")


if __name__ == "__main__":
    main()