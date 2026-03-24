#!/usr/bin/env python3
"""
Stage 1 Evaluation — v5.4 + LLM Action Labeling

변경사항:
  [fix-H] 키워드 매칭 → LLM 기반 action label 생성
          1순위: zero-shot classification (facebook/bart-large-mnli)
          2순위: sentence embedding similarity (all-MiniLM-L6-v2)
          3순위: 개선된 키워드 매칭 (fallback)
          + majority baseline 대비 lift 리포팅
          + per-class accuracy 리포팅
          + label 캐싱 (두 번째 실행부터 즉시)

사용법:
  python scripts/evaluate.py --config configs/latent_graph.yaml \
      --ckpt checkpoints/latent_graph/latent_graph_best.pt \
      --data_dir data/radar_text_dataset/test

  # labeling 방법 지정
  python scripts/evaluate.py ... --label_method zero_shot
  python scripts/evaluate.py ... --label_method sentence_similarity
  python scripts/evaluate.py ... --label_method keyword
"""

import os, sys, json, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.latent_graph_dynamics import LatentGraphDynamicsModel
from models.motion_encoder import build_motion_encoder


# ═══════════════════════════════════════════════════════════
# ★ LLM-based Action Labeling
# ═══════════════════════════════════════════════════════════

ACTION_CATEGORIES = [
    "walking or strolling",
    "running or jogging",
    "jumping or hopping",
    "sitting down or seated",
    "standing up or standing still",
    "kicking",
    "punching or striking",
    "throwing or tossing",
    "waving hands or arms",
    "picking up or grabbing an object",
    "pushing or pulling",
    "bending or bowing",
    "crouching or squatting",
    "turning or spinning",
    "dancing or rhythmic movement",
    "climbing or stepping up",
    "balancing on one leg",
    "stretching or reaching",
    "arm circles or arm exercises",
    "full body exercise or workout",
]

ACTION_SHORT_LABELS = [
    "walk", "run", "jump", "sit", "stand",
    "kick", "punch", "throw", "wave", "pick_up",
    "push", "bend", "crouch", "turn", "dance",
    "climb", "balance", "stretch", "arm_exercise", "exercise",
]


class ActionLabeler:
    def __init__(self, method="auto", cache_path=None):
        self.method = method
        self.cache_path = cache_path
        self.cache = {}
        self._classifier = None
        self._embedder = None
        self._cat_embeddings = None

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
        except Exception as e:
            print(f"[ActionLabeler] zero-shot unavailable: {e}")

        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._cat_embeddings = self._embedder.encode(
                ACTION_CATEGORIES, normalize_embeddings=True)
            print("[ActionLabeler] ✓ sentence-transformers loaded")
            return "sentence_similarity"
        except Exception as e:
            print(f"[ActionLabeler] sentence-transformers unavailable: {e}")

        print("[ActionLabeler] Using enhanced keyword matching")
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
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._cat_embeddings = self._embedder.encode(
                ACTION_CATEGORIES, normalize_embeddings=True)
        text_embs = self._embedder.encode(texts, normalize_embeddings=True,
                                           show_progress_bar=True)
        sims = text_embs @ self._cat_embeddings.T
        best_indices = sims.argmax(axis=1)
        return [ACTION_SHORT_LABELS[idx] for idx in best_indices]

    def _keyword_label(self, text):
        text_lower = text.lower()
        words = set(text_lower.split())

        ENHANCED_KEYWORDS = {
            "jump":     ["jump", "jumping", "jumps", "hop", "hopping", "leap",
                         "leaping", "jacks", "jack"],
            "kick":     ["kick", "kicking", "kicks"],
            "punch":    ["punch", "punching", "punches", "strike", "striking",
                         "jab", "boxing", "uppercut"],
            "throw":    ["throw", "throwing", "throws", "toss", "tossing", "hurl"],
            "pick_up":  ["pick", "picks", "picking", "grab", "grabs", "grabbing",
                         "grasp", "lift", "lifting"],
            "push":     ["push", "pushing", "pull", "pulling"],
            "wave":     ["wave", "waving", "waves", "beckon"],
            "dance":    ["dance", "dancing", "dances", "sway", "swaying", "groove",
                         "salsa", "waltz", "ballet"],
            "climb":    ["climb", "climbing", "climbs", "step up", "stepping",
                         "mountain climber"],
            "balance":  ["balance", "balancing", "one leg", "tightrope", "wobble"],
            "stretch":  ["stretch", "stretching", "stretches", "reach", "reaching",
                         "extend", "extending"],
            "bend":     ["bend", "bending", "bends", "bow", "bowing", "lean",
                         "leaning", "stoop"],
            "crouch":   ["crouch", "crouching", "squat", "squatting", "squats",
                         "kneel", "kneeling", "lunge", "lunging", "lunges"],
            "turn":     ["turn", "turning", "turns", "rotate", "rotating", "spin",
                         "spinning", "pivot", "pivoting"],
            "run":      ["run", "running", "runs", "jog", "jogging", "sprint",
                         "sprinting", "dash"],
            "walk":     ["walk", "walking", "walks", "stroll", "strolling", "step",
                         "steps", "pace", "pacing", "march", "marching"],
            "sit":      ["sit", "sitting", "sits", "sat", "seat", "seated"],
            "stand":    ["stand", "standing", "stands", "stood", "upright"],
            "exercise": ["exercise", "exercising", "workout", "pushup", "push-up",
                         "situp", "sit-up", "burpee", "plank", "crunches", "crunch",
                         "rep", "reps", "count", "counts"],
            "arm_exercise": ["arm circle", "arm swing", "shoulder", "flap",
                             "flapping", "windmill"],
        }

        for action, keywords in ENHANCED_KEYWORDS.items():
            for kw in keywords:
                if " " in kw:
                    if kw in text_lower:
                        return action
                else:
                    if kw in words:
                        return action
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

            dist = Counter(self._action_labels)
            print(f"[EvalDataset] Action distribution ({len(dist)} classes):")
            for action, count in dist.most_common():
                print(f"    {action:20s}: {count:4d} ({count/len(self.samples)*100:.1f}%)")
        else:
            self._action_labels = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        text = str(data["text"])

        if self._action_labels is not None:
            action = self._action_labels[idx]
        elif "action_label" in data:
            action = str(data["action_label"])
        else:
            action = "other"

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
# 4.2 Context Interpretability
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run_context_interpretability(model, loader, device):
    print("\n" + "=" * 60)
    print("Exp 4.2: Context Interpretability")
    print("=" * 60)
    model.eval()
    action_ctx_means = defaultdict(list)
    action_ctx_change = defaultdict(list)
    all_ctx_for_tsne = []
    all_ctx_labels = []

    for batch in tqdm(loader, desc="Context analysis"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        out = model.forward_sequence(pc, mask)
        ctx = out["context_history"]
        B, T, C = ctx.shape
        for b in range(B):
            Tv = int(mask[b].sum().item())
            action = batch["actions"][b]
            ct = ctx[b, :Tv].cpu().numpy()
            mean_c = ct.mean(axis=0)
            action_ctx_means[action].append(mean_c)
            if Tv > 1:
                deltas = np.linalg.norm(ct[1:] - ct[:-1], axis=-1)
                action_ctx_change[action].append(deltas.mean())
            step = max(1, Tv // 10)
            for t in range(0, Tv, step):
                all_ctx_for_tsne.append(ct[t])
                all_ctx_labels.append(action)

    actions_list = sorted(action_ctx_means.keys())
    valid_actions = [a for a in actions_list if len(action_ctx_means[a]) >= 2]

    if len(valid_actions) < 2:
        print("  ⚠ < 2 valid classes with ≥2 samples")
        separation, avg_inter, avg_intra = 0.0, 0.0, 0.0
    else:
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
    print(f"    valid classes (≥2 samples): {len(valid_actions)}/{len(actions_list)}")

    print(f"\n  Context temporal change rate:")
    change_by_action = {}
    for a in sorted(action_ctx_change.keys()):
        avg = np.mean(action_ctx_change[a])
        change_by_action[a] = float(avg)
        print(f"    {a:20s}: {avg:.4f} (n={len(action_ctx_change[a])})")

    return {
        "separability_ratio": float(separation),
        "avg_inter_dist": float(avg_inter),
        "avg_intra_dist": float(avg_intra),
        "change_rate_by_action": change_by_action,
        "tsne_data": np.stack(all_ctx_for_tsne).tolist() if all_ctx_for_tsne else [],
        "tsne_labels": all_ctx_labels,
        "n_actions": len(actions_list),
        "n_valid_actions": len(valid_actions),
    }


# ═══════════════════════════════════════════════════════════
# 4.3 Robustness
# ═══════════════════════════════════════════════════════════

def mask_lower_body_points(pc, point_mask, percentile=30):
    pc_c = pc.clone()
    mask_c = point_mask.clone()
    B, T, N, D = pc.shape
    for b in range(B):
        for t in range(T):
            if not point_mask[b, t].any():
                continue
            z_vals = pc[b, t, :, 2]
            valid = pc[b, t, :, :3].norm(dim=-1) > 1e-6
            if valid.sum() < 2:
                continue
            z_valid = z_vals[valid]
            threshold = torch.quantile(z_valid, percentile / 100.0)
            lower = valid & (z_vals <= threshold)
            pc_c[b, t, lower] = 0
            mask_c[b, t, lower] = False
    return pc_c, mask_c


@torch.no_grad()
def run_robustness(model, loader, device):
    print("\n" + "=" * 60)
    print("Exp 4.3: Robustness / Observability Analysis")
    print("=" * 60)
    model.eval()
    print("\n  Part A: Lower-body point masking")
    clean_alpha, masked_alpha = [], []
    clean_sigma, masked_sigma = [], []
    node_alpha_clean = defaultdict(list)
    node_alpha_masked = defaultdict(list)

    for batch in tqdm(loader, desc="Robustness (body mask)"):
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        pmask = pc[..., :3].norm(dim=-1) > 1e-6
        out_c = model.forward_sequence(pc, mask)
        pc_m, pmask_m = mask_lower_body_points(pc, pmask, percentile=30)
        out_m = model.forward_sequence(pc_m, mask)
        B, T = pc.shape[:2]
        for b in range(B):
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

    ca, ma = np.mean(clean_alpha), np.mean(masked_alpha)
    cs, ms = np.mean(clean_sigma), np.mean(masked_sigma)
    print(f"  Confidence (α):  clean={ca:.4f} → masked={ma:.4f} (Δ={ma-ca:+.4f})")
    print(f"  Uncertainty (Σ): clean={cs:.4f} → masked={ms:.4f} (Δ={ms-cs:+.4f})")
    print(f"\n  Per-node α change:")
    node_results = {}
    M = len(node_alpha_clean)
    for m in range(M):
        nc = np.mean(node_alpha_clean[m])
        nm = np.mean(node_alpha_masked[m])
        delta = nm - nc
        node_results[m] = {"clean": float(nc), "masked": float(nm), "delta": float(delta)}
        label = "↓ sensitive" if delta < -0.05 else "≈ stable"
        print(f"    Node {m}: {nc:.4f} → {nm:.4f} (Δ={delta:+.4f}) {label}")

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
            all_g.append(g.cpu())
            all_actions.extend(batch["actions"])
        g_all = torch.cat(all_g).numpy()
        unique = sorted(set(all_actions))
        if len(unique) >= 2:
            amap = {a: i for i, a in enumerate(unique)}
            y = np.array([amap.get(a, 0) for a in all_actions])
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
            "per_node": node_results,
        },
        "point_drop_curve": drop_results,
    }


# ═══════════════════════════════════════════════════════════
# 4.4 Ablation
# ═══════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "w/o_context": {"disable_context": True},
    "w/o_self_attn": {"disable_self_attn": True},
    "w/o_confidence": {"disable_confidence": True},
    "w/o_contrastive": {"lambda_motion": 0.0},
}

def create_ablation_config(base_config, ablation_name):
    import copy
    config = copy.deepcopy(base_config)
    lg = config["latent_graph"]
    for k, v in ABLATION_CONFIGS[ablation_name].items():
        lg[k] = v
    return config

@torch.no_grad()
def run_ablation(base_ckpt, ablation_dir, loader, device, config):
    print("\n" + "=" * 60)
    print("Exp 4.4: Ablation Studies")
    print("=" * 60)
    results = {}
    model = load_model(config, base_ckpt, device)
    emb = extract_all(model, loader, device)
    full_acc = _quick_knn(emb)
    full_chamfer = _quick_chamfer(model, loader, device)
    results["full"] = {"knn_acc": full_acc, "chamfer": full_chamfer}
    print(f"  Full model: kNN={full_acc:.4f}, Chamfer={full_chamfer:.6f}")
    for abl_name in ABLATION_CONFIGS:
        ckpt = os.path.join(ablation_dir, f"{abl_name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  {abl_name:20s}: ✗ not found")
            results[abl_name] = {"status": "missing"}
            continue
        abl_cfg = create_ablation_config(config, abl_name)
        try:
            abl_model = load_model(abl_cfg, ckpt, device)
            abl_emb = extract_all(abl_model, loader, device)
            acc = _quick_knn(abl_emb)
            chamfer = _quick_chamfer(abl_model, loader, device)
            results[abl_name] = {"knn_acc": acc, "chamfer": chamfer}
            print(f"  {abl_name:20s}: kNN={acc:.4f} (Δ={acc-full_acc:+.4f}), "
                  f"Chamfer={chamfer:.6f} (Δ={chamfer-full_chamfer:+.6f})")
        except Exception as e:
            print(f"  {abl_name:20s}: ✗ {e}")
            results[abl_name] = {"status": "error", "error": str(e)}
    return results

def _quick_knn(emb):
    g = emb["g_radar"].numpy()
    actions = emb["actions"]
    unique = sorted(set(actions))
    if len(unique) < 2: return 0.0
    amap = {a: i for i, a in enumerate(unique)}
    y = np.array([amap[a] for a in actions])
    perm = np.random.RandomState(42).permutation(len(y))
    sp = int(len(y) * 0.8)
    if sp < 2: return 0.0
    knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
    knn.fit(g[perm[:sp]], y[perm[:sp]])
    return float(accuracy_score(y[perm[sp:]], knn.predict(g[perm[sp:]])))

@torch.no_grad()
def _quick_chamfer(model, loader, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        pc = batch["point_cloud"].to(device)
        mask = batch["temporal_mask"].to(device)
        out = model.forward_sequence(pc, mask)
        recon = out["recon_sequence"]
        gt = pc[..., :3]
        gt_valid = gt.norm(dim=-1) > 1e-6
        B, T = pc.shape[:2]
        for b in range(B):
            Tv = int(mask[b].sum().item())
            for t in range(min(Tv, 5)):
                vm = gt_valid[b, t]
                if vm.sum() < 2: continue
                g, p = gt[b, t, vm], recon[b, t]
                gn = g - g.mean(0, keepdim=True)
                pn = p - p.mean(0, keepdim=True)
                d = (pn.unsqueeze(1) - gn.unsqueeze(0)).pow(2).sum(-1)
                total += (d.min(1).values.mean() + d.min(0).values.mean()).item()
                count += 1
    return total / max(count, 1)


# ═══════════════════════════════════════════════════════════
# 4.5 Semantic Readiness
# ═══════════════════════════════════════════════════════════

def run_linear_probe(embeddings, config):
    print("\n" + "=" * 60)
    print("Exp 4.5: Semantic Readiness (Linear Probing)")
    print("=" * 60)
    g = embeddings["g_radar"].numpy()
    actions = embeddings["actions"]
    unique = sorted(set(actions))
    if len(unique) < 2:
        print("  ✗ < 2 classes")
        return {"status": "skipped"}

    amap = {a: i for i, a in enumerate(unique)}
    y = np.array([amap[a] for a in actions])
    K = len(unique)

    dist = Counter(actions)
    print(f"  Classes ({K}): {unique}")
    print(f"  Distribution:")
    for a, c in dist.most_common():
        print(f"    {a:20s}: {c:4d} ({c/len(actions)*100:.1f}%)")

    majority_pct = max(dist.values()) / len(actions)
    print(f"  Majority baseline: {majority_pct:.4f}")
    print(f"  Random baseline:   {1.0/K:.4f}")

    N = len(y)
    perm = np.random.RandomState(42).permutation(N)
    sp = int(N * 0.8)
    tr_idx, te_idx = perm[:sp], perm[sp:]
    if sp < 2 or len(te_idx) < 1:
        return {"status": "insufficient_data"}

    results = {"majority_baseline": float(majority_pct),
               "random_baseline": 1.0/K, "num_classes": K}

    # k-NN
    knn = KNeighborsClassifier(n_neighbors=min(5, sp-1), metric="cosine")
    knn.fit(g[tr_idx], y[tr_idx])
    knn_pred = knn.predict(g[te_idx])
    knn_acc = accuracy_score(y[te_idx], knn_pred)
    knn_f1 = f1_score(y[te_idx], knn_pred, average="weighted", zero_division=0)
    knn_lift = knn_acc - majority_pct
    print(f"\n  k-NN (k=5): acc={knn_acc:.4f}, F1={knn_f1:.4f}, lift={knn_lift:+.4f}")
    results.update(knn_accuracy=knn_acc, knn_f1=knn_f1, knn_lift=float(knn_lift))

    # Linear probe
    D = g.shape[1]
    probe = nn.Linear(D, K)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Xt = torch.from_numpy(g[tr_idx]).float()
    yt = torch.from_numpy(y[tr_idx]).long()
    Xe = torch.from_numpy(g[te_idx]).float()
    for _ in range(50):
        probe.train()
        loss = F.cross_entropy(probe(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(Xe).argmax(1).numpy()
    lin_acc = accuracy_score(y[te_idx], pred)
    lin_f1 = f1_score(y[te_idx], pred, average="weighted", zero_division=0)
    lin_lift = lin_acc - majority_pct
    print(f"  Linear (50ep): acc={lin_acc:.4f}, F1={lin_f1:.4f}, lift={lin_lift:+.4f}")
    results.update(linear_accuracy=lin_acc, linear_f1=lin_f1, linear_lift=float(lin_lift))
    results["confusion_matrix"] = confusion_matrix(y[te_idx], pred).tolist()
    results["class_names"] = unique

    # Per-class accuracy
    print(f"\n  Per-class accuracy (linear probe):")
    per_class = {}
    for cls_idx, cls_name in enumerate(unique):
        cls_mask = y[te_idx] == cls_idx
        if cls_mask.sum() > 0:
            cls_acc = float((pred[cls_mask] == cls_idx).mean())
            per_class[cls_name] = {"accuracy": cls_acc, "n": int(cls_mask.sum())}
            print(f"    {cls_name:20s}: {cls_acc:.4f} (n={cls_mask.sum()})")
    results["per_class"] = per_class

    # Cross-modal retrieval
    if "g_motion" in embeddings:
        print("\n  Cross-modal Retrieval:")
        gr = F.normalize(embeddings["g_radar"], dim=-1).numpy()
        gm = F.normalize(embeddings["g_motion"], dim=-1).numpy()
        N_all = gr.shape[0]
        for name, q, gal in [("R→M", gr, gm), ("M→R", gm, gr)]:
            sim = q @ gal.T
            ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] + 1
                              for i in range(N_all)])
            r1, r5, r10 = (ranks<=1).mean(), (ranks<=5).mean(), (ranks<=10).mean()
            mrr = (1.0/ranks).mean()
            print(f"    {name}: R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, MRR={mrr:.4f}")
            results[f"retrieval_{name}"] = {
                "R@1": float(r1), "R@5": float(r5),
                "R@10": float(r10), "MRR": float(mrr)}

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Stage 1 Evaluation (v5.4)")
    p.add_argument("--config", default="configs/latent_graph.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--experiment", default="all",
                   choices=["all","context","robustness","ablation","linear_probe"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", default="results/stage1_eval.json")
    p.add_argument("--ablation_dir", default="checkpoints/ablations")
    p.add_argument("--label_method", default="auto",
                   choices=["auto","zero_shot","sentence_similarity","keyword"])
    p.add_argument("--label_cache", default="results/action_labels_cache.json")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate, num_workers=2)

    R = {"config": args.config, "ckpt": args.ckpt, "label_method": action_labeler.method}
    run_all = args.experiment == "all"

    emb = None
    if run_all or args.experiment == "linear_probe":
        emb = extract_all(model, loader, device, motion_enc)

    if run_all or args.experiment == "context":
        R["context"] = run_context_interpretability(model, loader, device)
    if run_all or args.experiment == "robustness":
        R["robustness"] = run_robustness(model, loader, device)
    if run_all or args.experiment == "ablation":
        R["ablation"] = run_ablation(args.ckpt, args.ablation_dir, loader, device, config)
    if run_all or args.experiment == "linear_probe":
        R["linear_probe"] = run_linear_probe(emb, config)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    class Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    with open(args.output, "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, cls=Enc)
    print(f"\nSaved: {args.output}")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"  Label method: {action_labeler.method}")
    if "context" in R:
        print(f"  Context separability: {R['context']['separability_ratio']:.4f}")
        print(f"    valid classes: {R['context']['n_valid_actions']}")
    if "robustness" in R:
        bm = R["robustness"]["body_mask"]
        print(f"  α change (body mask): {bm['alpha_clean']:.4f} → {bm['alpha_masked']:.4f}")
        print(f"  Σ change (body mask): {bm['sigma_clean']:.4f} → {bm['sigma_masked']:.4f}")
    if "linear_probe" in R and "linear_accuracy" in R["linear_probe"]:
        lp = R["linear_probe"]
        print(f"  Majority baseline: {lp.get('majority_baseline', '?'):.4f}")
        print(f"  Linear probe: {lp['linear_accuracy']:.4f} (lift: {lp.get('linear_lift', '?'):+.4f})")
        print(f"  k-NN:         {lp['knn_accuracy']:.4f} (lift: {lp.get('knn_lift', '?'):+.4f})")
        if "retrieval_R→M" in lp:
            rm = lp["retrieval_R→M"]
            print(f"  Retrieval R→M: R@1={rm['R@1']:.4f}, R@5={rm['R@5']:.4f}")


if __name__ == "__main__":
    main()