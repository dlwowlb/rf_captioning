"""
Stage 1 Validation: Embedding Visualization + Cosine Similarity.

1. t-SNE / PCA Visualization:
   - Extract Z_Frame-Query from test heatmaps
   - Visualize in 2D, color by action label
   - Check clustering: 'walking' tokens cluster together, 'sitting' separately

2. Cosine Similarity Check:
   - For each test radar input, compute cosine similarity with all text embeddings
   - Verify that the correct text has highest similarity
   - Report accuracy and average similarity gap

Usage:
    python -m radar_captioning.validate_stage1 \
        --data_dir data/toy_dataset \
        --ckpt output/stage1/best_stage1.pt \
        --output_dir output/validation/stage1
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from .config import FullConfig
from .data.toy_dataset import RadarCaptionDataset
from .models.radar_q_former import RadarQFormer
from .models.text_encoder import SurrogateTextEncoder


def collate_fn(batch):
    rd = torch.stack([b["rd"] for b in batch])
    ra = torch.stack([b["ra"] for b in batch])
    da = torch.stack([b["da"] for b in batch])
    texts = [b["text"] for b in batch]
    labels = [b["action_label"] for b in batch]
    return {"rd": rd, "ra": ra, "da": da, "texts": texts, "labels": labels}


def validate_stage1(
    data_dir: str,
    ckpt_path: str,
    output_dir: str,
    config: FullConfig = None,
):
    """Run Stage 1 validation: embedding visualization + cosine similarity."""
    if config is None:
        config = FullConfig()

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(config.device)

    print("=" * 60)
    print("Stage 1 Validation")
    print("=" * 60)

    # Load models
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

    text_encoder = SurrogateTextEncoder(embed_dim=config.stage1.z_text_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    radar_qformer.load_state_dict(ckpt["radar_qformer"])
    text_encoder.load_state_dict(ckpt["text_encoder"])
    print(f"Loaded checkpoint: {ckpt_path}")

    radar_qformer.eval()
    text_encoder.eval()

    # Load val dataset
    val_dataset = RadarCaptionDataset(
        data_dir, split="val", num_frames=config.data.num_frames,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # ================================================================
    # Extract embeddings
    # ================================================================
    all_z_radar = []
    all_z_text = []
    all_labels = []
    all_texts = []

    with torch.no_grad():
        for batch in val_loader:
            rd = batch["rd"].to(device)
            ra = batch["ra"].to(device)
            da = batch["da"].to(device)
            texts = batch["texts"]
            labels = batch["labels"]

            frame_queries = radar_qformer(rd, ra, da)
            z_radar = frame_queries.mean(dim=1)  # (B, D)
            z_text = text_encoder(texts)          # (B, D)

            all_z_radar.append(z_radar.cpu().numpy())
            all_z_text.append(z_text.cpu().numpy())
            all_labels.extend(labels)
            all_texts.extend(texts)

    z_radar_all = np.concatenate(all_z_radar, axis=0)  # (N, D)
    z_text_all = np.concatenate(all_z_text, axis=0)     # (N, D)
    N = z_radar_all.shape[0]
    print(f"Extracted {N} embeddings")

    # ================================================================
    # 1. t-SNE / PCA Visualization
    # ================================================================
    print("\n--- Embedding Visualization ---")

    # Get unique labels and assign colors
    unique_labels = sorted(set(all_labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_indices = np.array([label_to_idx[l] for l in all_labels])

    # Combine radar and text embeddings for joint visualization
    combined = np.concatenate([z_radar_all, z_text_all], axis=0)  # (2N, D)
    source_labels = ["radar"] * N + ["text"] * N
    action_labels_combined = all_labels + all_labels

    # PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PCA - colored by action
    ax = axes[0]
    for i, label in enumerate(unique_labels):
        mask_r = (np.array(all_labels) == label)
        mask_t = mask_r
        ax.scatter(combined_pca[:N][mask_r, 0], combined_pca[:N][mask_r, 1],
                   c=[colors[i]], marker="o", alpha=0.7, s=60, label=f"{label} (radar)")
        ax.scatter(combined_pca[N:][mask_t, 0], combined_pca[N:][mask_t, 1],
                   c=[colors[i]], marker="x", alpha=0.7, s=60, label=f"{label} (text)")
    ax.set_title("PCA: Radar (o) vs Text (x) Embeddings")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=7, loc="best", ncol=2)

    # t-SNE
    perplexity = min(30, len(combined) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    combined_tsne = tsne.fit_transform(combined)

    ax = axes[1]
    for i, label in enumerate(unique_labels):
        mask = (np.array(all_labels) == label)
        ax.scatter(combined_tsne[:N][mask, 0], combined_tsne[:N][mask, 1],
                   c=[colors[i]], marker="o", alpha=0.7, s=60, label=f"{label} (radar)")
        ax.scatter(combined_tsne[N:][mask, 0], combined_tsne[N:][mask, 1],
                   c=[colors[i]], marker="x", alpha=0.7, s=60, label=f"{label} (text)")
    ax.set_title("t-SNE: Radar (o) vs Text (x) Embeddings")
    ax.legend(fontsize=7, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_visualization.png"), dpi=150)
    plt.close()
    print(f"Saved embedding visualization → {output_dir}/embedding_visualization.png")

    # ================================================================
    # 2. Cosine Similarity Check
    # ================================================================
    print("\n--- Cosine Similarity Check ---")

    # Normalize
    z_radar_norm = z_radar_all / (np.linalg.norm(z_radar_all, axis=1, keepdims=True) + 1e-8)
    z_text_norm = z_text_all / (np.linalg.norm(z_text_all, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix: (N, N)
    sim_matrix = z_radar_norm @ z_text_norm.T

    # Retrieval accuracy: for each radar embedding, is the correct text the most similar?
    correct_matches = 0
    correct_top3 = 0
    sim_gaps = []

    for i in range(N):
        sorted_indices = np.argsort(-sim_matrix[i])
        if sorted_indices[0] == i:
            correct_matches += 1
        if i in sorted_indices[:3]:
            correct_top3 += 1

        # Similarity gap: correct vs best incorrect
        correct_sim = sim_matrix[i, i]
        best_incorrect_sim = max(sim_matrix[i, j] for j in range(N) if j != i)
        sim_gaps.append(correct_sim - best_incorrect_sim)

    top1_acc = correct_matches / N * 100
    top3_acc = correct_top3 / N * 100
    avg_gap = np.mean(sim_gaps)
    diag_sim = np.mean(np.diag(sim_matrix))

    print(f"  Retrieval Top-1 Accuracy: {top1_acc:.1f}%")
    print(f"  Retrieval Top-3 Accuracy: {top3_acc:.1f}%")
    print(f"  Mean Correct Cosine Sim: {diag_sim:.4f}")
    print(f"  Mean Similarity Gap (correct - best incorrect): {avg_gap:.4f}")

    # Group-level similarity: same action type vs different
    same_action_sims = []
    diff_action_sims = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if all_labels[i] == all_labels[j]:
                same_action_sims.append(sim_matrix[i, j])
            else:
                diff_action_sims.append(sim_matrix[i, j])

    avg_same = np.mean(same_action_sims) if same_action_sims else 0
    avg_diff = np.mean(diff_action_sims) if diff_action_sims else 0
    print(f"  Avg Cosine Sim (same action): {avg_same:.4f}")
    print(f"  Avg Cosine Sim (diff action): {avg_diff:.4f}")
    print(f"  Same/Diff Gap: {avg_same - avg_diff:.4f}")

    # Visualize similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("Text Embedding Index")
    ax.set_ylabel("Radar Embedding Index")
    ax.set_title(f"Cosine Similarity Matrix (Top-1: {top1_acc:.1f}%)")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_similarity_matrix.png"), dpi=150)
    plt.close()

    # Per-action similarity bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    per_action_correct = {}
    per_action_total = {}
    for i, label in enumerate(all_labels):
        per_action_total[label] = per_action_total.get(label, 0) + 1
        sorted_indices = np.argsort(-sim_matrix[i])
        if sorted_indices[0] == i:
            per_action_correct[label] = per_action_correct.get(label, 0) + 1

    actions = sorted(per_action_total.keys())
    accs = [per_action_correct.get(a, 0) / per_action_total[a] * 100 for a in actions]
    ax.bar(range(len(actions)), accs, tick_label=actions, color="steelblue")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Per-Action Retrieval Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_action_accuracy.png"), dpi=150)
    plt.close()

    # Save results
    results = {
        "num_samples": N,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "mean_correct_sim": float(diag_sim),
        "mean_sim_gap": float(avg_gap),
        "avg_same_action_sim": float(avg_same),
        "avg_diff_action_sim": float(avg_diff),
        "per_action_accuracy": {a: per_action_correct.get(a, 0) / per_action_total[a] * 100
                                 for a in actions},
    }
    with open(os.path.join(output_dir, "stage1_validation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Validation")
    parser.add_argument("--data_dir", type=str, default="data/toy_dataset")
    parser.add_argument("--ckpt", type=str, default="output/stage1/best_stage1.pt")
    parser.add_argument("--output_dir", type=str, default="output/validation/stage1")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    validate_stage1(args.data_dir, args.ckpt, args.output_dir, config)


if __name__ == "__main__":
    main()
