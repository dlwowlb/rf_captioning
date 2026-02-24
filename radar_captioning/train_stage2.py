"""
Stage 2 Training: Sentence Generation with Projector + LoRA.

Trains the MLP Projector and LoRA adapters on a frozen Q-Former + LLM backbone.
Uses JSON physical slots as intermediate representation for hallucination control.

The LLM generates:
  1. Physical parameters (JSON): distance, speed, angle, doppler info
  2. Natural language description based on those parameters

Loss: Causal LM loss only on answer tokens (JSON + description).

Usage:
    python -m radar_captioning.train_stage2 \
        --data_dir data/toy_dataset \
        --stage1_ckpt output/stage1/best_stage1.pt \
        --output_dir output/stage2 \
        --num_epochs 30
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import FullConfig
from .data.toy_dataset import RadarCaptionDataset, generate_toy_dataset
from .models.radar_q_former import RadarQFormer
from .models.projector import MLPProjector, build_training_prompt


def collate_fn_stage2(batch):
    """Collate with physical slots and texts."""
    rd = torch.stack([b["rd"] for b in batch])
    ra = torch.stack([b["ra"] for b in batch])
    da = torch.stack([b["da"] for b in batch])
    texts = [b["text"] for b in batch]
    physical_slots = [json.loads(b["physical_slots"]) for b in batch]
    return {
        "rd": rd, "ra": ra, "da": da,
        "texts": texts, "physical_slots": physical_slots,
    }


def train_stage2(
    data_dir: str,
    stage1_ckpt: str,
    output_dir: str,
    config: FullConfig = None,
    curriculum_phase: str = None,
):
    """
    Train Stage 2: Sentence Generation.

    Pipeline:
      1. Load frozen Q-Former from Stage 1
      2. Initialize Projector (trainable)
      3. Load LLM with LoRA (trainable LoRA, frozen base)
      4. Train on causal LM objective
    """
    if config is None:
        config = FullConfig()

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    print("=" * 60)
    print("Stage 2: Sentence Generation Training")
    print("=" * 60)

    # Generate dataset if not exists
    if not os.path.exists(os.path.join(data_dir, "metadata.json")):
        generate_toy_dataset(
            num_samples=config.data.num_samples,
            num_frames=config.data.num_frames,
            heatmap_size=config.data.heatmap_size,
            output_dir=data_dir,
            seed=config.seed,
        )

    # Load datasets
    train_dataset = RadarCaptionDataset(
        data_dir, split="train", curriculum_phase=curriculum_phase,
        num_frames=config.data.num_frames,
    )
    val_dataset = RadarCaptionDataset(
        data_dir, split="val", curriculum_phase=curriculum_phase,
        num_frames=config.data.num_frames,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.stage2.batch_size,
        shuffle=True, collate_fn=collate_fn_stage2, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.stage2.batch_size,
        shuffle=False, collate_fn=collate_fn_stage2,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ---- Load Frozen Q-Former ----
    radar_qformer = RadarQFormer(
        img_size=config.vit.img_size,
        patch_size=config.vit.patch_size,
        vit_embed_dim=config.vit.embed_dim,
        vit_depth=config.vit.depth,
        vit_num_heads=config.vit.num_heads,
        mask_ratio=0.0,  # no masking during Stage 2
        qformer_hidden_dim=config.qformer.hidden_dim,
        num_spatial_queries=config.qformer.num_spatial_queries,
        num_frame_queries=config.qformer.num_frame_queries,
        qformer_num_heads=config.qformer.num_heads,
        num_layers_spatial=config.qformer.num_layers_spatial,
        num_layers_temporal=config.qformer.num_layers_temporal,
        dropout=0.0,
    ).to(device)

    if stage1_ckpt and os.path.exists(stage1_ckpt):
        ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=False)
        radar_qformer.load_state_dict(ckpt["radar_qformer"])
        print(f"Loaded Q-Former from {stage1_ckpt}")
    else:
        print("WARNING: No Stage 1 checkpoint found, using random Q-Former weights")

    # Freeze Q-Former
    radar_qformer.eval()
    for p in radar_qformer.parameters():
        p.requires_grad = False

    # ---- Load LLM with LoRA ----
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"Loading LLM: {config.stage2.llm_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.stage2.llm_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(
            config.stage2.llm_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.stage2.lora_rank,
            lora_alpha=config.stage2.lora_alpha,
            lora_dropout=config.stage2.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()
        llm_hidden_dim = llm.config.hidden_size
        use_real_llm = True

    except Exception as e:
        print(f"Could not load LLM ({e}), using surrogate LM")
        tokenizer = None
        llm = None
        llm_hidden_dim = config.stage2.llm_hidden_dim
        use_real_llm = False

    # ---- Projector ----
    projector = MLPProjector(
        input_dim=config.qformer.hidden_dim,
        hidden_dim=config.stage2.projector_hidden_dim,
        output_dim=llm_hidden_dim,
    ).to(device)

    # ---- Surrogate LM (if real LLM not available) ----
    if not use_real_llm:
        surrogate_lm = SurrogateLM(
            input_dim=llm_hidden_dim,
            vocab_size=256,  # char-level
            max_length=256,
        ).to(device)
    else:
        surrogate_lm = None

    # ---- Optimizer ----
    trainable_params = list(projector.parameters())
    if use_real_llm:
        trainable_params += [p for p in llm.parameters() if p.requires_grad]
    else:
        trainable_params += list(surrogate_lm.parameters())

    optimizer = torch.optim.AdamW([
        {"params": projector.parameters(), "lr": config.stage2.lr_projector},
        {"params": (
            [p for p in llm.parameters() if p.requires_grad] if use_real_llm
            else list(surrogate_lm.parameters())
        ), "lr": config.stage2.lr_lora},
    ], weight_decay=0.01)

    total_steps = config.stage2.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps,
    )

    # ---- Training Loop ----
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(config.stage2.num_epochs):
        if use_real_llm:
            llm.train()
        else:
            surrogate_lm.train()
        projector.train()

        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            rd = batch["rd"].to(device)
            ra = batch["ra"].to(device)
            da = batch["da"].to(device)
            texts = batch["texts"]
            physical_slots = batch["physical_slots"]

            # Get frozen Q-Former embeddings
            with torch.no_grad():
                frame_queries = radar_qformer(rd, ra, da)  # (B, Nq, D)

            # Project to LLM space
            projected = projector(frame_queries)  # (B, Nq, D_llm)

            # Build training prompts with JSON slots
            full_prompts = []
            for ps, text in zip(physical_slots, texts):
                full_prompts.append(build_training_prompt(ps, text))

            if use_real_llm:
                loss = _compute_llm_loss(
                    llm, tokenizer, projected, full_prompts, device,
                    config.qformer.num_frame_queries,
                )
            else:
                loss = _compute_surrogate_loss(
                    surrogate_lm, projected, full_prompts, device,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if use_real_llm:
            llm.eval()
        else:
            surrogate_lm.eval()
        projector.eval()
        val_loss_total = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                rd = batch["rd"].to(device)
                ra = batch["ra"].to(device)
                da = batch["da"].to(device)
                texts = batch["texts"]
                physical_slots = batch["physical_slots"]

                frame_queries = radar_qformer(rd, ra, da)
                projected = projector(frame_queries)

                full_prompts = []
                for ps, text in zip(physical_slots, texts):
                    full_prompts.append(build_training_prompt(ps, text))

                if use_real_llm:
                    loss = _compute_llm_loss(
                        llm, tokenizer, projected, full_prompts, device,
                        config.qformer.num_frame_queries,
                    )
                else:
                    loss = _compute_surrogate_loss(
                        surrogate_lm, projected, full_prompts, device,
                    )
                val_loss_total += loss.item()
                n_val += 1

        val_loss = val_loss_total / max(n_val, 1)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{config.stage2.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                "epoch": epoch,
                "projector": projector.state_dict(),
                "val_loss": val_loss,
            }
            if use_real_llm:
                save_dict["llm_lora"] = {
                    k: v for k, v in llm.state_dict().items() if "lora" in k
                }
            else:
                save_dict["surrogate_lm"] = surrogate_lm.state_dict()
            torch.save(save_dict, os.path.join(output_dir, "best_stage2.pt"))

    # Save final
    save_dict = {
        "epoch": config.stage2.num_epochs,
        "projector": projector.state_dict(),
        "val_loss": val_loss,
    }
    if use_real_llm:
        save_dict["llm_lora"] = {
            k: v for k, v in llm.state_dict().items() if "lora" in k
        }
    else:
        save_dict["surrogate_lm"] = surrogate_lm.state_dict()
    torch.save(save_dict, os.path.join(output_dir, "final_stage2.pt"))

    with open(os.path.join(output_dir, "stage2_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nStage 2 complete. Best val loss: {best_val_loss:.4f}")
    return history


def _compute_llm_loss(llm, tokenizer, projected, prompts, device, num_radar_tokens):
    """Compute causal LM loss with radar token prefix."""
    B = projected.shape[0]

    # Tokenize prompts
    encoding = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=256,
    ).to(device)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Build labels: mask prompt prefix with -100
    # Find where "### Motion Description:" ends for each sample
    labels = input_ids.clone()
    # Simple approach: mask first 60% of tokens as prompt
    prompt_len = int(input_ids.shape[1] * 0.4)
    labels[:, :prompt_len] = -100

    # Get text embeddings
    text_embeds = llm.get_input_embeddings()(input_ids)

    # Concatenate radar tokens + text tokens
    combined = torch.cat([projected, text_embeds], dim=1)

    radar_attn = torch.ones(B, num_radar_tokens, device=device)
    combined_attn = torch.cat([radar_attn, attention_mask.float()], dim=1)

    radar_labels = torch.full((B, num_radar_tokens), -100, dtype=labels.dtype, device=device)
    combined_labels = torch.cat([radar_labels, labels], dim=1)

    outputs = llm(
        inputs_embeds=combined,
        attention_mask=combined_attn,
        labels=combined_labels,
    )
    return outputs.loss


def _compute_surrogate_loss(surrogate_lm, projected, prompts, device):
    """Compute loss with surrogate LM when real LLM is not available."""
    # Use the mean of projected tokens as context
    context = projected.mean(dim=1)  # (B, D)
    loss = surrogate_lm(context, prompts)
    return loss


class SurrogateLM(nn.Module):
    """Simple surrogate language model for toy experiments without real LLM."""

    def __init__(self, input_dim=896, vocab_size=256, max_length=256):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.lm_head = nn.Linear(512, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, context, texts):
        """
        Args:
            context: (B, D) context vector from projector
            texts: list of target text strings
        Returns:
            loss: scalar
        """
        B = context.shape[0]
        device = context.device

        # Simple char-level targets
        targets = []
        for text in texts:
            chars = [ord(c) % self.vocab_size for c in text[:self.max_length]]
            chars += [0] * (self.max_length - len(chars))
            targets.append(chars)
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        # Decode
        h = self.decoder(context)  # (B, 512)
        # Repeat for each position
        h = h.unsqueeze(1).expand(-1, self.max_length, -1)  # (B, L, 512)
        logits = self.lm_head(h)  # (B, L, vocab_size)

        loss = self.criterion(logits.view(-1, self.vocab_size), targets.view(-1))
        return loss


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Sentence Generation")
    parser.add_argument("--data_dir", type=str, default="data/toy_dataset")
    parser.add_argument("--stage1_ckpt", type=str, default="output/stage1/best_stage1.pt")
    parser.add_argument("--output_dir", type=str, default="output/stage2")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--curriculum_phase", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = FullConfig()
    config.stage2.num_epochs = args.num_epochs
    config.stage2.batch_size = args.batch_size
    config.device = args.device

    train_stage2(args.data_dir, args.stage1_ckpt, args.output_dir, config, args.curriculum_phase)


if __name__ == "__main__":
    main()
