"""
Radar Captioner — v8.1 (Qwen3-8B QLoRA)

v8.0 → v8.1 changes:
  [replaced] T5 → Qwen3-8B (HY-Motion과 동일 LM)
  [added]    QLoRA: 4-bit 양자화 + LoRA adapters
  [changed]  Encoder-decoder → Causal LM generation

핵심 이유:
  HY-Motion이 Qwen3-8B로 text→motion 학습 시, Qwen은 motion description의
  구조를 이미 이해하고 있음. 같은 모델을 쓰면 text space 재학습 불필요.
  T5를 쓰면 motion text space를 처음부터 배워야 함.

Architecture:
  Radar PC → LatentGraph → g_radar → hymotion_head → 201D MSE (auxiliary)
                         → radar_tokens (B,K,512)
                              → projection (512 → 4096)
                              → Qwen prefix (soft prompt)
                              → Qwen3-8B (QLoRA) → text generation

Generation 방식 (Causal LM):
  [radar_token_1] ... [radar_token_K] [task_tokens] → "a person walks..."
  
  Qwen은 prefix tokens를 읽고 이어서 text를 생성.
  T5의 encoder-decoder와 달리, 단일 decoder로 동작.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .latent_graph_dynamics import LatentGraphDynamicsModel

#device_id = config.get("captioner", {}).get("device", "cuda:0")
gpu_id = 0# int(device_id.split(":")[-1]) if ":" in device_id else 0
# ═══════════════════════════════════════════════════════════
# QLoRA Setup
# ═══════════════════════════════════════════════════════════

def setup_qlora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=None):
    """Qwen3-8B에 QLoRA 적용."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("peft 라이브러리 필요: pip install peft")

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[QLoRA] Trainable: {trainable:,} / {total:,} "
          f"({trainable/total*100:.2f}%)")
    return model


def load_qwen_4bit(model_path):
    """Qwen3-8B를 4-bit 양자화로 로드."""
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_id},
            trust_remote_code=True,
        )
        print(f"[Qwen] Loaded 4-bit from {model_path}")
        return model
    except ImportError:
        print("[Qwen] bitsandbytes not found, loading in float16")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": gpu_id},
            trust_remote_code=True,
        )
        return model


# ═══════════════════════════════════════════════════════════
# Radar Captioner (Qwen3-8B)
# ═══════════════════════════════════════════════════════════

class RadarCaptioner(nn.Module):
    """
    End-to-end Radar → Caption with Qwen3-8B (QLoRA).

    LatentGraph의 semantic tokens를 Qwen의 embedding space로 투영하여
    soft prompt로 사용. Qwen이 이어서 text를 생성.

    Causal LM 학습:
      Input:  [prefix_embeds] [target_text_tokens]
      Labels: [-100 ... -100]  [target_text_tokens]
      → prefix 부분은 loss에서 제외, text 부분만 학습
    """

    def __init__(self, config: dict):
        super().__init__()
        cap_cfg = config.get("captioner", {})
        lg_cfg = config.get("latent_graph", {})

        # ── LatentGraph (v8.0) ──
        self.latent_graph = LatentGraphDynamicsModel(config)
        radar_token_dim = lg_cfg.get("out_dim", 512)
        self.num_semantic_queries = lg_cfg.get("num_semantic_queries", 4)

        # ── Qwen3-8B 경로 결정 ──
        qwen_path = cap_cfg.get("qwen_path", "Qwen/Qwen3-8B")
        from pathlib import Path
        local_paths = [
            Path("HY-Motion-1.0/ckpts/Qwen3-8B"),
            Path("ckpts/Qwen3-8B"),
        ]
        for p in local_paths:
            if p.exists():
                qwen_path = str(p)
                break
        print(f"[RadarCaptioner] Loading Qwen from: {qwen_path}")

        # ── Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path, padding_side="right", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Qwen model (4-bit + QLoRA) ──
        use_qlora = cap_cfg.get("use_qlora", True)
        if use_qlora:
            self.qwen = load_qwen_4bit(qwen_path)
            self.qwen = setup_qlora(
                self.qwen,
                lora_r=cap_cfg.get("lora_r", 16),
                lora_alpha=cap_cfg.get("lora_alpha", 32),
                lora_dropout=cap_cfg.get("lora_dropout", 0.05),
            )
        else:
            self.qwen = AutoModelForCausalLM.from_pretrained(
                qwen_path, torch_dtype=torch.float16,
                device_map={"": gpu_id}, trust_remote_code=True)

        qwen_dim = self.qwen.config.hidden_size  # 4096

        # ── Projection: radar tokens → Qwen embedding space ──
        self.token_proj = nn.Sequential(
            nn.LayerNorm(radar_token_dim),
            nn.Linear(radar_token_dim, qwen_dim),
            nn.GELU(),
            nn.Dropout(0.1), 
            nn.Linear(qwen_dim, qwen_dim),
        )

        # ── Global context projection ──
        self.global_proj = nn.Sequential(
            nn.LayerNorm(radar_token_dim),
            nn.Linear(radar_token_dim, qwen_dim),
        )

        # ── Learnable task prefix ──
        num_task_tokens = cap_cfg.get("num_task_tokens", 4)
        self.task_prefix = nn.Parameter(
            torch.randn(1, num_task_tokens, qwen_dim) * 0.02)
        self.num_task_tokens = num_task_tokens

        # ── Loss weights ──
        self.lambda_caption = cap_cfg.get("lambda_caption", 1.0)

        self.num_prefix_tokens = num_task_tokens + 1 + self.num_semantic_queries
        self.qwen_dim = qwen_dim

        print(f"[RadarCaptioner] Qwen dim: {qwen_dim}")
        print(f"  Prefix: {self.num_task_tokens} task + 1 global + "
              f"{self.num_semantic_queries} semantic = "
              f"{self.num_prefix_tokens} tokens")

    def _build_prefix_embeds(self, out: dict) -> torch.Tensor:
        """
        LatentGraph 출력 → Qwen prefix embeddings.
        Returns: (B, P, qwen_dim)
        """
        B = out["g_radar"].shape[0]
        task = self.task_prefix.expand(B, -1, -1)
        g = self.global_proj(out["g_radar"]).unsqueeze(1)
        tokens = self.token_proj(out["radar_tokens"])
        return torch.cat([task, g, tokens], dim=1)

    def forward(self,
                point_cloud: torch.Tensor,
                temporal_mask: torch.Tensor,
                texts: List[str],
                motion_latent_201: Optional[torch.Tensor] = None,
                phase_labels: Optional[torch.Tensor] = None,
                phase_confidence: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Joint forward: LatentGraph losses + Qwen captioning loss.
        """
        device = point_cloud.device

        # ── LatentGraph forward ──
        lg_out = self.latent_graph(
            point_cloud,
            motion_latent_201=motion_latent_201,
            temporal_mask=temporal_mask,
            phase_labels=phase_labels,
            phase_confidence=phase_confidence,
        )

        # ── Build prefix embeddings ──
        prefix_embeds = self._build_prefix_embeds(lg_out)
        B, P, D = prefix_embeds.shape

        # ── Tokenize target text ──
        text_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        text_ids = text_encoding.input_ids
        text_mask = text_encoding.attention_mask
        S = text_ids.shape[1]

        # ── Get text embeddings from Qwen ──
        with torch.no_grad():
            text_embeds = self.qwen.get_input_embeddings()(text_ids)

        # ── Concatenate: [prefix | text] ──
        inputs_embeds = torch.cat([
            prefix_embeds.to(text_embeds.dtype),
            text_embeds,
        ], dim=1)

        # ── Attention mask ──
        prefix_mask = torch.ones(B, P, device=device, dtype=text_mask.dtype)
        attention_mask = torch.cat([prefix_mask, text_mask], dim=1)

        # ── Labels: prefix=-100 (loss 제외), text=target ──
        prefix_labels = torch.full(
            (B, P), -100, device=device, dtype=text_ids.dtype)
        labels = torch.cat([prefix_labels, text_ids], dim=1)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # ── Qwen forward ──
        qwen_out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss_caption = qwen_out.loss

        # ── Total loss ──
        loss_total = lg_out["loss"] + self.lambda_caption * loss_caption

        return {
            "loss": loss_total,
            "loss_obs": lg_out["loss_obs"],
            "loss_kl": lg_out["loss_kl"],
            "loss_latent": lg_out["loss_latent"],
            "loss_caption": loss_caption,
            "metric_div": lg_out["metric_div"],
            "pred_hymotion_latent": lg_out["pred_hymotion_latent"],
            "confidence": lg_out["confidence"],
            "node_history": lg_out["node_history"],
        }

    @torch.no_grad()
    def generate(self,
                 point_cloud: torch.Tensor,
                 temporal_mask: torch.Tensor,
                 max_new_tokens: int = 64,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 ) -> List[str]:
        """Inference: Radar → caption text."""
        self.eval()
        device = point_cloud.device

        out = self.latent_graph.forward_sequence(point_cloud, temporal_mask)
        prefix_embeds = self._build_prefix_embeds(out)
        B, P, D = prefix_embeds.shape
        prefix_mask = torch.ones(B, P, device=device, dtype=torch.long)

        outputs = self.qwen.generate(
            inputs_embeds=prefix_embeds.to(self.qwen.dtype),
            attention_mask=prefix_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        captions = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        return captions

    @torch.no_grad()
    def predict_201d(self, point_cloud, temporal_mask):
        """201D latent prediction."""
        return self.latent_graph.predict_hymotion_latent(
            point_cloud, temporal_mask)

    def freeze_latent_graph(self):
        """Stage 2: LatentGraph frozen."""
        for p in self.latent_graph.parameters():
            p.requires_grad = False
        print(f"[RadarCaptioner] LatentGraph frozen")

    def unfreeze_latent_graph(self):
        """Stage 3: 전체 fine-tune."""
        for p in self.latent_graph.parameters():
            p.requires_grad = True
        print("[RadarCaptioner] LatentGraph unfrozen")

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lg_total = sum(p.numel() for p in self.latent_graph.parameters())
        lg_train = sum(p.numel() for p in self.latent_graph.parameters()
                       if p.requires_grad)
        qwen_train = sum(p.numel() for p in self.qwen.parameters()
                         if p.requires_grad)
        proj_train = (sum(p.numel() for p in self.token_proj.parameters()
                          if p.requires_grad)
                      + sum(p.numel() for p in self.global_proj.parameters()
                            if p.requires_grad)
                      + (self.task_prefix.numel()
                         if self.task_prefix.requires_grad else 0))

        print(f"[RadarCaptioner] Parameter stats:")
        print(f"  Total:       {total:>12,}")
        print(f"  Trainable:   {trainable:>12,}")
        print(f"  LatentGraph: {lg_total:>12,} (trainable: {lg_train:,})")
        print(f"  Qwen LoRA:   {qwen_train:>12,}")
        print(f"  Projection:  {proj_train:>12,}")
