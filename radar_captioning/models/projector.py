"""
Stage 2: MLP Projector and LLM wrapper for sentence generation.

The Projector maps Q-Former frame queries (768-dim) to LLM input space.
LoRA adapters are applied to the LLM for efficient fine-tuning.

Prompt template includes JSON physical slots before sentence generation
to control hallucination.
"""

import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    """
    Simple MLP projector: Q-Former hidden_dim → LLM hidden_dim.

    Maps radar frame query embeddings into the LLM's input space.
    """

    def __init__(
        self,
        input_dim: int = 768,    # Q-Former output dim
        hidden_dim: int = 768,    # intermediate
        output_dim: int = 896,    # LLM hidden dim (Qwen2.5-0.5B = 896)
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, Nq, input_dim) frame query tokens
        Returns:
            (B, Nq, output_dim) projected tokens for LLM input
        """
        return self.proj(x)


# ============================================================================
# Prompt Template for JSON Physical Slots + Sentence Generation
# ============================================================================

STAGE2_PROMPT_TEMPLATE = """Based on the radar observation, first extract the physical parameters as JSON, then describe the motion.

### Physical Parameters (JSON):
{json_slot}

### Motion Description:
{description}"""

STAGE2_INFERENCE_PROMPT = """Based on the radar observation, first extract the physical parameters as JSON, then describe the motion.

### Physical Parameters (JSON):
"""


def build_training_prompt(physical_slots: Dict, description: str) -> str:
    """
    Build the full training prompt with JSON slots and description.

    Args:
        physical_slots: dict with distance_m, speed_ms, angle_deg, doppler_sign, action
        description: ground truth text description

    Returns:
        Full prompt string
    """
    json_str = json.dumps(physical_slots, indent=2)
    return STAGE2_PROMPT_TEMPLATE.format(json_slot=json_str, description=description)


def build_inference_prompt() -> str:
    """Build inference prompt (model generates JSON + description)."""
    return STAGE2_INFERENCE_PROMPT


class RadarLLMWrapper(nn.Module):
    """
    Wrapper that combines Q-Former embeddings with LLM for captioning.

    Architecture:
      [Radar Tokens (projected)] + [Prompt Tokens] → LLM → [JSON + Description]

    During training:
      - Q-Former is frozen
      - Projector is trainable
      - LLM has LoRA adapters (trainable), base weights frozen
      - Loss only on answer tokens (JSON + description), not prompt tokens
    """

    def __init__(
        self,
        projector: MLPProjector,
        tokenizer,
        llm_model,
        num_radar_tokens: int = 32,
    ):
        super().__init__()
        self.projector = projector
        self.tokenizer = tokenizer
        self.llm = llm_model
        self.num_radar_tokens = num_radar_tokens

        # Special tokens for radar input demarcation
        self.radar_start_token = "<radar>"
        self.radar_end_token = "</radar>"

    def forward(
        self,
        radar_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            radar_embeds: (B, Nq, D_qformer) frame queries from frozen Q-Former
            input_ids: (B, L) tokenized prompt + answer
            attention_mask: (B, L) attention mask
            labels: (B, L) labels with -100 for prompt tokens (masked)

        Returns:
            dict with 'loss' and 'logits'
        """
        B = radar_embeds.shape[0]

        # Project radar embeddings to LLM space
        radar_projected = self.projector(radar_embeds)  # (B, Nq, D_llm)

        # Get text embeddings from LLM's embedding layer
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, L, D_llm)

        # Concatenate: [radar_tokens] + [text_tokens]
        combined_embeds = torch.cat([radar_projected, text_embeds], dim=1)
        # (B, Nq + L, D_llm)

        # Extend attention mask for radar tokens (all 1s)
        radar_attn = torch.ones(B, self.num_radar_tokens, device=attention_mask.device)
        combined_attn = torch.cat([radar_attn, attention_mask.float()], dim=1)

        # Extend labels: -100 for radar tokens (don't compute loss on them)
        radar_labels = torch.full(
            (B, self.num_radar_tokens), -100, dtype=labels.dtype, device=labels.device
        )
        combined_labels = torch.cat([radar_labels, labels], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            labels=combined_labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        radar_embeds: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = False,
    ) -> List[str]:
        """
        Generate captions from radar embeddings.

        Args:
            radar_embeds: (B, Nq, D_qformer) frame queries
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            do_sample: whether to sample or greedy decode

        Returns:
            List of generated text strings
        """
        B = radar_embeds.shape[0]
        device = radar_embeds.device

        # Project radar embeddings
        radar_projected = self.projector(radar_embeds)  # (B, Nq, D_llm)

        # Encode the inference prompt
        prompt = build_inference_prompt()
        prompt_encoding = self.tokenizer(
            [prompt] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        prompt_embeds = self.llm.get_input_embeddings()(prompt_encoding["input_ids"])

        # Concatenate radar + prompt
        combined = torch.cat([radar_projected, prompt_embeds], dim=1)
        combined_attn = torch.ones(combined.shape[:2], device=device)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=combined,
            attention_mask=combined_attn,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts
