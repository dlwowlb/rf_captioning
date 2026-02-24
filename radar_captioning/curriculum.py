"""
Coarse-to-fine Curriculum Learning.

Organizes training into three progressive phases:

Phase 1 (Basic):
  - Simple actions: walking, sitting, standing
  - High LR for fast convergence
  - Stage 1 + Stage 2 Projector

Phase 2 (Composition):
  - Complex actions: walk+wave, talk+jump
  - Lower LR for fine-grained learning
  - Stage 1 + Stage 2 LoRA

Phase 3 (Occlusion):
  - 70% occluded + 30% clean data mixed
  - Very low LR for denoising fine-tuning
  - Stage 1 + Stage 2 LoRA

Each phase builds on the previous checkpoint.
"""

import json
import os

import torch

from .config import FullConfig, CurriculumConfig
from .train_stage1 import train_stage1
from .train_stage2 import train_stage2


def run_curriculum(
    data_dir: str,
    output_dir: str,
    config: FullConfig = None,
):
    """
    Execute full coarse-to-fine curriculum learning.

    Args:
        data_dir: path to toy dataset
        output_dir: root output directory
    """
    if config is None:
        config = FullConfig()

    print("=" * 60)
    print("Coarse-to-Fine Curriculum Learning")
    print("=" * 60)

    curriculum = config.curriculum
    results = {}

    # ================================================================
    # Phase 1: Basic Actions
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 1: BASIC ACTIONS (walk, sit, stand)")
    print(f"LR: {curriculum.basic_lr}, Epochs: {curriculum.basic_epochs}")
    print("Training: Stage 1 + Stage 2 Projector")
    print("=" * 60)

    phase1_dir = os.path.join(output_dir, "phase1_basic")

    # Stage 1 - Basic
    config.stage1.lr = curriculum.basic_lr
    config.stage1.num_epochs = curriculum.basic_epochs
    _, _, hist1_s1 = train_stage1(
        data_dir, os.path.join(phase1_dir, "stage1"),
        config, curriculum_phase="basic",
    )
    results["phase1_stage1"] = hist1_s1

    # Stage 2 - Basic (Projector only, no LoRA fine-tuning yet)
    config.stage2.lr_projector = curriculum.basic_lr
    config.stage2.lr_lora = 0.0  # no LoRA in phase 1
    config.stage2.num_epochs = curriculum.basic_epochs
    hist1_s2 = train_stage2(
        data_dir,
        os.path.join(phase1_dir, "stage1", "best_stage1.pt"),
        os.path.join(phase1_dir, "stage2"),
        config, curriculum_phase="basic",
    )
    results["phase1_stage2"] = hist1_s2

    # ================================================================
    # Phase 2: Composition Actions
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 2: COMPOSITION ACTIONS (walk+wave, talk+jump)")
    print(f"LR: {curriculum.composition_lr}, Epochs: {curriculum.composition_epochs}")
    print("Training: Stage 1 + Stage 2 LoRA")
    print("=" * 60)

    phase2_dir = os.path.join(output_dir, "phase2_composition")

    # Stage 1 - Composition (fine-tune from phase 1)
    config.stage1.lr = curriculum.composition_lr
    config.stage1.num_epochs = curriculum.composition_epochs
    _, _, hist2_s1 = train_stage1(
        data_dir, os.path.join(phase2_dir, "stage1"),
        config, curriculum_phase="composition",
    )
    results["phase2_stage1"] = hist2_s1

    # Stage 2 - Composition (with LoRA)
    config.stage2.lr_projector = curriculum.composition_lr * 0.5
    config.stage2.lr_lora = curriculum.composition_lr * 0.1
    config.stage2.num_epochs = curriculum.composition_epochs
    hist2_s2 = train_stage2(
        data_dir,
        os.path.join(phase2_dir, "stage1", "best_stage1.pt"),
        os.path.join(phase2_dir, "stage2"),
        config, curriculum_phase="composition",
    )
    results["phase2_stage2"] = hist2_s2

    # ================================================================
    # Phase 3: Occlusion
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 3: OCCLUSION (70% occluded + 30% clean)")
    print(f"LR: {curriculum.occlusion_lr}, Epochs: {curriculum.occlusion_epochs}")
    print("Training: Stage 1 + Stage 2 LoRA (denoising)")
    print("=" * 60)

    phase3_dir = os.path.join(output_dir, "phase3_occlusion")

    config.stage1.lr = curriculum.occlusion_lr
    config.stage1.num_epochs = curriculum.occlusion_epochs
    _, _, hist3_s1 = train_stage1(
        data_dir, os.path.join(phase3_dir, "stage1"),
        config, curriculum_phase="occlusion",
    )
    results["phase3_stage1"] = hist3_s1

    config.stage2.lr_projector = curriculum.occlusion_lr * 0.5
    config.stage2.lr_lora = curriculum.occlusion_lr * 0.1
    config.stage2.num_epochs = curriculum.occlusion_epochs
    hist3_s2 = train_stage2(
        data_dir,
        os.path.join(phase3_dir, "stage1", "best_stage1.pt"),
        os.path.join(phase3_dir, "stage2"),
        config, curriculum_phase="occlusion",
    )
    results["phase3_stage2"] = hist3_s2

    # Save overall results
    with open(os.path.join(output_dir, "curriculum_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Curriculum Learning Complete!")
    print("=" * 60)
    for phase, hist in results.items():
        if "train_loss" in hist:
            print(f"  {phase}: final train_loss = {hist['train_loss'][-1]:.4f}")

    return results
