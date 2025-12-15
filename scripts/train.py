#!/usr/bin/env python3
"""VLA-0 Training Script using TRL's SFTTrainer."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from trl import SFTConfig, SFTTrainer, TrlParser

from rv_train.collator import VLACollator
from rv_train.dataset import LiberoDataset
from rv_train.model import load_model_for_training, load_processor

cuda_visible_devices = 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    use_flash_attention: bool = field(default=False)
    finetune_mode: str = field(default="full")  # "full" | "lora"
    # LoRA params (used only when finetune_mode == "lora")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    repo_id: str = field(default="physical-intelligence/libero")
    history: int = field(default=1)
    horizon: int = field(default=8)
    img_size: int = field(default=224)
    crop_ratio: float = field(default=0.875)
    tile_images: bool = field(default=True)
    # If False, use only the main camera ("image") and skip wrist camera to save VRAM.
    use_wrist_image: bool = field(default=True)
    brightness_aug: float = field(default=0.2)
    contrast_aug: float = field(default=0.2)
    saturation_aug: float = field(default=0.2)
    hue_aug: float = field(default=0.05)


@dataclass
class VLATrainingArguments:
    action_mask_aug_pct: float = field(default=0.4)


def main():
    parser = TrlParser(dataclass_types=[ModelArguments, DataArguments, VLATrainingArguments, SFTConfig])
    model_args, data_args, vla_args, training_args = parser.parse_args_and_config()

    print(f"Loading model: {model_args.model_id}")
    model = load_model_for_training(
        model_id=model_args.model_id,
        use_flash_attention=model_args.use_flash_attention,
        finetune_mode=model_args.finetune_mode,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )

    # Save memory during training (KV cache not needed)
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    cam_list = ("image", "wrist_image") if data_args.use_wrist_image else ("image",)
    num_cams = len(cam_list)
    if num_cams > 1 and not data_args.tile_images:
        raise ValueError("Multi-camera training currently requires tile_images=true (the collator injects only one image).")

    processor = load_processor(
        model_id=model_args.model_id,
        img_size=data_args.img_size,
        num_cams=num_cams,
        tile_images=data_args.tile_images,
    )

    print("Loading dataset...")
    dataset = LiberoDataset(
        repo_id=data_args.repo_id,
        history=data_args.history,
        horizon=data_args.horizon,
        cam_list=cam_list,
        img_size=data_args.img_size,
        crop_ratio=data_args.crop_ratio,
        tile_images=data_args.tile_images,
        brightness_aug=data_args.brightness_aug,
        contrast_aug=data_args.contrast_aug,
        saturation_aug=data_args.saturation_aug,
        hue_aug=data_args.hue_aug,
    )

    # Save stats for inference
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{training_args.output_dir}/dataset_stats.json", "w") as f:
        json.dump(dataset.stats, f, indent=2)

    collator = VLACollator(
        processor=processor,
        action_mask_aug_pct=vla_args.action_mask_aug_pct,
    )

    # VLM-specific settings
    training_args.max_length = None  # Don't truncate images
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(f"{training_args.output_dir}/final")
    processor.save_pretrained(f"{training_args.output_dir}/final")


if __name__ == "__main__":
    main()
