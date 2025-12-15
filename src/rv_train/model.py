"""Model loading utilities for VLA-0."""

import json
from pathlib import Path
from typing import Optional

import torch
from qwen_vl_utils import process_vision_info
from transformers import LogitsProcessor, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


class NumberSpaceOnlyProcessor(LogitsProcessor):
    """Constrains generation to numbers (0-9), spaces, and EOS."""

    def __init__(self, tokenizer):
        self.allowed_tokens = set()
        for i in range(10):
            self.allowed_tokens.add(tokenizer.encode(str(i), add_special_tokens=False)[0])
        self.allowed_tokens.add(tokenizer.encode(" ", add_special_tokens=False)[0])
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0
        return scores + mask


def load_model_for_training(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    use_flash_attention: bool = False,
    finetune_mode: str = "full",  # "full" | "lora"
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Qwen2_5_VLForConditionalGeneration:
    """Load Qwen2.5-VL for training.

    Notes:
    - `finetune_mode="full"` does full fine-tuning (paper setting; needs lots of VRAM).
    - `finetune_mode="lora"` applies PEFT LoRA adapters to drastically reduce VRAM.
    """
    kwargs = {"torch_dtype": torch.bfloat16}
    if use_flash_attention:
        # switch to kernels-community/flash-attn2 / flash_attention_3 / flash_attention_2 if necessary
        kwargs["attn_implementation"] = "kernels-community/flash-attn3"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    # KV cache is not needed for training and can waste memory.
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    if finetune_mode not in {"full", "lora"}:
        raise ValueError(f"Unknown finetune_mode={finetune_mode!r}. Expected 'full' or 'lora'.")

    if finetune_mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "finetune_mode='lora' requires `peft`. Install with `pip install peft` "
                "or `uv pip install -e '.[lora]'`."
            ) from e

        # Try to infer a good default set of target module names from the loaded model.
        candidate_targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        found = set()
        for name, _module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in candidate_targets:
                found.add(leaf)

        if not found:
            raise ValueError(
                "Could not infer LoRA target modules for this model. "
                "Please update `candidate_targets` in rv_train/model.py for this architecture."
            )

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=sorted(found),
        )
        model = get_peft_model(model, peft_config)

    return model


def load_processor(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    img_size: int = 224,
    num_cams: int = 2,
    tile_images: bool = True,
) -> Qwen2_5_VLProcessor:
    """Load Qwen2.5-VL processor with correct pixel settings."""
    pixel_count = img_size * img_size
    if tile_images:
        pixel_count *= num_cams

    return Qwen2_5_VLProcessor.from_pretrained(
        model_id,
        min_pixels=pixel_count,
        max_pixels=pixel_count,
    )


class QwenVLActor:
    """Wrapper for inference with trained VLA model."""

    def __init__(
        self,
        model_path: str,
        *,
        base_model_id: Optional[str] = None,
        stats_path: Optional[str] = None,
        horizon: int = 8,
        action_dim: int = 7,
        num_bins: int = 1000,
        device: str = "cuda",
        torch_compile: bool = False,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device

        model_dir = Path(model_path)
        adapter_cfg = model_dir / "adapter_config.json"

        # Support LoRA checkpoints saved via PEFT/TRL: the directory contains adapter weights,
        # not full model weights. In that case, load base model then attach adapters.
        if adapter_cfg.exists():
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "This checkpoint looks like a LoRA adapter (adapter_config.json present) "
                    "but `peft` is not installed. Install with `pip install peft` or `uv pip install -e '.[lora]'`."
                ) from e

            adapter_conf = json.loads(adapter_cfg.read_text())
            inferred_base = adapter_conf.get("base_model_name_or_path")
            base_id = base_model_id or inferred_base
            if not base_id:
                raise ValueError(
                    "Could not infer base model id for LoRA adapter. Pass --base_model_id (eval.py) "
                    "or ensure adapter_config.json contains base_model_name_or_path."
                )

            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_id, torch_dtype=torch.bfloat16, device_map=device
            )
            # KV cache not needed for our generate settings and can waste memory.
            if hasattr(base, "config") and getattr(base.config, "use_cache", None) is True:
                base.config.use_cache = False

            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device
            )
        self.model.eval()

        if torch_compile:
            self.model = torch.compile(self.model)

        # Prefer loading processor from checkpoint dir (train.py saves it there); fallback to base model.
        try:
            self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        except Exception:
            if base_model_id is None:
                raise
            self.processor = Qwen2_5_VLProcessor.from_pretrained(base_model_id)
        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

        # Load stats
        if stats_path:
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
        else:
            self.stats = None

        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next "
            f"{horizon} timesteps. Each action has {action_dim} dimensions. "
            f"Output a single sequence of {horizon * action_dim} integers "
            f"(0-{num_bins} each), representing the {horizon} timesteps "
            f"sequentially. Provide only space separated numbers. Nothing else."
        )

    @torch.no_grad()
    def predict(self, image, instruction: str, temperature: float = 0.1):
        """Predict actions given image and instruction."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
        )

        image_inputs = process_vision_info(messages)[0]

        inputs = self.processor(text=[text], images=[image_inputs], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # NOTE: you must increase this value if you use longer horizon! for horizon=8, 208 tokens
        gen_kwargs = {"max_new_tokens": 256, "logits_processor": [self.logits_processor]}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        action_text = self.processor.decode(generated, skip_special_tokens=True)

        return self._text_to_action(action_text)

    def _text_to_action(self, text: str):
        """Convert action text to tensor."""
        if self.stats is None:
            raise ValueError("Stats not loaded")

        stats = self.stats["out_ori_act"]
        min_act = torch.tensor(stats["min"])
        max_act = torch.tensor(stats["max"])

        try:
            tokens = [int(x) for x in text.strip().split()]
            actions = torch.tensor(tokens, dtype=torch.float32)
            actions = actions.reshape(-1, self.action_dim)

            if actions.shape[0] < self.horizon:
                pad = actions[-1:].repeat(self.horizon - actions.shape[0], 1)
                actions = torch.cat([actions, pad], dim=0)
            actions = actions[: self.horizon]

            actions = (actions / self.num_bins) * (max_act - min_act) + min_act
        except Exception:
            actions = ((min_act + max_act) / 2).repeat(self.horizon, 1)

        return actions
