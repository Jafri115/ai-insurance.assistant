"""
Common utilities for loading the Insurance Assistant model and running inference.

Mirrors the notebook logic:
- Loads base model (Phi-3-mini-4k-instruct) and merges local LoRA adapter.
- Uses chat templates when available, with a fallback prompt format.
- Supports bitsandbytes 4-bit on CUDA when available; otherwise falls back gracefully.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _maybe_get_bnb_config(load_in_4bit: bool = True):
    """Return BitsAndBytesConfig if bitsandbytes + CUDA are available and requested, else None."""
    if not load_in_4bit:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        import bitsandbytes as bnb  # noqa: F401
        from transformers import BitsAndBytesConfig  # type: ignore

        # Prefer bfloat16 compute for stability on modern GPUs
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception as e:
        logger.info(f"bitsandbytes quantization not available: {e}")
        return None


def load_model_and_tokenizer(
    base_model: str = "microsoft/Phi-3-mini-4k-instruct",
    adapter_path: str = "models/insurance-assistant-gpu",
    trust_remote_code: bool = True,
    load_in_4bit: bool = True,
):
    """Load tokenizer, base model, and apply local LoRA adapter.

    Returns (model, tokenizer). The model is placed on CUDA if available (device_map="auto").
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding works best for generation with chat templates
    tokenizer.padding_side = "left"

    # Model load kwargs
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs: Dict[str, Any] = dict(
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        dtype=dtype,
        use_safetensors=True,
        # Prefer eager attention to avoid flash-attn dependency issues
        attn_implementation="eager",
    )

    if not torch.cuda.is_available():
        # CPU-friendly loading
        model_kwargs.update({"low_cpu_mem_usage": True})

    # Optional 4-bit
    bnb_config = _maybe_get_bnb_config(load_in_4bit=load_in_4bit)
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Apply LoRA adapter
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        # Keep in adapter mode for runtime efficiency; merging is optional
    except Exception as e:
        raise RuntimeError(
            f"Failed to load LoRA adapter from '{adapter_path}'. Ensure the path exists and is a valid PEFT adapter. Error: {e}"
        )

    # Disable cache for better memory usage if needed
    if hasattr(model.config, "use_cache"):
        try:
            model.config.use_cache = True  # Enable cache for inference
        except Exception:
            pass

    return model, tokenizer


def build_prompt_text(tokenizer: AutoTokenizer, instruction: str, input_text: str) -> str:
    """Build a prompt using chat templates when available; fallback to a simple format."""
    try:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text or instruction},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.debug(f"Chat template unavailable, using fallback: {e}")
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask").to(model.device)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p, top_k=top_k))

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    # Decode only the newly generated tokens
    new_tokens = outputs[0, input_ids.shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response_text.strip()
