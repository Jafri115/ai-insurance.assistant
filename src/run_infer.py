"""Run single-turn inference with the Insurance Assistant model.

Example (PowerShell):
  python src\run_infer.py --question "How do deductibles work?"

Optional flags let you change temperature, max tokens, and adapter path.
"""
import argparse
import logging
from typing import Optional

import torch

from inference_utils import load_model_and_tokenizer, build_prompt_text, generate_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_infer")


def main():
    parser = argparse.ArgumentParser(description="Insurance Assistant inference")
    parser.add_argument("--question", type=str, required=True, help="User question or input text")
    parser.add_argument(
        "--instruction",
        type=str,
        default="You are a helpful insurance assistant. Answer the user's question clearly and concisely.",
        help="System instruction",
    )
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter_path", type=str, default="models/insurance-assistant-gpu")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_sample", action="store_true", help="Disable sampling (greedy/beam-like)")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading even if available")

    args = parser.parse_args()

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        trust_remote_code=True,
        load_in_4bit=not args.no_4bit,
    )

    device = model.device
    logger.info(f"Model device: {device}")

    prompt = build_prompt_text(tokenizer, args.instruction, args.question)
    output = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
    )

    print(output)


if __name__ == "__main__":
    main()
