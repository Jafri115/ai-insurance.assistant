"""Evaluate the Insurance Assistant model on simple intrinsic metrics.

Metrics:
- Toxicity (Detoxify)
- Semantic similarity between question and answer (SentenceTransformers cosine)
- Domain relevance (keyword-based heuristic)

This mirrors the evaluation section from the notebook and prints a small report.
"""
import argparse
import logging
from typing import List

import numpy as np

from inference_utils import load_model_and_tokenizer, build_prompt_text, generate_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


def _calc_toxicity(texts: List[str]) -> float:
    try:
        from detoxify import Detoxify

        model = Detoxify("original")
        results = model.predict(texts)
        tox = results.get("toxicity")
        if tox is None:
            return 0.0
        return float(np.mean(tox))
    except Exception as e:
        logger.warning(f"Toxicity calculation failed: {e}")
        return 0.0


def _calc_semantic_similarity(questions: List[str], answers: List[str]) -> float:
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = sbert.encode(questions)
        a_emb = sbert.encode(answers)
        sims = [cosine_similarity([q], [a])[0][0] for q, a in zip(q_emb, a_emb)]
        return float(np.mean(sims))
    except Exception as e:
        logger.warning(f"Similarity calculation failed: {e}")
        return 0.0


def _calc_domain_relevance(texts: List[str]) -> float:
    keywords = {
        "insurance",
        "policy",
        "premium",
        "deductible",
        "coverage",
        "claim",
        "liability",
        "risk",
        "beneficiary",
        "insurer",
        "auto",
        "health",
        "life",
        "property",
    }
    scores = []
    for t in texts:
        tl = t.lower()
        kcount = sum(1 for k in keywords if k in tl)
        wcount = max(len(t.split()), 1)
        scores.append(kcount / wcount)
    return float(np.mean(scores) * 10.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Insurance Assistant model")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter_path", type=str, default="models/insurance-assistant-gpu")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model, adapter_path=args.adapter_path, load_in_4bit=not args.no_4bit
    )

    instruction = "You are a helpful insurance assistant. Answer the user's question clearly."
    test_questions = [
        "What is comprehensive auto insurance and what does it cover?",
        "How do insurance deductibles work?",
        "Explain the concept of an insurance premium.",
    ]

    logger.info("Generating responses for evaluation...")
    answers: List[str] = []
    for q in test_questions:
        prompt = build_prompt_text(tokenizer, instruction, q)
        out = generate_text(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        answers.append(out)
        print(f"Q: {q}\nA: {out}\n")

    logger.info("Calculating metrics...")
    toxicity = _calc_toxicity(answers)
    similarity = _calc_semantic_similarity(test_questions, answers)
    relevance = _calc_domain_relevance(answers)

    print("\n" + "=" * 52)
    print("EVALUATION REPORT")
    print("=" * 52)
    print(f"Mean Toxicity Score:      {toxicity:.4f} (Lower is better)")
    print(f"Domain Relevance Score:   {relevance:.4f} (Higher is better)")
    print(f"Semantic Similarity:      {similarity:.4f} (Higher is better)")
    print("=" * 52)


if __name__ == "__main__":
    main()
