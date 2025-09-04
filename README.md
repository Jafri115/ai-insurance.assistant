# AI Insurance Assistant

An insurance-focused AI assistant fine-tuned with LoRA on Microsoft Phi-3 Mini (4k instruct). This repo includes training, single-prompt inference, a Gradio chat app, and simple evaluation utilities.

## What’s included

- LoRA fine-tuning script (`src/fine_tune.py`)
- Inference utilities and CLI (`src/inference_utils.py`, `src/run_infer.py`)
- Gradio demo chat app (`src/demo_app.py`)
- Lightweight evaluation script (`src/evaluate.py`)
- Trained LoRA adapters under `models/insurance-assistant-gpu/`

Base model: `microsoft/Phi-3-mini-4k-instruct`

## Project structure (key files)

```
ai-insurance-assistant/
├── README.md
├── requirements.txt
├── infer_and_demo.ipynb
├── models/
│   └── insurance-assistant-gpu/           # LoRA adapter (PEFT)
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_evaluation.ipynb
└── src/
        ├── fine_tune.py
        ├── inference_utils.py
        ├── run_infer.py            # single-prompt CLI
        ├── run_inference.py        # wrapper -> run_infer.py
        ├── evaluate.py             # simple metrics
        └── demo_app.py             # Gradio chat UI
```

## Setup

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes for Windows/CPU:
- bitsandbytes 4-bit isn’t available on Windows CPU; use the `--no_4bit` flag in commands below.
- The loader uses eager attention to avoid flash-attn dependency warnings.

## Inference (single question)

Run a one-off answer from the fine-tuned adapter in `models\insurance-assistant-gpu`.

```powershell
python .\src\run_infer.py --question "How do insurance deductibles work?" ^
    --adapter_path models\insurance-assistant-gpu ^
    --max_new_tokens 256 --temperature 0.7 --top_p 0.95 --no_4bit --no_sample
```

Common toggles:
- Remove `--no_sample` to enable sampling (more diverse answers)
- Adjust `--max_new_tokens` if responses are too short/long

## Gradio demo app (chat UI)

Launch a local chat interface backed by the same model+adapter.

```powershell
python .\src\demo_app.py --adapter_path models\insurance-assistant-gpu --no_4bit --server_port 7860
# Open http://127.0.0.1:7860
```

Optional:
- `--share` to create a public Gradio link
- `--server_name 0.0.0.0` to bind on all interfaces

## Evaluation (lightweight)

The evaluation mirrors the notebook’s intrinsic checks (toxicity, domain relevance, semantic similarity) on a small test set.

```powershell
python .\src\evaluate.py --adapter_path models\insurance-assistant-gpu --no_4bit
```

Example results (from `infer_and_demo.ipynb`):
- Mean Toxicity: 0.0009 (lower is better)
- Domain Relevance: 0.7965 (higher is better)
- Semantic Similarity: 0.7853 (higher is better)

Evaluation report:

```
==================================================
EVALUATION REPORT
==================================================
Mean Toxicity Score:      0.0009 (Lower is better)
Domain Relevance Score:   0.7965 (Higher is better)
Semantic Similarity:      0.7853 (Higher is better)
==================================================
```

Heads up:
- The script will download `sentence-transformers` and Detoxify weights on first run.
- These numbers are indicative, not a gold-standard benchmark.

## Training (optional)

Fine-tune a base model with QLoRA + PEFT using your processed datasets.

```powershell
python .\src\fine_tune.py --model_name microsoft/Phi-3-mini-4k-instruct ^
    --dataset_path data/processed/train.json ^
    --validation_path data/processed/validation.json ^
    --output_dir models/insurance-assistant
```

The produced adapter (in `output_dir`) can then be copied or referenced as `--adapter_path` for inference/app/eval.

## Troubleshooting

- Stuck at “Loading checkpoint shards”: ensure PyTorch is installed and `--no_4bit` is used on Windows CPU. Try reducing `--max_new_tokens` for a quick sanity check.
- Flash-attn warnings are expected; the code forces eager attention for compatibility.
- If VRAM is limited, keep `--max_new_tokens` small and avoid sampling.

## License

MIT – see [LICENSE](LICENSE).

Disclaimer: This assistant is for educational purposes; verify insurance information with qualified professionals.
