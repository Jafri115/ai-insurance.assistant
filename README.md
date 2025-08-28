# AI Insurance Assistant

A comprehensive AI assistant fine-tuned for insurance-related queries using state-of-the-art language models with LoRA/QLoRA techniques.

## 🎯 Project Overview

This project demonstrates advanced AI techniques applied to the insurance domain:

- **Fine-tuning**: LoRA/QLoRA fine-tuning of LLaMA, Mistral, or Gemma models
- **Prompt Engineering**: Specialized prompt templates for insurance Q&A
- **Dataset Curation**: Insurance-specific datasets with quality filtering
- **Evaluation**: Comprehensive metrics including perplexity, toxicity, and relevance
- **Deployment**: Production-ready FastAPI and Gradio interfaces
- **Monitoring**: Real-time monitoring with Prometheus and Grafana

## 🚀 Features

- **Multi-Model Support**: Compatible with LLaMA 2/3, Mistral 7B, Gemma 2B/7B
- **Efficient Training**: QLoRA for memory-efficient fine-tuning
- **Safety First**: Built-in toxicity detection and content filtering
- **Production Ready**: Scalable deployment with monitoring and logging
- **Interactive UI**: User-friendly Gradio interface for testing

## 📁 Project Structure

```
ai-insurance-assistant/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                # Raw insurance datasets
│   ├── processed/          # Cleaned and curated data
│   └── preprocess.py       # Data preprocessing pipeline
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── fine_tune.py        # QLoRA training script
│   ├── prompts.py          # Prompt templates and engineering
│   ├── evaluate.py         # Evaluation metrics and benchmarks
│   ├── app.py              # FastAPI/Gradio deployment
│   └── monitor.py          # Monitoring and metrics collection
├── models/                 # Fine-tuned model checkpoints
├── dashboards/             # Grafana dashboards and logs
│   ├── grafana.json
│   └── logs/
└── LICENSE
```

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-insurance-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Hardware Requirements

- **Minimum**: 16GB RAM, CUDA-compatible GPU with 8GB VRAM
- **Recommended**: 32GB RAM, RTX 3090/4090 or A100 GPU
- **Cloud**: Google Colab Pro, AWS EC2 g4dn instances, or similar

### 3. Dataset Preparation

```bash
# Download and preprocess insurance datasets
python data/preprocess.py --source insurance_qa --output data/processed/
```

### 4. Model Training

```bash
# Fine-tune with QLoRA
python src/fine_tune.py --model_name microsoft/DialoGPT-medium \
                        --dataset_path data/processed/insurance_qa.json \
                        --output_dir models/insurance-assistant-v1
```

### 5. Evaluation

```bash
# Run comprehensive evaluation
python src/evaluate.py --model_path models/insurance-assistant-v1 \
                       --test_data data/processed/test_set.json
```

### 6. Deployment

```bash
# Launch Gradio interface
python src/app.py --interface gradio --port 7860

# Or launch FastAPI server
python src/app.py --interface fastapi --port 8000
```

## 📊 Datasets

The project works with several insurance-related datasets:

1. **Insurance QA**: Question-answer pairs about insurance policies
2. **Claims Processing**: Automated claims handling scenarios
3. **Policy Explanations**: Complex policy terms simplified
4. **Regulatory Compliance**: Insurance law and regulation queries

### Data Sources
- Kaggle Insurance datasets
- Publicly available insurance FAQs
- Synthetic data generation using GPT-4
- Web scraped insurance documentation (with proper permissions)

## 🔧 Training Configuration

### Supported Models
- **LLaMA 2 (7B/13B)**: Meta's flagship model
- **Mistral 7B**: Efficient and powerful
- **Gemma (2B/7B)**: Google's lightweight option
- **Custom Models**: Easy to adapt for other architectures

### QLoRA Parameters
```python
{
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,          # LoRA scaling
    "lora_dropout": 0.1,       # Dropout rate
    "target_modules": ["q_proj", "v_proj"],
    "task_type": "CAUSAL_LM"
}
```

## 📈 Evaluation Metrics

### Automated Metrics
- **Perplexity**: Language model confidence
- **BLEU/ROUGE**: Text similarity scores
- **Toxicity**: Content safety assessment
- **Relevance**: Domain-specific accuracy

### Human Evaluation
- **Helpfulness**: User satisfaction ratings
- **Accuracy**: Factual correctness
- **Clarity**: Response comprehensibility
- **Safety**: Harmful content detection

## 🚀 Deployment Options

### Local Development
```bash
# Gradio interface
python src/app.py --mode gradio

# FastAPI with docs
python src/app.py --mode fastapi
# Access: http://localhost:8000/docs
```

### Production Deployment
```bash
# Docker deployment
docker build -t ai-insurance-assistant .
docker run -p 8000:8000 ai-insurance-assistant

# Kubernetes deployment
kubectl apply -f k8s/deployment.yaml
```

## 📊 Monitoring & Observability

### Metrics Tracked
- **Response Latency**: API response times
- **Toxicity Rate**: Percentage of flagged responses
- **Model Drift**: Performance degradation over time
- **User Satisfaction**: Feedback scores

### Grafana Dashboard
- Import `dashboards/grafana.json` for pre-built visualizations
- Monitor real-time inference metrics
- Track model performance trends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is for educational and research purposes. Always verify insurance information with qualified professionals.
