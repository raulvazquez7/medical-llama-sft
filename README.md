# Medical LLM Fine-Tuning with QLoRA 🏥

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raulvazquez7/medical-llama-sft/blob/main/notebooks/medical_llm_finetuning.ipynb)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> An educational demonstration of Supervised Fine-Tuning (SFT) for specializing Llama 3.1-8B-Instruct in medical reasoning using QLoRA, inspired by [HuatuoGPT-o1](https://arxiv.org/abs/2412.18925).

## 📋 Overview

This project provides a **complete, reproducible workflow** to fine-tune a Large Language Model for medical reasoning tasks. Designed with **clarity and education** in mind, it demonstrates:

- ✅ **Efficient Training**: QLoRA enables fine-tuning 8B models on free Google Colab (T4 GPU)
- ✅ **Reproducible**: Complete notebooks with step-by-step explanations
- ✅ **Educational**: Detailed documentation of hyperparameters, design decisions, and trade-offs
- ✅ **Production-Ready Vision**: Includes roadmap for scaling to cloud infrastructure

## 🎯 Project Goals

1. Fine-tune Llama 3.1-8B-Instruct on medical reasoning data
2. Implement QLoRA for memory-efficient training
3. Create reproducible training pipeline
4. Evaluate model performance before and after fine-tuning
5. Document scalability path to production

## 🏗️ Project Structure

```
SFT Demo/
├── notebooks/                              # Main interface (Colab-ready)
│   ├── 01_model_finetuning.ipynb    # Complete fine-tuning pipeline
│   └── 02_model_evaluation.ipynb          # Baseline vs fine-tuned comparison
├── data/                                   # Training data (JSONL format)
│   ├── train_data.jsonl                   # Training set (~450 examples)
│   └── test_data.jsonl                    # Test set (~50 examples)
├── checkpoints/                            # Model checkpoints (gitignored)
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

**Note**: This is a **notebook-first** project optimized for reproducibility and education. All logic (data prep, training, evaluation) is self-contained in notebooks for maximum clarity.

## 🚀 Quick Start (5 minutes)

### Option 1: Google Colab (Recommended - No setup required)

1. **Click the Colab badge** at the top of this README
2. **Follow the HuggingFace setup** (first-time only, see below)
3. **Run all cells** - Training takes ~2-3 hours on free T4 GPU

### Option 2: Local Setup (Advanced)

**Prerequisites:**
- Python 3.10+
- CUDA-capable GPU with 16GB+ VRAM (or use Colab)
- ~50GB disk space

**Installation:**
```bash
git clone <your-repo-url>
cd "SFT Demo"
pip install -r requirements.txt
jupyter notebook notebooks/01_model_finetuning.ipynb
```

---

## 🔑 HuggingFace Setup (Required)

This project uses **Llama 3.1-8B-Instruct**, which requires HuggingFace authentication and model access approval.

### Step 1: Create HuggingFace Account
1. Go to [huggingface.co/join](https://huggingface.co/join)
2. Sign up with your email

### Step 2: Request Llama 3.1 Access
1. Visit [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
2. Click **"Request Access"** button
3. Fill out Meta's form (typically approved within 1-24 hours)
4. **Wait for approval email** before proceeding

### Step 3: Generate Access Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Name it (e.g., "medical-llm-finetuning")
4. Select **"Read"** permission (sufficient for downloading models)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

### Step 4: Authenticate in Notebook

**In Google Colab:**
```python
from huggingface_hub import login
login()  # Will prompt for token - paste and press Enter
```

**Or set as environment variable:**
```python
import os
os.environ['HF_TOKEN'] = 'hf_your_token_here'  # Replace with your token
```

**Security Note:** Never commit tokens to Git. Use Colab secrets or environment variables.

---

## 📊 Dataset

This project uses the [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset, which contains:

- Medical questions requiring complex reasoning
- Chain-of-thought reasoning (Complex CoT)
- Final medical responses

**Format:**
```json
{
  "question": "Medical question...",
  "complex_cot": "Step-by-step reasoning...",
  "response": "Final answer..."
}
```

## 🔧 Technical Details

### Why Google Colab?

This project is optimized for **Google Colab Free (T4 GPU)** for several reasons:

1. **Hardware Constraints**: Mac M1 doesn't support the CUDA-based stack required for QLoRA (bitsandbytes, CUDA kernels)
2. **Accessibility**: Free T4 GPU (~16GB VRAM) is sufficient for 4-bit quantized 8B parameter models
3. **Reproducibility**: Anyone can run this without local GPU hardware
4. **Cost-Effective**: $0 for PoC vs $20-30/run on cloud VMs

**For production or larger experiments**, see the [Production Roadmap](#️-production-roadmap) section below for cloud deployment strategies.

---

### QLoRA Configuration (Optimized for Colab T4)

**Model Architecture:**
- Base: Llama 3.1-8B-Instruct
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Memory footprint: ~4-5GB (down from ~16GB full precision)

**LoRA Hyperparameters:**
- **Rank (r)**: 16 - Balances expressiveness vs parameter efficiency
- **Alpha**: 32 - Standard scaling factor (typically 2×rank)
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- **Dropout**: 0.05 - Minimal regularization

**Quantization:**
- **Type**: NF4 (4-bit NormalFloat) - Optimal for LLM weight distributions
- **Double quantization**: Enabled - Further reduces memory
- **Compute dtype**: bfloat16 - More stable than float16 for training

**Training Hyperparameters:**
- **Batch size**: 2 per device
- **Gradient accumulation**: 4 steps (effective batch size: 8)
- **Learning rate**: 2e-4 with cosine decay
- **Warmup**: 3% of total steps
- **Epochs**: 3
- **Max sequence length**: 2048 tokens
- **Optimizer**: Paged AdamW 32-bit

**Why these values?**
- Small batches + gradient accumulation: Fits in T4's 15GB VRAM
- 2e-4 LR: Standard for LoRA, lower than full fine-tuning to prevent catastrophic forgetting
- 3 epochs on ~450 examples: Enough for pattern learning without overfitting

---

### Training Strategy

The model learns to follow HuatuoGPT-o1's two-stage reasoning format:

```
## Thinking
[Step-by-step medical reasoning with intermediate conclusions]

## Final Response
[Concise, actionable medical answer]
```

This format improves:
- **Interpretability**: Reasoning is explicit, not hidden
- **Reliability**: Structured thinking reduces hallucinations
- **Usability**: Users can verify the logic before trusting the conclusion

## 📈 Results

> **Note**: Results will be updated after training completion.

### Performance Metrics

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Perplexity | TBD | TBD | TBD |
| Reasoning Length | TBD | TBD | TBD |
| Format Compliance | TBD | TBD | TBD |

### Qualitative Examples

Coming soon...

## ☁️ Production Roadmap

This is currently an **educational proof-of-concept**. Here's how I would scale it to production:

### Phase 1: MLOps Foundation (Weeks 1-2)

**Experiment Tracking:**
- Integrate **MLflow** or **Weights & Biases** for tracking hyperparameters, metrics, and artifacts
- Version datasets with **DVC** (Data Version Control)
- Implement structured logging (JSON format) for better observability

**Code Organization:**
- Refactor notebooks into modular Python scripts:
  - `src/data/dataset.py` - Data loading and preprocessing
  - `src/training/trainer.py` - Training orchestration
  - `src/evaluation/metrics.py` - Evaluation logic
- Add `hydra` or `pydantic-settings` for configuration management
- Implement unit tests for data processing and evaluation

### Phase 2: Cloud Infrastructure (Weeks 3-4)

**Training Pipeline (Google Cloud Vertex AI):**
```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Data Pipeline  │─────▶│  Training Job    │─────▶│  Model Registry │
│  (Dataflow)     │      │  (Vertex AI)     │      │  (Artifact Reg) │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         │                        │                          │
         │                        ▼                          │
         │               ┌──────────────────┐              │
         │               │  Evaluation Job  │              │
         │               │  (Cloud Run)     │              │
         │               └──────────────────┘              │
         │                        │                          │
         └────────────────────────┴──────────────────────────┘
                                  │
                          ┌───────▼────────┐
                          │  Monitoring    │
                          │  (Cloud Logs)  │
                          └────────────────┘
```

**Infrastructure as Code:**
- **Terraform** for provisioning GPU instances and storage
- **Docker** containers for reproducible environments
- **GitHub Actions** for CI/CD pipeline

**Compute Configuration:**
- **Instance type:** `a2-highgpu-1g` (1x NVIDIA A100 80GB)
- **Training time:** ~4-6 hours for full dataset
- **Estimated cost:** ~$20-30 per training run

### Phase 3: Production Deployment (Weeks 5-6)

**Model Serving:**
- Deploy via **Vertex AI Endpoints** or **AWS SageMaker**
- Use **TGI (Text Generation Inference)** or **vLLM** for optimized serving
- Implement **LoRA adapter swapping** for multi-model serving

**Monitoring & Observability:**
- **Prometheus + Grafana** for metrics (latency, throughput, GPU utilization)
- **Sentry** for error tracking
- **Custom metrics:** Format compliance rate, reasoning length distribution

**Safety & Guardrails:**
- Input validation (medical context detection)
- Output filtering (hallucination detection)
- Rate limiting and quota management

### Cost Estimation (GCP)

| Component | Configuration | Monthly Cost (USD) |
|-----------|--------------|-------------------|
| Training (monthly) | A100 80GB × 6h × 4 runs | ~$100 |
| Inference | n1-standard-4 + T4 GPU | ~$250 |
| Storage | 500GB GCS | ~$10 |
| Monitoring | Cloud Logging + Metrics | ~$50 |
| **Total** | | **~$410/month** |

### Key Trade-offs

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Notebooks (Current)** | Max reproducibility, fast iteration | Not production-ready | Research, education, PoC |
| **Modular Scripts** | Better testability, reusable | More complex setup | Team projects, iterative dev |
| **Cloud Pipeline** | Scalable, automated, monitored | Higher cost, complexity | Production, large-scale |

### What I Would Build First

1. **Week 1:** MLflow integration + refactor to modular scripts
2. **Week 2:** Docker containerization + basic CI/CD
3. **Week 3:** Vertex AI training pipeline
4. **Week 4+:** Model serving + monitoring

**Rationale:** Focus on **experiment tracking** and **reproducibility** before scaling infrastructure. This allows rapid iteration while building toward production-grade deployment.

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace model library
- **TRL**: Transformer Reinforcement Learning (SFTTrainer)
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA/QLoRA)
- **bitsandbytes**: 4-bit quantization
- **Accelerate**: Multi-device training orchestration

## 📚 References

- [HuatuoGPT-o1 Paper](https://arxiv.org/abs/2412.18925)
- [HuatuoGPT-o1 Repository](https://github.com/FreedomIntelligence/HuatuoGPT-o1)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a portfolio project for educational purposes. Feedback and suggestions are welcome!

*This project is inspired by the excellent work of the FreedomIntelligence team on HuatuoGPT-o1.*
https://github.com/FreedomIntelligence/HuatuoGPT-o1/tree/main

