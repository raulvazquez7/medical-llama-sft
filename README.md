# Medical SFT Pipeline 🏥

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

> A professional Supervised Fine-Tuning pipeline for specializing Large Language Models in medical reasoning, inspired by HuatuoGPT-o1.

## 📋 Overview

This project demonstrates how to fine-tune a Large Language Model (Llama 3.1-8B-Instruct) to enhance its medical reasoning capabilities using QLoRA (Quantized Low-Rank Adaptation). The pipeline is designed to be:

- **Efficient**: Runs on local hardware (Mac M1) with memory-optimized configurations
- **Scalable**: Includes configurations for cloud deployment (GCP/AWS)
- **Production-ready**: Follows best practices and modern ML engineering standards
- **Educational**: Comprehensive documentation and notebooks for learning

## 🎯 Project Goals

1. Fine-tune Llama 3.1-8B-Instruct on medical reasoning data
2. Implement QLoRA for memory-efficient training
3. Create reproducible training pipeline
4. Evaluate model performance before and after fine-tuning
5. Document scalability path to production

## 🏗️ Architecture

```
SFT Demo/
├── data/                           # Data preparation and analysis
│   ├── prepare_dataset.py          # Dataset preparation script
│   └── dataset_analysis.ipynb      # Exploratory data analysis
├── configs/                        # Training configurations
│   ├── training_local.yaml         # Mac M1 configuration
│   └── training_cloud.yaml         # Cloud (GCP/AWS) configuration
├── src/                            # Source code
│   ├── train.py                    # Main training script
│   ├── evaluate.py                 # Evaluation script
│   └── utils.py                    # Utility functions
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Data exploration
│   ├── 02_before_finetuning.ipynb  # Baseline evaluation
│   └── 03_after_finetuning.ipynb   # Post-training evaluation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 16GB+ RAM (for Mac M1 local training)
- ~50GB disk space

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd SFT\ Demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Prepare Data

```bash
python data/prepare_dataset.py
```

#### 2. Train Model (Local)

```bash
python src/train.py --config configs/training_local.yaml
```

#### 3. Evaluate Model

```bash
python src/evaluate.py --model_path ./checkpoints/final_model
```

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

### QLoRA Configuration

**Local (Mac M1):**
- Rank: 16
- Alpha: 32
- Quantization: 4-bit (nf4)
- Target modules: q_proj, k_proj, v_proj, o_proj
- Batch size: 1 (gradient accumulation: 8)
- Learning rate: 2e-4
- Max sequence length: 2048

**Cloud (GCP/AWS):**
- Higher batch sizes
- Multi-GPU support with DeepSpeed
- Full configuration in `configs/training_cloud.yaml`

### Training Strategy

The model is trained to follow this response pattern:

```
## Thinking
[Step-by-step medical reasoning]

## Final Response
[Concise medical answer]
```

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

## ☁️ Scaling to Production

### Cloud Deployment (GCP)

For production-scale training, we recommend:

1. **Compute Engine VM Configuration:**
   - Instance type: `a2-highgpu-1g` (1x A100 80GB)
   - OS: Ubuntu 22.04 LTS with Deep Learning VM
   - Disk: 500GB SSD

2. **Setup:**
```bash
# SSH into VM
gcloud compute ssh your-instance-name

# Clone repo and install dependencies
git clone <your-repo-url>
cd SFT\ Demo
pip install -r requirements.txt

# Run training with cloud config
python src/train.py --config configs/training_cloud.yaml
```

3. **Cost Estimation:**
   - A100 80GB: ~$3.67/hour
   - Training time: ~4-6 hours
   - Total cost: ~$15-25

### Alternative: Google Colab Pro

See `notebooks/colab_training.ipynb` for a Colab-ready version.

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

## 👤 Author

**Raúl Vázquez**
- AI Engineer in Training
- Focus: LLM Fine-tuning, Medical AI, MLOps

---

*This project is inspired by the excellent work of the FreedomIntelligence team on HuatuoGPT-o1.*

