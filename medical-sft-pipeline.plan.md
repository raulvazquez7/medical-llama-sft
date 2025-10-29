# Medical SFT Pipeline - HuatuoGPT-o1 Inspired

## Project Objectives

Develop a complete and professional Supervised Fine-Tuning pipeline that demonstrates how to specialize an LLM (Llama 3.1-8B-Instruct) in complex medical reasoning, following state-of-the-art best practices.

**Strategy:** Local data preparation (Mac M1) + QLoRA training on Google Colab (GPU)

## Proposed Architecture

### Project Structure

```
/
├── data/
│ └── prepare_dataset.py # Script to prepare and filter data
├── configs/
│ └── training_config.yaml # Config for Colab (QLoRA)
├── src/
│ ├── train.py # Training script (modular)
│ ├── evaluate.py # Evaluation script
│ └── utils.py # Utilities (formatting, metrics)
├── notebooks/
│ ├── colab_training.ipynb # 🌟 MAIN NOTEBOOK (all-in-one)
│ ├── 01_data_exploration.ipynb # Medical dataset exploration
│ └── 02_evaluation.ipynb # Results evaluation
├── requirements.txt # Dependencies (with bitsandbytes)
├── README.md
└── .gitignore
```

## Implementation Phases

### Phase 1: Setup and Data Preparation (Local - Mac M1)

- ✅ Configure project structure
- ✅ Create `utils.py` with reusable functions
- 🔄 Create `prepare_dataset.py`:
  - Download `FreedomIntelligence/medical-o1-reasoning-SFT` dataset
  - Create subset of ~500-1000 examples for training
  - Train/test split (90%/10%)
  - Save in JSONL format
  - Basic exploratory analysis
- Expected structure: `{"question": ..., "complex_cot": ..., "response": ...}`

### Phase 2: Training Notebook (Google Colab - GPU)

**Main Notebook: `notebooks/colab_training.ipynb`**

Notebook sections:

1. **Setup and Configuration**
   - Install dependencies with pip
   - Verify available GPU (T4/V100/A100)
   - Configure HuggingFace authentication

2. **Data Loading and Preparation**
   - Load processed dataset (or download directly)
   - Visualize examples
   - Format for Llama 3.1 chat template

3. **QLoRA Configuration**
   - Load Llama 3.1-8B-Instruct
   - 4-bit quantization configuration (NF4)
   - LoRA config: rank=16, alpha=32
   - Target modules: q_proj, k_proj, v_proj, o_proj
   - Tokenizer setup

4. **Training (SFT)**
   - SFTTrainer from trl
   - Training arguments:
     - Batch size: 4-8 (with gradient accumulation)
     - Learning rate: 2e-4
     - Epochs: 2-3
     - Max seq length: 2048
     - FP16/BF16 training
   - Progress bars and logging
   - Save checkpoints

5. **Evaluation**
   - Load fine-tuned model
   - Baseline vs fine-tuned comparison
   - Qualitative examples (thinking + response)
   - Metrics: perplexity, format compliance

6. **Save and Export**
   - Save LoRA adapters
   - (Optional) Upload to HuggingFace Hub
   - Download to Google Drive

### Phase 3: Detailed Evaluation (Local Notebook)

**Notebook: `notebooks/02_evaluation.ipynb`**

- Load adapters from Google Drive
- Qualitative evaluation with more examples
- Success/failure case analysis
- Structured comparison

### Phase 4: Documentation and Presentation

**Professional README.md:**

- "Open in Colab" badge with direct link
- Project description and motivation
- Hybrid architecture (local + cloud)
- Screenshots of notebook results
- Reproduction instructions
- Justified technical decisions
- "Why Colab?" section explaining the choice

**Demo notebooks:**

- `01_data_exploration.ipynb`: Dataset analysis
- `colab_training.ipynb`: Complete training pipeline
- `02_evaluation.ipynb`: Results evaluation

## Key Technologies and Libraries

- **transformers**: Model and tokenizer management
- **trl**: SFTTrainer for fine-tuning
- **peft**: LoRA/QLoRA implementation
- **bitsandbytes**: 4-bit quantization (CUDA on Colab)
- **accelerate**: Hardware management
- **datasets**: HuggingFace data loading and processing

## Hardware Strategy

### Local (Mac M1)
- Data preparation
- Exploratory analysis
- Results evaluation
- Code development

### Google Colab (GPU - T4/V100/A100)
- QLoRA 4-bit training
- Complete fine-tuning
- Integrated evaluation
- **Advantages:**
  - Free/cheap GPU
  - Reproducible for recruiters
  - Real QLoRA (no compromises)
  - No local installation required

## Important Considerations

1. Output format must follow HuatuoGPT-o1 pattern: reasoning first (thinking), then final response
2. Use frequent checkpoints in Colab (limited sessions)
3. Subset must be representative (balance medical question types)
4. Clearly document why we chose Colab vs local/cloud
5. Git: DO NOT upload models, only LoRA adapters (if small) or instructions

## Costs

- **Colab Free**: $0 - T4 GPU, sufficient for this project
- **Colab Pro**: $10/month - Better GPUs and longer sessions (recommended)
- **Total estimated**: $0-10 (vs $15-25 on GCP)

### To-dos

- [x] Create project folder and file structure
- [x] Create utils.py with reusable functions
- [ ] Complete prepare_dataset.py
- [ ] Execute data preparation locally
- [ ] Create colab_training.ipynb notebook (complete pipeline)
- [ ] Execute training on Colab
- [ ] Create evaluation notebook
- [ ] Update README with results and Colab badge
