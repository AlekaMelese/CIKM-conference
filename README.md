# Hybrid PEFT+RAG for Clinical Discharge Summarization

This repository contains the implementation for **"Reducing Hallucination in Clinical Discharge Summarization via Hybrid PEFT and RAG"**, submitted to CHIL 2026.

## Overview

We present a hybrid framework combining Parameter-Efficient Fine-Tuning (PEFT) with Retrieval-Augmented Generation (RAG) for generating structured clinical discharge summaries with reduced hallucination rates.

### Key Features
- **Phase 1**: QLoRA fine-tuning of 6 LLMs (Llama-3.1-8B, Mistral-7B, BioMistral-7B, Gemma-2-9B, Phi-3.5-Mini, Qwen2.5-7B)
- **Phase 2**: Hybrid PEFT+RAG with S-PubMedBERT embeddings and MedCPT cross-encoder reranking
- **Phase 3**: Clinical transparency via confidence scoring, factual alignment, and evidence attribution
- **Dual Format**: Both structured (11-section) and narrative output formats

## Installation

```bash
# Clone repository
git clone https://github.com/[ANONYMOUS]/CHIL2026-Hybrid-PEFT-RAG.git
cd CHIL2026-Hybrid-PEFT-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data

This project uses the **MIMIC-IV** database. Due to data use agreements, we cannot redistribute the data.

To obtain access:
1. Complete CITI training at https://physionet.org/
2. Sign the data use agreement
3. Download from https://physionet.org/content/mimiciv/

See `data/README.md` for preprocessing instructions.

## Project Structure

```
├── src/
│   ├── finetuning/          # Phase 1: QLoRA fine-tuning scripts
│   │   ├── llama/
│   │   ├── mistral/
│   │   ├── biomistral/
│   │   ├── gemma/
│   │   ├── phi/
│   │   └── qwen/
│   ├── rag/                 # Phase 2: RAG pipeline
│   │   ├── structured/      # Structured format RAG
│   │   └── narrative/       # Narrative format RAG
│   ├── explainability/      # Phase 3: Clinical transparency
│   └── evaluation/          # Metrics computation
├── configs/                 # Hyperparameters
├── scripts/                 # Shell scripts for running experiments
└── data/                    # Data instructions (not actual data)
```

## Usage

### Phase 1: Fine-tuning

```bash
# Fine-tune Llama for structured format
python src/finetuning/llama/llama_finetune.py

# Fine-tune Llama for narrative format
python src/finetuning/llama/llama_narrative_finetune.py
```

### Phase 2: RAG Inference

```bash
# Build RAG corpus
python src/rag/structured/llama/2_build_rag_corpus.py

# Run RAG-enhanced generation
python src/rag/structured/llama/4_evaluate_rag.py
```

### Phase 3: Explainability

```bash
# Run all explainability analyses
python src/explainability/run_all_explainability.py
```

## Model Configurations

All models use the following hyperparameters (see `configs/model_configs.yaml`):

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 64 |
| LoRA alpha (α) | 64 |
| Learning rate | 2e-4 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Epochs | 3 |
| Quantization | 4-bit (QLoRA) |

## Results

| Model | Format | Method | ROUGE-1 | ROUGE-2 | Hallucination Rate |
|-------|--------|--------|---------|---------|-------------------|
| Llama-3.1-8B | Structured | Hybrid | **0.466** | **0.334** | **18.5%** |
| Gemma-2-9B | Structured | Hybrid | 0.428 | 0.324 | 23.9% |
| Qwen2.5-7B | Structured | Hybrid | 0.439 | 0.334 | 27.8% |

## Citation

```bibtex
@inproceedings{anonymous2026hybrid,
  title={Reducing Hallucination in Clinical Discharge Summarization via Hybrid PEFT and RAG},
  author={Anonymous},
  booktitle={Conference on Health, Inference, and Learning (CHIL)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MIMIC-IV dataset provided by PhysioNet
- Unsloth for optimized fine-tuning
- Hugging Face Transformers
