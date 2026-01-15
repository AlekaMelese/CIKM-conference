# Hybrid PEFT+RAG for Clinical Discharge Summarization

This repository contains the implementation for **"Reducing Hallucination in Clinical Discharge Summarization via Hybrid PEFT and RAG"**, submitted to CHIL 2026.

## Overview

### Problem
Clinical discharge summaries are essential for care continuity but require 5-10 minutes per patient to write. Large Language Models (LLMs) can automate this task but suffer from **hallucination** generating clinically plausible but factually incorrect information which poses serious patient safety risks.

### Approach
We present a three-phase hybrid framework:

1. **Phase 1 - QLoRA Fine-tuning**: Parameter-efficient adaptation of 6 LLMs on MIMIC-IV discharge summaries using 4-bit quantization
2. **Phase 2 - Hybrid PEFT+RAG**: Retrieved similar cases serve as **structural templates** (not content sources), guiding output format while all facts come strictly from the source note
3. **Phase 3 - Clinical Transparency**: Confidence scoring, factual alignment verification, and evidence attribution for safe deployment

### Key Contributions
- Hybrid PEFT+RAG reduces hallucination by **up to 40%** compared to fine-tuning alone
- Structured 11-section format achieves **100% section completeness** with 14.3 percentage points lower hallucination than narrative
- **Llama-3.1-8B** emerges as optimal with 18.5% hallucination rate and highest ROUGE scores
- Confidence scoring flags 77-83% of summaries as suitable for clinical use

### Models Evaluated
Six LLMs across two output formats (structured and narrative):
- Llama-3.1-8B, Mistral-7B, BioMistral-7B, Gemma-2-9B, Phi-3.5-Mini, Qwen2.5-7B

### Technical Components
- **Embeddings**: S-PubMedBERT-MS-MARCO for medical-domain semantic similarity
- **Reranking**: MedCPT cross-encoder for retrieval refinement
- **Retrieval**: FAISS index with top-20 candidates, reranked to top-3

### Explainability (Phase 3)
Three mechanisms for clinical transparency:

1. **Confidence Scoring**: Multi-factor reliability assessment combining:
   - Retrieval quality (0.25 weight)
   - Structure completeness (0.20 weight)
   - Input-output consistency (0.25 weight)
   - Length appropriateness (0.10 weight)
   - Entity preservation (0.20 weight)

2. **Factual Alignment**: Sentence-level verification using S-PubMedBERT embeddings to classify each generated sentence as:
   - SUPPORTED (≥0.70 similarity to source)
   - PARTIAL (0.50-0.70 similarity)
   - UNSUPPORTED (<0.50 similarity, potential hallucination)

3. **Evidence Attribution**: Traces which retrieved cases influenced each generated section through post-hoc embedding analysis, achieving 91% traceability

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
│   │   ├── structured/      # Structured (11-section) format
│   │   │   ├── llama/
│   │   │   ├── mistral/
│   │   │   ├── biomistral/
│   │   │   ├── gemma/
│   │   │   ├── phi/
│   │   │   └── qwen/
│   │   └── narrative/       # Narrative (paragraph) format
│   │       ├── llama/
│   │       ├── mistral/
│   │       ├── biomistral/
│   │       ├── gemma/
│   │       ├── phi/
│   │       └── qwen/
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
python src/finetuning/structured/llama/llama_finetune.py

# Fine-tune Llama for narrative format
python src/finetuning/narrative/llama/llama_narrative_finetune.py
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
| Learning rate | 1e-4 (Phi: 5e-5) |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Effective batch size | 8 |
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
