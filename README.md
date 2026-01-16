# Hybrid PEFT+RAG for Clinical Discharge Summarization

This repository contains the implementation for **"Reducing Hallucination in Clinical Discharge Summarization via Hybrid PEFT and RAG"**, submitted to CHIL 2026.

## Overview

### Problem
Clinical discharge summaries are essential for care continuity but require 5-10 minutes per patient to write. Large Language Models (LLMs) can automate this task but suffer from **hallucination**—generating clinically plausible but factually incorrect information—which poses serious patient safety risks.

### Approach
We present a three-phase hybrid framework:

1. **Phase 1 - QLoRA Fine-tuning**: Parameter-efficient adaptation of 6 LLMs on MIMIC-IV discharge summaries using 4-bit quantization
2. **Phase 2 - Hybrid PEFT+RAG**: Retrieved similar cases serve as **structural templates** (not content sources), guiding output format while all facts come strictly from the source note
3. **Phase 3 - Clinical Transparency**: Confidence scoring, factual alignment verification, and evidence attribution for safe deployment

### Key Contributions
- Hybrid PEFT+RAG reduces hallucination by **up to 34%** compared to fine-tuning alone
- Structured 11-section format achieves **100% section completeness** with 14.3 percentage points lower hallucination than narrative
- **Llama-3.1-8B** emerges as optimal with 18.5% hallucination rate and highest ROUGE scores
- Confidence scoring flags 77-83% of summaries as suitable for clinical use

### Models Evaluated
Six LLMs across two output formats (structured and narrative):
- Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, BioMistral-7B, Gemma-2-9B-IT, Phi-3-Medium-14B-Instruct, Qwen2.5-7B-Instruct

---

## Phase 1: QLoRA Fine-tuning

Phase 1 adapts pre-trained LLMs to generate clinical discharge summaries using **QLoRA (Quantized Low-Rank Adaptation)** with Unsloth optimization for memory efficiency.

### Methodology

1. **4-bit Quantization**: Base models are loaded in 4-bit precision (NF4) to reduce GPU memory requirements by ~75%
2. **LoRA Adapters**: Low-rank matrices are added to transformer attention and feed-forward layers, enabling training of only ~1% of parameters
3. **Dual-Format Training**: Each model is trained separately for:
   - **Structured format**: 11-section clinical template
   - **Narrative format**: Free-form paragraph summaries

### Structured 11-Section Format
The structured output organizes discharge summaries into standardized clinical sections:

| # | Section | Description |
|---|---------|-------------|
| 1 | Case Type | Primary diagnosis/procedure category |
| 2 | Patient & Service | Demographics and admitting service |
| 3 | Chief Complaint | Reason for hospitalization |
| 4 | History of Present Illness | Detailed illness narrative |
| 5 | Past Medical/Surgical History | Relevant medical background |
| 6 | Medications | Admission and discharge medications |
| 7 | Physical Examination | Key examination findings |
| 8 | Investigations/Labs/Imaging | Diagnostic results |
| 9 | Assessment/Impression | Clinical conclusions |
| 10 | Discharge Condition | Patient status at discharge |
| 11 | Follow-Up & Recommendations | Post-discharge instructions |

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank (r) | 64 | 16 for Qwen |
| LoRA alpha (α) | 64 | 32 for Qwen |
| LoRA dropout | 0.1 | dropout |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + MLP |
| Learning rate | 1e-4 | 5e-5 for Phi |
| Batch size | 4 | Per device |
| Gradient accumulation | 2 | Effective batch: 8 |
| Epochs | 3 | |
| Quantization | 4-bit (NF4) via Unsloth | QLoRA |
| Max sequence length | 4,096 | Tokens |
| Optimizer | AdamW 8-bit | Paged for Qwen |
| Weight decay | 0.05 | Regularization |


### Few-Shot Prompt Template

```
You are a medical AI assistant specialized in generating structured discharge summaries.

Here are two examples of the correct format:

EXAMPLE 1:
[Structured example from training data]

EXAMPLE 2:
[Structured example from training data]

Now generate a structured discharge summary for the following clinical note:
{input_clinical_note}

Generate the summary with these 11 sections:
- Case Type
- Patient & Service
- Chief Complaint / Admission Context
- History of Present Illness (HPI)
- Past Medical / Surgical History
- Medications (Discharge / Ongoing)
- Physical Examination (summarized)
- Investigations / Labs / Imaging
- Assessment / Impression
- Discharge Condition
- Follow-Up & Recommendations
```

---

## Phase 2: Hybrid PEFT+RAG

Phase 2 augments fine-tuned models with **Retrieval-Augmented Generation (RAG)** to reduce hallucination while maintaining structural consistency.

### Key Innovation: Templates, Not Content

Unlike traditional RAG that retrieves factual content, our approach uses retrieved cases as **structural templates only**:
- Retrieved cases guide the **format and organization** of the output
- All **factual content** comes strictly from the input clinical note
- This prevents cross-patient information leakage while leveraging structural patterns

### RAG Architecture

```
Input Clinical Note
        │
        ▼
┌─────────────────────┐
│  Dense Embedding    │  S-PubMedBERT-MS-MARCO (768d)
│  (Query Encoding)   │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   FAISS Index       │  Cosine similarity search
│   (Top-20 retrieval)│  Training + Validation corpus
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Cross-Encoder      │  MedCPT-Cross-Encoder
│  Reranking (Top-3)  │  Medical domain reranking
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Context Assembly   │  Retrieved cases as templates
│  + Prompt Building  │  + Input note for facts
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Fine-tuned LLM     │  QLoRA-adapted model
│  Generation         │  from Phase 1
└─────────────────────┘
        │
        ▼
   Structured and Narrative Summary
```

### RAG Components

| Component | Model/Tool | Specification |
|-----------|------------|---------------|
| Dense Embedding | S-PubMedBERT-MS-MARCO | 768 dimensions, medical domain |
| Vector Index | FAISS | Flat index, cosine similarity |
| Cross-Encoder Reranking | MedCPT-Cross-Encoder | Medical domain reranker |
| Initial Retrieval | Top-k | k=20 candidates |
| Final Selection | Reranked Top-k | k=3 cases |

### RAG Corpus Construction

The retrieval corpus comprises training and validation sets, with test set excluded to prevent data leakage:

```
Corpus = Training_Set ∪ Validation_Set
Corpus ∩ Test_Set = ∅
```

### RAG Prompt Template

```
You are a medical AI assistant generating structured discharge summaries.

RETRIEVED SIMILAR CASES (for format reference only):
{retrieved_case_1}
{retrieved_case_2}
{retrieved_case_3}

IMPORTANT: Use the above cases ONLY as structural templates.
Extract ALL factual content ONLY from the patient's clinical note below.
Do NOT copy any patient information from the retrieved cases.

PATIENT CLINICAL NOTE:
{input_clinical_note}

Generate a structured discharge summary with these 11 sections:
[Case Type, Patient & Service, Chief Complaint, HPI, Past Medical History,
Medications, Physical Exam, Investigations, Assessment, Discharge Condition, Follow-Up]
```

### Generation Parameters

| Model | Temperature | Top-P | Repetition Penalty | Max Tokens |
|-------|-------------|-------|-------------------|------------|
| Llama-3.1-8B | 0.4 | 0.95 | 1.10 | 2,048 |
| Mistral-7B | 0.5 | 0.95 | 1.12 | 2,048 |
| BioMistral-7B | 0.5 | 0.90 | 1.15 | 2,048 |
| Gemma-2-9B | 0.6 | 0.95 | 1.10 | 2,048 |
| Phi-3-Medium-14B | 0.7 | 0.95 | 1.12 | 2,048 |
| Qwen2.5-7B | 0.5 | 0.95 | 1.10 | 2,048 |

### Inference Performance

| Metric | Value |
|--------|-------|
| Retrieval Time (per sample) | 2-3 seconds |
| Generation Time (per sample) | 10-15 seconds |
| Total Inference Time | 12-18 seconds |
| Throughput | 3-5 samples/minute |
| GPU Memory (Inference) | 12-18 GB |

---

## Phase 3: Clinical Transparency (Explainability)

Three mechanisms for clinical transparency:

### 1. Confidence Scoring
Multi-factor reliability assessment combining:
- Retrieval quality (0.25 weight)
- Structure completeness (0.20 weight)
- Input-output consistency (0.25 weight)
- Length appropriateness (0.10 weight)
- Entity preservation (0.20 weight)

**Confidence Categories:**
- `HIGH` (≥0.80): Summary is reliable for clinical use
- `MODERATE` (0.60-0.80): Recommend careful review
- `LOW` (0.40-0.60): Requires thorough review
- `VERY LOW` (<0.40): Should not be used without regeneration

### 2. Factual Alignment
Sentence-level verification using S-PubMedBERT embeddings to classify each generated sentence as:
- `SUPPORTED` (≥0.70 similarity): Factually grounded in source note
- `PARTIAL` (0.50-0.70 similarity): Partially supported, requires review
- `UNSUPPORTED` (<0.50 similarity): Potential hallucination

### 3. Evidence Attribution
Traces which retrieved cases influenced each generated section through post-hoc embedding analysis, achieving **91% traceability**.

---

## Installation

```bash
# Clone repository
git clone https://github.com/AlekaMelese/CHIL2026-Hybrid-PEFT-RAG.git
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

## Project Structure

```
├── src/
│   ├── finetuning/              # Phase 1: QLoRA fine-tuning scripts
│   │   ├── structured/          # Structured (11-section) format
│   │   │   ├── llama/
│   │   │   ├── mistral/
│   │   │   ├── biomistral/
│   │   │   ├── gemma/
│   │   │   ├── phi/
│   │   │   └── qwen/
│   │   └── narrative/           # Narrative (paragraph) format
│   │       ├── llama/
│   │       ├── mistral/
│   │       ├── biomistral/
│   │       ├── gemma/
│   │       ├── phi/
│   │       └── qwen/
│   ├── rag/                     # Phase 2: RAG pipeline
│   │   ├── structured/          # Structured format RAG
│   │   └── narrative/           # Narrative format RAG
│   ├── explainability/          # Phase 3: Clinical transparency
│   │   ├── 1_evidence_attribution.py
│   │   ├── 2_confidence_scoring.py
│   │   ├── 4_factual_alignment.py
│   │   └── run_all_explainability.py
│   └── evaluation/              # Metrics computation
├── configs/                     # Hyperparameters
├── scripts/                     # Shell scripts for running experiments
├── supplementary/               # Supplementary data files
└── data_processing/             # Data preprocessing scripts
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
python src/rag/structured/llama/3_rag_inference.py

# Evaluate results
python src/rag/structured/llama/4_evaluate_rag.py
```

### Phase 3: Explainability

```bash
# Run all explainability analyses
python src/explainability/run_all_explainability.py

# Or run individual methods
python src/explainability/1_evidence_attribution.py
python src/explainability/2_confidence_scoring.py
python src/explainability/4_factual_alignment.py
```

## Results

### Performance Comparison (Structured Format, Hybrid PEFT+RAG)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Hallucination Rate |
|-------|---------|---------|---------|-------------------|
| **Llama-3.1-8B** | **0.466** | **0.334** | **0.421** | **18.5%** |
| Gemma-2-9B | 0.428 | 0.324 | 0.387 | 23.9% |
| Qwen2.5-7B | 0.439 | 0.334 | 0.395 | 27.8% |
| Phi-3-Medium-14B | 0.432 | 0.318 | 0.389 | 25.2% |
| Mistral-7B | 0.425 | 0.312 | 0.383 | 26.4% |
| BioMistral-7B | 0.418 | 0.305 | 0.376 | 24.8% |

### Explainability Results

| Metric | Value |
|--------|-------|
| HIGH/MODERATE Confidence | 77-83% |
| Evidence Traceability | 91% |
| Factual Alignment (SUPPORTED) | 100% |
| Structure Completeness | 100% |

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
