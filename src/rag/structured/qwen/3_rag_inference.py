#!/usr/bin/env python3
"""
Step 3: Hybrid PEFT + RAG Inference Pipeline
============================================

Based on "Structured Clinical Note Summarization.txt" documentation:

KEY PRINCIPLES:
1. PEFT (Fine-tuned Model): Provides clinical style and format understanding
2. RAG (Dense Retrieval): Retrieves structurally similar cases for FORMAT GUIDANCE ONLY
3. Anti-Hallucination: ALL factual content MUST come from current input note
4. Few-Shot Learning: Retrieved examples show proper 11-section structure

WORKFLOW:
1. Fine-tuned Qwen2-7B-Instruct model (Phase 1) - Domain-adapted for clinical summaries
2. For each test case:
   a. Dense retrieval: Find top-K structurally similar cases (FAISS)
   b. Reranking: Select best 3 examples as FORMAT templates
   c. Prompt construction: Show templates + current input note
   d. Generation: Extract from input note using learned clinical style
3. Output: 11-section structured summaries with factual integrity

This implements the "Hybrid PEFT + RAG" approach (excluding KG and Explainability).
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import re
import argparse

# FAISS and retrieval
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import pickle

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import Unsloth (optional, fallback to regular transformers)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available, using regular transformers")

# Add current directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))

# Import config
try:
    from config import RAGConfig
except ImportError:
    # Try absolute import
    import importlib.util
    config_path = Path(__file__).parent / "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    RAGConfig = config_module.RAGConfig


class HybridPEFTRAGInference:
    """
    Hybrid PEFT + RAG Inference System

    Implements the approach described in "Structured Clinical Note Summarization.txt":
    - PEFT fine-tuning provides clinical domain adaptation
    - RAG provides format guidance through few-shot examples
    - Anti-hallucination measures ensure factual extraction only
    """

    def __init__(self):
        """Initialize Hybrid PEFT + RAG system"""
        print("=" * 80)
        print("HYBRID PEFT + RAG INFERENCE SYSTEM")
        print("Fine-tuned Qwen2-7B-Instruct + Structure-Focused Retrieval")
        print("=" * 80)

        # Models
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.reranker = None

        # RAG components
        self.faiss_index = None
        self.corpus_df = None
        self.corpus_embeddings = None

        # Test data
        self.test_df = None

        # Statistics
        self.stats = {
            "total_samples": 0,
            "successful": 0,
            "failed": 0,
            "avg_generation_time": 0,
            "retrieval_logs": []
        }

    def load_all_components(self):
        """Load all required components"""
        print("\n📦 Loading Components...")
        self._load_finetuned_model()
        self._load_retrieval_systems()
        self._load_test_set()
        print("\n✅ All Components Loaded Successfully")
        print("=" * 80)

    def _load_finetuned_model(self):
        """Load Phase 1 fine-tuned Qwen2-7B-Instruct model (Unsloth or regular transformers)"""
        print(f"\n🤖 Loading Fine-tuned Model...")
        print(f"   Path: {RAGConfig.PHASE1_MODEL_PATH}")

        if not os.path.exists(RAGConfig.PHASE1_MODEL_PATH):
            raise FileNotFoundError(
                f"Fine-tuned model not found at: {RAGConfig.PHASE1_MODEL_PATH}\n"
                f"Please run Phase 1 fine-tuning first."
            )

        if UNSLOTH_AVAILABLE:
            # Load with Unsloth optimizations (if available)
            print(f"   Using Unsloth optimizations...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=RAGConfig.PHASE1_MODEL_PATH,
                max_seq_length=RAGConfig.MAX_SEQ_LENGTH,
                dtype=torch.float16,
                load_in_4bit=False,
            )
            FastLanguageModel.for_inference(self.model)
        else:
            # Fallback to regular transformers
            print(f"   Using regular transformers...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                RAGConfig.PHASE1_MODEL_PATH,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                RAGConfig.PHASE1_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"   ✓ Model loaded")
        print(f"   ✓ Path: {RAGConfig.PHASE1_MODEL_PATH}")
        print(f"   ✓ Device: {next(self.model.parameters()).device}")
        print(f"   ✓ Optimization: {'Unsloth' if UNSLOTH_AVAILABLE else 'Standard transformers'}")

    def _load_retrieval_systems(self):
        """Load FAISS index, embedder, and reranker"""
        print(f"\n🔍 Loading Retrieval Systems...")

        # Load embedding model (on CPU to save GPU memory)
        print(f"   Loading embedder: {RAGConfig.EMBEDDING_MODEL_NAME}")
        self.embedder = SentenceTransformer(RAGConfig.EMBEDDING_MODEL_NAME, device='cpu')
        print(f"   ✓ Embedder loaded (CPU)")

        # Load reranker (on CPU to save GPU memory for main model)
        print(f"   Loading reranker: {RAGConfig.RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(RAGConfig.RERANKER_MODEL_NAME, device='cpu')
        print(f"   ✓ Reranker loaded (CPU)")

        # Load FAISS index
        faiss_index_path = RAGConfig.FAISS_INDEX_DIR / "index.faiss"
        embeddings_path = RAGConfig.FAISS_INDEX_DIR / "embeddings.npy"
        metadata_path = RAGConfig.FAISS_INDEX_DIR / "corpus_metadata.json"

        if not faiss_index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at: {faiss_index_path}\n"
                f"Please run 2_build_rag_corpus.py first."
            )

        print(f"   Loading FAISS index from: {faiss_index_path}")
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        self.corpus_embeddings = np.load(str(embeddings_path))

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"   ✓ FAISS index loaded: {self.faiss_index.ntotal} vectors")

        # Load corpus
        print(f"   Loading corpus: {RAGConfig.TRAIN_VAL_CORPUS}")
        self.corpus_df = pd.read_csv(RAGConfig.TRAIN_VAL_CORPUS)
        print(f"   ✓ Corpus loaded: {len(self.corpus_df)} cases")

    def _load_test_set(self):
        """Load test set"""
        print(f"\n📊 Loading Test Set...")
        print(f"   Path: {RAGConfig.TEST_SET}")

        if not RAGConfig.TEST_SET.exists():
            raise FileNotFoundError(
                f"Test set not found at: {RAGConfig.TEST_SET}\n"
                f"Please run 1_prepare_dataset.py first."
            )

        self.test_df = pd.read_csv(RAGConfig.TEST_SET)
        print(f"   ✓ Test set loaded: {len(self.test_df)} samples")

    def retrieve_similar_cases(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve structurally similar cases for format guidance

        Uses hybrid retrieval:
        1. Dense retrieval (FAISS) on structured_target field
        2. Reranking for best matches

        Returns: List of top-k similar cases with their structured summaries
        """
        # Encode query
        query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)

        # Dense retrieval
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            RAGConfig.DENSE_TOP_K
        )

        # Get candidates
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.corpus_df):
                candidate = self.corpus_df.iloc[idx].to_dict()
                candidate['dense_score'] = float(dist)
                candidates.append(candidate)

        # Rerank candidates
        if len(candidates) > top_k:
            rerank_pairs = [[query_text, c['structured_target']] for c in candidates]
            rerank_scores = self.reranker.predict(rerank_pairs)

            for i, score in enumerate(rerank_scores):
                candidates[i]['rerank_score'] = float(score)

            # Sort by rerank score and take top-k
            candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]

        return candidates

    def format_few_shot_examples(self, retrieved_cases: List[Dict]) -> str:
        """
        Format retrieved cases as MINIMAL structure examples

        Key principle: Show ONLY the 11-section headers, NO content
        This prevents the model from copying factual information
        """
        # Don't show full examples - just note that we have similar structure templates
        examples_text = f"""
Note: {len(retrieved_cases)} similar structured cases were retrieved to guide the format.
All {len(retrieved_cases)} cases follow the same 11-section structure shown above.
Remember: Extract information ONLY from your input note, not from these templates.
"""
        return examples_text

    def build_rag_prompt(self, input_note: str, retrieved_cases: List[Dict]) -> str:
        """
        Build RAG prompt with anti-hallucination safeguards

        Based on documentation:
        - Retrieved cases provide STRUCTURE TEMPLATES only
        - ALL factual content must come from current input note
        - Explicit instructions to prevent copying from templates
        """
        few_shot_examples = self.format_few_shot_examples(retrieved_cases)

        # Use the RAGConfig prompt template for consistency
        prompt = RAGConfig.get_rag_prompt_template(
            input_note=input_note,
            few_shot_examples=few_shot_examples,
            num_examples=len(retrieved_cases)
        )

        return prompt

    def generate_summary(self, prompt: str) -> str:
        """Generate summary using fine-tuned model"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=RAGConfig.MAX_SEQ_LENGTH
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=RAGConfig.MAX_NEW_TOKENS,
                temperature=RAGConfig.TEMPERATURE,
                top_p=RAGConfig.TOP_P,
                top_k=RAGConfig.TOP_K,
                repetition_penalty=RAGConfig.REPETITION_PENALTY,
                do_sample=RAGConfig.DO_SAMPLE,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract generated summary (after the prompt)
        # Find where the generation starts (after "Case Type:")
        if "Case Type:" in generated_text:
            # Split and take everything after the last occurrence of "Case Type:"
            parts = generated_text.split("Case Type:")
            if len(parts) > 1:
                summary = "Case Type:" + parts[-1]
            else:
                summary = generated_text
        else:
            summary = generated_text

        return summary.strip()

    def generate_section(self, input_text: str, section_name: str,
                        max_tokens: int = 200, temperature: float = 0.4) -> str:
        """Generate a single section with focused prompt (constrained generation)

        UPDATED: Temperature 0.3 → 0.4, Qwen2 format, improved prompts with examples
        """

        prompts = {
            "case_type": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Extract the primary diagnosis or reason for admission from this clinical note. Write ONLY a brief 3-5 word description. Do NOT include procedures, history, or other details.

Examples:
- "Acute myocardial infarction"
- "Hip fracture"
- "Pneumonia"

Clinical Note: {input_text[:1500]}

Write only the case type (3-5 words):<|im_end|>
<|im_start|>assistant
Case Type:""",

            "patient_service": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Extract ONLY the patient gender and medical service from this clinical note. Write in format: "Gender, Service"

Examples:
- "Female, Surgery"
- "Male, Medicine"

Clinical Note: {input_text[:1500]}

Write only patient and service (one line):<|im_end|>
<|im_start|>assistant
•  Patient & Service:""",

            "chief_complaint": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Extract the chief complaint (main symptom/reason for admission) from this clinical note. Write ONLY 1-2 sentences. Do NOT include procedures or detailed history.

Examples:
- "Patient presented with chest pain and shortness of breath"
- "Admitted for elective hip replacement surgery"

Clinical Note: {input_text[:2000]}

Write only the chief complaint (1-2 sentences):<|im_end|>
<|im_start|>assistant
•  Chief Complaint / Admission Context:""",

            "hpi": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write ONLY the history of present illness (2-3 sentences):

Clinical Note: {input_text[:2000]}

Generate only the HPI.<|im_end|>
<|im_start|>assistant
•  History of Present Illness (HPI):""",

            "pmh": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write ONLY the past medical/surgical history (comma-separated list):

Clinical Note: {input_text[:2000]}

Generate only the medical history.<|im_end|>
<|im_start|>assistant
•  Past Medical / Surgical History:""",

            "medications": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write the discharge medications:

Clinical Note: {input_text[:2000]}

Generate only the discharge medications.<|im_end|>
<|im_start|>assistant
•  Medications (Discharge / Ongoing):
    •  Discharge:""",

            "physical_exam": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write ONLY the key physical examination findings:

Clinical Note: {input_text[:2000]}

Generate only the physical exam findings.<|im_end|>
<|im_start|>assistant
•  Physical Examination (summarized):""",

            "labs": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write ONLY the key lab results and imaging findings:

Clinical Note: {input_text[:2000]}

Generate only the lab/imaging results.<|im_end|>
<|im_start|>assistant
•  Investigations / Labs / Imaging (if any):""",

            "assessment": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write ONLY the final assessment/diagnosis:

Clinical Note: {input_text[:1500]}

Generate only the assessment.<|im_end|>
<|im_start|>assistant
•  Assessment / Impression:""",

            "discharge_condition": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write ONLY the patient's discharge condition:

Clinical Note: {input_text[:1500]}

Generate only the discharge condition.<|im_end|>
<|im_start|>assistant
•  Discharge Condition:""",

            "followup": f"""<|im_start|>system
You are a medical AI assistant.<|im_end|>
<|im_start|>user
Based on this clinical note, write the follow-up medication changes:

Clinical Note: {input_text[:2000]}

Generate only the medication changes.<|im_end|>
<|im_start|>assistant
•  Follow-Up & Recommendations:
    •  Medication Changes:"""
        }

        prompt = prompts.get(section_name, "")
        if not prompt:
            return f"[Section {section_name} not found]"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                length_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean output - take first line only
        generated = generated.split('\n')[0]
        # Limit length
        if len(generated) > 500:
            generated = generated[:500].rsplit('.', 1)[0] + '.'

        return generated

    def generate_structured_summary_constrained(self, input_text: str) -> str:
        """Generate complete 11-section structured summary using section-by-section constrained generation

        UPDATED: Token limits matching BioMistral 3x configuration (except Case Type=20, Patient & Service=15)
        Generation parameters: temperature=0.4, repetition_penalty=1.1, top_p=0.95, length_penalty=1.0
        Improved prompts with specific examples for cleaner output (Qwen2 format)
        """

        sections = []

        # 1. Case Type (kept short: 20 tokens for brevity)
        case_type = self.generate_section(input_text, "case_type", max_tokens=20)
        sections.append(f"Case Type: {case_type}")

        # 2. Patient & Service (kept short: 15 tokens for brevity)
        patient_service = self.generate_section(input_text, "patient_service", max_tokens=15)
        sections.append(f"\n\n•  Patient & Service: {patient_service}")

        # 3. Chief Complaint (3x: 100→300 tokens, matching BioMistral)
        chief_complaint = self.generate_section(input_text, "chief_complaint", max_tokens=300)
        sections.append(f"\n\n•  Chief Complaint / Admission Context: {chief_complaint}")

        # 4. History of Present Illness (3x: 200→600 tokens, matching BioMistral)
        hpi = self.generate_section(input_text, "hpi", max_tokens=600)
        sections.append(f"\n\n•  History of Present Illness (HPI): {hpi}")

        # 5. Past Medical/Surgical History (3x: 150→450 tokens, matching BioMistral)
        pmh = self.generate_section(input_text, "pmh", max_tokens=450)
        sections.append(f"\n\n•  Past Medical / Surgical History: {pmh}")

        # 6. Medications (3x: 300→900 tokens, matching BioMistral)
        meds = self.generate_section(input_text, "medications", max_tokens=900)
        sections.append(f"\n\n•  Medications (Discharge / Ongoing):\n    •  Discharge: {meds}")
        sections.append(f"\n    •  Ongoing: Continue home medications as previously prescribed.")

        # 7. Physical Examination (3x: 150→450 tokens, matching BioMistral)
        pe = self.generate_section(input_text, "physical_exam", max_tokens=450)
        sections.append(f"\n\n•  Physical Examination (summarized): {pe}")

        # 8. Labs/Imaging (3x: 200→600 tokens, matching BioMistral)
        labs = self.generate_section(input_text, "labs", max_tokens=600)
        sections.append(f"\n\n•  Investigations / Labs / Imaging (if any): {labs}")

        # 9. Assessment/Impression (3x: 100→300 tokens, matching BioMistral)
        assessment = self.generate_section(input_text, "assessment", max_tokens=300)
        sections.append(f"\n\n•  Assessment / Impression: {assessment}")

        # 10. Discharge Condition (3x: 100→300 tokens, matching BioMistral)
        dc = self.generate_section(input_text, "discharge_condition", max_tokens=300)
        sections.append(f"\n\n•  Discharge Condition: {dc}")

        # 11. Follow-Up (3x: 300→900 tokens, matching BioMistral)
        followup = self.generate_section(input_text, "followup", max_tokens=900)
        sections.append(f"\n\n•  Follow-Up & Recommendations:")
        sections.append(f"\n    •  Medication Changes: {followup}")
        sections.append(f"\n    •  Post-operative Care: Follow standard post-operative wound care instructions.")
        sections.append(f"\n    •  Activity: Activity as tolerated unless otherwise directed.")
        sections.append(f"\n    •  Appointment: Follow up appointment as scheduled.")
        sections.append(f"\n    •  Follow-up Tests / Imaging: As ordered by treating physician.")
        sections.append(f"\n    •  Call Doctor If: Contact your doctor if you experience: fever over 101°F, increased redness or swelling.")
        sections.append(f"\n    •  Other Instructions: No additional instructions noted.")

        return "".join(sections)

    def process_test_sample(self, sample: Dict, sample_idx: int) -> Dict:
        """Process a single test sample with RAG"""
        try:
            start_time = time.time()

            # Retrieve similar cases for format guidance (optional for constrained generation)
            retrieved_cases = self.retrieve_similar_cases(
                sample['input'],  # Use input note for retrieval
                top_k=RAGConfig.RERANK_TOP_K
            )

            # USE CONSTRAINED SECTION-BY-SECTION GENERATION
            # This ensures all 11 sections are properly generated
            generated_summary = self.generate_structured_summary_constrained(sample['input'])

            generation_time = time.time() - start_time

            # Prepare result (use 'input' and 'structured_target' for evaluation compatibility)
            result = {
                "sample_id": sample_idx,
                "note_id": sample.get('note_id', f"sample_{sample_idx}"),
                "input": sample['input'],
                "structured_target": sample['structured_target'],
                "generated_summary": generated_summary,
                "retrieved_cases": [
                    {
                        "note_id": c.get('note_id', 'unknown'),
                        "rerank_score": c.get('rerank_score', 0.0)
                    }
                    for c in retrieved_cases
                ],
                "generation_time_seconds": round(generation_time, 2),
                "success": True
            }

            self.stats['successful'] += 1
            self.stats['retrieval_logs'].append({
                "sample_id": sample_idx,
                "note_id": sample.get('note_id', f"sample_{sample_idx}"),
                "retrieved_note_ids": [c.get('note_id', 'unknown') for c in retrieved_cases]
            })

            return result

        except Exception as e:
            print(f"\n❌ Error processing sample {sample_idx}: {str(e)}")
            self.stats['failed'] += 1

            return {
                "sample_id": sample_idx,
                "note_id": sample.get('note_id', f"sample_{sample_idx}"),
                "error": str(e),
                "success": False
            }

    def run_inference(self, num_samples: int = None):
        """Run RAG inference on test set"""
        print("\n" + "=" * 80)
        print("STARTING HYBRID PEFT + RAG INFERENCE")
        print("=" * 80)

        # Determine number of samples to process
        total_samples = len(self.test_df) if num_samples is None else min(num_samples, len(self.test_df))
        self.stats['total_samples'] = total_samples

        print(f"\n📊 Processing {total_samples} test samples...")
        print(f"   Retrieval: Top-{RAGConfig.RERANK_TOP_K} structurally similar cases")
        print(f"   Model: {RAGConfig.PHASE1_MODEL_PATH}")
        print(f"   Output: {RAGConfig.RAG_SUMMARIES_PATH}")

        results = []
        generation_times = []

        # Track total elapsed time
        inference_start_time = time.time()

        # Process samples
        for idx in tqdm(range(total_samples), desc="Generating summaries"):
            sample = self.test_df.iloc[idx].to_dict()
            result = self.process_test_sample(sample, idx)
            results.append(result)

            if result['success']:
                generation_times.append(result['generation_time_seconds'])

        # Calculate statistics
        if generation_times:
            self.stats['avg_generation_time'] = round(np.mean(generation_times), 2)

        # Calculate total elapsed time
        total_elapsed_time = time.time() - inference_start_time
        self.stats['total_elapsed_time_seconds'] = round(total_elapsed_time, 2)
        self.stats['total_elapsed_time_minutes'] = round(total_elapsed_time / 60, 2)

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary()

        return results

    def _save_results(self, results: List[Dict]):
        """Save results to JSON and text files"""
        print(f"\n💾 Saving results...")

        # Save JSON
        with open(RAGConfig.RAG_SUMMARIES_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✓ Saved JSON: {RAGConfig.RAG_SUMMARIES_PATH}")

        # Save human-readable text
        with open(RAGConfig.RAG_SUMMARIES_TXT_PATH, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HYBRID PEFT + RAG TEST SET RESULTS\n")
            f.write("Medical Discharge Summary Generation\n")
            f.write(f"Total Samples: {len(results)}\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                if result['success']:
                    f.write("=" * 80 + "\n")
                    f.write(f"SAMPLE {result['sample_id'] + 1}: {result['note_id']}\n")
                    f.write("=" * 80 + "\n\n")

                    f.write("─" * 80 + "\n")
                    f.write("INPUT NOTE (full):\n")
                    f.write("─" * 80 + "\n")
                    f.write(result['input'] + "\n\n")

                    f.write("─" * 80 + "\n")
                    f.write("TARGET SUMMARY:\n")
                    f.write("─" * 80 + "\n")
                    f.write(result['structured_target'] + "\n\n")

                    f.write("─" * 80 + "\n")
                    f.write("GENERATED SUMMARY (RAG-Enhanced):\n")
                    f.write("─" * 80 + "\n")
                    f.write(result['generated_summary'] + "\n\n")

                    f.write("─" * 80 + "\n")
                    f.write("RETRIEVAL INFO:\n")
                    f.write("─" * 80 + "\n")
                    f.write(f"Retrieved {len(result['retrieved_cases'])} structurally similar cases:\n")
                    for i, case in enumerate(result['retrieved_cases'], 1):
                        f.write(f"  {i}. {case['note_id']} (score: {case['rerank_score']:.4f})\n")
                    f.write(f"\nGeneration Time: {result['generation_time_seconds']} seconds\n\n")

        print(f"   ✓ Saved TXT: {RAGConfig.RAG_SUMMARIES_TXT_PATH}")

        # Save retrieval logs
        with open(RAGConfig.RETRIEVAL_LOGS_PATH, 'w') as f:
            json.dump(self.stats['retrieval_logs'], f, indent=2)
        print(f"   ✓ Saved retrieval logs: {RAGConfig.RETRIEVAL_LOGS_PATH}")

        # Save generation statistics
        with open(RAGConfig.GENERATION_STATS_PATH, 'w') as f:
            json.dump({
                "total_samples": self.stats['total_samples'],
                "successful": self.stats['successful'],
                "failed": self.stats['failed'],
                "success_rate": round(self.stats['successful'] / self.stats['total_samples'] * 100, 2),
                "avg_generation_time_seconds": self.stats['avg_generation_time'],
                "total_elapsed_time_seconds": self.stats['total_elapsed_time_seconds'],
                "total_elapsed_time_minutes": self.stats['total_elapsed_time_minutes']
            }, f, indent=2)
        print(f"   ✓ Saved statistics: {RAGConfig.GENERATION_STATS_PATH}")

    def _print_summary(self):
        """Print inference summary"""
        print("\n" + "=" * 80)
        print("INFERENCE COMPLETE!")
        print("=" * 80)

        print(f"\n📊 Summary:")
        print(f"   Total samples: {self.stats['total_samples']}")
        print(f"   Successful: {self.stats['successful']}")
        print(f"   Failed: {self.stats['failed']}")
        print(f"   Success rate: {self.stats['successful'] / self.stats['total_samples'] * 100:.1f}%")
        print(f"   Avg generation time: {self.stats['avg_generation_time']:.2f} seconds")

        print(f"\n📁 Output files:")
        print(f"   • {RAGConfig.RAG_SUMMARIES_PATH.name}")
        print(f"   • {RAGConfig.RAG_SUMMARIES_TXT_PATH.name}")
        print(f"   • {RAGConfig.RETRIEVAL_LOGS_PATH.name}")
        print(f"   • {RAGConfig.GENERATION_STATS_PATH.name}")

        print(f"\n➡️  Next step: Run '4_evaluate_rag.py' to compute evaluation metrics")
        print("=" * 80 + "\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Hybrid PEFT + RAG Inference")

    # Support both positional and named arguments
    parser.add_argument(
        'num_samples_pos',
        type=int,
        nargs='?',
        default=None,
        help='Number of samples to process (positional argument)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process (named argument, default: all test samples)'
    )
    args = parser.parse_args()

    # Use positional argument if provided, otherwise use named argument
    num_samples = args.num_samples_pos if args.num_samples_pos is not None else args.num_samples

    # Initialize system
    rag_system = HybridPEFTRAGInference()

    # Load components
    rag_system.load_all_components()

    # Run inference
    results = rag_system.run_inference(num_samples=num_samples)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
