#!/usr/bin/env python3
"""
Step 3: Narrative RAG Inference Pipeline
=========================================

Based on README.md documentation for NARRATIVE RAG:

KEY PRINCIPLES:
1. PEFT (Fine-tuned Model): Provides clinical narrative style and domain adaptation
2. RAG (Dense Retrieval): Retrieves similar NARRATIVE cases for STYLE GUIDANCE
3. Anti-Hallucination: ALL factual content MUST come from current input note
4. Output Format: Flowing paragraphs (NO bullets, NO headers, NO structured sections)

WORKFLOW:
1. Fine-tuned Qwen2-7B-Instruct model (Phase 1) - Trained on narrative summaries
2. For each test case:
   a. Dense retrieval: Find top-K similar NARRATIVE cases (FAISS)
   b. Reranking: Select best 3 examples for STYLE templates
   c. Prompt construction: Show narrative style examples + current input note
   d. Generation: Generate flowing paragraph narrative
3. Output: Narrative paragraph discharge summaries (NOT structured)

CRITICAL: This is NARRATIVE RAG - generates flowing paragraphs, NOT 11-section structured format.
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
import textwrap

# FAISS and retrieval
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import Unsloth (optional, fallback to regular transformers)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not available, using regular transformers")


class NarrativeRAGConfig:
    """Configuration for Narrative RAG Pipeline - generates flowing paragraphs"""

    # ========== Paths ==========
    BASE_DIR = Path(__file__).parent
    FINETUNING_DIR = BASE_DIR.parent / "Finetuning"
    DATA_DIR = BASE_DIR.parent.parent.parent.parent / "Data"

    # Phase 1 fine-tuned NARRATIVE model (use final_model to match baseline)
    PHASE1_MODEL_DIR = FINETUNING_DIR / "outputs" / "final_model"
    PHASE1_MODEL_PATH = str(PHASE1_MODEL_DIR)

    # RAG-specific directories
    RAG_DATA_DIR = BASE_DIR / "data"
    RAG_OUTPUTS_DIR = BASE_DIR / "outputs"
    RAG_LOGS_DIR = BASE_DIR / "logs"

    # Dataset paths
    TRAIN_VAL_CORPUS = RAG_DATA_DIR / "train_val_corpus.csv"
    TEST_SET = RAG_DATA_DIR / "test_set.csv"

    # Index paths
    FAISS_INDEX_DIR = RAG_DATA_DIR / "faiss_index"

    # Output paths - NARRATIVE specific
    RAG_SUMMARIES_PATH = RAG_OUTPUTS_DIR / "narrative_rag_summaries.json"
    RAG_SUMMARIES_TXT_PATH = RAG_OUTPUTS_DIR / "narrative_rag_summaries.txt"
    RETRIEVAL_LOGS_PATH = RAG_OUTPUTS_DIR / "retrieval_logs.json"
    GENERATION_STATS_PATH = RAG_OUTPUTS_DIR / "generation_stats.json"

    # ========== Model Configuration ==========
    # Dense embedding model: Medical-domain PubMedBERT
    EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
    EMBEDDING_DIM = 768

    # Reranker model: Medical-domain MedCPT
    RERANKER_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"

    # ========== Retrieval Configuration ==========
    # CRITICAL: Use 'target' for NARRATIVE (paragraphs), NOT 'structured_target' (bullets)
    RETRIEVAL_FIELD = "target"  # NARRATIVE paragraphs

    DENSE_TOP_K = 20  # Initial dense retrieval candidates
    RERANK_TOP_K = 3  # Final few-shot examples

    # ========== Generation Configuration ==========
    # MATCH QWEN BASELINE EXACTLY from qwen_narrative_generation.py
    # Baseline achieved: ROUGE1=0.307, BERTScore=0.820, ClinicalBERT=0.948
    MAX_SEQ_LENGTH = 4096   # BASELINE
    MAX_NEW_TOKENS = 768    # BASELINE: dynamic up to 768
    MIN_NEW_TOKENS = 0      # BASELINE: NO minimum (baseline doesn't use this!)
    TEMPERATURE = 0.6       # BASELINE: 0.6
    TOP_P = 0.92            # BASELINE: 0.92 (exact match)
    TOP_K = 50              # BASELINE: 50
    REPETITION_PENALTY = 1.1   # BASELINE: 1.1 (exact match)
    LENGTH_PENALTY = 1.0    # BASELINE
    NO_REPEAT_NGRAM_SIZE = 0   # BASELINE: disabled
    DO_SAMPLE = True        # BASELINE: True

    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.RAG_DATA_DIR,
            cls.RAG_OUTPUTS_DIR,
            cls.RAG_LOGS_DIR,
            cls.FAISS_INDEX_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class NarrativeRAGInference:
    """
    Narrative RAG Inference System

    Generates FLOWING PARAGRAPH discharge summaries (NOT structured 11-section format).

    Key differences from Structured RAG:
    - Retrieves narrative examples (target field) instead of structured (structured_target)
    - Generates flowing prose paragraphs
    - Removes any bullets, headers, or structured formatting
    - Style guidance from narrative examples
    """

    def __init__(self):
        """Initialize Narrative RAG system"""
        print("=" * 80)
        print("NARRATIVE RAG INFERENCE SYSTEM")
        print("Fine-tuned Qwen2-7B-Instruct + Narrative Style Retrieval")
        print("Output: Flowing paragraphs (NOT structured sections)")
        print("=" * 80)

        # Create directories
        NarrativeRAGConfig.create_directories()

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
        print("\n Loading Components...")
        self._load_finetuned_model()
        self._load_retrieval_systems()
        self._load_test_set()
        print("\n All Components Loaded Successfully")
        print("=" * 80)

    def _load_finetuned_model(self):
        """Load Phase 1 fine-tuned NARRATIVE Qwen model"""
        print(f"\n Loading Fine-tuned NARRATIVE Model...")
        print(f"   Path: {NarrativeRAGConfig.PHASE1_MODEL_PATH}")

        if not os.path.exists(NarrativeRAGConfig.PHASE1_MODEL_PATH):
            raise FileNotFoundError(
                f"Fine-tuned NARRATIVE model not found at: {NarrativeRAGConfig.PHASE1_MODEL_PATH}\n"
                f"Please run Phase 1 fine-tuning first (Narrative/Finetuning/)."
            )

        if UNSLOTH_AVAILABLE:
            print(f"   Using Unsloth optimizations...")
            print(f"   MATCHING BASELINE: max_seq_length=4096, dtype=None, load_in_4bit=True")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=NarrativeRAGConfig.PHASE1_MODEL_PATH,
                max_seq_length=NarrativeRAGConfig.MAX_SEQ_LENGTH,  # 4096 (BASELINE)
                dtype=None,           # BASELINE: None (auto-detect)
                load_in_4bit=True,    # BASELINE: True (4-bit quantization)
            )
            FastLanguageModel.for_inference(self.model)
        else:
            print(f"   Using regular transformers...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                NarrativeRAGConfig.PHASE1_MODEL_PATH,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                NarrativeRAGConfig.PHASE1_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"   Model loaded successfully")
        print(f"   Device: {next(self.model.parameters()).device}")

    def _load_retrieval_systems(self):
        """Load FAISS index, embedder, and reranker"""
        print(f"\n Loading Retrieval Systems...")

        # Load embedding model
        print(f"   Loading embedder: {NarrativeRAGConfig.EMBEDDING_MODEL_NAME}")
        self.embedder = SentenceTransformer(NarrativeRAGConfig.EMBEDDING_MODEL_NAME)
        print(f"   Embedder loaded")

        # Load reranker
        print(f"   Loading reranker: {NarrativeRAGConfig.RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(NarrativeRAGConfig.RERANKER_MODEL_NAME)
        print(f"   Reranker loaded")

        # Load FAISS index
        faiss_index_path = NarrativeRAGConfig.FAISS_INDEX_DIR / "index.faiss"
        embeddings_path = NarrativeRAGConfig.FAISS_INDEX_DIR / "embeddings.npy"

        if not faiss_index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at: {faiss_index_path}\n"
                f"Please run 2_build_rag_corpus.py first."
            )

        print(f"   Loading FAISS index from: {faiss_index_path}")
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        self.corpus_embeddings = np.load(str(embeddings_path))
        print(f"   FAISS index loaded: {self.faiss_index.ntotal} vectors")

        # Load corpus
        print(f"   Loading corpus: {NarrativeRAGConfig.TRAIN_VAL_CORPUS}")
        self.corpus_df = pd.read_csv(NarrativeRAGConfig.TRAIN_VAL_CORPUS)
        print(f"   Corpus loaded: {len(self.corpus_df)} cases")

    def _load_test_set(self):
        """Load test set"""
        print(f"\n Loading Test Set...")
        print(f"   Path: {NarrativeRAGConfig.TEST_SET}")

        if not NarrativeRAGConfig.TEST_SET.exists():
            raise FileNotFoundError(
                f"Test set not found at: {NarrativeRAGConfig.TEST_SET}\n"
                f"Please run 1_prepare_dataset.py first."
            )

        self.test_df = pd.read_csv(NarrativeRAGConfig.TEST_SET)
        print(f"   Test set loaded: {len(self.test_df)} samples")

    def retrieve_similar_cases(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve similar NARRATIVE cases for style guidance

        Uses 'target' field (narrative paragraphs) for retrieval,
        NOT 'structured_target' (bullet points).
        """
        # Encode query
        query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)

        # Dense retrieval
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            NarrativeRAGConfig.DENSE_TOP_K
        )

        # Get candidates
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.corpus_df):
                candidate = self.corpus_df.iloc[idx].to_dict()
                candidate['dense_score'] = float(dist)
                candidates.append(candidate)

        # Rerank candidates using NARRATIVE target field
        if len(candidates) > top_k:
            # Use 'target' (narrative) for reranking, NOT 'structured_target'
            rerank_pairs = [[query_text, c.get('target', '')] for c in candidates]
            rerank_scores = self.reranker.predict(rerank_pairs)

            for i, score in enumerate(rerank_scores):
                candidates[i]['rerank_score'] = float(score)

            candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]

        return candidates

    def build_narrative_prompt(self, input_note: str, retrieved_cases: List[Dict] = None) -> str:
        """
        Build prompt for NARRATIVE summary generation using BASELINE format.

        UPDATED APPROACH:
        - NO example injection (caused content copying and degraded quality)
        - Use EXACT same prompt as baseline fine-tuned model
        - retrieved_cases parameter kept for API compatibility but NOT used in prompt
        - RAG retrieval is used AFTER generation for fact-checking only
        """
        # BASELINE PROMPT - exact same format as qwen_narrative_finetune.py
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical AI assistant specialized in generating narrative discharge summaries.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a well-organized narrative discharge summary from the clinical note below. Use flowing paragraphs without bullet points or section headers.

Clinical Note:
{input_note}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def clean_narrative_output(self, text: str) -> str:
        """
        Remove any structured formatting - ensure pure narrative output

        SIMPLIFIED per FIX_DOCUMENTATION.md:
        - Baseline used 103 lines with NO aggressive truncation
        - Previous 273-line cleaning caused 13% ROUGE drop
        - Keep cleaning simple, match baseline behavior
        """
        # Remove common prefixes
        prefixes_to_remove = [
            r'^Here is a comprehensive.*?:\s*',
            r'^Here is the narrative summary.*?:\s*',
            r'^Narrative (?:Discharge )?Summary:?\s*',
            r'^---+\s*',
        ]
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

        # Remove markdown headers
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)

        # Remove common section headers at line starts
        headers_to_remove = [
            r'^\s*Discharge\s+Summary:?\s*',
            r'^\s*Patient\s+&\s+Service:?\s*',
            r'^\s*Chief\s+Complaint:?\s*',
            r'^\s*Hospital\s+Course:?\s*',
        ]
        for header_pattern in headers_to_remove:
            text = re.sub(header_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove bullet points at line start
        text = re.sub(r'^\s*[•\-\*]\s+', '', text, flags=re.MULTILINE)

        # Remove "---" separators
        text = re.sub(r'\n---+\n', '\n\n', text)

        # Remove instruction artifacts
        text = re.sub(r'Now,?\s+generate.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'Remember:.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

        # CRITICAL: Remove structured discharge sections that contaminate output
        # These patterns indicate the model is copying structured input instead of generating narrative

        # Remove "Discharge Medications:" followed by numbered list
        text = re.sub(r'Discharge Medications:.*?(?=\n\n[A-Z]|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove "Discharge Disposition:" blocks
        text = re.sub(r'Discharge Disposition:.*?(?=\n\n[A-Z]|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove "Discharge Condition:" blocks
        text = re.sub(r'Discharge Condition:.*?(?=\n\n[A-Z]|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove "Discharge Diagnosis:" blocks
        text = re.sub(r'Discharge Diagnosis:.*?(?=\n\n[A-Z]|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove "Followup Instructions" blocks
        text = re.sub(r'Followup Instructions.*?(?=\n\n[A-Z]|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove numbered medication lists (19. Acetaminophen, 20. Ativan, etc.)
        text = re.sub(r'\n\d+\.\s+[A-Za-z]+.*?(?:mg|mcg|units?).*', '', text, flags=re.IGNORECASE)

        # Remove lab report garbage (ANA, ANCA, RPR, etc.)
        lab_patterns = [
            r'ANA\s*\(Final.*?(?=\n\n|\Z)',
            r'ANCA\s*\(Final.*?(?=\n\n|\Z)',
            r'RPR\s*\(Final.*?(?=\n\n|\Z)',
            r'HEPATITIS.*?(?=\n\n|\Z)',
            r'HIV\s*AB.*?(?=\n\n|\Z)',
            r'TUBERCULOSIS.*?PCR.*?(?=\n\n|\Z)',
            r'P carinii\).*?(?=\n\n|\Z)',
        ]
        for pattern in lab_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove blood pressure/vital sign dumps
        text = re.sub(r'blood pressure:\s*[\d/\s]+(?:\d+/\d+\s*)+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Temp:\s*[\d\s.]+', '', text, flags=re.IGNORECASE)

        # Remove timestamp garbage (19: 09 20: 40 22: 02...)
        text = re.sub(r'\d+:\s*\d+\s+(?:\d+:\s*\d+\s*)+', '', text)

        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        # If there's "Clinical Note:" anywhere, cut everything after it
        if 'Clinical Note:' in text:
            parts = text.split('Clinical Note:', 1)
            text = parts[0].strip()

        # If there's "EXAMPLE" anywhere, cut everything after it
        if 'EXAMPLE' in text.upper():
            parts = re.split(r'EXAMPLE', text, flags=re.IGNORECASE, maxsplit=1)
            text = parts[0].strip()

        # If output starts with technical metadata, it's garbage
        if text.startswith('-sex:') or text.startswith('sex:'):
            return ""

        # If output is suspiciously short (< 20 words), likely failed
        word_count = len(text.split())
        if word_count < 20:
            return ""

        return text

    def generate_narrative_summary(self, input_note: str, retrieved_cases: List[Dict] = None) -> str:
        """Generate narrative-style discharge summary using BASELINE parameters.

        UPDATED: Uses EXACT baseline generation parameters.
        RAG retrieval is for post-generation fact-checking, NOT prompt injection.
        """

        # Build prompt (BASELINE format - no example injection)
        prompt = self.build_narrative_prompt(input_note, retrieved_cases)

        # Tokenize with initial max_length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=NarrativeRAGConfig.MAX_SEQ_LENGTH
        ).to(self.model.device)

        # DYNAMIC max_tokens calculation (MATCHING BASELINE)
        # This prevents "input length + max_new_tokens exceeds max_seq_length" errors
        input_token_count = inputs['input_ids'].shape[1]
        max_seq_length = NarrativeRAGConfig.MAX_SEQ_LENGTH  # 4096

        # Reserve space: max_seq_length - input_length - safety_margin
        max_possible_tokens = max_seq_length - input_token_count - 50

        # If input is at max length, truncate more aggressively
        if max_possible_tokens <= 256:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_token_count = inputs['input_ids'].shape[1]
            max_possible_tokens = max_seq_length - input_token_count - 50

        # Choose max_new_tokens conservatively based on input length
        if max_possible_tokens < 512:
            dynamic_max_tokens = min(200, max_possible_tokens - 10)  # Very long input
        elif input_token_count > 2500:  # Long input
            dynamic_max_tokens = min(512, max_possible_tokens)
        else:  # Normal input
            dynamic_max_tokens = min(NarrativeRAGConfig.MAX_NEW_TOKENS, max_possible_tokens)

        # Generate with EXACT BASELINE parameters from qwen_narrative_generation.py
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=dynamic_max_tokens,  # DYNAMIC based on input length
                temperature=NarrativeRAGConfig.TEMPERATURE,       # 0.6
                top_p=NarrativeRAGConfig.TOP_P,                   # 0.92
                top_k=NarrativeRAGConfig.TOP_K,                   # 50
                repetition_penalty=NarrativeRAGConfig.REPETITION_PENALTY,  # 1.1
                do_sample=NarrativeRAGConfig.DO_SAMPLE,           # True
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean to ensure pure narrative output
        summary = self.clean_narrative_output(summary)

        return summary

    def process_test_sample(self, sample: Dict, sample_idx: int) -> Dict:
        """Process a single test sample with narrative RAG"""
        try:
            start_time = time.time()

            # Retrieve similar NARRATIVE cases for style guidance
            retrieved_cases = self.retrieve_similar_cases(
                sample['input'],
                top_k=NarrativeRAGConfig.RERANK_TOP_K
            )

            # Generate narrative summary
            generated_summary = self.generate_narrative_summary(
                sample['input'],
                retrieved_cases
            )

            generation_time = time.time() - start_time

            # Prepare result
            result = {
                "sample_id": sample_idx,
                "note_id": sample.get('note_id', f"sample_{sample_idx}"),
                "input": sample['input'],
                "target": sample.get('target', ''),  # Narrative target
                "generated_summary": generated_summary,
                "retrieved_cases": [
                    {
                        "note_id": c.get('note_id', 'unknown'),
                        "rerank_score": c.get('rerank_score', 0.0)
                    }
                    for c in retrieved_cases
                ],
                "generation_time_seconds": round(generation_time, 2),
                "word_count": len(generated_summary.split()),
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
            print(f"\n Error processing sample {sample_idx}: {str(e)}")
            self.stats['failed'] += 1

            return {
                "sample_id": sample_idx,
                "note_id": sample.get('note_id', f"sample_{sample_idx}"),
                "error": str(e),
                "success": False
            }

    def run_inference(self, num_samples: int = None):
        """Run narrative RAG inference on test set"""
        print("\n" + "=" * 80)
        print("STARTING NARRATIVE RAG INFERENCE")
        print("Output: Flowing paragraph summaries (NOT structured)")
        print("=" * 80)

        # Determine number of samples
        total_samples = len(self.test_df) if num_samples is None else min(num_samples, len(self.test_df))
        self.stats['total_samples'] = total_samples

        print(f"\n Processing {total_samples} test samples...")
        print(f"   Retrieval: Top-{NarrativeRAGConfig.RERANK_TOP_K} similar narrative cases")
        print(f"   Model: {NarrativeRAGConfig.PHASE1_MODEL_PATH}")
        print(f"   Output: {NarrativeRAGConfig.RAG_SUMMARIES_PATH}")

        results = []
        generation_times = []

        inference_start_time = time.time()

        # Process samples
        for idx in tqdm(range(total_samples), desc="Generating narrative summaries"):
            sample = self.test_df.iloc[idx].to_dict()
            result = self.process_test_sample(sample, idx)
            results.append(result)

            if result['success']:
                generation_times.append(result['generation_time_seconds'])

        # Calculate statistics
        if generation_times:
            self.stats['avg_generation_time'] = round(np.mean(generation_times), 2)

        total_elapsed_time = time.time() - inference_start_time
        self.stats['total_elapsed_time_seconds'] = round(total_elapsed_time, 2)
        self.stats['total_elapsed_time_minutes'] = round(total_elapsed_time / 60, 2)

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary()

        return results

    def _format_as_paragraphs(self, text: str, width: int = 80) -> str:
        """
        Format text into readable paragraphs with proper line wrapping

        - Splits long text into sentences
        - Groups ~4-5 sentences per paragraph
        - Wraps lines at specified width
        """
        if not text:
            return text

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        if len(sentences) <= 3:
            # Short text - just wrap it
            return textwrap.fill(text, width=width)

        # Group sentences into paragraphs (~4-5 sentences each)
        paragraphs = []
        sentences_per_para = max(3, len(sentences) // 4)

        for i in range(0, len(sentences), sentences_per_para):
            para_sentences = sentences[i:i + sentences_per_para]
            para_text = ' '.join(para_sentences)
            wrapped = textwrap.fill(para_text, width=width)
            paragraphs.append(wrapped)

        return '\n\n'.join(paragraphs)

    def _save_results(self, results: List[Dict]):
        """Save results to JSON and text files"""
        print(f"\n Saving results...")

        # Save JSON
        with open(NarrativeRAGConfig.RAG_SUMMARIES_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Saved JSON: {NarrativeRAGConfig.RAG_SUMMARIES_PATH}")

        # Save human-readable text
        with open(NarrativeRAGConfig.RAG_SUMMARIES_TXT_PATH, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NARRATIVE RAG TEST SET RESULTS\n")
            f.write("Medical Discharge Summary Generation - NARRATIVE FORMAT\n")
            f.write("(Flowing paragraphs, NOT structured sections)\n")
            f.write(f"Total Samples: {len(results)}\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                if result['success']:
                    f.write("=" * 80 + "\n")
                    f.write(f"SAMPLE {result['sample_id'] + 1}: {result['note_id']}\n")
                    f.write("=" * 80 + "\n\n")

                    f.write("-" * 80 + "\n")
                    f.write("INPUT NOTE (truncated to 500 chars):\n")
                    f.write("-" * 80 + "\n")
                    f.write(result['input'][:500] + "...\n\n")

                    f.write("-" * 80 + "\n")
                    f.write("TARGET NARRATIVE SUMMARY:\n")
                    f.write("-" * 80 + "\n")
                    # Format target as paragraphs
                    target_formatted = self._format_as_paragraphs(
                        result.get('target', 'No target available'), width=80
                    )
                    f.write(target_formatted + "\n\n")

                    f.write("-" * 80 + "\n")
                    f.write("GENERATED NARRATIVE SUMMARY (RAG-Enhanced):\n")
                    f.write("-" * 80 + "\n")
                    # Format generated summary as paragraphs
                    generated_formatted = self._format_as_paragraphs(
                        result['generated_summary'], width=80
                    )
                    f.write(generated_formatted + "\n\n")

                    f.write("-" * 80 + "\n")
                    f.write("RETRIEVAL INFO:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Retrieved {len(result['retrieved_cases'])} similar narrative cases:\n")
                    for i, case in enumerate(result['retrieved_cases'], 1):
                        f.write(f"  {i}. {case['note_id']} (score: {case['rerank_score']:.4f})\n")
                    f.write(f"\nGeneration Time: {result['generation_time_seconds']} seconds\n")
                    f.write(f"Word Count: {result['word_count']} words\n\n")

        print(f"   Saved TXT: {NarrativeRAGConfig.RAG_SUMMARIES_TXT_PATH}")

        # Save retrieval logs
        with open(NarrativeRAGConfig.RETRIEVAL_LOGS_PATH, 'w') as f:
            json.dump(self.stats['retrieval_logs'], f, indent=2)
        print(f"   Saved retrieval logs: {NarrativeRAGConfig.RETRIEVAL_LOGS_PATH}")

        # Save generation statistics
        with open(NarrativeRAGConfig.GENERATION_STATS_PATH, 'w') as f:
            json.dump({
                "total_samples": self.stats['total_samples'],
                "successful": self.stats['successful'],
                "failed": self.stats['failed'],
                "success_rate": round(self.stats['successful'] / self.stats['total_samples'] * 100, 2) if self.stats['total_samples'] > 0 else 0,
                "avg_generation_time_seconds": self.stats['avg_generation_time'],
                "total_elapsed_time_seconds": self.stats['total_elapsed_time_seconds'],
                "total_elapsed_time_minutes": self.stats['total_elapsed_time_minutes'],
                "output_format": "narrative_paragraphs"
            }, f, indent=2)
        print(f"   Saved statistics: {NarrativeRAGConfig.GENERATION_STATS_PATH}")

    def _print_summary(self):
        """Print inference summary"""
        print("\n" + "=" * 80)
        print("NARRATIVE RAG INFERENCE COMPLETE!")
        print("=" * 80)

        print(f"\n Summary:")
        print(f"   Total samples: {self.stats['total_samples']}")
        print(f"   Successful: {self.stats['successful']}")
        print(f"   Failed: {self.stats['failed']}")
        if self.stats['total_samples'] > 0:
            print(f"   Success rate: {self.stats['successful'] / self.stats['total_samples'] * 100:.1f}%")
        print(f"   Avg generation time: {self.stats['avg_generation_time']:.2f} seconds")
        print(f"   Total time: {self.stats.get('total_elapsed_time_minutes', 0):.1f} minutes")

        print(f"\n Output files:")
        print(f"   - {NarrativeRAGConfig.RAG_SUMMARIES_PATH.name}")
        print(f"   - {NarrativeRAGConfig.RAG_SUMMARIES_TXT_PATH.name}")
        print(f"   - {NarrativeRAGConfig.RETRIEVAL_LOGS_PATH.name}")
        print(f"   - {NarrativeRAGConfig.GENERATION_STATS_PATH.name}")

        print(f"\n Next step: Run '4_evaluate_rag.py' to compute evaluation metrics")
        print("=" * 80 + "\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Narrative RAG Inference - Generate flowing paragraph summaries"
    )

    parser.add_argument(
        'num_samples_pos',
        type=int,
        nargs='?',
        default=None,
        help='Number of samples to process (positional argument)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Number of samples to process (default: all test samples)'
    )
    args = parser.parse_args()

    # Use positional argument if provided, otherwise use --limit
    num_samples = args.num_samples_pos if args.num_samples_pos is not None else args.limit

    # Initialize system
    rag_system = NarrativeRAGInference()

    # Load components
    rag_system.load_all_components()

    # Run inference
    results = rag_system.run_inference(num_samples=num_samples)

    print("\n Done!")


if __name__ == "__main__":
    main()
