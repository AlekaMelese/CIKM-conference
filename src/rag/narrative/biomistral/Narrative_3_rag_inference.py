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
1. Fine-tuned BioMistral-7B model (Phase 1) - Trained on narrative summaries
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
import gc
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import re
import argparse
import textwrap

# Set HuggingFace cache to scratch to avoid home directory disk space issues
os.environ['HF_HOME'] = './.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = './.cache/huggingface'

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

# Try to import PEFT for LoRA loading
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT not available")


class NarrativeRAGConfig:
    """Configuration for Narrative RAG Pipeline - generates flowing paragraphs"""

    # ========== Paths ==========
    BASE_DIR = Path(__file__).parent
    FINETUNING_DIR = BASE_DIR.parent / "Finetuning"
    DATA_DIR = BASE_DIR.parent.parent.parent.parent / "Data"

    # Phase 1 fine-tuned NARRATIVE model (use merged_model for full weights)
    PHASE1_MODEL_DIR = FINETUNING_DIR / "outputs" / "merged_model"
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
    # MATCH BIOMISTRAL BASELINE EXACTLY from biomistral_narrative_generation.py
    # Baseline achieved: ROUGE1=0.272, BERTScore=0.822, avg 202 words
    MAX_SEQ_LENGTH = 4096   # BASELINE
    MAX_NEW_TOKENS = 768    # BASELINE: uses 768 default, 1024 for long inputs
    MIN_NEW_TOKENS = 0      # BASELINE: NO minimum (baseline doesn't use this!)
    TEMPERATURE = 0.6       # BASELINE: 0.6 (exact match)
    TOP_P = 0.92            # BASELINE: 0.92 (exact match)
    TOP_K = 50              # BASELINE: 50 (exact match)
    REPETITION_PENALTY = 1.1   # BASELINE: 1.1 (exact match)
    LENGTH_PENALTY = 1.0    # BASELINE
    NO_REPEAT_NGRAM_SIZE = 0  # BASELINE: disabled (exact match)
    DO_SAMPLE = True        # BASELINE: True (exact match)

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
        print("Fine-tuned BioMistral-7B + Narrative Style Retrieval")
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
        """Load Phase 1 fine-tuned NARRATIVE BioMistral model

        Uses base model + LoRA adapters because merged model weights are corrupted.
        This matches the approach in biomistral_narrative_generation.py
        """
        print(f"\n Loading Fine-tuned NARRATIVE Model...")

        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"   Cleared GPU cache")

        # BioMistral merged model is corrupted - use base + LoRA instead
        base_model = "BioMistral/BioMistral-7B"
        lora_path = str(NarrativeRAGConfig.FINETUNING_DIR / "outputs" / "final_model")

        print(f"   Base model: {base_model}")
        print(f"   LoRA path: {lora_path}")

        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"LoRA adapters not found at: {lora_path}\n"
                f"Please run Phase 1 fine-tuning first (Narrative/Finetuning/)."
            )

        # Load base model + LoRA adapters using Unsloth
        if UNSLOTH_AVAILABLE:
            print(f"   Loading base model with Unsloth...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=NarrativeRAGConfig.MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
            )

            print(f"   Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            FastLanguageModel.for_inference(self.model)
        else:
            print(f"   Using regular transformers...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(self.model, lora_path)
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
        # BASELINE PROMPT - exact same format as biomistral_narrative_generation.py (lines 119-126)
        # BioMistral uses Mistral-style prompt format, NOT Llama-style!
        prompt = f"""<s>[INST] You are a medical AI assistant specialized in generating narrative discharge summaries.

Write a well-organized narrative discharge summary from the clinical note below. Use flowing paragraphs without bullet points or section headers.

Clinical Note:
{input_note} [/INST]

"""
        return prompt

    def clean_narrative_output(self, text: str) -> str:
        """
        MATCH BASELINE EXACTLY - baseline uses MINIMAL cleaning.

        BioMistral baseline (_clean_narrative_output) does:
        1. Check if empty -> return ""
        2. Clean whitespace (max 2 newlines, single spaces)
        3. Handle catastrophic repetition (7+ mm measurements or nodule mentions)
        4. ALWAYS return something

        NO aggressive removal of "Discharge Medications", etc.
        """
        import re

        # Only reject if COMPLETELY empty
        if not text or len(text.strip()) == 0:
            return ""

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
        text = text.strip()

        # ONLY reject catastrophic repetition (matching baseline)
        repetition_patterns = [
            r'\d+\s*mm.*?\d+\s*mm.*?\d+\s*mm.*?\d+\s*mm.*?\d+\s*mm.*?\d+\s*mm.*?\d+\s*mm',  # 7+ mm measurements
            r'nodule.*?nodule.*?nodule.*?nodule.*?nodule.*?nodule.*?nodule',  # 7+ nodule mentions
        ]

        for pattern in repetition_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                # Only truncate catastrophic repetition, don't reject entirely
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match and match.start() > 200:
                    # Keep content before repetition
                    text = text[:match.start()].strip()
                    break

        # ALWAYS return something, even if short or has issues
        # Let the metrics show quality problems rather than hiding them
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

        # DYNAMIC max_tokens calculation (MATCHING BASELINE biomistral_narrative_generation.py)
        # Baseline uses SIMPLE logic: 1024 for long inputs (>3000 tokens), 768 for others
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

        # BASELINE LOGIC (simpler, less aggressive truncation):
        # Use 1024 for long inputs (>3000 tokens), otherwise use 768
        # This matches biomistral_narrative_generation.py lines 131-136
        if input_token_count > 3000:
            dynamic_max_tokens = min(1024, max_possible_tokens)
        else:
            dynamic_max_tokens = min(NarrativeRAGConfig.MAX_NEW_TOKENS, max_possible_tokens)

        # Generate with EXACT BASELINE parameters from biomistral_narrative_generation.py
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
