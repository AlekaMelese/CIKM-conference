#!/usr/bin/env python3
"""
Phase 2 - Step 2: Build RAG Corpus with Medical-Domain Embeddings
Creates FAISS and BM25 indices based on structured_target similarity
Uses pritamdeka/S-PubMedBert-MS-MARCO for medical-domain semantic matching
"""

import os
import sys

# CRITICAL: Set cache location BEFORE importing anything from transformers/sentence_transformers
# This avoids stale file handle errors from old cache
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE = os.path.join(SCRIPT_DIR, "models", "huggingface_cache")
os.makedirs(LOCAL_CACHE, exist_ok=True)

os.environ['HF_HOME'] = LOCAL_CACHE
os.environ['TRANSFORMERS_CACHE'] = LOCAL_CACHE
os.environ['SENTENCE_TRANSFORMERS_HOME'] = LOCAL_CACHE

# Now safe to import
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

sys.path.append(SCRIPT_DIR)
from config import RAGConfig

class StructuralRAGCorpusBuilder:
    """Builds RAG corpus with structure-focused embeddings"""

    def __init__(self):
        self.config = RAGConfig
        self.embedding_model = None
        self.corpus_df = None
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None

    def load_corpus(self):
        """Load train+val corpus for RAG retrieval"""
        print("=" * 80)
        print("STEP 2: BUILD RAG CORPUS (Structure-Focused)")
        print("=" * 80)
        print(f"\nLoading RAG corpus from: {self.config.TRAIN_VAL_CORPUS}")

        self.corpus_df = pd.read_csv(self.config.TRAIN_VAL_CORPUS)

        print(f"✓ Loaded {len(self.corpus_df)} records for RAG corpus")
        print(f"  Retrieval field: {self.config.RETRIEVAL_FIELD}")

        # Verify retrieval field exists
        if self.config.RETRIEVAL_FIELD not in self.corpus_df.columns:
            raise ValueError(f"Retrieval field '{self.config.RETRIEVAL_FIELD}' not found in corpus")

        return self.corpus_df

    def load_embedding_model(self):
        """Load medical sentence transformer model (PubMedBert MS-MARCO)"""
        print(f"\n📥 Loading embedding model: {self.config.EMBEDDING_MODEL_NAME}")
        device = "cuda" if torch.cuda.is_available() and self.config.USE_GPU else "cpu"
        print(f"  Device: {device}")
        print(f"  Cache: {LOCAL_CACHE}")

        # Force local_files_only=False to bypass stale cache and download fresh
        try:
            self.embedding_model = SentenceTransformer(
                self.config.EMBEDDING_MODEL_NAME,
                device=device,
                cache_folder=LOCAL_CACHE,
                local_files_only=False  # Force fresh download if needed
            )
        except Exception as e:
            print(f"  Error with local_files_only=False: {e}")
            print(f"  Attempting direct download...")
            # Try with model_kwargs to force download
            self.embedding_model = SentenceTransformer(
                self.config.EMBEDDING_MODEL_NAME,
                device=device,
                cache_folder=LOCAL_CACHE,
                model_kwargs={'local_files_only': False, 'force_download': True}
            )

        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded")
        print(f"  Dimension: {embedding_dim}")

        # Verify dimension matches config
        if embedding_dim != self.config.EMBEDDING_DIM:
            print(f"  ⚠️  Warning: Model dimension ({embedding_dim}) != config ({self.config.EMBEDDING_DIM})")
            self.config.EMBEDDING_DIM = embedding_dim

    def create_embeddings(self):
        """
        Create structure-focused embeddings
        Uses structured_target field for matching similar structural patterns
        """
        print("\n" + "─" * 80)
        print("CREATING STRUCTURE-FOCUSED DENSE EMBEDDINGS")
        print("─" * 80)

        # Prepare documents from structured_target (structure-focused)
        documents = []
        for idx, row in self.corpus_df.iterrows():
            # Use structured_target for structure-focused matching
            struct_target = row.get(self.config.RETRIEVAL_FIELD, '')

            # Also include input for richer context
            input_text = row.get('input', '')

            # Combine: emphasize structure but include input context
            doc = f"STRUCTURE:\n{struct_target}\n\nCONTEXT:\n{input_text}"
            documents.append(doc)

        print(f"  Documents to encode: {len(documents)}")
        print(f"  Model: {self.config.EMBEDDING_MODEL_NAME}")
        print(f"  Retrieval focus: Structure similarity via '{self.config.RETRIEVAL_FIELD}'")
        print(f"\n  Encoding...")

        # Create embeddings with progress bar
        self.embeddings = self.embedding_model.encode(
            documents,
            batch_size=16,  # Smaller batch for medical model (PubMedBert)
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print(f"\n✓ Created embeddings: {self.embeddings.shape}")
        print(f"  Shape: ({self.embeddings.shape[0]} documents, {self.embeddings.shape[1]} dimensions)")

    def build_faiss_index(self):
        """Build FAISS index for dense retrieval"""
        print("\n" + "─" * 80)
        print("BUILDING FAISS INDEX")
        print("─" * 80)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Create flat index for inner product (cosine similarity)
        embedding_dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)

        # Add embeddings to index
        self.faiss_index.add(self.embeddings)

        print(f"✓ FAISS index built")
        print(f"  Total vectors: {self.faiss_index.ntotal}")
        print(f"  Dimension: {embedding_dim}")
        print(f"  Index type: Flat (Inner Product / Cosine Similarity)")

    def build_bm25_index(self):
        """Build BM25 index for sparse retrieval"""
        print("\n" + "─" * 80)
        print("BUILDING BM25 INDEX")
        print("─" * 80)

        # Prepare tokenized documents from structured_target
        tokenized_corpus = []
        for _, row in self.corpus_df.iterrows():
            # Use structured_target for BM25 as well (structure-focused)
            struct_target = row.get(self.config.RETRIEVAL_FIELD, '')
            # Simple whitespace tokenization
            tokens = struct_target.lower().split()
            tokenized_corpus.append(tokens)

        print(f"  Documents: {len(tokenized_corpus)}")
        print(f"  Building BM25 index...")

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"✓ BM25 index built")
        print(f"  Total documents: {len(tokenized_corpus)}")
        print(f"  Average doc length: {np.mean([len(doc) for doc in tokenized_corpus]):.1f} tokens")

    def save_indices(self):
        """Save FAISS and BM25 indices to disk"""
        print("\n" + "─" * 80)
        print("SAVING INDICES")
        print("─" * 80)

        # Save FAISS index
        faiss_index_path = self.config.FAISS_INDEX_DIR / "index.faiss"
        faiss.write_index(self.faiss_index, str(faiss_index_path))
        print(f"✓ FAISS index: {faiss_index_path.name}")

        # Save embeddings
        embeddings_path = self.config.FAISS_INDEX_DIR / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        print(f"✓ Embeddings: {embeddings_path.name}")

        # Save corpus metadata
        corpus_metadata_path = self.config.FAISS_INDEX_DIR / "corpus_metadata.json"
        metadata = {
            "corpus_size": len(self.corpus_df),
            "embedding_model": self.config.EMBEDDING_MODEL_NAME,
            "embedding_dim": self.config.EMBEDDING_DIM,
            "retrieval_field": self.config.RETRIEVAL_FIELD,
            "index_type": "structure-focused",
            "corpus_path": str(self.config.TRAIN_VAL_CORPUS)
        }
        with open(corpus_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata: {corpus_metadata_path.name}")

        # Save BM25 index
        with open(self.config.BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(self.bm25, f)
        print(f"✓ BM25 index: {self.config.BM25_INDEX_PATH.name}")

        print(f"\n  All indices saved to: {self.config.RAG_DATA_DIR}")

    def test_retrieval(self):
        """Test retrieval system with a sample query"""
        print("\n" + "=" * 80)
        print("TESTING RETRIEVAL SYSTEM")
        print("=" * 80)

        # Get a sample from test set (if available) or use first corpus entry
        try:
            test_df = pd.read_csv(self.config.TEST_SET)
            sample_row = test_df.iloc[0]
            print("Using sample from test set...")
        except:
            sample_row = self.corpus_df.iloc[0]
            print("Using sample from corpus (test set not available)...")

        sample_input = sample_row['input']
        sample_target = sample_row.get(self.config.RETRIEVAL_FIELD, '')

        print(f"\n📝 Sample query (first 200 chars):")
        print(f"   {sample_input[:200]}...")

        # Create query embedding (using same structure-focused approach)
        query_doc = f"STRUCTURE:\n{sample_target}\n\nCONTEXT:\n{sample_input}"
        query_embedding = self.embedding_model.encode(
            [query_doc],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Dense retrieval (FAISS)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, k=5)

        print(f"\n🔍 Dense retrieval (FAISS) - Top 5:")
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            note_id = self.corpus_df.iloc[idx]['note_id']
            print(f"   {rank}. note_id={note_id}, similarity={score:.4f}")

        # Sparse retrieval (BM25)
        query_tokens = sample_target.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]

        print(f"\n📊 Sparse retrieval (BM25) - Top 5:")
        for rank, idx in enumerate(top_bm25_indices, 1):
            note_id = self.corpus_df.iloc[idx]['note_id']
            score = bm25_scores[idx]
            print(f"   {rank}. note_id={note_id}, score={score:.2f}")

        print("\n✅ Retrieval test complete!")

def main():
    """Main execution"""
    # Create directories
    RAGConfig.create_directories()

    # Validate corpus exists
    if not RAGConfig.TRAIN_VAL_CORPUS.exists():
        print(f"\n❌ Error: RAG corpus not found at {RAGConfig.TRAIN_VAL_CORPUS}")
        print("   Please run '1_prepare_dataset.py' first")
        return

    # Initialize builder
    builder = StructuralRAGCorpusBuilder()

    # Load corpus
    builder.load_corpus()

    # Load embedding model
    builder.load_embedding_model()

    # Create embeddings
    builder.create_embeddings()

    # Build FAISS index
    builder.build_faiss_index()

    # Build BM25 index
    builder.build_bm25_index()

    # Save indices
    builder.save_indices()

    # Test retrieval
    builder.test_retrieval()

    print("\n" + "=" * 80)
    print("✅ RAG CORPUS BUILDING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Created indices:")
    print(f"  • FAISS index (dense retrieval)")
    print(f"  • BM25 index (sparse retrieval)")
    print(f"  • Embeddings and metadata")

    print(f"\n📊 Summary:")
    print(f"  • Corpus size: {len(builder.corpus_df)} samples")
    print(f"  • Embedding model: {RAGConfig.EMBEDDING_MODEL_NAME}")
    print(f"  • Retrieval focus: Structure similarity")
    print(f"  • Ready for inference")

    print(f"\n➡️  Next step: Run '3_rag_inference.py' to generate summaries")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
