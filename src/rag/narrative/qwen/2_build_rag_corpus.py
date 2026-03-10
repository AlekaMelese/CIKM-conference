import os
import sys

# CRITICAL: Set cache location BEFORE importing anything from transformers/sentence_transformers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE = os.path.join(SCRIPT_DIR, "models", "huggingface_cache")
os.makedirs(LOCAL_CACHE, exist_ok=True)

os.environ['HF_HOME'] = LOCAL_CACHE
os.environ['TRANSFORMERS_CACHE'] = LOCAL_CACHE
os.environ['SENTENCE_TRANSFORMERS_HOME'] = LOCAL_CACHE

# Now safe to import
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import faiss

sys.path.append(SCRIPT_DIR)
from config import NarrativeRAGConfig

class NarrativeRAGCorpusBuilder:
    """Builds RAG corpus with narrative-focused embeddings (dense-only, no BM25)"""

    def __init__(self):
        self.config = NarrativeRAGConfig
        self.embedding_model = None
        self.corpus_df = None
        self.embeddings = None
        self.faiss_index = None

    def load_corpus(self):
        """Load train+val corpus for RAG retrieval"""
        print("=" * 80)
        print("STEP 2: BUILD RAG CORPUS - NARRATIVE FORMAT (Dense-only)")
        print("=" * 80)
        print(f"\nLoading RAG corpus from: {self.config.TRAIN_VAL_CORPUS}")

        self.corpus_df = pd.read_csv(self.config.TRAIN_VAL_CORPUS)

        print(f"✓ Loaded {len(self.corpus_df)} records for RAG corpus")
        print(f"  Retrieval field: {self.config.RETRIEVAL_FIELD} (narrative paragraphs)")

        # Verify retrieval field exists
        if self.config.RETRIEVAL_FIELD not in self.corpus_df.columns:
            raise ValueError(f"Retrieval field '{self.config.RETRIEVAL_FIELD}' not found in corpus")

        # Show narrative statistics
        if 'target' in self.corpus_df.columns:
            avg_words = self.corpus_df['target'].str.split().str.len().mean()
            print(f"  Average narrative length: {avg_words:.0f} words")

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
        Create narrative-focused embeddings
        Uses 'target' field (narrative paragraphs) for matching similar narrative styles
        """
        print("\n" + "─" * 80)
        print("CREATING NARRATIVE-FOCUSED DENSE EMBEDDINGS")
        print("─" * 80)

        # Prepare documents from narrative target field
        documents = []
        for idx, row in self.corpus_df.iterrows():
            # Use narrative target for narrative-focused matching
            narrative_target = row.get(self.config.RETRIEVAL_FIELD, '')

            # Also include input for richer context
            input_text = row.get('input', '')

            # Combine: emphasize narrative style but include input context
            doc = f"NARRATIVE SUMMARY:\n{narrative_target}\n\nCLINICAL NOTE:\n{input_text}"
            documents.append(doc)

        print(f"  Documents to encode: {len(documents)}")
        print(f"  Model: {self.config.EMBEDDING_MODEL_NAME}")
        print(f"  Retrieval focus: Narrative similarity via '{self.config.RETRIEVAL_FIELD}'")
        print(f"\n  Encoding...")

        # Create embeddings with progress bar
        # GPU-optimized batch size (increase if GPU has more memory)
        batch_size = 64 if torch.cuda.is_available() else 16
        print(f"  Batch size: {batch_size} ({'GPU' if torch.cuda.is_available() else 'CPU'})")

        self.embeddings = self.embedding_model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print(f"\n✓ Created embeddings: {self.embeddings.shape}")
        print(f"  Shape: ({self.embeddings.shape[0]} documents, {self.embeddings.shape[1]} dimensions)")

    def build_faiss_index(self):
        """Build FAISS index for dense retrieval"""
        print("\n" + "─" * 80)
        print("BUILDING FAISS INDEX (Dense-only)")
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

    def save_indices(self):
        """Save FAISS index to disk (no BM25)"""
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
            "index_type": "narrative-focused dense-only",
            "corpus_path": str(self.config.TRAIN_VAL_CORPUS),
            "format": "NARRATIVE (flowing paragraphs)",
            "bm25_enabled": False
        }
        with open(corpus_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata: {corpus_metadata_path.name}")

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

        # Create query embedding (using same narrative-focused approach)
        query_doc = f"NARRATIVE SUMMARY:\n{sample_target}\n\nCLINICAL NOTE:\n{sample_input}"
        query_embedding = self.embedding_model.encode(
            [query_doc],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Dense retrieval (FAISS)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, k=5)

        print(f"\n🔍 Dense retrieval (FAISS) - Top 5 narrative matches:")
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            note_id = self.corpus_df.iloc[idx]['note_id']
            print(f"   {rank}. note_id={note_id}, similarity={score:.4f}")

        print("\n✅ Retrieval test complete!")

def main():
    """Main execution"""
    # Create directories
    NarrativeRAGConfig.create_directories()

    # Validate corpus exists
    if not NarrativeRAGConfig.TRAIN_VAL_CORPUS.exists():
        print(f"\n❌ Error: RAG corpus not found at {NarrativeRAGConfig.TRAIN_VAL_CORPUS}")
        print("   Please run '1_prepare_dataset.py' first")
        return

    # Initialize builder
    builder = NarrativeRAGCorpusBuilder()

    # Load corpus
    builder.load_corpus()

    # Load embedding model
    builder.load_embedding_model()

    # Create embeddings
    builder.create_embeddings()

    # Build FAISS index
    builder.build_faiss_index()

    # Save indices (no BM25)
    builder.save_indices()

    # Test retrieval
    builder.test_retrieval()

    print("\n" + "=" * 80)
    print("✅ RAG CORPUS BUILDING COMPLETE - NARRATIVE FORMAT!")
    print("=" * 80)
    print(f"\n📁 Created indices:")
    print(f"  • FAISS index (dense retrieval)")
    print(f"  • Embeddings and metadata")
    print(f"  • NO BM25 (dense-only retrieval)")

    print(f"\n📊 Summary:")
    print(f"  • Corpus size: {len(builder.corpus_df)} samples")
    print(f"  • Embedding model: {NarrativeRAGConfig.EMBEDDING_MODEL_NAME}")
    print(f"  • Retrieval focus: Narrative similarity (flowing paragraphs)")
    print(f"  • Ready for inference")

    print(f"\n➡️  Next step: Run '3_rag_inference.py' to generate narrative summaries")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
