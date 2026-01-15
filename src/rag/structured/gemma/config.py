"""
Phase 2 RAG Configuration - Hybrid PEFT + RAG Protocol
Based on "Structured Clinical Note Summarization.txt" documentation

This configuration implements the Hybrid Fine-Tuning + RAG approach:
1. PEFT (Phase 1): Fine-tuned Gemma-2-9B model provides clinical domain adaptation
2. RAG: Dense retrieval provides structural format guidance through few-shot examples
3. Anti-Hallucination: Prompts enforce factual extraction from input note only

Dataset: 5000_structured.csv with custom splits (4500 train, 300 val, 200 test)
"""

import os
from pathlib import Path


class RAGConfig:
    """Configuration for Hybrid PEFT + RAG Pipeline"""

    # ========== Paths ==========
    # Base directories
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT = BASE_DIR.parent.parent.parent.parent / "Llama"  # ./Llama
    DATA_DIR = PROJECT_ROOT / "Data"
    FINETUNING_DIR = BASE_DIR.parent / "Finetuning"

    # Phase 1 fine-tuned model (merged model - no download needed)
    PHASE1_MODEL_DIR = FINETUNING_DIR / "outputs" / "merged_model"

    # RAG-specific directories
    RAG_DATA_DIR = BASE_DIR / "data"
    RAG_MODELS_DIR = BASE_DIR / "models"
    RAG_OUTPUTS_DIR = BASE_DIR / "outputs2"  # UPDATED: Save to outputs2 to preserve original results
    RAG_LOGS_DIR = BASE_DIR / "logs"

    # Dataset paths
    ORIGINAL_DATASET = DATA_DIR / "5000_structured.csv"
    TRAIN_VAL_CORPUS = RAG_DATA_DIR / "train_val_corpus.csv"
    TEST_SET = RAG_DATA_DIR / "test_set.csv"
    TRAIN_SET = RAG_DATA_DIR / "train_set.csv"
    VAL_SET = RAG_DATA_DIR / "val_set.csv"

    # Index paths
    FAISS_INDEX_DIR = RAG_DATA_DIR / "faiss_index"
    BM25_INDEX_PATH = RAG_DATA_DIR / "bm25_index.pkl"

    # Output paths
    RAG_SUMMARIES_PATH = RAG_OUTPUTS_DIR / "rag_summaries_hybrid.json"
    RAG_SUMMARIES_TXT_PATH = RAG_OUTPUTS_DIR / "rag_summaries_hybrid.txt"
    RETRIEVAL_LOGS_PATH = RAG_OUTPUTS_DIR / "retrieval_logs.json"
    EVALUATION_RESULTS_PATH = RAG_OUTPUTS_DIR / "evaluation_results.json"
    GENERATION_STATS_PATH = RAG_OUTPUTS_DIR / "generation_stats.json"

    # ========== Dataset Split Configuration ==========
    TEST_SIZE = 200
    VAL_SIZE = 300
    TRAIN_SIZE = 4500
    RANDOM_SEED = 42

    # ========== Model Configuration ==========
    # Dense embedding model: Medical-domain PubMedBERT embeddings
    # Using pritamdeka/S-PubMedBert-MS-MARCO (specialized for medical retrieval)
    EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
    EMBEDDING_DIM = 768  # PubMedBERT uses 768-dim embeddings

    # Reranker model: Medical-domain MedCPT reranker (medical cross-encoder)
    # Using ncbi/MedCPT-Cross-Encoder (same as successful Mistral-7B-Instruct/RAG/ medical experiments)
    RERANKER_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"

    # Phase 1 fine-tuned Gemma model path
    PHASE1_MODEL_NAME = "unsloth/gemma-2-9b-it-bnb-4bit"
    PHASE1_MODEL_PATH = str(PHASE1_MODEL_DIR)

    # Sparse retrieval (BM25) - Not used in this dense-only configuration
    SPARSE_TOP_K = 0  # Dense-only RAG (no BM25)

    # ========== Retrieval Configuration ==========
    # Retrieval field: Use 'structured_target' for structure-focused retrieval
    # or 'input' for content-focused retrieval
    RETRIEVAL_FIELD = "structured_target"  # Structure-focused as per documentation

    # Dense retrieval (FAISS) - Initial candidates
    DENSE_TOP_K = 20

    # Reranking - Final few-shot examples (reduced to prevent copying)
    RERANK_TOP_K = 3  # Show 3 examples as format templates

    # ========== Generation Configuration ==========
    # Model generation parameters
    MAX_SEQ_LENGTH = 6144  # Maximum sequence length for model
    MAX_NEW_TOKENS = 3072  # Maximum new tokens to generate
    TEMPERATURE = 0.7  # Sampling temperature (0.7 for balanced creativity)
    TOP_P = 0.95  # Nucleus sampling
    TOP_K = 100  # Top-k sampling
    REPETITION_PENALTY = 1.15  # Penalty for repetition
    DO_SAMPLE = True  # Enable sampling

    # ========== Hardware Configuration ==========
    USE_GPU = True
    DEVICE = "cuda:0"
    DTYPE = "float16"

    # ========== Logging Configuration ==========
    LOG_LEVEL = "INFO"
    SAVE_RETRIEVAL_LOGS = True
    VERBOSE = True

    # ========== RAG Prompt Template ==========
    # Based on documentation: Few-shot examples for FORMAT GUIDANCE ONLY
    # Anti-hallucination measures: Extract from input note only

    @classmethod
    def get_rag_prompt_template(cls, input_note: str, few_shot_examples: str, num_examples: int) -> str:
        """
        Build RAG prompt matching EXACT Phase 1 training format

        This format was used during fine-tuning and produces proper 11-section structure.
        Few-shot examples from RAG retrieval are intentionally NOT shown to prevent copying.
        """
        prompt = f"""<start_of_turn>user
You are a medical AI assistant. Generate a structured discharge summary from the clinical note below.

Clinical Note:
{input_note}

Generate a structured summary with these sections:
- Case Type
- Patient & Service
- Chief Complaint / Admission Context
- History of Present Illness (HPI)
- Past Medical / Surgical History
- Medications (Discharge / Ongoing)
- Physical Examination (summarized)
- Investigations / Labs / Imaging (if any)
- Assessment / Impression
- Discharge Condition
- Follow-Up & Recommendations
<end_of_turn>
<start_of_turn>model
"""

        return prompt

    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.RAG_DATA_DIR,
            cls.RAG_MODELS_DIR,
            cls.RAG_OUTPUTS_DIR,
            cls.RAG_LOGS_DIR,
            cls.FAISS_INDEX_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print("✓ Created all necessary directories")

    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist"""
        required_files = [
            (cls.ORIGINAL_DATASET, "Original dataset (5000_structured.csv)"),
            (cls.PHASE1_MODEL_DIR, "Phase 1 fine-tuned model directory"),
        ]

        missing = []
        for path, name in required_files:
            if not path.exists():
                missing.append(f"  ✗ {name}: {path}")

        if missing:
            raise FileNotFoundError(
                f"\nMissing required files:\n" + "\n".join(missing)
            )

        print("✓ All required paths validated")

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "=" * 80)
        print("HYBRID PEFT + RAG CONFIGURATION")
        print("Based on 'Structured Clinical Note Summarization.txt'")
        print("=" * 80)

        print(f"\n📁 DATASET")
        print(f"  Source: {cls.ORIGINAL_DATASET.name}")
        print(f"  Total: 5000 samples")
        print(f"  Split: {cls.TRAIN_SIZE} train + {cls.VAL_SIZE} val + {cls.TEST_SIZE} test")

        print(f"\n🤖 MODELS")
        print(f"  Generator (PEFT): Fine-tuned Gemma-2-9B (Phase 1)")
        print(f"  Location: {cls.PHASE1_MODEL_DIR}")
        print(f"  Embeddings (Dense): {cls.EMBEDDING_MODEL_NAME}")
        print(f"  Reranker: {cls.RERANKER_MODEL_NAME}")

        print(f"\n🔍 RETRIEVAL (Structure-Focused)")
        print(f"  Match field: {cls.RETRIEVAL_FIELD}")
        print(f"  Dense Top-K: {cls.DENSE_TOP_K}")
        print(f"  Rerank Top-K: {cls.RERANK_TOP_K} (few-shot examples)")

        print(f"\n⚠️  ETHICAL SAFEGUARDS (Anti-Hallucination)")
        print(f"  ✓ Few-shot examples for FORMAT guidance only")
        print(f"  ✓ Explicit anti-copying prompts")
        print(f"  ✓ Factual extraction from input note only")
        print(f"  ✓ Test set isolation from retrieval corpus")

        print(f"\n🎯 HYBRID APPROACH")
        print(f"  1. PEFT: Clinical domain adaptation (Phase 1 fine-tuning)")
        print(f"  2. RAG: Format guidance through few-shot retrieval")
        print(f"  3. Generation: Combine fine-tuned style + structural templates")

        print(f"\n📊 OUTPUT")
        print(f"  Format: 11-section structured summaries")
        print(f"  Directory: {cls.RAG_OUTPUTS_DIR}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test configuration
    print("Testing RAG Configuration...")

    RAGConfig.create_directories()
    RAGConfig.print_config()

    try:
        RAGConfig.validate_paths()
        print("✅ Configuration test passed!\n")
    except FileNotFoundError as e:
        print(f"❌ Configuration test failed:\n{e}\n")
