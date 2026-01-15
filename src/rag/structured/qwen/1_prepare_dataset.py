#!/usr/bin/env python3
"""
Phase 2 - Step 1: Dataset Preparation
Splits 5000_structured.csv into custom splits: 200 test, 300 val, 4500 train
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from config import RAGConfig

def load_dataset():
    """Load the 5000_structured.csv dataset"""
    print("=" * 80)
    print("STEP 1: DATASET PREPARATION")
    print("=" * 80)
    print(f"\nLoading dataset from: {RAGConfig.ORIGINAL_DATASET}")

    df = pd.read_csv(RAGConfig.ORIGINAL_DATASET)

    print(f"✓ Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")

    # Check for required columns
    required_cols = ['note_id', 'input', 'target', 'structured_target']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Warning: Missing columns: {missing_cols}")

    return df

def custom_split(df, test_size=200, val_size=300, random_seed=42):
    """
    Custom split: 200 test, 300 validation, 4500 training
    """
    print(f"\nPerforming custom dataset split...")
    print(f"  Target split: {RAGConfig.TRAIN_SIZE} train / {RAGConfig.VAL_SIZE} val / {RAGConfig.TEST_SIZE} test")

    # First split: test vs rest
    rest_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )

    # Second split: train vs val from rest
    train_df, val_df = train_test_split(
        rest_df,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True
    )

    print(f"\n✓ Split results:")
    print(f"    Train: {len(train_df):>4} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Val:   {len(val_df):>4} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Test:  {len(test_df):>4} records ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    ─────────────────────")
    print(f"    Total: {len(train_df) + len(val_df) + len(test_df):>4} records")

    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df):
    """Save train/val/test splits and create RAG corpus"""
    print(f"\nSaving dataset splits to: {RAGConfig.RAG_DATA_DIR}")

    # Save individual splits
    train_df.to_csv(RAGConfig.TRAIN_SET, index=False)
    print(f"  ✓ Train set: {RAGConfig.TRAIN_SET.name} ({len(train_df)} records)")

    val_df.to_csv(RAGConfig.VAL_SET, index=False)
    print(f"  ✓ Val set: {RAGConfig.VAL_SET.name} ({len(val_df)} records)")

    test_df.to_csv(RAGConfig.TEST_SET, index=False)
    print(f"  ✓ Test set: {RAGConfig.TEST_SET.name} ({len(test_df)} records)")

    # Create RAG corpus (train + val, exclude test for ethical integrity)
    corpus_df = pd.concat([train_df, val_df], ignore_index=True)
    corpus_df.to_csv(RAGConfig.TRAIN_VAL_CORPUS, index=False)
    print(f"  ✓ RAG corpus: {RAGConfig.TRAIN_VAL_CORPUS.name} ({len(corpus_df)} records)")
    print(f"    → Corpus = Train + Val (Test excluded for academic integrity)")

    return corpus_df

def analyze_splits(train_df, val_df, test_df):
    """Analyze dataset statistics for each split"""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    for name, df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        print(f"\n{name} SET ({len(df)} records)")
        print(f"  {'Metric':<30} {'Mean':<15} {'Std':<15}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")

        # Character lengths
        print(f"  {'Input length (chars)':<30} {df['input'].str.len().mean():>10.0f}     {df['input'].str.len().std():>10.0f}")
        print(f"  {'Target length (chars)':<30} {df['target'].str.len().mean():>10.0f}     {df['target'].str.len().std():>10.0f}")

        # Token lengths
        if 'input_token_len' in df.columns:
            print(f"  {'Input tokens':<30} {df['input_token_len'].mean():>10.0f}     {df['input_token_len'].std():>10.0f}")
        if 'target_token_len' in df.columns:
            print(f"  {'Target tokens':<30} {df['target_token_len'].mean():>10.0f}     {df['target_token_len'].std():>10.0f}")

def save_metadata(train_df, val_df, test_df, corpus_df):
    """Save dataset metadata for reference"""
    metadata = {
        "dataset_info": {
            "source": str(RAGConfig.ORIGINAL_DATASET),
            "total_records": len(train_df) + len(val_df) + len(test_df),
            "random_seed": RAGConfig.RANDOM_SEED,
            "split_type": "custom",
            "description": "Structurally-Focused Hybrid RAG Protocol"
        },
        "splits": {
            "train": {
                "count": len(train_df),
                "target_size": RAGConfig.TRAIN_SIZE,
                "path": str(RAGConfig.TRAIN_SET)
            },
            "val": {
                "count": len(val_df),
                "target_size": RAGConfig.VAL_SIZE,
                "path": str(RAGConfig.VAL_SET)
            },
            "test": {
                "count": len(test_df),
                "target_size": RAGConfig.TEST_SIZE,
                "path": str(RAGConfig.TEST_SET)
            }
        },
        "rag_corpus": {
            "count": len(corpus_df),
            "composition": "train + val",
            "description": "Test set excluded to prevent contamination",
            "path": str(RAGConfig.TRAIN_VAL_CORPUS),
            "retrieval_field": RAGConfig.RETRIEVAL_FIELD
        },
        "statistics": {
            "train": {
                "avg_input_len": int(train_df['input'].str.len().mean()),
                "avg_target_len": int(train_df['target'].str.len().mean()),
            },
            "val": {
                "avg_input_len": int(val_df['input'].str.len().mean()),
                "avg_target_len": int(val_df['target'].str.len().mean()),
            },
            "test": {
                "avg_input_len": int(test_df['input'].str.len().mean()),
                "avg_target_len": int(test_df['target'].str.len().mean()),
            }
        },
        "ethical_considerations": {
            "test_set_isolation": "Test set completely excluded from RAG retrieval corpus",
            "retrieval_focus": "Structure-focused retrieval using structured_target field",
            "anti_copying": "Prompt designed to prevent factual copying from retrieved examples"
        }
    }

    # Add token statistics if available
    if 'input_token_len' in train_df.columns:
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            metadata["statistics"][split_name]["avg_input_tokens"] = int(df['input_token_len'].mean())
            if 'target_token_len' in df.columns:
                metadata["statistics"][split_name]["avg_target_tokens"] = int(df['target_token_len'].mean())

    metadata_path = RAGConfig.RAG_DATA_DIR / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata: {metadata_path.name}")

def main():
    """Main execution"""
    # Create directories
    RAGConfig.create_directories()

    # Validate original dataset exists
    if not RAGConfig.ORIGINAL_DATASET.exists():
        print(f"\n❌ Error: Dataset not found at {RAGConfig.ORIGINAL_DATASET}")
        print("   Please ensure 5000_structured.csv exists in Data/ directory")
        return

    # Load dataset
    df = load_dataset()

    # Perform custom split
    train_df, val_df, test_df = custom_split(
        df,
        test_size=RAGConfig.TEST_SIZE,
        val_size=RAGConfig.VAL_SIZE,
        random_seed=RAGConfig.RANDOM_SEED
    )

    # Save splits
    corpus_df = save_splits(train_df, val_df, test_df)

    # Analyze splits
    analyze_splits(train_df, val_df, test_df)

    # Save metadata
    save_metadata(train_df, val_df, test_df, corpus_df)

    print("\n" + "=" * 80)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Created files:")
    print(f"  • {RAGConfig.TRAIN_SET.name} - Training set")
    print(f"  • {RAGConfig.VAL_SET.name} - Validation set")
    print(f"  • {RAGConfig.TEST_SET.name} - Test set")
    print(f"  • {RAGConfig.TRAIN_VAL_CORPUS.name} - RAG corpus (train+val)")
    print(f"  • dataset_metadata.json - Statistics and metadata")

    print(f"\n📊 Summary:")
    print(f"  • RAG Corpus: {len(corpus_df)} samples (for retrieval)")
    print(f"  • Test Set: {len(test_df)} samples (for evaluation only)")
    print(f"  • Retrieval Field: {RAGConfig.RETRIEVAL_FIELD} (structure-focused)")

    print(f"\n➡️  Next step: Run '2_build_rag_corpus.py' to create embeddings and indices")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
