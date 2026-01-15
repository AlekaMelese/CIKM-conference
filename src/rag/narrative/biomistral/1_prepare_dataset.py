#!/usr/bin/env python3
"""
Phase 2 - Step 1: Dataset Preparation for NARRATIVE RAG
Splits 5000_structured.csv into custom splits: 200 test, 300 val, 4500 train
Uses 'target' column (narrative paragraphs) NOT 'structured_target'
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from config import NarrativeRAGConfig

def load_dataset():
    """Load the 5000_structured.csv dataset"""
    print("=" * 80)
    print("STEP 1: DATASET PREPARATION - NARRATIVE FORMAT")
    print("=" * 80)
    print(f"\nLoading dataset from: {NarrativeRAGConfig.ORIGINAL_DATASET}")

    df = pd.read_csv(NarrativeRAGConfig.ORIGINAL_DATASET)

    print(f"✓ Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")

    # Check for required columns
    required_cols = ['note_id', 'input', 'target']  # NARRATIVE uses 'target' not 'structured_target'
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Warning: Missing columns: {missing_cols}")

    # Show target field info
    if 'target' in df.columns:
        avg_len = df['target'].str.len().mean()
        print(f"  Narrative target avg length: {avg_len:.0f} characters")

    return df

def custom_split(df, test_size=200, val_size=300, random_seed=42):
    """
    Custom split: 200 test, 300 validation, 4500 training
    """
    print(f"\nPerforming custom dataset split...")
    print(f"  Target split: {NarrativeRAGConfig.TRAIN_SIZE} train / {NarrativeRAGConfig.VAL_SIZE} val / {NarrativeRAGConfig.TEST_SIZE} test")

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
    print(f"\nSaving dataset splits to: {NarrativeRAGConfig.RAG_DATA_DIR}")

    # Save individual splits
    train_df.to_csv(NarrativeRAGConfig.TRAIN_SET, index=False)
    print(f"  ✓ Train set: {NarrativeRAGConfig.TRAIN_SET.name} ({len(train_df)} records)")

    val_df.to_csv(NarrativeRAGConfig.VAL_SET, index=False)
    print(f"  ✓ Val set: {NarrativeRAGConfig.VAL_SET.name} ({len(val_df)} records)")

    test_df.to_csv(NarrativeRAGConfig.TEST_SET, index=False)
    print(f"  ✓ Test set: {NarrativeRAGConfig.TEST_SET.name} ({len(test_df)} records)")

    # Create RAG corpus (train + val, exclude test for ethical integrity)
    corpus_df = pd.concat([train_df, val_df], ignore_index=True)
    corpus_df.to_csv(NarrativeRAGConfig.TRAIN_VAL_CORPUS, index=False)
    print(f"  ✓ RAG corpus: {NarrativeRAGConfig.TRAIN_VAL_CORPUS.name} ({len(corpus_df)} records)")
    print(f"    → Corpus = Train + Val (Test excluded for academic integrity)")

    return corpus_df

def analyze_splits(train_df, val_df, test_df):
    """Analyze dataset statistics for each split"""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS - NARRATIVE FORMAT")
    print("=" * 80)

    for name, df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        print(f"\n{name} SET ({len(df)} records)")
        print(f"  {'Metric':<30} {'Mean':<15} {'Std':<15}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")

        # Character lengths
        print(f"  {'Input length (chars)':<30} {df['input'].str.len().mean():>10.0f}     {df['input'].str.len().std():>10.0f}")
        print(f"  {'Narrative target (chars)':<30} {df['target'].str.len().mean():>10.0f}     {df['target'].str.len().std():>10.0f}")

        # Word counts for narrative
        if 'target' in df.columns:
            word_counts = df['target'].str.split().str.len()
            print(f"  {'Narrative target (words)':<30} {word_counts.mean():>10.0f}     {word_counts.std():>10.0f}")

def save_metadata(train_df, val_df, test_df, corpus_df):
    """Save dataset metadata for reference"""
    metadata = {
        "dataset_info": {
            "source": str(NarrativeRAGConfig.ORIGINAL_DATASET),
            "total_records": len(train_df) + len(val_df) + len(test_df),
            "random_seed": NarrativeRAGConfig.RANDOM_SEED,
            "split_type": "custom",
            "format": "NARRATIVE (flowing paragraphs)",
            "description": "Narrative-Focused Hybrid RAG Protocol"
        },
        "splits": {
            "train": {
                "count": len(train_df),
                "target_size": NarrativeRAGConfig.TRAIN_SIZE,
                "path": str(NarrativeRAGConfig.TRAIN_SET)
            },
            "val": {
                "count": len(val_df),
                "target_size": NarrativeRAGConfig.VAL_SIZE,
                "path": str(NarrativeRAGConfig.VAL_SET)
            },
            "test": {
                "count": len(test_df),
                "target_size": NarrativeRAGConfig.TEST_SIZE,
                "path": str(NarrativeRAGConfig.TEST_SET)
            }
        },
        "rag_corpus": {
            "count": len(corpus_df),
            "composition": "train + val",
            "description": "Test set excluded to prevent contamination",
            "path": str(NarrativeRAGConfig.TRAIN_VAL_CORPUS),
            "retrieval_field": NarrativeRAGConfig.RETRIEVAL_FIELD
        },
        "statistics": {
            "train": {
                "avg_input_len": int(train_df['input'].str.len().mean()),
                "avg_target_len": int(train_df['target'].str.len().mean()),
                "avg_target_words": int(train_df['target'].str.split().str.len().mean())
            },
            "val": {
                "avg_input_len": int(val_df['input'].str.len().mean()),
                "avg_target_len": int(val_df['target'].str.len().mean()),
                "avg_target_words": int(val_df['target'].str.split().str.len().mean())
            },
            "test": {
                "avg_input_len": int(test_df['input'].str.len().mean()),
                "avg_target_len": int(test_df['target'].str.len().mean()),
                "avg_target_words": int(test_df['target'].str.split().str.len().mean())
            }
        },
        "ethical_considerations": {
            "test_set_isolation": "Test set completely excluded from RAG retrieval corpus",
            "retrieval_focus": "Narrative-focused retrieval using 'target' field (flowing paragraphs)",
            "anti_copying": "Prompt designed to prevent factual copying from retrieved examples"
        }
    }

    metadata_path = NarrativeRAGConfig.RAG_DATA_DIR / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata: {metadata_path.name}")

def main():
    """Main execution"""
    # Create directories
    NarrativeRAGConfig.create_directories()

    # Validate original dataset exists
    if not NarrativeRAGConfig.ORIGINAL_DATASET.exists():
        print(f"\n❌ Error: Dataset not found at {NarrativeRAGConfig.ORIGINAL_DATASET}")
        print("   Please ensure 5000_structured.csv exists in Data/ directory")
        return

    # Load dataset
    df = load_dataset()

    # Perform custom split
    train_df, val_df, test_df = custom_split(
        df,
        test_size=NarrativeRAGConfig.TEST_SIZE,
        val_size=NarrativeRAGConfig.VAL_SIZE,
        random_seed=NarrativeRAGConfig.RANDOM_SEED
    )

    # Save splits
    corpus_df = save_splits(train_df, val_df, test_df)

    # Analyze splits
    analyze_splits(train_df, val_df, test_df)

    # Save metadata
    save_metadata(train_df, val_df, test_df, corpus_df)

    print("\n" + "=" * 80)
    print("✅ DATASET PREPARATION COMPLETE - NARRATIVE FORMAT!")
    print("=" * 80)
    print(f"\n📁 Created files:")
    print(f"  • {NarrativeRAGConfig.TRAIN_SET.name} - Training set")
    print(f"  • {NarrativeRAGConfig.VAL_SET.name} - Validation set")
    print(f"  • {NarrativeRAGConfig.TEST_SET.name} - Test set")
    print(f"  • {NarrativeRAGConfig.TRAIN_VAL_CORPUS.name} - RAG corpus (train+val)")
    print(f"  • dataset_metadata.json - Statistics and metadata")

    print(f"\n📊 Summary:")
    print(f"  • RAG Corpus: {len(corpus_df)} samples (for retrieval)")
    print(f"  • Test Set: {len(test_df)} samples (for evaluation only)")
    print(f"  • Retrieval Field: {NarrativeRAGConfig.RETRIEVAL_FIELD} (narrative paragraphs)")
    print(f"  • Format: NARRATIVE (flowing prose, NO bullets/headers)")

    print(f"\n➡️  Next step: Run '2_build_rag_corpus.py' to create embeddings and indices")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
