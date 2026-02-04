#!/usr/bin/env python3
"""
RAG Corpus Preparation Script

Prepares the retrieval corpus for RAG by:
1. Loading processed dataset
2. Creating train/val/test splits with custom sizes
3. Building the RAG corpus (train + val, excluding test)

Usage:
    python prepare_rag_corpus.py --input data/processed/discharge_notes.csv --output data/rag_corpus/
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


class RAGCorpusPreparator:
    """Prepares corpus for RAG retrieval"""

    def __init__(self, input_path: str, output_path: str,
                 test_size: int = 15%, val_size: int = 15%, random_seed: int = 42):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

    def load_dataset(self) -> pd.DataFrame:
        """Load the processed dataset"""
        print("=" * 80)
        print("RAG CORPUS PREPARATION")
        print("=" * 80)
        print(f"\nLoading dataset from: {self.input_path}")

        df = pd.read_csv(self.input_path)

        print(f"✓ Loaded {len(df)} records")
        print(f"  Columns: {list(df.columns)}")

        return df

    def custom_split(self, df: pd.DataFrame) -> tuple:
        """
        Custom split: test_size test, val_size validation, rest for training
        """
        print(f"\nPerforming custom dataset split...")
        train_size = len(df) - self.test_size - self.val_size
        print(f"  Target split: {train_size} train / {self.val_size} val / {self.test_size} test")

        # First split: test vs rest
        rest_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True
        )

        # Second split: train vs val from rest
        train_df, val_df = train_test_split(
            rest_df,
            test_size=self.val_size,
            random_state=self.random_seed,
            shuffle=True
        )

        print(f"\n✓ Split results:")
        print(f"    Train: {len(train_df):>4} records ({len(train_df)/len(df)*100:.1f}%)")
        print(f"    Val:   {len(val_df):>4} records ({len(val_df)/len(df)*100:.1f}%)")
        print(f"    Test:  {len(test_df):>4} records ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def create_rag_corpus(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RAG corpus from train + val sets (excluding test to prevent data leakage)
        """
        print(f"\nCreating RAG corpus...")

        # Combine train and validation for retrieval corpus
        corpus_df = pd.concat([train_df, val_df], ignore_index=True)

        print(f"✓ RAG corpus: {len(corpus_df)} documents")
        print(f"    Source: {len(train_df)} train + {len(val_df)} val")

        return corpus_df

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    test_df: pd.DataFrame, corpus_df: pd.DataFrame):
        """Save all splits and corpus"""
        print(f"\nSaving splits to {self.output_path}...")

        # Save individual splits
        train_df.to_csv(self.output_path / "train.csv", index=False)
        val_df.to_csv(self.output_path / "val.csv", index=False)
        test_df.to_csv(self.output_path / "test.csv", index=False)

        # Save RAG corpus
        corpus_df.to_csv(self.output_path / "rag_corpus.csv", index=False)

        # Save as JSON for easy loading
        splits = {
            'train': train_df.to_dict(orient='records'),
            'val': val_df.to_dict(orient='records'),
            'test': test_df.to_dict(orient='records'),
        }

        with open(self.output_path / "splits.json", 'w') as f:
            json.dump(splits, f, indent=2)

        # Save split indices for reproducibility
        split_info = {
            'train_indices': train_df.index.tolist(),
            'val_indices': val_df.index.tolist(),
            'test_indices': test_df.index.tolist(),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'corpus_size': len(corpus_df),
            'random_seed': self.random_seed,
        }

        with open(self.output_path / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"✓ Saved files:")
        print(f"    train.csv:      {len(train_df)} records")
        print(f"    val.csv:        {len(val_df)} records")
        print(f"    test.csv:       {len(test_df)} records")
        print(f"    rag_corpus.csv: {len(corpus_df)} records")
        print(f"    splits.json:    All splits in JSON format")
        print(f"    split_info.json: Split metadata")

    def run(self):
        """Run the corpus preparation pipeline"""
        # Load dataset
        df = self.load_dataset()

        # Create splits
        train_df, val_df, test_df = self.custom_split(df)

        # Create RAG corpus
        corpus_df = self.create_rag_corpus(train_df, val_df)

        # Save everything
        self.save_splits(train_df, val_df, test_df, corpus_df)

        print("\n" + "=" * 80)
        print("RAG CORPUS PREPARATION COMPLETE")
        print("=" * 80)
        print(f"\nReady for Phase 2 RAG pipeline:")
        print(f"  - Use rag_corpus.csv for building FAISS index")
        print(f"  - Use test.csv for inference/evaluation")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Prepare RAG corpus from processed dataset')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--test_size', type=int, default=200, help='Number of test samples')
    parser.add_argument('--val_size', type=int, default=300, help='Number of validation samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    preparator = RAGCorpusPreparator(
        input_path=args.input,
        output_path=args.output,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed
    )
    preparator.run()


if __name__ == "__main__":
    main()
