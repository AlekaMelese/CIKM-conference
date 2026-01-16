#!/usr/bin/env python3
"""
MIMIC-IV Dataset Preparation Script

Extracts and preprocesses discharge summaries from MIMIC-IV database
for clinical summarization fine-tuning.

Usage:
    python prepare_dataset.py --mimic_path /path/to/mimiciv/ --output_path data/processed/
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import argparse


class MIMICPreprocessor:
    """Preprocesses MIMIC-IV discharge summaries for clinical summarization"""

    def __init__(self, mimic_path: str, output_path: str):
        self.mimic_path = Path(mimic_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Clinical abbreviation expansions
        self.abbreviations = {
            'HTN': 'hypertension',
            'DM': 'diabetes mellitus',
            'CAD': 'coronary artery disease',
            'CHF': 'congestive heart failure',
            'COPD': 'chronic obstructive pulmonary disease',
            'CKD': 'chronic kidney disease',
            'CVA': 'cerebrovascular accident',
            'MI': 'myocardial infarction',
            'DVT': 'deep vein thrombosis',
            'PE': 'pulmonary embolism',
            'UTI': 'urinary tract infection',
            'GERD': 'gastroesophageal reflux disease',
            'BPH': 'benign prostatic hyperplasia',
            'RA': 'rheumatoid arthritis',
            'OA': 'osteoarthritis',
            'ESRD': 'end-stage renal disease',
            'AKI': 'acute kidney injury',
            'ARDS': 'acute respiratory distress syndrome',
            'GI': 'gastrointestinal',
            'SOB': 'shortness of breath',
            'CP': 'chest pain',
            'HA': 'headache',
            'N/V': 'nausea/vomiting',
            'BID': 'twice daily',
            'TID': 'three times daily',
            'QID': 'four times daily',
            'PRN': 'as needed',
            'PO': 'by mouth',
            'IV': 'intravenous',
            'IM': 'intramuscular',
            'SQ': 'subcutaneous',
        }

    def load_mimic_data(self) -> pd.DataFrame:
        """Load discharge summaries from MIMIC-IV"""
        print("Loading MIMIC-IV data...")

        # Load discharge notes
        discharge_path = self.mimic_path / "hosp" / "discharge.csv"
        if not discharge_path.exists():
            # Try alternative path structures
            discharge_path = self.mimic_path / "discharge.csv"

        if not discharge_path.exists():
            raise FileNotFoundError(f"Discharge file not found at {discharge_path}")

        df = pd.read_csv(discharge_path)
        print(f"  Loaded {len(df)} discharge records")

        return df

    def clean_text(self, text: str) -> str:
        """Clean and normalize clinical text"""
        if pd.isna(text):
            return ""

        text = str(text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep clinical punctuation
        text = re.sub(r'[^\w\s.,;:()/-]', '', text)

        # Normalize numbers with units
        text = re.sub(r'(\d+)\s+(mg|ml|mcg|g|kg|lb|cm|mm|L)', r'\1\2', text)

        return text.strip()

    def expand_abbreviations(self, text: str) -> str:
        """Expand common clinical abbreviations"""
        for abbrev, expansion in self.abbreviations.items():
            # Match whole words only
            pattern = r'\b' + abbrev + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

        return text

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract clinical sections from discharge note"""
        sections = {}

        section_patterns = {
            'chief_complaint': r'chief complaint[:\s]+(.*?)(?=history|past medical|$)',
            'hpi': r'history of present illness[:\s]+(.*?)(?=past medical|medications|$)',
            'pmh': r'past medical history[:\s]+(.*?)(?=past surgical|medications|$)',
            'medications': r'medications[:\s]+(.*?)(?=allergies|physical exam|$)',
            'allergies': r'allergies[:\s]+(.*?)(?=physical exam|pertinent|$)',
            'physical_exam': r'physical exam(?:ination)?[:\s]+(.*?)(?=pertinent|labs|$)',
            'labs': r'pertinent results?[:\s]+(.*?)(?=discharge|$)',
            'discharge_diagnosis': r'discharge diagnosis[:\s]+(.*?)(?=discharge condition|$)',
            'discharge_condition': r'discharge condition[:\s]+(.*?)(?=discharge instructions|$)',
            'discharge_instructions': r'discharge instructions?[:\s]+(.*?)(?=follow|$)',
            'followup': r'follow[\s-]?up[:\s]+(.*?)(?=$)',
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = self.clean_text(match.group(1))

        return sections

    def create_input_output_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create input-output pairs for training"""
        print("Creating input-output pairs...")

        records = []

        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processing record {idx}/{len(df)}")

            # Get the discharge note text
            text = row.get('text', row.get('note', ''))

            if pd.isna(text) or len(str(text)) < 100:
                continue

            # Clean the text
            cleaned_text = self.clean_text(text)
            expanded_text = self.expand_abbreviations(cleaned_text)

            # Extract sections
            sections = self.extract_sections(expanded_text)

            # Create input (full clinical note)
            input_text = expanded_text

            # Create target (summary - using discharge diagnosis and condition)
            target_parts = []
            if 'discharge_diagnosis' in sections:
                target_parts.append(f"Diagnosis: {sections['discharge_diagnosis']}")
            if 'discharge_condition' in sections:
                target_parts.append(f"Condition: {sections['discharge_condition']}")
            if 'discharge_instructions' in sections:
                target_parts.append(f"Instructions: {sections['discharge_instructions']}")

            target_text = " ".join(target_parts) if target_parts else cleaned_text[:500]

            # Create record
            record = {
                'note_id': row.get('note_id', f"note_{idx}"),
                'input': input_text,
                'target': target_text,
                'input_tokens': len(input_text.split()),
                'target_tokens': len(target_text.split()),
            }

            records.append(record)

        result_df = pd.DataFrame(records)
        print(f"  Created {len(result_df)} input-output pairs")

        return result_df

    def split_dataset(self, df: pd.DataFrame,
                      train_ratio: float = 0.70,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test sets"""
        print(f"\nSplitting dataset: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True
        )

        # Second split: train vs val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_seed,
            shuffle=True
        )

        print(f"  Train: {len(train_df)} records")
        print(f"  Val:   {len(val_df)} records")
        print(f"  Test:  {len(test_df)} records")

        return train_df, val_df, test_df

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save train/val/test splits as JSON files"""
        print("\nSaving dataset splits...")

        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            output_file = self.output_path / f"{name}.json"

            records = df.to_dict(orient='records')

            with open(output_file, 'w') as f:
                json.dump(records, f, indent=2)

            print(f"  Saved {output_file}: {len(records)} records")

        # Also save as CSV
        combined_df = pd.concat([train_df, val_df, test_df])
        csv_file = self.output_path / "discharge_notes.csv"
        combined_df.to_csv(csv_file, index=False)
        print(f"  Saved {csv_file}: {len(combined_df)} records")

    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute dataset statistics"""
        stats = {
            'total_records': len(df),
            'avg_input_chars': df['input'].str.len().mean(),
            'avg_target_chars': df['target'].str.len().mean(),
            'avg_input_tokens': df['input_tokens'].mean(),
            'avg_target_tokens': df['target_tokens'].mean(),
            'max_input_tokens': df['input_tokens'].max(),
            'max_target_tokens': df['target_tokens'].max(),
            'compression_ratio': df['input_tokens'].mean() / df['target_tokens'].mean(),
        }

        return stats

    def run(self):
        """Run the complete preprocessing pipeline"""
        print("=" * 80)
        print("MIMIC-IV PREPROCESSING PIPELINE")
        print("=" * 80)

        # Load data
        df = self.load_mimic_data()

        # Create input-output pairs
        processed_df = self.create_input_output_pairs(df)

        # Compute statistics
        stats = self.compute_statistics(processed_df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        # Split dataset
        train_df, val_df, test_df = self.split_dataset(processed_df)

        # Save splits
        self.save_splits(train_df, val_df, test_df)

        # Save statistics
        stats_file = self.output_path / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved statistics to {stats_file}")

        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-IV for clinical summarization')
    parser.add_argument('--mimic_path', '-m', required=True, help='Path to MIMIC-IV directory')
    parser.add_argument('--output_path', '-o', required=True, help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')

    args = parser.parse_args()

    preprocessor = MIMICPreprocessor(args.mimic_path, args.output_path)
    preprocessor.run()


if __name__ == "__main__":
    main()
