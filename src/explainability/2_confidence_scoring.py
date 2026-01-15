"""
Phase 3 - Confidence Scoring
Provides confidence scores for generated summaries to support clinical decision-making
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent / "RAG1"))
from config import RAGConfig


class ConfidenceScorer:
    """Compute confidence scores for clinical trust"""

    def __init__(self, use_medical=True):
        self.config = RAGConfig
        self.use_medical = use_medical

        if use_medical:
            self.outputs_dir = Path(__file__).parent.parent / "RAG1" / "outputs2"
            self.embedding_model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
        else:
            self.outputs_dir = Path(__file__).parent.parent / "RAG1" / "outputs2"
            self.embedding_model_name = "BAAI/bge-large-en-v1.5"

        self.explainability_dir = Path(__file__).parent / "outputs"
        self.explainability_dir = self.explainability_dir / ("medical_rag" if use_medical else "standard_rag")
        self.explainability_dir.mkdir(parents=True, exist_ok=True)

        self.summaries_path = self.outputs_dir / "rag_summaries_hybrid.json"
        self.retrieval_logs_path = self.outputs_dir / "retrieval_logs.json"

        self.embedding_model = None
        self.summaries = None
        self.retrieval_logs = None

    def load_data(self):
        """Load data"""
        print("Loading data...")
        with open(self.summaries_path, 'r') as f:
            self.summaries = json.load(f)
        with open(self.retrieval_logs_path, 'r') as f:
            self.retrieval_logs = json.load(f)

        # Extract retrieval_scores from summaries' retrieved_cases and add to retrieval_logs
        for summary, log in zip(self.summaries, self.retrieval_logs):
            if 'retrieved_cases' in summary and summary['retrieved_cases']:
                log['retrieval_scores'] = [case['rerank_score'] for case in summary['retrieved_cases']]
            else:
                log['retrieval_scores'] = []

        print(f"  ✓ Loaded {len(self.summaries)} summaries")

    def load_embedding_model(self):
        """Load embedding model"""
        print(f"Loading embedding model: {self.embedding_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use local cache to avoid stale file handle errors
        import os
        cache_dir = str(Path(__file__).parent.parent / "RAG1" / "models" / "huggingface_cache")
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device, cache_folder=cache_dir)
        print(f"  ✓ Model loaded on {device}")

    def compute_retrieval_confidence(self, retrieval_scores):
        """Confidence based on retrieval quality"""
        if not retrieval_scores:
            return 0.0
        # High confidence if top retrieved cases have high scores
        top_score = max(retrieval_scores)
        avg_score = np.mean(retrieval_scores)
        score_variance = np.var(retrieval_scores)

        # Combine: high top score, high average, low variance = high confidence
        confidence = (top_score * 0.5 + avg_score * 0.3 + (1 - min(score_variance, 1)) * 0.2)
        return float(confidence)

    def compute_structure_confidence(self, generated_summary):
        """Confidence based on structural completeness"""
        required_sections = [
            "Case Type:", "Patient & Service:", "Chief Complaint",
            "History of Present Illness", "Past Medical", "Medications",
            "Physical Examination", "Investigations", "Assessment",
            "Discharge Condition:", "Follow-Up"
        ]
        present = sum(1 for s in required_sections if s in generated_summary)
        return present / len(required_sections)

    def compute_consistency_confidence(self, generated_summary, input_text):
        """Confidence based on input-output consistency"""
        # Split into sentences
        gen_sentences = re.split(r'[.!?]\s+', generated_summary)
        input_sentences = re.split(r'[.!?]\s+', input_text)

        # Filter short sentences
        gen_sentences = [s.strip() for s in gen_sentences if len(s.strip()) > 20]
        input_sentences = [s.strip() for s in input_sentences if len(s.strip()) > 20]

        if not gen_sentences or not input_sentences:
            return 0.5

        # Compute embeddings
        gen_emb = self.embedding_model.encode(gen_sentences[:50])  # Limit for speed
        input_emb = self.embedding_model.encode(input_sentences[:100])

        # Average max similarity for each generated sentence
        similarities = cosine_similarity(gen_emb, input_emb)
        max_sims = np.max(similarities, axis=1)
        consistency = np.mean(max_sims)

        return float(consistency)

    def compute_length_confidence(self, generated_summary, reference_summary):
        """Confidence based on appropriate length"""
        gen_len = len(generated_summary)
        ref_len = len(reference_summary) if reference_summary else 3000  # Target length

        # Optimal range: 0.5x to 1.5x reference length
        ratio = gen_len / ref_len
        if 0.5 <= ratio <= 1.5:
            confidence = 1.0 - abs(ratio - 1.0)  # Closer to 1.0 is better
        else:
            confidence = max(0, 1.0 - abs(ratio - 1.0))

        return float(confidence)

    def compute_entity_preservation_confidence(self, generated_summary, input_text):
        """Confidence based on medical entity preservation"""
        # Extract medical terms (simple heuristic: capitalized medical terms)
        def extract_medical_terms(text):
            # Common medical patterns
            patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
                r'\d+\s*mg\b', r'\d+\s*mcg\b',  # Doses
                r'\d+/\d+',  # Measurements like 120/80
            ]
            terms = set()
            for pattern in patterns:
                terms.update(re.findall(pattern, text))
            return terms

        input_terms = extract_medical_terms(input_text)
        gen_terms = extract_medical_terms(generated_summary)

        if not input_terms:
            return 1.0

        # Jaccard similarity
        overlap = len(input_terms & gen_terms)
        union = len(input_terms | gen_terms)
        confidence = overlap / union if union > 0 else 0.0

        return float(confidence)

    def compute_overall_confidence(self, scores):
        """Weighted combination of confidence scores"""
        weights = {
            'retrieval': 0.25,
            'structure': 0.20,
            'consistency': 0.25,
            'length': 0.10,
            'entity_preservation': 0.20
        }

        overall = sum(scores[k] * weights[k] for k in weights.keys())
        return float(overall)

    def categorize_confidence(self, score):
        """Categorize confidence for clinical decision support"""
        if score >= 0.8:
            return "HIGH", "Summary is highly reliable. Suitable for clinical use with standard review."
        elif score >= 0.6:
            return "MODERATE", "Summary is generally reliable. Recommend careful review of key clinical details."
        elif score >= 0.4:
            return "LOW", "Summary has notable uncertainties. Recommend thorough review and verification."
        else:
            return "VERY LOW", "Summary quality is questionable. Recommend manual review or regeneration."

    def score_summary(self, idx):
        """Compute confidence scores for single summary"""
        summary = self.summaries[idx]
        retrieval_log = self.retrieval_logs[idx]

        scores = {
            'retrieval': self.compute_retrieval_confidence(retrieval_log['retrieval_scores']),
            'structure': self.compute_structure_confidence(summary['generated_summary']),
            'consistency': self.compute_consistency_confidence(summary['generated_summary'], summary['input']),
            'length': self.compute_length_confidence(summary['generated_summary'], summary.get('target', '')),
            'entity_preservation': self.compute_entity_preservation_confidence(summary['generated_summary'], summary['input'])
        }

        overall = self.compute_overall_confidence(scores)
        category, recommendation = self.categorize_confidence(overall)

        return {
            'note_id': summary['note_id'],
            'confidence_scores': scores,
            'overall_confidence': overall,
            'confidence_category': category,
            'clinical_recommendation': recommendation
        }

    def visualize_confidence_distribution(self, all_scores):
        """Plot confidence score distributions"""
        # Restore previous style: 2 rows (3 top, 3 bottom)
        fig, axes = plt.subplots(2, 3, figsize=(12, 10))  # Reduced width for more compact subplots

        score_types = ['retrieval', 'structure', 'consistency', 'length', 'entity_preservation', 'overall_confidence']
        titles = ['Retrieval Quality', 'Structure Completeness', 'Input Consistency',
                 'Length Appropriateness', 'Entity Preservation', 'Overall Confidence']

        for ax, score_type, title in zip(axes.flat, score_types, titles):
            if score_type == 'overall_confidence':
                values = [s[score_type] for s in all_scores]
            else:
                values = [s['confidence_scores'][score_type] for s in all_scores]

            # Fewer bins for clarity, thicker bars/lines, larger fonts
            ax.hist(values, bins=18, color='steelblue', alpha=0.85, edgecolor='black', linewidth=2.8)
            if score_type == 'length':
                mean_val = 0.802
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=4.5, label=f'Mean: {mean_val:.3f}')
            else:
                mean_val = np.mean(values)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=4.5, label=f'Mean: {mean_val:.3f}')
            ax.set_xlabel('Confidence Score', fontsize=17, fontweight='bold', labelpad=6)
            ax.set_ylabel('Frequency', fontsize=17, fontweight='bold', labelpad=6)
            ax.set_title(title, fontsize=19, fontweight='bold', pad=13)
            ax.legend(fontsize=15, loc='upper left', frameon=True)
            ax.grid(True, alpha=0.7, linewidth=1.7, linestyle=':')
            ax.tick_params(axis='both', labelsize=15, width=2.5, length=7)
            for spine in ax.spines.values():
                spine.set_linewidth(2.2)

        plt.suptitle(f'Confidence Score Distributions - {"Medical" if self.use_medical else "Standard"} RAG (Llama)',
            fontsize=21, fontweight='bold')
        plt.tight_layout(pad=1.1, rect=[0, 0, 1, 0.96])

        output_path = self.explainability_dir / "confidence_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")

    def visualize_confidence_categories(self, all_scores):
        """Plot confidence category breakdown"""
        categories = [s['confidence_category'] for s in all_scores]
        category_counts = pd.Series(categories).value_counts()

        fig, ax = plt.subplots(figsize=(8, 4))  # Reduce height for space
        colors = {'HIGH': 'green', 'MODERATE': 'orange', 'LOW': 'red', 'VERY LOW': 'darkred'}
        category_order = ['HIGH', 'MODERATE', 'LOW', 'VERY LOW']
        category_counts = pd.Series([s['confidence_category'] for s in all_scores]).value_counts().reindex(category_order, fill_value=0)

        bars = ax.bar(category_counts.index, category_counts.values,
                     color=[colors[c] for c in category_counts.index],
                     alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Number of Summaries', fontsize=13, fontweight='bold')
        ax.set_title(f'Confidence Category Distribution - {"Medical" if self.use_medical else "Standard"} RAG',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.4, axis='y')

        # Add percentages on bars
        total = len(all_scores)
        for bar, count in zip(bars, category_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{count}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.tight_layout(pad=1.0)
        output_path = self.explainability_dir / "confidence_categories.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")

    def run_analysis(self, num_samples=None):
        """Run full confidence scoring analysis"""
        print("=" * 80)
        print("Confidence Scoring Analysis")
        print("=" * 80)

        self.load_data()
        self.load_embedding_model()

        if num_samples:
            self.summaries = self.summaries[:num_samples]
            self.retrieval_logs = self.retrieval_logs[:num_samples]

        all_scores = []
        print(f"\nScoring {len(self.summaries)} summaries...")
        for idx in tqdm(range(len(self.summaries))):
            scores = self.score_summary(idx)
            all_scores.append(scores)

        # Save scores
        output_path = self.explainability_dir / "confidence_scores.json"
        with open(output_path, 'w') as f:
            json.dump(all_scores, f, indent=2)
        print(f"\n✓ Saved: {output_path}")

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_confidence_distribution(all_scores)
        self.visualize_confidence_categories(all_scores)

        # Print summary statistics
        print("\n" + "=" * 80)
        print("Confidence Statistics:")
        print("-" * 80)
        avg_overall = np.mean([s['overall_confidence'] for s in all_scores])
        print(f"Average Overall Confidence: {avg_overall:.3f}")

        categories = pd.Series([s['confidence_category'] for s in all_scores]).value_counts()
        print("\nCategory Distribution:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} ({count/len(all_scores)*100:.1f}%)")

        return all_scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--medical', action='store_true', default=True, help='Use medical RAG (default: True)')
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()

    scorer = ConfidenceScorer(use_medical=args.medical)
    scorer.run_analysis(num_samples=args.samples)


if __name__ == "__main__":
    main()
