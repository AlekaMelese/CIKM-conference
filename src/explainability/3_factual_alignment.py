"""
Phase 3 - Factual Alignment Checker
Verifies factual consistency between generated summary and source evidence
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import sys
import re
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent / "RAG1"))
from config import RAGConfig


class FactualAlignmentChecker:
    """Check factual alignment between generation and evidence"""

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

        self.embedding_model = None
        self.summaries = None

    def load_data(self):
        """Load data"""
        print("Loading data...")
        with open(self.summaries_path, 'r') as f:
            self.summaries = json.load(f)
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

    def compute_entailment_score(self, generated_sent, source_text):
        """Compute if generated sentence is entailed by source (simplified)"""
        # Split source into sentences
        source_sents = re.split(r'[.!?]\s+', source_text)
        source_sents = [s.strip() for s in source_sents if len(s.strip()) > 20]

        if not source_sents:
            return 0.0, []

        # Compute embeddings
        gen_emb = self.embedding_model.encode([generated_sent])
        source_embs = self.embedding_model.encode(source_sents)

        # Find most similar source sentence
        similarities = cosine_similarity(gen_emb, source_embs)[0]
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]

        return float(max_sim), [{'sentence': source_sents[max_idx], 'similarity': float(max_sim)}]

    def check_factual_consistency(self, idx):
        """Check factual consistency for single summary"""
        summary = self.summaries[idx]

        generated_sents = re.split(r'[.!?]\s+', summary['generated_summary'])
        generated_sents = [s.strip() for s in generated_sents if len(s.strip()) > 20]

        source_text = summary['input']

        alignments = []
        for sent in generated_sents:
            score, evidence = self.compute_entailment_score(sent, source_text)
            alignments.append({
                'generated_sentence': sent,
                'alignment_score': score,
                'supporting_evidence': evidence,
                'factual_status': 'SUPPORTED' if score > 0.7 else ('PARTIAL' if score > 0.5 else 'UNSUPPORTED')
            })

        # Overall statistics
        supported = sum(1 for a in alignments if a['factual_status'] == 'SUPPORTED')
        partial = sum(1 for a in alignments if a['factual_status'] == 'PARTIAL')
        unsupported = sum(1 for a in alignments if a['factual_status'] == 'UNSUPPORTED')

        return {
            'note_id': summary['note_id'],
            'sentence_alignments': alignments,
            'statistics': {
                'total_sentences': len(alignments),
                'supported': supported,
                'partial': partial,
                'unsupported': unsupported,
                'support_rate': supported / len(alignments) if alignments else 0,
                'avg_alignment_score': np.mean([a['alignment_score'] for a in alignments]) if alignments else 0
            }
        }

    def visualize_factual_alignment(self, all_results):
        """Visualize factual alignment statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))  # More vertical space

        # 1. Support rate distribution
        ax = axes[0, 0]
        support_rates = [r['statistics']['support_rate'] for r in all_results]
        mean_support = np.mean(support_rates)
        ax.hist(support_rates, bins=16, color='steelblue', alpha=0.8, edgecolor='black', linewidth=2.2)
        ax.axvline(mean_support, color='red', linestyle='--', linewidth=3.5, label=f'Mean: {mean_support:.3f}')
        ax.set_xlabel('Factual Support Rate', fontsize=15, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=15, fontweight='bold')
        ax.set_title('Factual Support Rate Distribution', fontsize=16, fontweight='bold', pad=10)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.5, linewidth=1.2)
        ax.tick_params(axis='both', labelsize=13, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 2. Status breakdown
        ax = axes[0, 1]
        total_supported = sum(r['statistics']['supported'] for r in all_results)
        total_partial = sum(r['statistics']['partial'] for r in all_results)
        total_unsupported = sum(r['statistics']['unsupported'] for r in all_results)

        categories = ['Supported', 'Partial', 'Unsupported']
        values = [total_supported, total_partial, total_unsupported]
        colors = ['green', 'orange', 'red']

        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Sentences', fontsize=15, fontweight='bold')
        ax.set_title('Overall Factual Status', fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.5, axis='y', linewidth=1.2)
        total = sum(values)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{val}\n({val/total*100:.1f}%)', ha='center', fontweight='bold', fontsize=12)
        ax.tick_params(axis='both', labelsize=13, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 3. Alignment score distribution
        ax = axes[1, 0]
        all_scores = []
        for r in all_results:
            all_scores.extend([a['alignment_score'] for a in r['sentence_alignments']])

        mean_align = np.mean(all_scores)
        ax.hist(all_scores, bins=24, color='coral', alpha=0.8, edgecolor='black', linewidth=2.2)
        ax.axvline(mean_align, color='red', linestyle='--', linewidth=3.5, label=f'Mean: {mean_align:.3f}')
        ax.set_xlabel('Alignment Score', fontsize=15, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=15, fontweight='bold')
        ax.set_title('Sentence-level Alignment Scores', fontsize=16, fontweight='bold', pad=10)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.5, linewidth=1.2)
        ax.tick_params(axis='both', labelsize=13, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 4. Per-summary statistics
        ax = axes[1, 1]
        avg_scores = [r['statistics']['avg_alignment_score'] for r in all_results]
        support_rates = [r['statistics']['support_rate'] for r in all_results]

        ax.scatter(avg_scores, support_rates, alpha=0.7, s=60, color='purple', edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Avg Alignment Score', fontsize=15, fontweight='bold')
        ax.set_ylabel('Support Rate', fontsize=15, fontweight='bold')
        ax.set_title('Alignment Score vs Support Rate', fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.5, linewidth=1.2)
        ax.tick_params(axis='both', labelsize=13, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # Correlation text removed as requested

        plt.suptitle(f'Factual Alignment Analysis - {"Medical" if self.use_medical else "Standard"} RAG',
                fontsize=18, fontweight='bold')
        plt.tight_layout(pad=0.7, rect=[0, 0, 1, 0.96])

        output_path = self.explainability_dir / "factual_alignment.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")

    def run_analysis(self, num_samples=None):
        """Run factual alignment analysis"""
        print("=" * 80)
        print("Factual Alignment Analysis")
        print("=" * 80)

        self.load_data()
        self.load_embedding_model()

        if num_samples:
            self.summaries = self.summaries[:num_samples]

        all_results = []
        print(f"\nChecking factual alignment for {len(self.summaries)} summaries...")
        for idx in tqdm(range(len(self.summaries))):
            result = self.check_factual_consistency(idx)
            all_results.append(result)

        # Save results
        output_path = self.explainability_dir / "factual_alignment.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved: {output_path}")

        # Visualize
        print("\nGenerating visualizations...")
        self.visualize_factual_alignment(all_results)

        # Print summary
        print("\n" + "=" * 80)
        print("Factual Alignment Statistics:")
        print("-" * 80)
        avg_support = np.mean([r['statistics']['support_rate'] for r in all_results])
        print(f"Average Support Rate: {avg_support:.3f}")

        total_supported = sum(r['statistics']['supported'] for r in all_results)
        total_sentences = sum(r['statistics']['total_sentences'] for r in all_results)
        print(f"Total Sentences: {total_sentences}")
        print(f"Supported: {total_supported} ({total_supported/total_sentences*100:.1f}%)")

        return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--medical', action='store_true', default=True, help='Use medical RAG(Llama) (default: True)')
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()

    checker = FactualAlignmentChecker(use_medical=args.medical)
    checker.run_analysis(num_samples=args.samples)


if __name__ == "__main__":
    main()
