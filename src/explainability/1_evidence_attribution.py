"""
Phase 3 - Evidence Attribution
Traces which retrieved evidence sentences influenced each part of generated summary
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


class EvidenceAttributor:
    """Attribute generated text to retrieved evidence using embeddings"""

    def __init__(self, use_medical=True):
        self.config = RAGConfig
        self.use_medical = use_medical

        # Paths - Updated to use RAG outputs directory
        base_path = Path(__file__).parent.parent / "RAG1" / "outputs2"
        self.outputs_dir = base_path
        self.embedding_model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
        self.corpus_path = self.config.TRAIN_VAL_CORPUS

        self.explainability_dir = Path(__file__).parent / "outputs"
        if use_medical:
            self.explainability_dir = self.explainability_dir / "medical_rag"
        else:
            self.explainability_dir = self.explainability_dir / "standard_rag"
        self.explainability_dir.mkdir(parents=True, exist_ok=True)

        self.summaries_path = self.outputs_dir / "rag_summaries_hybrid.json"
        self.retrieval_logs_path = self.outputs_dir / "retrieval_logs.json"

        self.embedding_model = None
        self.summaries = None
        self.retrieval_logs = None
        self.corpus_df = None

    def load_data(self):
        """Load RAG results"""
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

        self.corpus_df = pd.read_csv(self.corpus_path)
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

    def extract_sections(self, summary_text):
        """Extract 11 sections"""
        sections = {}
        patterns = [
            ("Case Type", r"Case Type:\s*(.+?)(?=\n•|$)"),
            ("Patient & Service", r"•\s*Patient & Service:\s*(.+?)(?=\n•|$)"),
            ("Chief Complaint", r"•\s*Chief Complaint[^:]*:\s*(.+?)(?=\n•|$)"),
            ("HPI", r"•\s*History of Present Illness[^:]*:\s*(.+?)(?=\n•|$)"),
            ("Past Medical", r"•\s*Past Medical[^:]*:\s*(.+?)(?=\n•|$)"),
            ("Medications", r"•\s*Medications[^:]*:(.+?)(?=\n•\s*(?!Discharge|Ongoing)|$)"),
            ("Physical Exam", r"•\s*Physical Examination[^:]*:\s*(.+?)(?=\n•|$)"),
            ("Investigations", r"•\s*Investigations[^:]*:\s*(.+?)(?=\n•|$)"),
            ("Assessment", r"•\s*Assessment[^:]*:\s*(.+?)(?=\n•|$)"),
            ("Discharge Condition", r"•\s*Discharge Condition:\s*(.+?)(?=\n•|$)"),
            ("Follow-Up", r"•\s*Follow-Up[^:]*:(.+?)$")
        ]

        for name, pattern in patterns:
            match = re.search(pattern, summary_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[name] = match.group(1).strip()
        return sections

    def compute_sentence_attribution(self, generated_sentence, retrieved_docs):
        """Compute similarity between generated sentence and retrieved evidence"""
        retrieved_sentences = []
        for doc in retrieved_docs:
            doc_text = doc.get('structured_target', doc.get('target', ''))
            sentences = re.split(r'[.!?]\s+', doc_text)
            for sent in sentences:
                if len(sent.strip()) > 20:
                    retrieved_sentences.append({
                        'text': sent.strip(),
                        'doc_id': doc['note_id'],
                        'retrieval_score': doc.get('retrieval_score', 0)
                    })

        if not retrieved_sentences or not generated_sentence.strip():
            return []

        gen_embedding = self.embedding_model.encode([generated_sentence])
        ret_embeddings = self.embedding_model.encode([s['text'] for s in retrieved_sentences])
        similarities = cosine_similarity(gen_embedding, ret_embeddings)[0]

        for sent, sim in zip(retrieved_sentences, similarities):
            sent['similarity'] = float(sim)

        return sorted(retrieved_sentences, key=lambda x: x['similarity'], reverse=True)[:5]

    def explain_summary(self, idx):
        """Generate evidence attribution for single summary"""
        summary = self.summaries[idx]
        retrieval_log = self.retrieval_logs[idx]

        retrieved_docs = []
        for ret_id, score in zip(retrieval_log['retrieved_note_ids'], retrieval_log['retrieval_scores']):
            doc = self.corpus_df[self.corpus_df['note_id'] == ret_id].iloc[0].to_dict()
            doc['retrieval_score'] = score
            retrieved_docs.append(doc)

        sections = self.extract_sections(summary['generated_summary'])
        section_attributions = {}

        for section_name, section_text in sections.items():
            sentences = re.split(r'[.!?]\s+', section_text)
            sentence_attributions = []
            for sent in sentences:
                if len(sent.strip()) > 10:
                    evidence = self.compute_sentence_attribution(sent, retrieved_docs)
                    sentence_attributions.append({'sentence': sent.strip(), 'evidence': evidence})
            section_attributions[section_name] = sentence_attributions

        return {
            'note_id': summary['note_id'],
            'retrieved_cases': [{'note_id': d['note_id'], 'score': d['retrieval_score']} for d in retrieved_docs],
            'section_attributions': section_attributions
        }

    def visualize_heatmap(self, explanation, output_path):
        """Create attribution heatmap"""
        sections = list(explanation['section_attributions'].keys())
        cases = [c['note_id'] for c in explanation['retrieved_cases']]

        matrix = np.zeros((len(sections), len(cases)))
        for i, section in enumerate(sections):
            doc_sims = {case: [] for case in cases}
            for sent_attr in explanation['section_attributions'][section]:
                for ev in sent_attr['evidence']:
                    doc_sims[ev['doc_id']].append(ev['similarity'])
            for j, case in enumerate(cases):
                if doc_sims[case]:
                    matrix[i, j] = np.mean(doc_sims[case])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[f"Case {i+1}" for i in range(len(cases))],
                   yticklabels=sections, ax=ax, cbar_kws={'label': 'Similarity'})
        ax.set_title(f'Evidence Attribution - Note {explanation["note_id"]}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_analysis(self, num_samples=None, num_visualize=5):
        """Run full evidence attribution analysis"""
        print("=" * 80)
        print("Evidence Attribution Analysis")
        print("=" * 80)

        self.load_data()
        self.load_embedding_model()

        if num_samples:
            self.summaries = self.summaries[:num_samples]
            self.retrieval_logs = self.retrieval_logs[:num_samples]

        explanations = []
        print(f"\nAnalyzing {len(self.summaries)} summaries...")
        for idx in tqdm(range(len(self.summaries))):
            exp = self.explain_summary(idx)
            explanations.append(exp)

            if idx < num_visualize:
                viz_path = self.explainability_dir / f"heatmap_{exp['note_id']}.png"
                self.visualize_heatmap(exp, viz_path)

        # Save all explanations
        output_path = self.explainability_dir / "evidence_attributions.json"
        with open(output_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        print(f"\n✓ Saved: {output_path}")

        return explanations


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--medical', action='store_true', default=True, help='Use medical RAG (default: True)')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--visualize', type=int, default=5)
    args = parser.parse_args()

    attributor = EvidenceAttributor(use_medical=args.medical)
    attributor.run_analysis(num_samples=args.samples, num_visualize=args.visualize)


if __name__ == "__main__":
    main()
