"""
Phase 2 - Step 4: Evaluate RAG Performance
Compares Phase 1 (baseline) vs Phase 2 (RAG) for hallucination reduction
Includes all Phase 1 metrics plus RAG-specific metrics
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from datetime import datetime
import re
import time

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not available. Install with: pip install bert-score")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spacy not available. Install with: pip install spacy")

from config import RAGConfig


class RAGEvaluator:
    """Evaluate RAG performance vs baseline with comprehensive metrics"""

    def __init__(self):
        self.config = RAGConfig
        self.rag_results = None
        self.test_df = None
        self.metrics = defaultdict(list)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if ROUGE_AVAILABLE else None
        self.smoothing = SmoothingFunction().method1 if BLEU_AVAILABLE else None
        
        # Load spacy model for entity extraction if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None

    def parse_txt_file(self, txt_path):
        """Parse TXT file to extract input, target, and generated summaries"""
        with open(txt_path, 'r') as f:
            content = f.read()

        results = []

        # Split by sample separators
        samples = re.split(r'={80}\nSAMPLE \d+: (.+?)\n={80}', content)

        # Skip the header (first element)
        for i in range(1, len(samples), 2):
            if i + 1 >= len(samples):
                break

            note_id = samples[i].strip()
            sample_content = samples[i + 1]

            # Extract INPUT NOTE
            input_match = re.search(r'INPUT NOTE \(full\):\n─{80}\n(.*?)\n\n─{80}', sample_content, re.DOTALL)
            input_text = input_match.group(1).strip() if input_match else ""

            # Extract TARGET SUMMARY
            target_match = re.search(r'TARGET SUMMARY:\n─{80}\n(.*?)\n\n─{80}', sample_content, re.DOTALL)
            target_text = target_match.group(1).strip() if target_match else ""

            # Extract GENERATED SUMMARY
            generated_match = re.search(r'GENERATED SUMMARY \(RAG-Enhanced\):\n─{80}\n(.*?)\n\n─{80}', sample_content, re.DOTALL)
            generated_text = generated_match.group(1).strip() if generated_match else ""

            # Extract retrieval info
            retrieval_match = re.search(r'Retrieved (\d+) structurally similar cases:', sample_content)
            num_retrieved = int(retrieval_match.group(1)) if retrieval_match else 0

            results.append({
                'note_id': note_id,
                'input': input_text,
                'structured_target': target_text,
                'generated_summary': generated_text,
                'num_retrieved': num_retrieved,
                'retrieved_cases': [{'note_id': f'case_{j}', 'rerank_score': 0.0} for j in range(num_retrieved)]
            })

        return results

    def load_data(self):
        """Load RAG results and test set"""
        print("Loading evaluation data...")

        # Load RAG results from TXT file
        txt_path = self.config.RAG_SUMMARIES_TXT_PATH
        print(f"  Reading from TXT file: {txt_path}")
        self.rag_results = self.parse_txt_file(txt_path)
        print(f"  ✓ Loaded {len(self.rag_results)} RAG summaries")

        # Load test set
        self.test_df = pd.read_csv(self.config.TEST_SET)
        print(f"  ✓ Loaded {len(self.test_df)} test samples")

    def compute_rouge_scores(self, generated, reference):
        """Compute ROUGE scores (matching Phase 1 format)"""
        if not ROUGE_AVAILABLE:
            return {}

        scores = self.rouge_scorer.score(reference, generated)

        return {
            'rouge1': scores['rouge1'].fmeasure,  # Match Phase 1 format
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            # Detailed scores for analysis
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }

    def compute_bleu_score(self, generated, reference):
        """Compute BLEU score"""
        if not BLEU_AVAILABLE:
            return 0.0

        # Tokenize
        reference_tokens = reference.split()
        generated_tokens = generated.split()

        # Compute BLEU with smoothing
        smoothie = SmoothingFunction().method4
        score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothie)

        return score

    def compute_length_metrics(self, generated, reference):
        """Compute length-based metrics"""
        gen_len = len(generated.split())
        ref_len = len(reference.split())

        return {
            'generated_length': gen_len,
            'reference_length': ref_len,
            'length_ratio': gen_len / ref_len if ref_len > 0 else 0,
            'length_diff': abs(gen_len - ref_len)
        }

    def compute_coverage_metrics(self, generated, reference, input_text):
        """Compute information coverage metrics"""
        # Simple word-level coverage
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        input_words = set(input_text.lower().split())

        # Coverage: how much of reference is in generated
        coverage = len(ref_words & gen_words) / len(ref_words) if len(ref_words) > 0 else 0

        # Hallucination proxy: words in generated but not in input or reference
        # (This is a simple heuristic, not perfect)
        potential_hallucinations = gen_words - (input_words | ref_words)
        hallucination_rate = len(potential_hallucinations) / len(gen_words) if len(gen_words) > 0 else 0

        return {
            'coverage': coverage,
            'hallucination_rate': hallucination_rate,
            'unique_words_generated': len(gen_words),
            'unique_words_reference': len(ref_words)
        }

    def compute_bertscore_metrics(self, generated, reference):
        """Compute BERTScore metrics (matching Phase 1)"""
        if not BERTSCORE_AVAILABLE:
            return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
        
        try:
            P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
            return {
                'bertscore_precision': float(P[0]),
                'bertscore_recall': float(R[0]),
                'bertscore_f1': float(F1[0])
            }
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}

    def compute_meteor_score(self, generated, reference):
        """Compute METEOR score (matching Phase 1)"""
        if not BLEU_AVAILABLE:
            return 0.0
        
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            nltk.download('wordnet', quiet=True)
            
            # Tokenize
            gen_tokens = generated.lower().split()
            ref_tokens = reference.lower().split()
            
            return meteor_score([ref_tokens], gen_tokens)
        except Exception as e:
            print(f"Warning: METEOR computation failed: {e}")
            return 0.0

    def compute_structure_metrics(self, generated):
        """Compute structure completeness metrics (matching Phase 1)"""
        # Count bullet points
        bullet_count = len(re.findall(r'[•\-\*]\s+', generated))
        
        # Check for common medical sections
        sections = [
            'chief complaint', 'history', 'physical exam', 'assessment', 
            'plan', 'discharge', 'medication', 'follow-up', 'diagnosis'
        ]
        found_sections = sum(1 for section in sections if section.lower() in generated.lower())
        
        # Structure completeness (simplified)
        structure_complete = 1.0 if bullet_count > 5 and found_sections >= 3 else 0.0
        section_coverage = found_sections / len(sections)
        
        return {
            'bullet_count': bullet_count,
            'section_coverage': section_coverage,
            'structure_complete': structure_complete,
            'avg_bullet_count': bullet_count,  # Match Phase 1 format
            'section_coverage_pct': section_coverage * 100,
            'structure_complete_pct': structure_complete * 100
        }

    def compute_entity_metrics(self, generated, reference):
        """Compute entity extraction metrics (matching Phase 1)"""
        if not self.nlp:
            return {'entity_f1': 0.0, 'entity_coverage': 0.0, 'medication_f1': 0.0}
        
        try:
            # Extract entities from both texts
            gen_doc = self.nlp(generated)
            ref_doc = self.nlp(reference)
            
            # Get all entities
            gen_entities = set((ent.text.lower(), ent.label_) for ent in gen_doc.ents)
            ref_entities = set((ent.text.lower(), ent.label_) for ent in ref_doc.ents)
            
            # Compute F1 for all entities
            if len(ref_entities) == 0:
                entity_f1 = 1.0 if len(gen_entities) == 0 else 0.0
            else:
                precision = len(gen_entities & ref_entities) / len(gen_entities) if len(gen_entities) > 0 else 0
                recall = len(gen_entities & ref_entities) / len(ref_entities)
                entity_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Medication-specific F1
            gen_meds = set(ent.text.lower() for ent in gen_doc.ents if ent.label_ in ['DRUG', 'CHEMICAL'])
            ref_meds = set(ent.text.lower() for ent in ref_doc.ents if ent.label_ in ['DRUG', 'CHEMICAL'])
            
            if len(ref_meds) == 0:
                medication_f1 = 1.0 if len(gen_meds) == 0 else 0.0
            else:
                med_precision = len(gen_meds & ref_meds) / len(gen_meds) if len(gen_meds) > 0 else 0
                med_recall = len(gen_meds & ref_meds) / len(ref_meds)
                medication_f1 = 2 * med_precision * med_recall / (med_precision + med_recall) if (med_precision + med_recall) > 0 else 0
            
            # Entity coverage
            entity_coverage = len(gen_entities & ref_entities) / len(ref_entities) if len(ref_entities) > 0 else 0
            
            return {
                'entity_f1': entity_f1,
                'entity_coverage': entity_coverage,
                'medication_f1': medication_f1
            }
        except Exception as e:
            print(f"Warning: Entity extraction failed: {e}")
            return {'entity_f1': 0.0, 'entity_coverage': 0.0, 'medication_f1': 0.0}

    def compute_clinical_bert_similarity(self, generated, reference):
        """Compute clinical BERT similarity (simplified version)"""
        # This is a simplified version - in practice you'd use a clinical BERT model
        # For now, we'll use word overlap as a proxy
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if len(ref_words) == 0:
            return 1.0 if len(gen_words) == 0 else 0.0
        
        overlap = len(gen_words & ref_words)
        similarity = overlap / len(ref_words)
        return similarity

    def compute_factual_consistency(self, generated, reference, input_text):
        """Compute factual consistency (simplified version)"""
        # This is a simplified heuristic - in practice you'd use a more sophisticated model
        # Check for contradictions between generated and input/reference
        
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        input_words = set(input_text.lower().split())
        
        # Words in generated that contradict input/reference
        contradictory_words = gen_words - (ref_words | input_words)
        
        # Simple consistency score
        if len(gen_words) == 0:
            return 0.0
        
        consistency = 1.0 - (len(contradictory_words) / len(gen_words))
        return max(0.0, consistency)

    def compute_rag_specific_metrics(self, result):
        """Compute RAG-specific evaluation metrics"""
        generated = result['generated_summary']
        input_text = result.get('input', result.get('input_note', ''))
        num_retrieved = result.get('num_retrieved', len(result.get('retrieved_cases', [])))
        
        # Retrieval utilization
        retrieval_utilization = min(1.0, num_retrieved / 5.0)  # Assuming max 5 retrieved
        
        # Context relevance (simplified - check if generated content relates to input)
        gen_words = set(generated.lower().split())
        input_words = set(input_text.lower().split())
        
        # Medical term overlap
        medical_terms = ['patient', 'diagnosis', 'treatment', 'medication', 'procedure', 
                        'surgery', 'discharge', 'follow-up', 'condition', 'symptom']
        gen_medical = gen_words & set(medical_terms)
        input_medical = input_words & set(medical_terms)
        
        medical_relevance = len(gen_medical & input_medical) / len(input_medical) if len(input_medical) > 0 else 0
        
        # Information density
        info_density = len(gen_words) / len(input_words) if len(input_words) > 0 else 0
        
        return {
            'retrieval_utilization': retrieval_utilization,
            'medical_relevance': medical_relevance,
            'information_density': info_density,
            'num_retrieved': num_retrieved
        }

    def evaluate_sample(self, result):
        """Evaluate a single RAG result with comprehensive metrics"""
        generated = result['generated_summary']
        # Use structured_target (from CSV) saved as target_summary (in JSON) as reference
        # This is the golden 11-section summary from the test set
        reference = result.get('structured_target', result.get('target_summary', result.get('target', '')))
        input_text = result.get('input', result.get('input_note', ''))

        sample_metrics = {}

        # ROUGE scores - compare against structured_target (golden 11-section summary)
        if ROUGE_AVAILABLE:
            rouge_scores = self.compute_rouge_scores(generated, reference)
            sample_metrics.update(rouge_scores)

        # BLEU score - compare against structured_target
        if BLEU_AVAILABLE:
            bleu_score = self.compute_bleu_score(generated, reference)
            sample_metrics['bleu'] = bleu_score

        # METEOR score - compare against structured_target
        meteor_score = self.compute_meteor_score(generated, reference)
        sample_metrics['meteor'] = meteor_score

        # BERTScore metrics - compare against structured_target
        bertscore_metrics = self.compute_bertscore_metrics(generated, reference)
        sample_metrics.update(bertscore_metrics)

        # Length metrics
        length_metrics = self.compute_length_metrics(generated, reference)
        sample_metrics.update(length_metrics)

        # Structure metrics
        structure_metrics = self.compute_structure_metrics(generated)
        sample_metrics.update(structure_metrics)

        # Entity metrics - compare against structured_target
        entity_metrics = self.compute_entity_metrics(generated, reference)
        sample_metrics.update(entity_metrics)

        # Clinical BERT similarity - compare against structured_target
        clinical_similarity = self.compute_clinical_bert_similarity(generated, reference)
        sample_metrics['clinical_bert_similarity'] = clinical_similarity

        # Factual consistency
        factual_consistency = self.compute_factual_consistency(generated, reference, input_text)
        sample_metrics['factual_consistency'] = factual_consistency

        # Coverage metrics - compare against structured_target
        coverage_metrics = self.compute_coverage_metrics(generated, reference, input_text)
        sample_metrics.update(coverage_metrics)

        # RAG-specific metrics
        rag_metrics = self.compute_rag_specific_metrics(result)
        sample_metrics.update(rag_metrics)

        return sample_metrics

    def evaluate_all(self):
        """Evaluate all RAG results"""
        print("\n" + "=" * 80)
        print("Evaluating RAG Results")
        print("=" * 80)

        all_metrics = []

        for result in self.rag_results:
            metrics = self.evaluate_sample(result)
            metrics['note_id'] = result['note_id']
            all_metrics.append(metrics)

        return all_metrics

    def aggregate_metrics(self, all_metrics):
        """Aggregate metrics across all samples"""
        print("\nAggregating metrics...")

        aggregated = {}

        # Get all metric keys (excluding note_id)
        metric_keys = [k for k in all_metrics[0].keys() if k != 'note_id']

        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }

        return aggregated

    def print_results(self, aggregated):
        """Print comprehensive evaluation results"""
        print("\n" + "=" * 80)
        print("Phase 2 RAG Evaluation Results (Comprehensive)")
        print("=" * 80)

        # Basic metrics (matching Phase 1 format)
        print(f"\n📊 Basic Metrics:")
        print(f"  Number of samples: {len(self.rag_results)}")
        
        if 'structure_complete_pct' in aggregated:
            print(f"  Structure complete: {aggregated['structure_complete_pct']['mean']:.1f}%")
        if 'section_coverage_pct' in aggregated:
            print(f"  Section coverage: {aggregated['section_coverage_pct']['mean']:.1f}%")
        if 'avg_bullet_count' in aggregated:
            print(f"  Avg bullet count: {aggregated['avg_bullet_count']['mean']:.1f}")
        if 'avg_length' in aggregated:
            print(f"  Avg length: {aggregated['avg_length']['mean']:.1f} words")

        # ROUGE scores (matching Phase 1)
        if 'rouge1' in aggregated:
            print(f"\n🎯 ROUGE Scores:")
            print(f"  ROUGE-1: {aggregated['rouge1']['mean']:.4f} ± {aggregated['rouge1']['std']:.4f}")
            print(f"  ROUGE-2: {aggregated['rouge2']['mean']:.4f} ± {aggregated['rouge2']['std']:.4f}")
            print(f"  ROUGE-L: {aggregated['rougeL']['mean']:.4f} ± {aggregated['rougeL']['std']:.4f}")

        # BLEU and METEOR
        if 'bleu' in aggregated:
            print(f"\n📝 Generation Quality:")
            print(f"  BLEU: {aggregated['bleu']['mean']:.4f} ± {aggregated['bleu']['std']:.4f}")
        if 'meteor' in aggregated:
            print(f"  METEOR: {aggregated['meteor']['mean']:.4f} ± {aggregated['meteor']['std']:.4f}")

        # BERTScore
        if 'bertscore_f1' in aggregated:
            print(f"\n🧠 Semantic Similarity (BERTScore):")
            print(f"  Precision: {aggregated['bertscore_precision']['mean']:.4f} ± {aggregated['bertscore_precision']['std']:.4f}")
            print(f"  Recall: {aggregated['bertscore_recall']['mean']:.4f} ± {aggregated['bertscore_recall']['std']:.4f}")
            print(f"  F1: {aggregated['bertscore_f1']['mean']:.4f} ± {aggregated['bertscore_f1']['std']:.4f}")

        # Entity metrics
        if 'entity_f1' in aggregated:
            print(f"\n🏥 Medical Entity Extraction:")
            print(f"  Entity F1: {aggregated['entity_f1']['mean']:.4f} ± {aggregated['entity_f1']['std']:.4f}")
            print(f"  Entity Coverage: {aggregated['entity_coverage']['mean']:.4f} ± {aggregated['entity_coverage']['std']:.4f}")
            print(f"  Medication F1: {aggregated['medication_f1']['mean']:.4f} ± {aggregated['medication_f1']['std']:.4f}")

        # Clinical similarity and consistency
        if 'clinical_bert_similarity' in aggregated:
            print(f"\n🔬 Clinical Quality:")
            print(f"  Clinical BERT Similarity: {aggregated['clinical_bert_similarity']['mean']:.4f} ± {aggregated['clinical_bert_similarity']['std']:.4f}")
        if 'factual_consistency' in aggregated:
            print(f"  Factual Consistency: {aggregated['factual_consistency']['mean']:.4f} ± {aggregated['factual_consistency']['std']:.4f}")

        # Hallucination analysis
        if 'coverage' in aggregated:
            print(f"\n🚫 Hallucination Analysis:")
            print(f"  Coverage: {aggregated['coverage']['mean']:.4f} ± {aggregated['coverage']['std']:.4f}")
            print(f"  Hallucination Rate: {aggregated['hallucination_rate']['mean']:.4f} ± {aggregated['hallucination_rate']['std']:.4f}")

        # RAG-specific metrics
        if 'retrieval_utilization' in aggregated:
            print(f"\n🔍 RAG-Specific Metrics:")
            print(f"  Retrieval Utilization: {aggregated['retrieval_utilization']['mean']:.4f} ± {aggregated['retrieval_utilization']['std']:.4f}")
            print(f"  Medical Relevance: {aggregated['medical_relevance']['mean']:.4f} ± {aggregated['medical_relevance']['std']:.4f}")
            print(f"  Information Density: {aggregated['information_density']['mean']:.4f} ± {aggregated['information_density']['std']:.4f}")
            print(f"  Avg Retrieved Cases: {aggregated['num_retrieved']['mean']:.1f}")

        print("\n" + "=" * 80)

    def convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_results(self, all_metrics, aggregated):
        """Save evaluation results"""
        results = {
            'evaluation_date': datetime.now().isoformat(),
            'num_samples': len(all_metrics),
            'aggregated_metrics': aggregated,
            'per_sample_metrics': all_metrics,
            'config': {
                'embedding_model': self.config.EMBEDDING_MODEL_NAME,
                'reranker_model': self.config.RERANKER_MODEL_NAME,
                'dense_top_k': self.config.DENSE_TOP_K,
                'rerank_top_k': self.config.RERANK_TOP_K
            }
        }

        # Convert any NumPy types to Python native types for JSON serialization
        results = self.convert_numpy_types(results)

        # Save with unique filename to avoid overwriting existing results
        output_path = self.config.RAG_OUTPUTS_DIR / "evaluation_results_from_txt.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Evaluation results saved: {output_path}")

    def create_summary_report(self, aggregated):
        """Create human-readable summary report"""
        report_path = self.config.RAG_OUTPUTS_DIR / "evaluation_summary_from_txt.md"

        with open(report_path, 'w') as f:
            f.write("# Phase 2 RAG Evaluation Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Samples**: {len(self.rag_results)}\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- **Embedding Model**: {self.config.EMBEDDING_MODEL_NAME}\n")
            f.write(f"- **Reranker Model**: {self.config.RERANKER_MODEL_NAME}\n")
            f.write(f"- **Retrieval Strategy**: Dense ({self.config.DENSE_TOP_K}) → Rerank ({self.config.RERANK_TOP_K})\n\n")

            if ROUGE_AVAILABLE:
                f.write("## ROUGE Scores\n\n")
                f.write("| Metric | Mean | Std | Min | Max |\n")
                f.write("|--------|------|-----|-----|-----|\n")
                f.write(f"| ROUGE-1 F1 | {aggregated['rouge1_f']['mean']:.4f} | {aggregated['rouge1_f']['std']:.4f} | {aggregated['rouge1_f']['min']:.4f} | {aggregated['rouge1_f']['max']:.4f} |\n")
                f.write(f"| ROUGE-2 F1 | {aggregated['rouge2_f']['mean']:.4f} | {aggregated['rouge2_f']['std']:.4f} | {aggregated['rouge2_f']['min']:.4f} | {aggregated['rouge2_f']['max']:.4f} |\n")
                f.write(f"| ROUGE-L F1 | {aggregated['rougeL_f']['mean']:.4f} | {aggregated['rougeL_f']['std']:.4f} | {aggregated['rougeL_f']['min']:.4f} | {aggregated['rougeL_f']['max']:.4f} |\n\n")

            if BLEU_AVAILABLE:
                f.write("## BLEU Score\n\n")
                f.write(f"- **BLEU**: {aggregated['bleu']['mean']:.4f} ± {aggregated['bleu']['std']:.4f}\n\n")

            f.write("## Length Analysis\n\n")
            f.write(f"- **Average Generated Length**: {aggregated['generated_length']['mean']:.1f} words\n")
            f.write(f"- **Average Reference Length**: {aggregated['reference_length']['mean']:.1f} words\n")
            f.write(f"- **Length Ratio**: {aggregated['length_ratio']['mean']:.2f}\n\n")

            f.write("## Hallucination Analysis\n\n")
            f.write(f"- **Coverage**: {aggregated['coverage']['mean']:.4f} (how much of reference is captured)\n")
            f.write(f"- **Hallucination Rate**: {aggregated['hallucination_rate']['mean']:.4f} (lower is better)\n\n")

            f.write("## Interpretation\n\n")
            f.write("- **Coverage**: Measures how well the generated summary covers the reference content.\n")
            f.write("- **Hallucination Rate**: Proxy metric for potential hallucinations (words in generated but not in input/reference).\n")
            f.write("- **Note**: This is a simplified metric. Manual review is recommended for clinical validation.\n\n")

        print(f"✓ Summary report saved: {report_path}")


def main():
    """Main execution"""
    print("=" * 80)
    print("Phase 2 - Step 4: RAG Evaluation")
    print("=" * 80)

    evaluator = RAGEvaluator()

    # Load data
    evaluator.load_data()

    # Evaluate all samples
    all_metrics = evaluator.evaluate_all()

    # Aggregate metrics
    aggregated = evaluator.aggregate_metrics(all_metrics)

    # Print results
    evaluator.print_results(aggregated)

    # Save results
    evaluator.save_results(all_metrics, aggregated)

    # Create summary report
    evaluator.create_summary_report(aggregated)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)
    print(f"\nResults saved to: {RAGConfig.RAG_OUTPUTS_DIR}")

if __name__ == "__main__":
    main()
