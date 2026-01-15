#!/usr/bin/env python3
"""
Advanced Medical Evaluation Metrics

Includes:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- METEOR
- BERTScore
- BLEU Score
- Factual Consistency (NLI-based)
- Medical Entity F1 (Precision/Recall for medical terms)
- Entity Coverage
- Medication Accuracy
- Readability (Flesch-Kincaid)
- Clinical Semantic Similarity (ClinicalBERT)
- Hallucination Analysis
- RAG-Specific Metrics
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    print("⚠️  BLEU not available. Install: pip install nltk")
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("⚠️  ROUGE not available. Install: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score
    import nltk
    METEOR_AVAILABLE = True
except ImportError:
    print("⚠️  METEOR not available. Install: pip install nltk")
    METEOR_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not available. Install: pip install bert-score")
    BERTSCORE_AVAILABLE = False

try:
    import textstat
    READABILITY_AVAILABLE = True
except ImportError:
    print("⚠️  Readability not available. Install: pip install textstat")
    READABILITY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SEMANTIC_AVAILABLE = True
    # Load ClinicalBERT model (lazy loading)
    _clinical_bert_model = None
except ImportError:
    print("⚠️  Sentence-transformers not available. Install: pip install sentence-transformers")
    SEMANTIC_AVAILABLE = False

try:
    from transformers import pipeline
    NLI_AVAILABLE = True
    # Lazy loading for NLI model
    _nli_model = None
except ImportError:
    print("⚠️  NLI not available. transformers already installed, but may need specific model")
    NLI_AVAILABLE = False


class MedicalMetricsCalculator:
    """Calculate advanced medical evaluation metrics"""

    def __init__(self):
        """Initialize medical metrics calculator"""
        self.clinical_bert = None
        self.nli_model = None

        # Medical entity patterns (simplified - can be expanded)
        self.medication_patterns = [
            r'\b[A-Z][a-z]+(?:mycin|cillin|pril|olol|azole|statin|oxin)\b',  # Drug suffixes
            r'\b(?:aspirin|metformin|insulin|warfarin|heparin|morphine|acetaminophen)\b',  # Common drugs
        ]

        self.diagnosis_patterns = [
            r'\b(?:hypertension|diabetes|pneumonia|sepsis|failure|disease|syndrome|infection)\b',
            r'\b(?:HTN|DM|CHF|COPD|CAD|MI|CVA|PE|DVT)\b',  # Abbreviations
        ]

        self.procedure_patterns = [
            r'\b(?:surgery|procedure|operation|catheterization|intubation|resection)\b',
            r'\b(?:CT|MRI|X-ray|ultrasound|EKG|ECG|echo)\b',  # Imaging
        ]

    def get_clinical_bert(self):
        """Lazy load ClinicalBERT model"""
        if not SEMANTIC_AVAILABLE:
            return None

        if self.clinical_bert is None:
            try:
                print("  Loading ClinicalBERT model (first time only)...")
                self.clinical_bert = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
                print("  ✓ ClinicalBERT loaded")
            except Exception as e:
                print(f"  ⚠️  ClinicalBERT failed to load: {e}")
                return None
        return self.clinical_bert

    def get_nli_model(self):
        """Lazy load NLI model"""
        if not NLI_AVAILABLE:
            return None

        if self.nli_model is None:
            try:
                print("  Loading NLI model for factual consistency (first time only)...")
                # Use a proper NLI model (trained on MNLI/SNLI datasets)
                self.nli_model = pipeline("text-classification",
                                         model="cross-encoder/nli-deberta-v3-small",
                                         device=0 if torch.cuda.is_available() else -1)
                print("  ✓ NLI model loaded")
            except Exception as e:
                print(f"  ⚠️  NLI model failed to load: {e}")
                print(f"     Trying fallback NLI model...")
                try:
                    # Fallback to another NLI model
                    self.nli_model = pipeline("zero-shot-classification",
                                             model="facebook/bart-large-mnli",
                                             device=0 if torch.cuda.is_available() else -1)
                    print("  ✓ Fallback NLI model loaded")
                except Exception as e2:
                    print(f"  ⚠️  Fallback NLI model also failed: {e2}")
                    return None
        return self.nli_model

    def extract_medical_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract medical entities using regex patterns"""
        entities = {
            'medications': set(),
            'diagnoses': set(),
            'procedures': set()
        }

        text_lower = text.lower()

        # Extract medications
        for pattern in self.medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['medications'].update([m.lower() for m in matches])

        # Extract diagnoses
        for pattern in self.diagnosis_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['diagnoses'].update([m.lower() for m in matches])

        # Extract procedures
        for pattern in self.procedure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['procedures'].update([m.lower() for m in matches])

        return entities

    def calculate_rouge(self, generated: str, target: str) -> Dict[str, float]:
        """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)"""
        if not ROUGE_AVAILABLE:
            return {
                'rouge1_precision': 0.0,
                'rouge1_recall': 0.0,
                'rouge1_f1': 0.0,
                'rouge2_precision': 0.0,
                'rouge2_recall': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_precision': 0.0,
                'rougeL_recall': 0.0,
                'rougeL_f1': 0.0,
            }

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(target, generated)

            return {
                'rouge1_precision': scores['rouge1'].precision,
                'rouge1_recall': scores['rouge1'].recall,
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_precision': scores['rouge2'].precision,
                'rouge2_recall': scores['rouge2'].recall,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_precision': scores['rougeL'].precision,
                'rougeL_recall': scores['rougeL'].recall,
                'rougeL_f1': scores['rougeL'].fmeasure,
            }
        except Exception as e:
            print(f"  ⚠️  ROUGE calculation failed: {e}")
            return {
                'rouge1_precision': 0.0,
                'rouge1_recall': 0.0,
                'rouge1_f1': 0.0,
                'rouge2_precision': 0.0,
                'rouge2_recall': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_precision': 0.0,
                'rougeL_recall': 0.0,
                'rougeL_f1': 0.0,
            }

    def calculate_meteor(self, generated: str, target: str) -> float:
        """Calculate METEOR score"""
        if not METEOR_AVAILABLE:
            return 0.0

        try:
            # Ensure NLTK wordnet is downloaded
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("  Downloading NLTK wordnet for METEOR...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)

            reference = [target.split()]
            hypothesis = generated.split()
            return meteor_score(reference, hypothesis)
        except Exception as e:
            print(f"  ⚠️  METEOR calculation failed: {e}")
            return 0.0

    def calculate_bertscore(self, generated: str, target: str) -> Dict[str, float]:
        """Calculate BERTScore (Precision, Recall, F1)"""
        if not BERTSCORE_AVAILABLE:
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
            }

        try:
            P, R, F1 = bert_score([generated], [target], lang="en", verbose=False)
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item(),
            }
        except Exception as e:
            print(f"  ⚠️  BERTScore calculation failed: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
            }

    def calculate_bleu(self, generated: str, target: str) -> float:
        """Calculate BLEU score"""
        if not BLEU_AVAILABLE:
            return 0.0

        try:
            reference = [target.split()]
            hypothesis = generated.split()
            smoothing = SmoothingFunction().method1
            return sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
        except Exception as e:
            print(f"  ⚠️  BLEU calculation failed: {e}")
            return 0.0

    def calculate_entity_f1(self, generated: str, target: str) -> Dict[str, float]:
        """Calculate F1 for medical entities"""
        gen_entities = self.extract_medical_entities(generated)
        tgt_entities = self.extract_medical_entities(target)

        results = {}

        for entity_type in ['medications', 'diagnoses', 'procedures']:
            gen_set = gen_entities[entity_type]
            tgt_set = tgt_entities[entity_type]

            if len(tgt_set) == 0:
                results[f'{entity_type}_precision'] = 1.0 if len(gen_set) == 0 else 0.0
                results[f'{entity_type}_recall'] = 1.0
                results[f'{entity_type}_f1'] = 1.0 if len(gen_set) == 0 else 0.0
                continue

            # Calculate precision, recall, F1
            true_positives = len(gen_set & tgt_set)
            precision = true_positives / len(gen_set) if len(gen_set) > 0 else 0.0
            recall = true_positives / len(tgt_set) if len(tgt_set) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[f'{entity_type}_precision'] = precision
            results[f'{entity_type}_recall'] = recall
            results[f'{entity_type}_f1'] = f1

        # Overall entity metrics
        all_gen = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']
        all_tgt = tgt_entities['medications'] | tgt_entities['diagnoses'] | tgt_entities['procedures']

        if len(all_tgt) > 0:
            tp = len(all_gen & all_tgt)
            precision = tp / len(all_gen) if len(all_gen) > 0 else 0.0
            recall = tp / len(all_tgt) if len(all_tgt) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results['overall_entity_precision'] = precision
            results['overall_entity_recall'] = recall
            results['overall_entity_f1'] = f1
        else:
            results['overall_entity_precision'] = 1.0 if len(all_gen) == 0 else 0.0
            results['overall_entity_recall'] = 1.0
            results['overall_entity_f1'] = 1.0 if len(all_gen) == 0 else 0.0

        return results

    def calculate_entity_coverage(self, generated: str, input_note: str) -> Dict[str, float]:
        """Calculate what % of input entities are covered in generated summary"""
        gen_entities = self.extract_medical_entities(generated)
        input_entities = self.extract_medical_entities(input_note)

        coverage = {}

        for entity_type in ['medications', 'diagnoses', 'procedures']:
            gen_set = gen_entities[entity_type]
            input_set = input_entities[entity_type]

            if len(input_set) == 0:
                coverage[f'{entity_type}_coverage'] = 1.0
            else:
                covered = len(gen_set & input_set)
                coverage[f'{entity_type}_coverage'] = covered / len(input_set)

        # Overall coverage
        all_gen = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']
        all_input = input_entities['medications'] | input_entities['diagnoses'] | input_entities['procedures']

        if len(all_input) > 0:
            coverage['overall_entity_coverage'] = len(all_gen & all_input) / len(all_input)
        else:
            coverage['overall_entity_coverage'] = 1.0

        return coverage

    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability scores"""
        if not READABILITY_AVAILABLE:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0
            }

        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text)
            }
        except Exception as e:
            print(f"  ⚠️  Readability calculation failed: {e}")
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0
            }

    def calculate_clinical_similarity(self, generated: str, target: str) -> float:
        """Calculate semantic similarity using ClinicalBERT embeddings"""
        model = self.get_clinical_bert()
        if model is None:
            return 0.0

        try:
            gen_embedding = model.encode(generated, convert_to_tensor=True)
            tgt_embedding = model.encode(target, convert_to_tensor=True)
            similarity = util.cos_sim(gen_embedding, tgt_embedding).item()
            return similarity
        except Exception as e:
            print(f"  ⚠️  Clinical similarity calculation failed: {e}")
            return 0.0

    def calculate_factual_consistency(self, generated: str, input_note: str) -> float:
        """
        Calculate factual consistency using NLI
        Checks if generated summary is entailed by input note

        Returns a score between 0 and 1, where:
        - 1.0 = All generated sentences are entailed by input (fully consistent)
        - 0.0 = No generated sentences are entailed (potential hallucinations)
        """
        model = self.get_nli_model()
        if model is None:
            return 0.0

        try:
            # Split into sentences
            gen_sentences = [s.strip() for s in generated.split('.') if s.strip()]

            # Filter out very short sentences and section headers
            gen_sentences = [s for s in gen_sentences if len(s.split()) >= 5]

            # Truncate input if too long (NLI models typically have 512 token limit)
            input_truncated = input_note[:1500]

            entailment_scores = []

            # Check if this is zero-shot-classification pipeline (BART-MNLI)
            is_zero_shot = hasattr(model, 'model') and 'bart' in str(type(model.model)).lower()

            for sentence in gen_sentences[:15]:  # Check first 15 sentences
                try:
                    if is_zero_shot:
                        # Zero-shot classification format
                        result = model(sentence, candidate_labels=["entailment", "contradiction", "neutral"],
                                     hypothesis_template=input_truncated + " This means: {}")
                        # Get entailment score
                        for i, label in enumerate(result['labels']):
                            if label == 'entailment':
                                entailment_scores.append(result['scores'][i])
                                break
                    else:
                        # Cross-encoder NLI format (premise, hypothesis)
                        # For cross-encoder: input is premise, generated is hypothesis
                        result = model(f"{input_truncated} [SEP] {sentence}",
                                     truncation=True, max_length=512)

                        # Look for entailment/ENTAILMENT/entailed label
                        found_entailment = False
                        if isinstance(result, list):
                            for item in result:
                                label = item.get('label', '').lower()
                                if 'entail' in label:
                                    entailment_scores.append(item['score'])
                                    found_entailment = True
                                    break

                        if not found_entailment:
                            # If no explicit entailment label, use neutral assumption
                            entailment_scores.append(0.33)  # Neutral score

                except Exception as sent_error:
                    # Skip problematic sentences
                    continue

            if len(entailment_scores) == 0:
                return 0.0

            # Return mean entailment score
            return float(np.mean(entailment_scores))

        except Exception as e:
            print(f"  ⚠️  Factual consistency calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def calculate_medication_accuracy(self, generated: str, target: str) -> Dict[str, float]:
        """Calculate medication-specific accuracy"""
        gen_entities = self.extract_medical_entities(generated)
        tgt_entities = self.extract_medical_entities(target)

        gen_meds = gen_entities['medications']
        tgt_meds = tgt_entities['medications']

        if len(tgt_meds) == 0:
            return {
                'medication_precision': 1.0 if len(gen_meds) == 0 else 0.0,
                'medication_recall': 1.0,
                'medication_f1': 1.0 if len(gen_meds) == 0 else 0.0,
                'medication_count_gen': 0,
                'medication_count_target': 0
            }

        tp = len(gen_meds & tgt_meds)
        precision = tp / len(gen_meds) if len(gen_meds) > 0 else 0.0
        recall = tp / len(tgt_meds) if len(tgt_meds) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'medication_precision': precision,
            'medication_recall': recall,
            'medication_f1': f1,
            'medication_count_gen': len(gen_meds),
            'medication_count_target': len(tgt_meds)
        }

    def calculate_hallucination_metrics(self, generated: str, input_note: str) -> Dict[str, float]:
        """
        Calculate hallucination metrics

        Coverage: What proportion of generated content is supported by input
        Hallucination Rate: What proportion of generated content is NOT supported by input
        """
        if not input_note:
            return {
                'hallucination_coverage': 0.0,
                'hallucination_rate': 1.0
            }

        try:
            # Extract entities from both texts
            gen_entities = self.extract_medical_entities(generated)
            input_entities = self.extract_medical_entities(input_note)

            # Combine all entities
            all_gen = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']
            all_input = input_entities['medications'] | input_entities['diagnoses'] | input_entities['procedures']

            if len(all_gen) == 0:
                # No entities generated, no hallucinations
                return {
                    'hallucination_coverage': 1.0,
                    'hallucination_rate': 0.0
                }

            # Coverage: proportion of generated entities found in input
            supported = len(all_gen & all_input)
            coverage = supported / len(all_gen)
            hallucination_rate = 1.0 - coverage

            return {
                'hallucination_coverage': coverage,
                'hallucination_rate': hallucination_rate
            }

        except Exception as e:
            print(f"  ⚠️  Hallucination metrics calculation failed: {e}")
            return {
                'hallucination_coverage': 0.0,
                'hallucination_rate': 1.0
            }

    def calculate_rag_metrics(self, generated: str, retrieved_cases: List[str] = None,
                             input_note: str = None) -> Dict[str, float]:
        """
        Calculate RAG-specific metrics

        Args:
            generated: Generated summary
            retrieved_cases: List of retrieved similar cases (optional)
            input_note: Original clinical note

        Returns:
            Dictionary with RAG metrics:
            - retrieval_utilization: How much of retrieved context is used
            - medical_relevance: Relevance of generated text to medical domain
            - information_density: Information content per word
        """
        metrics = {}

        # Medical Relevance: proportion of medical entities in generated text
        try:
            gen_entities = self.extract_medical_entities(generated)
            all_entities = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']

            words = generated.split()
            if len(words) > 0:
                # Estimate medical relevance based on entity density
                medical_terms_count = len(all_entities)
                metrics['medical_relevance'] = min(1.0, medical_terms_count / (len(words) * 0.1))
            else:
                metrics['medical_relevance'] = 0.0

        except Exception as e:
            print(f"  ⚠️  Medical relevance calculation failed: {e}")
            metrics['medical_relevance'] = 0.0

        # Retrieval Utilization: if retrieved cases are provided
        if retrieved_cases and len(retrieved_cases) > 0:
            try:
                # Check how many retrieved cases contribute to generation
                utilized_count = 0
                for case in retrieved_cases:
                    # Simple overlap check - in practice, could use more sophisticated methods
                    case_entities = self.extract_medical_entities(case)
                    gen_entities = self.extract_medical_entities(generated)

                    case_all = case_entities['medications'] | case_entities['diagnoses'] | case_entities['procedures']
                    gen_all = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']

                    if len(case_all & gen_all) > 0:
                        utilized_count += 1

                metrics['retrieval_utilization'] = utilized_count / len(retrieved_cases) if len(retrieved_cases) > 0 else 0.0
                metrics['avg_retrieved_cases'] = len(retrieved_cases)
            except Exception as e:
                print(f"  ⚠️  Retrieval utilization calculation failed: {e}")
                metrics['retrieval_utilization'] = 0.0
                metrics['avg_retrieved_cases'] = 0
        else:
            metrics['retrieval_utilization'] = 0.0
            metrics['avg_retrieved_cases'] = 0

        # Information Density: unique medical entities per 100 words
        try:
            gen_entities = self.extract_medical_entities(generated)
            all_entities = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']

            words = generated.split()
            if len(words) > 0:
                metrics['information_density'] = (len(all_entities) / len(words)) * 100
            else:
                metrics['information_density'] = 0.0
        except Exception as e:
            print(f"  ⚠️  Information density calculation failed: {e}")
            metrics['information_density'] = 0.0

        return metrics

    def calculate_all_metrics(self, generated: str, target: str, input_note: str = None,
                             retrieved_cases: List[str] = None) -> Dict:
        """
        Calculate all medical metrics

        Args:
            generated: Generated summary text
            target: Target/reference summary text
            input_note: Original clinical note (optional, for coverage and consistency)
            retrieved_cases: List of retrieved similar cases (optional, for RAG metrics)

        Returns:
            Dictionary of all medical metrics
        """

        metrics = {}

        # ROUGE scores
        rouge_metrics = self.calculate_rouge(generated, target)
        metrics.update(rouge_metrics)

        # METEOR
        metrics['meteor'] = self.calculate_meteor(generated, target)

        # BERTScore
        bertscore_metrics = self.calculate_bertscore(generated, target)
        metrics.update(bertscore_metrics)

        # BLEU
        metrics['bleu'] = self.calculate_bleu(generated, target)

        # Entity F1
        entity_metrics = self.calculate_entity_f1(generated, target)
        metrics.update(entity_metrics)

        # Medication accuracy
        med_metrics = self.calculate_medication_accuracy(generated, target)
        metrics.update(med_metrics)

        # Readability
        readability = self.calculate_readability(generated)
        metrics.update(readability)

        # Clinical semantic similarity
        metrics['clinical_bert_similarity'] = self.calculate_clinical_similarity(generated, target)

        # Metrics requiring input note
        if input_note:
            # Entity coverage
            coverage = self.calculate_entity_coverage(generated, input_note)
            metrics.update(coverage)

            # Factual consistency
            metrics['factual_consistency'] = self.calculate_factual_consistency(generated, input_note)

            # Hallucination metrics
            hallucination = self.calculate_hallucination_metrics(generated, input_note)
            metrics.update(hallucination)

            # RAG metrics
            rag_metrics = self.calculate_rag_metrics(generated, retrieved_cases, input_note)
            metrics.update(rag_metrics)
        else:
            # Set to 0 if input not provided
            metrics['medications_coverage'] = 0.0
            metrics['diagnoses_coverage'] = 0.0
            metrics['procedures_coverage'] = 0.0
            metrics['overall_entity_coverage'] = 0.0
            metrics['factual_consistency'] = 0.0
            metrics['hallucination_coverage'] = 0.0
            metrics['hallucination_rate'] = 0.0
            metrics['retrieval_utilization'] = 0.0
            metrics['medical_relevance'] = 0.0
            metrics['information_density'] = 0.0
            metrics['avg_retrieved_cases'] = 0

        return metrics


def calculate_medical_metrics(generated: str, target: str, input_note: str = None,
                             retrieved_cases: List[str] = None) -> Dict:
    """
    Convenience function to calculate all medical metrics

    Args:
        generated: Generated summary text
        target: Target/reference summary text
        input_note: Original clinical note (optional, for coverage and consistency)
        retrieved_cases: List of retrieved similar cases (optional, for RAG metrics)

    Returns:
        Dictionary of all medical metrics including:
        - ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)
        - METEOR
        - BERTScore (precision, recall, F1)
        - BLEU
        - Medical Entity F1 (medications, diagnoses, procedures)
        - Entity Coverage
        - Medication Accuracy
        - Readability Metrics
        - ClinicalBERT Similarity
        - Factual Consistency (NLI-based)
        - Hallucination Metrics
        - RAG-specific Metrics
    """
    calculator = MedicalMetricsCalculator()
    return calculator.calculate_all_metrics(generated, target, input_note, retrieved_cases)


if __name__ == "__main__":
    # Test the metrics
    print("Testing Medical Metrics Calculator")
    print("="*80)

    generated = "Patient with hypertension and diabetes. Started on metformin and aspirin. Surgery performed successfully."
    target = "Patient has hypertension, diabetes mellitus. Medications: metformin, aspirin, lisinopril. Underwent surgery."
    input_note = "Patient admitted with hypertension and diabetes mellitus type 2. Medications initiated: metformin 500mg, aspirin 81mg, lisinopril 10mg. Surgical procedure completed without complications."

    # Simulate retrieved cases for RAG metrics
    retrieved_cases = [
        "Patient with diabetes on metformin. Blood glucose controlled.",
        "Hypertension managed with aspirin therapy.",
        "Post-surgical patient recovery normal."
    ]

    calculator = MedicalMetricsCalculator()
    metrics = calculator.calculate_all_metrics(generated, target, input_note, retrieved_cases)

    print("\nMedical Metrics Results:")
    print("="*80)

    # Group metrics by category
    categories = {
        'ROUGE Scores': ['rouge1_', 'rouge2_', 'rougeL_'],
        'Generation Quality': ['bleu', 'meteor'],
        'BERTScore': ['bertscore_'],
        'Medical Entity Metrics': ['medications_f1', 'diagnoses_f1', 'procedures_f1', 'overall_entity_'],
        'Entity Coverage': ['_coverage'],
        'Medication Metrics': ['medication_'],
        'Clinical Quality': ['clinical_bert_similarity', 'factual_consistency'],
        'Hallucination Analysis': ['hallucination_'],
        'RAG Metrics': ['retrieval_', 'medical_relevance', 'information_density', 'avg_retrieved'],
        'Readability': ['flesch_', 'gunning_fog']
    }

    for category, patterns in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for key, value in sorted(metrics.items()):
            if any(pattern in key for pattern in patterns):
                if isinstance(value, float):
                    print(f"  {key:.<50} {value:.4f}")
                else:
                    print(f"  {key:.<50} {value}")

    print("\n" + "="*80)
    print(f"Total metrics calculated: {len(metrics)}")
    print("="*80)
