import re
import numpy as np
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    BLEU_AVAILABLE = True
    METEOR_AVAILABLE = True
except ImportError:
    print("⚠️  BLEU/METEOR not available. Install: pip install nltk")
    BLEU_AVAILABLE = False
    METEOR_AVAILABLE = False

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
        """Lazy load NLI model

        Uses DeBERTa-v3-large-mnli for best accuracy (as per FACTUAL_CONSISTENCY_FIX_README.md)
        """
        if not NLI_AVAILABLE:
            return None

        if self.nli_model is None:
            try:
                print("  Loading NLI model for factual consistency (first time only)...")
                # Use DeBERTa-v3-large-mnli for best accuracy (FIXED model)
                # Labels: LABEL_0 (contradiction), LABEL_1 (neutral), LABEL_2 (entailment)
                self.nli_model = pipeline("text-classification",
                                         model="microsoft/deberta-large-mnli",
                                         device=0 if torch.cuda.is_available() else -1)
                print("  ✓ NLI model loaded (DeBERTa-v3-large-mnli)")
            except Exception as e:
                print(f"  ⚠️  Primary NLI model failed to load: {e}")
                print(f"     Trying fallback NLI model (DeBERTa-large-mnli)...")
                try:
                    # Fallback to smaller DeBERTa model
                    self.nli_model = pipeline("text-classification",
                                             model="microsoft/deberta-large-mnli",
                                             device=0 if torch.cuda.is_available() else -1)
                    print("  ✓ Fallback NLI model loaded (DeBERTa-large-mnli)")
                except Exception as e2:
                    print(f"  ⚠️  Fallback DeBERTa failed, trying BART-MNLI: {e2}")
                    try:
                        # Last fallback to zero-shot BART
                        self.nli_model = pipeline("zero-shot-classification",
                                                 model="facebook/bart-large-mnli",
                                                 device=0 if torch.cuda.is_available() else -1)
                        print("  ✓ BART-MNLI model loaded (zero-shot fallback)")
                    except Exception as e3:
                        print(f"  ⚠️  All NLI models failed: {e3}")
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

    def calculate_meteor(self, generated: str, target: str) -> float:
        """Calculate METEOR score"""
        if not METEOR_AVAILABLE:
            return 0.0

        try:
            # METEOR requires tokenized text
            reference = target.split()
            hypothesis = generated.split()
            return meteor_score([reference], hypothesis)
        except Exception as e:
            print(f"  ⚠️  METEOR calculation failed: {e}")
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

        Fixed version (2025-12-01):
        - Correct DeBERTa label parsing (LABEL_0/1/2)
        - Increased truncation (1500 → 4000 chars)
        - More sentences checked (15 → 30)
        - Better error handling
        """
        model = self.get_nli_model()
        if model is None:
            return 0.0

        try:
            # Split into sentences
            gen_sentences = [s.strip() for s in generated.split('.') if s.strip()]

            # Filter out very short sentences and section headers
            gen_sentences = [s for s in gen_sentences if len(s.split()) >= 5]

            # FIX #2: Increased truncation from 1500 to 4000 characters (55% coverage)
            # Medical notes average 7,331 characters, need more context
            input_truncated = input_note[:4000]

            entailment_scores = []

            # Check if this is zero-shot-classification pipeline (BART-MNLI)
            is_zero_shot = hasattr(model, 'model') and 'bart' in str(type(model.model)).lower()

            # FIX #3: Check first 30 sentences instead of 15 (86% coverage)
            # Summaries average ~35 sentences
            for sentence in gen_sentences[:30]:
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

                        # FIX #1: Proper DeBERTa label parsing
                        # DeBERTa-v3-mnli uses LABEL_0/1/2, not "entailment" string
                        if isinstance(result, list):
                            for item in result:
                                label = item.get('label', '')
                                score = item.get('score', 0.0)
                                label_lower = label.lower()

                                # Map DeBERTa labels correctly
                                if label == 'LABEL_2' or 'entail' in label_lower:
                                    # Entailment - use model confidence
                                    entailment_scores.append(score)
                                    break
                                elif label == 'LABEL_1' or 'neutral' in label_lower:
                                    # Neutral - partial entailment
                                    entailment_scores.append(0.33)
                                    break
                                elif label == 'LABEL_0' or 'contradiction' in label_lower:
                                    # Contradiction - no entailment
                                    entailment_scores.append(0.0)
                                    break
                            else:
                                # FIX #4: Only use fallback on genuine parsing failure
                                # (not on label format mismatch which is now handled above)
                                entailment_scores.append(0.33)

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

    def calculate_structure_metrics(self, generated: str) -> Dict[str, float]:
        """
        Calculate structure completeness metrics

        Checks for presence of expected medical discharge summary sections
        """
        # Expected sections in discharge summaries
        expected_sections = [
            'chief complaint', 'history of present illness', 'past medical history',
            'medications', 'physical exam', 'hospital course',
            'discharge condition', 'discharge medications', 'follow-up',
            'discharge diagnosis', 'procedures'
        ]

        generated_lower = generated.lower()

        # Count how many sections are present
        sections_present = sum(1 for section in expected_sections
                              if section in generated_lower)

        # Count bullet points (lines starting with -, •, or *)
        bullet_pattern = r'^[\s]*[-•\*]\s+'
        bullet_count = len(re.findall(bullet_pattern, generated, re.MULTILINE))

        # Check if structure is complete (has most sections)
        structure_complete = 1.0 if sections_present >= len(expected_sections) * 0.7 else 0.0

        return {
            'structure_complete': structure_complete,
            'section_coverage': sections_present / len(expected_sections),
            'bullet_count': bullet_count,
            'sections_present': sections_present
        }

    def calculate_hallucination_metrics(self, generated: str, input_note: str) -> Dict[str, float]:
        """
        Calculate hallucination-related metrics

        Measures how much of the generated content is grounded in the input
        """
        # Split into sentences
        gen_sentences = [s.strip() for s in generated.split('.') if s.strip() and len(s.split()) >= 3]
        input_lower = input_note.lower()

        # Count how many generated sentences have key terms present in input
        grounded_count = 0
        for sentence in gen_sentences:
            # Extract key terms (words > 4 chars, excluding common words)
            words = [w.lower() for w in sentence.split() if len(w) > 4]
            if not words:
                continue

            # Check if at least 30% of key terms are in input
            matches = sum(1 for w in words if w in input_lower)
            if matches / len(words) >= 0.3:
                grounded_count += 1

        coverage = grounded_count / len(gen_sentences) if gen_sentences else 0.0
        hallucination_rate = 1.0 - coverage

        return {
            'hallucination_coverage': coverage,
            'hallucination_rate': hallucination_rate,
            'grounded_sentences': grounded_count,
            'total_sentences': len(gen_sentences)
        }

    def calculate_all_metrics(self, generated: str, target: str, input_note: str = None) -> Dict:
        """Calculate all medical metrics"""

        metrics = {}

        # BLEU and METEOR
        metrics['bleu'] = self.calculate_bleu(generated, target)
        metrics['meteor'] = self.calculate_meteor(generated, target)

        # Entity F1
        entity_metrics = self.calculate_entity_f1(generated, target)
        metrics.update(entity_metrics)

        # Medication accuracy
        med_metrics = self.calculate_medication_accuracy(generated, target)
        metrics.update(med_metrics)

        # Structure metrics
        structure_metrics = self.calculate_structure_metrics(generated)
        metrics.update(structure_metrics)

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
            hallucination_metrics = self.calculate_hallucination_metrics(generated, input_note)
            metrics.update(hallucination_metrics)
        else:
            # Set to 0 if input not provided
            metrics['medications_coverage'] = 0.0
            metrics['diagnoses_coverage'] = 0.0
            metrics['procedures_coverage'] = 0.0
            metrics['overall_entity_coverage'] = 0.0
            metrics['factual_consistency'] = 0.0
            metrics['hallucination_coverage'] = 0.0
            metrics['hallucination_rate'] = 0.0
            metrics['grounded_sentences'] = 0
            metrics['total_sentences'] = 0

        return metrics


def calculate_medical_metrics(generated: str, target: str, input_note: str = None) -> Dict:
    """
    Convenience function to calculate all medical metrics

    Args:
        generated: Generated summary text
        target: Target/reference summary text
        input_note: Original clinical note (optional, for coverage and consistency)

    Returns:
        Dictionary of all medical metrics
    """
    calculator = MedicalMetricsCalculator()
    return calculator.calculate_all_metrics(generated, target, input_note)


if __name__ == "__main__":
    # Test the metrics
    print("Testing Medical Metrics Calculator\n")

    generated = "Patient with hypertension and diabetes. Started on metformin and aspirin. Surgery performed successfully."
    target = "Patient has hypertension, diabetes mellitus. Medications: metformin, aspirin, lisinopril. Underwent surgery."
    input_note = "Patient admitted with hypertension and diabetes mellitus type 2. Medications initiated: metformin 500mg, aspirin 81mg, lisinopril 10mg. Surgical procedure completed without complications."

    calculator = MedicalMetricsCalculator()
    metrics = calculator.calculate_all_metrics(generated, target, input_note)

    print("Medical Metrics Results:")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.4f}")
        else:
            print(f"  {key:.<40} {value}")
