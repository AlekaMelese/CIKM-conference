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

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from transformers import pipeline
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False
    print("Warning: transformers not available for NLI. Install with: pip install transformers")

try:
    import textstat
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    print("Warning: textstat not available. Install with: pip install textstat")

from config import NarrativeRAGConfig


class NarrativeRAGEvaluator:
    """Evaluate NARRATIVE RAG performance with comprehensive metrics"""

    def __init__(self):
        self.config = NarrativeRAGConfig
        self.rag_results = None
        self.test_df = None
        self.metrics = defaultdict(list)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if ROUGE_AVAILABLE else None
        self.smoothing = SmoothingFunction().method1 if BLEU_AVAILABLE else None

        # Models for clinical metrics (lazy loading)
        self.clinical_bert = None
        self.nli_model = None

        # Medical entity patterns (MATCHING llama_metrics.py exactly)
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

        # Load spacy model for entity extraction if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None

    def get_clinical_bert(self):
        """Lazy load ClinicalBERT model (MATCHING llama_metrics.py)"""
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
        """Lazy load NLI model (MATCHING llama_metrics.py)"""
        if not NLI_AVAILABLE:
            return None

        if self.nli_model is None:
            try:
                print("  Loading NLI model for factual consistency (first time only)...")
                self.nli_model = pipeline("text-classification",
                                         model="cross-encoder/nli-deberta-v3-small",
                                         device=0 if torch.cuda.is_available() else -1)
                print("  ✓ NLI model loaded")
            except Exception as e:
                print(f"  ⚠️  NLI model failed to load: {e}")
                print(f"     Trying fallback NLI model...")
                try:
                    self.nli_model = pipeline("zero-shot-classification",
                                             model="facebook/bart-large-mnli",
                                             device=0 if torch.cuda.is_available() else -1)
                    print("  ✓ Fallback NLI model loaded")
                except Exception as e2:
                    print(f"  ⚠️  Fallback NLI model also failed: {e2}")
                    return None
        return self.nli_model

    def extract_medical_entities(self, text: str):
        """Extract medical entities using regex patterns (MATCHING llama_metrics.py)"""
        entities = {
            'medications': set(),
            'diagnoses': set(),
            'procedures': set()
        }

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

    def load_data(self):
        """Load RAG results and test set"""
        print("Loading evaluation data...")

        # Load RAG results
        with open(self.config.RAG_SUMMARIES_PATH, 'r') as f:
            self.rag_results = json.load(f)
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

        # Coverage: how much of reference is in generated
        coverage = len(ref_words & gen_words) / len(ref_words) if len(ref_words) > 0 else 0

        return {
            'coverage': coverage,
            'unique_words_generated': len(gen_words),
            'unique_words_reference': len(ref_words)
        }

    def compute_hallucination_metrics(self, generated, input_text):
        """
        Compute hallucination metrics (MATCHING Phase 1 llama_metrics.py exactly)

        Measures how much of the generated content is grounded in the input
        """
        # Split into sentences
        gen_sentences = [s.strip() for s in generated.split('.') if s.strip() and len(s.split()) >= 3]
        input_lower = input_text.lower()

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

        hallucination_coverage = grounded_count / len(gen_sentences) if gen_sentences else 0.0
        hallucination_rate = 1.0 - hallucination_coverage

        return {
            'hallucination_coverage': hallucination_coverage,
            'hallucination_rate': hallucination_rate,
            'grounded_sentences': grounded_count,
            'total_sentences': len(gen_sentences)
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

    def compute_narrative_quality_metrics(self, generated):
        """Compute narrative-specific quality metrics"""
        # Word count
        word_count = len(generated.split())

        # Paragraph count (separated by double newlines)
        paragraphs = [p.strip() for p in generated.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)

        # Average words per paragraph
        avg_words_per_para = word_count / paragraph_count if paragraph_count > 0 else 0

        # Narrative purity (no bullets, headers, lists)
        has_bullets = bool(re.search(r'^\s*[•\-\*]\s+', generated, re.MULTILINE))
        has_headers = bool(re.search(r'^[A-Z][^:]*:\s*$', generated, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+\.\s+', generated, re.MULTILINE))

        narrative_purity = 1.0 if not (has_bullets or has_headers or has_numbers) else 0.0

        return {
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_para': avg_words_per_para,
            'narrative_purity': narrative_purity
        }

    def compute_readability_metrics(self, generated):
        """
        Compute readability metrics (ADDED to match finetuning evaluation)
        Measures how readable the generated summaries are
        """
        if not READABILITY_AVAILABLE:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0
            }

        try:
            flesch_ease = textstat.flesch_reading_ease(generated)
            flesch_grade = textstat.flesch_kincaid_grade(generated)

            return {
                'flesch_reading_ease': flesch_ease,
                'flesch_kincaid_grade': flesch_grade
            }
        except Exception as e:
            print(f"  ⚠️  Readability calculation failed: {e}")
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0
            }

    def compute_entity_metrics(self, generated, reference):
        """
        Compute entity extraction metrics (MATCHING llama_metrics.py)
        Uses regex-based medical entity extraction instead of spaCy
        """
        gen_entities = self.extract_medical_entities(generated)
        ref_entities = self.extract_medical_entities(reference)

        # Overall entity F1 (all entity types combined)
        all_gen = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']
        all_ref = ref_entities['medications'] | ref_entities['diagnoses'] | ref_entities['procedures']

        if len(all_ref) > 0:
            tp = len(all_gen & all_ref)
            precision = tp / len(all_gen) if len(all_gen) > 0 else 0.0
            recall = tp / len(all_ref) if len(all_ref) > 0 else 0.0
            entity_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            entity_f1 = 1.0 if len(all_gen) == 0 else 0.0

        # Medication-specific F1
        gen_meds = gen_entities['medications']
        ref_meds = ref_entities['medications']

        if len(ref_meds) == 0:
            medication_f1 = 1.0 if len(gen_meds) == 0 else 0.0
        else:
            tp = len(gen_meds & ref_meds)
            precision = tp / len(gen_meds) if len(gen_meds) > 0 else 0.0
            recall = tp / len(ref_meds) if len(ref_meds) > 0 else 0.0
            medication_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Entity coverage (will be updated in next method to use input)
        entity_coverage = len(all_gen & all_ref) / len(all_ref) if len(all_ref) > 0 else 0.0

        return {
            'entity_f1': entity_f1,
            'entity_coverage': entity_coverage,  # This will be overwritten by compute_entity_coverage_from_input
            'medication_f1': medication_f1
        }

    def compute_entity_coverage_from_input(self, generated, input_text):
        """
        Calculate what % of input entities are covered in generated summary
        (MATCHING llama_metrics.py)
        This is the CORRECT entity_coverage that compares generated vs INPUT
        """
        gen_entities = self.extract_medical_entities(generated)
        input_entities = self.extract_medical_entities(input_text)

        # Overall coverage
        all_gen = gen_entities['medications'] | gen_entities['diagnoses'] | gen_entities['procedures']
        all_input = input_entities['medications'] | input_entities['diagnoses'] | input_entities['procedures']

        if len(all_input) > 0:
            entity_coverage = len(all_gen & all_input) / len(all_input)
        else:
            entity_coverage = 1.0

        return entity_coverage

    def compute_clinical_bert_similarity(self, generated, reference):
        """
        Calculate semantic similarity using ClinicalBERT embeddings
        (MATCHING llama_metrics.py)
        """
        model = self.get_clinical_bert()
        if model is None:
            # Fallback to word overlap if model not available
            gen_words = set(generated.lower().split())
            ref_words = set(reference.lower().split())

            if len(ref_words) == 0:
                return 1.0 if len(gen_words) == 0 else 0.0

            overlap = len(gen_words & ref_words)
            return overlap / len(ref_words)

        try:
            gen_embedding = model.encode(generated, convert_to_tensor=True)
            ref_embedding = model.encode(reference, convert_to_tensor=True)
            similarity = util.cos_sim(gen_embedding, ref_embedding).item()
            return similarity
        except Exception as e:
            print(f"  ⚠️  Clinical similarity calculation failed: {e}")
            # Fallback to word overlap
            gen_words = set(generated.lower().split())
            ref_words = set(reference.lower().split())

            if len(ref_words) == 0:
                return 1.0 if len(gen_words) == 0 else 0.0

            overlap = len(gen_words & ref_words)
            return overlap / len(ref_words)

    def compute_factual_consistency(self, generated, input_text):
        """
        Calculate factual consistency using NLI (MATCHING llama_metrics.py)
        Checks if generated summary is entailed by input note

        Returns a score between 0 and 1, where:
        - 1.0 = All generated sentences are entailed by input (fully consistent)
        - 0.0 = No generated sentences are entailed (potential hallucinations)
        """
        model = self.get_nli_model()
        if model is None:
            # Fallback to simple word overlap if NLI model not available
            gen_words = set(generated.lower().split())
            input_words = set(input_text.lower().split())

            contradictory_words = gen_words - input_words

            if len(gen_words) == 0:
                return 0.0

            consistency = 1.0 - (len(contradictory_words) / len(gen_words))
            return max(0.0, consistency)

        try:
            # Split into sentences
            gen_sentences = [s.strip() for s in generated.split('.') if s.strip()]

            # Filter out very short sentences and section headers
            gen_sentences = [s for s in gen_sentences if len(s.split()) >= 5]

            # Truncate input if too long (NLI models typically have 512 token limit)
            input_truncated = input_text[:1500]

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
            # Fallback to simple method
            gen_words = set(generated.lower().split())
            input_words = set(input_text.lower().split())

            contradictory_words = gen_words - input_words

            if len(gen_words) == 0:
                return 0.0

            consistency = 1.0 - (len(contradictory_words) / len(gen_words))
            return max(0.0, consistency)

    def compute_rag_specific_metrics(self, result):
        """Compute RAG-specific evaluation metrics"""
        generated = result.get('generated_summary', result.get('generated', ''))
        input_text = result.get('input', '')
        num_retrieved = len(result.get('retrieved_cases', []))
        
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
        # Support both key formats: 'generated_summary' (Narrative_3_rag_inference.py) or 'generated' (older scripts)
        generated = result.get('generated_summary', result.get('generated', ''))
        # Use 'target' field from narrative RAG inference output (reference summary)
        # Fallback to 'reference' for backwards compatibility
        reference = result.get('target', result.get('reference', ''))
        input_text = result.get('input', '')

        sample_metrics = {}

        # ROUGE scores - compare against reference (golden narrative summary)
        if ROUGE_AVAILABLE:
            rouge_scores = self.compute_rouge_scores(generated, reference)
            sample_metrics.update(rouge_scores)

        # BLEU score - compare against reference
        if BLEU_AVAILABLE:
            bleu_score = self.compute_bleu_score(generated, reference)
            sample_metrics['bleu'] = bleu_score

        # METEOR score - compare against reference
        meteor_score = self.compute_meteor_score(generated, reference)
        sample_metrics['meteor'] = meteor_score

        # BERTScore metrics - compare against reference
        bertscore_metrics = self.compute_bertscore_metrics(generated, reference)
        sample_metrics.update(bertscore_metrics)

        # Length metrics
        length_metrics = self.compute_length_metrics(generated, reference)
        sample_metrics.update(length_metrics)

        # Narrative quality metrics (instead of structure metrics)
        narrative_metrics = self.compute_narrative_quality_metrics(generated)
        sample_metrics.update(narrative_metrics)

        # Readability metrics (ADDED to match finetuning evaluation)
        readability_metrics = self.compute_readability_metrics(generated)
        sample_metrics.update(readability_metrics)

        # Entity metrics - compare against reference (for F1)
        entity_metrics = self.compute_entity_metrics(generated, reference)
        sample_metrics.update(entity_metrics)

        # Entity coverage - CORRECTED: compare generated vs INPUT (not reference)
        # This matches llama_metrics.py behavior
        entity_coverage_corrected = self.compute_entity_coverage_from_input(generated, input_text)
        sample_metrics['entity_coverage'] = entity_coverage_corrected  # Overwrite with correct value

        # Clinical BERT similarity - compare against reference
        clinical_similarity = self.compute_clinical_bert_similarity(generated, reference)
        sample_metrics['clinical_bert_similarity'] = clinical_similarity

        # Factual consistency - CORRECTED: use NLI model with input (not reference)
        # This matches llama_metrics.py behavior
        factual_consistency = self.compute_factual_consistency(generated, input_text)
        sample_metrics['factual_consistency'] = factual_consistency

        # Coverage metrics - compare against reference
        coverage_metrics = self.compute_coverage_metrics(generated, reference, input_text)
        sample_metrics.update(coverage_metrics)

        # Hallucination metrics - CORRECTED: MATCHING Phase 1 implementation
        hallucination_metrics = self.compute_hallucination_metrics(generated, input_text)
        sample_metrics.update(hallucination_metrics)

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
        print("NARRATIVE RAG Evaluation Results (Comprehensive)")
        print("=" * 80)

        # Basic metrics
        print(f"\n📊 Basic Metrics:")
        print(f"  Number of samples: {len(self.rag_results)}")
        
        # Narrative quality metrics
        if 'word_count' in aggregated:
            print(f"  Avg word count: {aggregated['word_count']['mean']:.1f} words")
        if 'paragraph_count' in aggregated:
            print(f"  Avg paragraph count: {aggregated['paragraph_count']['mean']:.1f}")
        if 'narrative_purity' in aggregated:
            print(f"  Narrative purity: {aggregated['narrative_purity']['mean']*100:.1f}%")

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

        # Hallucination analysis (MATCHING Phase 1 output)
        if 'hallucination_coverage' in aggregated:
            print(f"\n🚫 Hallucination Analysis:")
            print(f"  Coverage: {aggregated['hallucination_coverage']['mean']:.4f} ± {aggregated['hallucination_coverage']['std']:.4f}")
            print(f"  Hallucination Rate: {aggregated['hallucination_rate']['mean']:.4f} ± {aggregated['hallucination_rate']['std']:.4f}")

        # Readability metrics (ADDED to match finetuning evaluation)
        if 'flesch_reading_ease' in aggregated:
            print(f"\n📖 Readability Metrics:")
            print(f"  Flesch Reading Ease: {aggregated['flesch_reading_ease']['mean']:.2f} ± {aggregated['flesch_reading_ease']['std']:.2f}")
            print(f"  Flesch-Kincaid Grade: {aggregated['flesch_kincaid_grade']['mean']:.2f} ± {aggregated['flesch_kincaid_grade']['std']:.2f}")

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

        with open(self.config.EVALUATION_RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Evaluation results saved: {self.config.EVALUATION_RESULTS_PATH}")

    def create_summary_report(self, aggregated):
        """Create human-readable summary report for NARRATIVE RAG"""
        report_path = self.config.RAG_OUTPUTS_DIR / "evaluation_summary.md"

        with open(report_path, 'w') as f:
            f.write("# Narrative RAG Evaluation Results\n\n")
            f.write(f"**Model**: Fine-tuned Llama-3.1-8B-Instruct (Narrative) + Hybrid RAG\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Samples**: {len(self.rag_results)}\n\n")
            f.write("=" * 80 + "\n\n")

            f.write("## System Configuration\n\n")
            f.write(f"- **Base Model**: Llama-3.1-8B-Instruct (narrative fine-tuned)\n")
            f.write(f"- **Embedding Model**: {self.config.EMBEDDING_MODEL_NAME}\n")
            f.write(f"- **Reranker Model**: {self.config.RERANKER_MODEL_NAME}\n")
            f.write(f"- **Retrieval Strategy**: Dense ({self.config.DENSE_TOP_K}) → Rerank ({self.config.RERANK_TOP_K})\n")
            f.write(f"- **Temperature**: {self.config.TEMPERATURE}\n")
            f.write(f"- **Repetition Penalty**: {self.config.REPETITION_PENALTY}\n\n")
            f.write("=" * 80 + "\n\n")

            # ROUGE scores
            f.write("## 🎯 ROUGE Scores\n\n")
            f.write("| Metric | Mean | Std | Min | Max |\n")
            f.write("|--------|------|-----|-----|-----|\n")
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                if metric in aggregated:
                    m = aggregated[metric]
                    f.write(f"| **{metric.upper()}** | **{m['mean']:.4f}** | {m['std']:.4f} | {m['min']:.4f} | {m['max']:.4f} |\n")
            f.write("\n")

            # Generation quality
            f.write("## 📝 Generation Quality\n\n")
            if 'meteor' in aggregated:
                f.write(f"- **METEOR**: {aggregated['meteor']['mean']:.4f} ± {aggregated['meteor']['std']:.4f}\n")
            if 'bleu' in aggregated:
                f.write(f"- **BLEU**: {aggregated['bleu']['mean']:.4f} ± {aggregated['bleu']['std']:.4f}\n")
            f.write("\n")

            # BERTScore
            f.write("## 🧠 Semantic Similarity (BERTScore)\n\n")
            for metric in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
                if metric in aggregated:
                    m = aggregated[metric]
                    label = metric.replace('bertscore_', '').capitalize()
                    f.write(f"- **{label}**: {m['mean']:.4f} ± {m['std']:.4f}\n")
            f.write("\n")

            # Medical metrics
            f.write("## 🏥 Medical Entity Extraction\n\n")
            if 'entity_f1' in aggregated:
                f.write(f"- **Entity F1**: {aggregated['entity_f1']['mean']:.4f} ± {aggregated['entity_f1']['std']:.4f}\n")
            if 'entity_coverage' in aggregated:
                f.write(f"- **Entity Coverage**: {aggregated['entity_coverage']['mean']:.4f} ± {aggregated['entity_coverage']['std']:.4f}\n")
            if 'medication_f1' in aggregated:
                f.write(f"- **Medication F1**: {aggregated['medication_f1']['mean']:.4f} ± {aggregated['medication_f1']['std']:.4f}\n")
            f.write("\n")

            # Clinical quality
            f.write("## 🔬 Clinical Quality\n\n")
            if 'clinical_bert_similarity' in aggregated:
                m = aggregated['clinical_bert_similarity']
                f.write(f"- **Clinical BERT Similarity**: **{m['mean']:.4f}** ± {m['std']:.4f}\n")
            if 'factual_consistency' in aggregated:
                m = aggregated['factual_consistency']
                f.write(f"- **Factual Consistency**: {m['mean']:.4f} ± {m['std']:.4f}\n")
            f.write("\n")

            # Hallucination
            f.write("## 🚫 Hallucination Analysis\n\n")
            if 'hallucination_coverage' in aggregated:
                m = aggregated['hallucination_coverage']
                f.write(f"- **Hallucination Coverage**: **{m['mean']:.4f}** ± {m['std']:.4f}\n")
            if 'hallucination_rate' in aggregated:
                m = aggregated['hallucination_rate']
                f.write(f"- **Hallucination Rate**: **{m['mean']:.4f}** ± {m['std']:.4f} (lower is better)\n")
            f.write("\n")

            # Narrative quality
            f.write("## 📖 Narrative Quality Metrics\n\n")
            if 'word_count' in aggregated:
                m = aggregated['word_count']
                f.write(f"- **Average Word Count**: {m['mean']:.1f} words\n")
            if 'paragraph_count' in aggregated:
                m = aggregated['paragraph_count']
                f.write(f"- **Average Paragraph Count**: {m['mean']:.1f} paragraphs\n")
            if 'avg_words_per_para' in aggregated:
                m = aggregated['avg_words_per_para']
                f.write(f"- **Average Words/Paragraph**: {m['mean']:.1f}\n")
            if 'narrative_purity' in aggregated:
                m = aggregated['narrative_purity']
                f.write(f"- **Narrative Purity**: {m['mean']*100:.1f}% (no bullets/headers/lists)\n")
            f.write("\n")

            # Readability
            f.write("## 📊 Readability Metrics\n\n")
            if 'flesch_reading_ease' in aggregated:
                m = aggregated['flesch_reading_ease']
                f.write(f"- **Flesch Reading Ease**: {m['mean']:.2f} ± {m['std']:.2f}\n")
            if 'flesch_kincaid_grade' in aggregated:
                m = aggregated['flesch_kincaid_grade']
                f.write(f"- **Flesch-Kincaid Grade**: {m['mean']:.2f} ± {m['std']:.2f}\n")
            f.write("\n")

            f.write("=" * 80 + "\n\n")
            f.write("✓ Full results: evaluation_results.json\n")

        print(f"✓ Summary report saved: {report_path}")


def main():
    """Main execution"""
    print("=" * 80)
    print("Phase 2 - Step 4: NARRATIVE RAG Evaluation")
    print("=" * 80)

    evaluator = NarrativeRAGEvaluator()

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
    print(f"\nResults saved to: {NarrativeRAGConfig.RAG_OUTPUTS_DIR}")

if __name__ == "__main__":
    main()
