import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import medical metrics
import sys
sys.path.append('./Final/Llama/Finetuning')
from llama_metrics import MedicalMetricsCalculator

# Import evaluation metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from bert_score import score as bert_score
    import nltk
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Some metric libraries not available.")
    print("   Install with: pip install rouge-score nltk bert-score")
    METRICS_AVAILABLE = False


class LlamaNarrativeGenerator:
    """Generate NARRATIVE medical summaries using fine-tuned Llama"""

    def __init__(self, model_path: str = "./Final/Llama/Narrative/Finetuning/outputs/merged_model"):
        """Load the narrative fine-tuned Llama model"""

        print("="*80)
        print("LLAMA-3.1-8B NARRATIVE MEDICAL SUMMARY GENERATOR")
        print("Generates well-organized flowing paragraph summaries")
        print("="*80)

        print("\n📥 Loading narrative fine-tuned model...")
        print(f"   Model path: {model_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(self.model)
        print("✅ Narrative Llama model loaded successfully\n")

    def generate_narrative_summary(self, input_text: str, temperature: float = 0.6,
                                   max_tokens: int = 1024, min_words: int = 180) -> str:
        """
        Generate comprehensive narrative paragraph summary

        Parameters:
        - temperature: 0.6 (balanced creativity for narrative flow)
        - max_tokens: 1024 (allows for comprehensive summaries up to ~800 words)
        - min_words: 180 (minimum target word count for comprehensive summaries)
        """

        # Prompt MUST match training format exactly (CLEAN and SIMPLE)
        # This matches the updated llama_narrative_finetune.py format (lines 125-135)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical AI assistant specialized in generating narrative discharge summaries.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a well-organized narrative discharge summary from the clinical note below. Use flowing paragraphs without bullet points or section headers.

Clinical Note:
{input_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Calculate min_length in tokens (approx 180 words * 1.33 tokens/word = 240 tokens)
        min_length_tokens = int(min_words * 1.33)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=min_length_tokens,  # Ensure at least 180 words (~240 tokens)
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,  # Reduced from 1.15 - too strong causes garbage output
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                length_penalty=1.2,  # Increased from 1.0 to encourage longer outputs
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Post-process to clean output
        generated = self._clean_narrative_output(generated)

        return generated

    def _clean_narrative_output(self, text: str) -> str:
        """Remove any unwanted formatting that may have slipped through"""
        import re

        # Remove common prefixes that indicate model is describing what it's doing
        prefixes_to_remove = [
            r'^Here is a comprehensive.*?:\s*',
            r'^Here is the narrative summary.*?:\s*',
            r'^Clinical Note:.*?(?=\n\n|\Z)',  # Remove repeated clinical notes
            r'^Narrative (?:Discharge )?Summary:.*?(?=\n\n|\Z)',
            r'^---+\s*',
        ]
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

        # Remove example-like patterns (lines starting with "EXAMPLE", "Clinical Note:", etc.)
        text = re.sub(r'^(?:EXAMPLE|Clinical Note:|Narrative Summary:).*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove markdown headers
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)

        # Remove common section headers if they appear at line starts
        headers_to_remove = [
            r'^\s*Narrative\s+(?:Discharge\s+)?Summary:?\s*',
            r'^\s*Discharge\s+Summary:?\s*',
            r'^\s*Patient\s+&\s+Service:?\s*',
            r'^\s*Chief\s+Complaint:?\s*',
            r'^\s*Hospital\s+Course:?\s*',
        ]
        for header_pattern in headers_to_remove:
            text = re.sub(header_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove bullet points if any appeared
        text = re.sub(r'^\s*[•\-\*]\s+', '', text, flags=re.MULTILINE)

        # Remove "---" separators that come from training examples
        text = re.sub(r'\n---+\n', '\n\n', text)

        # CRITICAL: Remove "Now, generate..." instruction blocks (most common artifact)
        text = re.sub(
            r'Now,?\s+generate.*?(?:Professional:.*?)?(?=\n\n|$)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove "Remember:" instruction blocks
        text = re.sub(
            r'Remember:.*?(?:Professional:.*?)?(?=\n\n|$)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove other instruction-like patterns
        instruction_patterns = [
            r'^Organize\s+logically:.*?$',
            r'^Write\s+fluently:.*?$',
            r'^Thorough:.*?$',
            r'^Professional:.*?$',
            r'^NO\s+bullets.*?$',
            r'^NO\s+headers.*?$',
            r'^Include\s+everything.*?$',
            r'^\s*Dis\s*charge\s+Information\s*>.*?$',  # "Dis charge Information >" artifact
        ]
        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove "Generate a..." instruction artifacts at the end
        text = re.sub(
            r'Generate\s+a\s+well[- ]organiz.*?discharge\s+summ.*?[.!]?\s*$',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove trailing instruction artifacts
        text = re.sub(r'\n\s*Generate.*?$', '', text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
        text = text.strip()

        # If there's "Clinical Note:" anywhere, cut everything after it
        if 'Clinical Note:' in text:
            parts = text.split('Clinical Note:', 1)
            text = parts[0].strip()

        # If there's "EXAMPLE" anywhere, cut everything after it
        if 'EXAMPLE' in text.upper():
            parts = re.split(r'EXAMPLE', text, flags=re.IGNORECASE, maxsplit=1)
            text = parts[0].strip()

        # If output starts with technical metadata (like "-sex:F Service:"), it's garbage
        if text.startswith('-sex:') or text.startswith('sex:'):
            return ""  # Return empty string for failed generations

        # If output is suspiciously short (< 20 words), likely failed generation
        word_count = len(text.split())
        if word_count < 20:
            return ""  # Return empty string for failed generations

        return text


def calculate_narrative_metrics(generated: str, target: str, input_note: str = None,
                                med_calculator: MedicalMetricsCalculator = None) -> Dict:
    """Calculate all metrics for a narrative summary"""

    metrics = {}

    # Narrative quality metrics
    word_count = len(generated.split())
    char_count = len(generated)

    # Count paragraphs (blocks of text separated by double newlines or single newlines)
    paragraphs = [p.strip() for p in generated.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in generated.split('\n') if p.strip()]
    paragraph_count = len(paragraphs)

    # Check for unwanted formatting
    has_bullets = '•' in generated or generated.count('\n- ') > 0 or generated.count('\n* ') > 0
    has_headers = bool(re.search(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*$', generated, re.MULTILINE))
    has_numbered_lists = bool(re.search(r'^\d+\.\s+', generated, re.MULTILINE))

    is_pure_narrative = not (has_bullets or has_headers or has_numbered_lists)

    metrics['narrative_quality'] = {
        'word_count': word_count,
        'char_count': char_count,
        'paragraph_count': paragraph_count,
        'avg_words_per_paragraph': word_count / paragraph_count if paragraph_count > 0 else 0,
        'is_pure_narrative': is_pure_narrative,
        'has_bullets': has_bullets,
        'has_headers': has_headers,
        'has_numbered_lists': has_numbered_lists,
    }

    # ROUGE scores
    if METRICS_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(target, generated)
        metrics['rouge1'] = scores['rouge1'].fmeasure
        metrics['rouge2'] = scores['rouge2'].fmeasure
        metrics['rougeL'] = scores['rougeL'].fmeasure

        # METEOR
        try:
            metrics['meteor'] = meteor_score([target.split()], generated.split())
        except:
            metrics['meteor'] = 0.0

        # BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
        try:
            smoothing = SmoothingFunction().method1
            reference = [target.split()]
            candidate = generated.split()

            # Individual BLEU scores
            metrics['bleu1'] = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            metrics['bleu2'] = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            metrics['bleu3'] = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            metrics['bleu4'] = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        except:
            metrics['bleu1'] = 0.0
            metrics['bleu2'] = 0.0
            metrics['bleu3'] = 0.0
            metrics['bleu4'] = 0.0
    else:
        metrics['rouge1'] = 0.0
        metrics['rouge2'] = 0.0
        metrics['rougeL'] = 0.0
        metrics['meteor'] = 0.0
        metrics['bleu1'] = 0.0
        metrics['bleu2'] = 0.0
        metrics['bleu3'] = 0.0
        metrics['bleu4'] = 0.0

    # Medical metrics (if calculator provided)
    if med_calculator is not None:
        medical_metrics = med_calculator.calculate_all_metrics(generated, target, input_note)
        metrics['medical'] = medical_metrics
    else:
        metrics['medical'] = {}

    return metrics


def calculate_bertscore_batch(generated_list: List[str], target_list: List[str]) -> Dict[str, List[float]]:
    """Calculate BERTScore for batch"""
    if not METRICS_AVAILABLE or len(generated_list) == 0:
        return {
            'precision': [0.0] * len(generated_list),
            'recall': [0.0] * len(generated_list),
            'f1': [0.0] * len(generated_list)
        }

    try:
        print("      Computing BERTScore (this may take a few minutes)...")
        P, R, F1 = bert_score(generated_list, target_list, lang='en', verbose=False,
                             device='cuda' if torch.cuda.is_available() else 'cpu')
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
    except Exception as e:
        print(f"      ⚠️  BERTScore failed: {e}")
        return {
            'precision': [0.0] * len(generated_list),
            'recall': [0.0] * len(generated_list),
            'f1': [0.0] * len(generated_list)
        }


def export_for_word_format(results: List[Dict], output_file: str, num_samples: int = None):
    """
    Export test results to Word-friendly text format with proper formatting

    Args:
        results: List of result dictionaries from evaluation
        output_file: Path to output text file
        num_samples: Number of samples to export (None = all)
    """
    if num_samples:
        results = results[:num_samples]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("LLAMA-3.1-8B-INSTRUCT NARRATIVE TEST SET RESULTS\n")
        f.write("Well-Organized Medical Discharge Summary Generation\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write("="*100 + "\n\n")

        for idx, sample in enumerate(results):
            note_id = sample.get('note_id', f'sample_{idx}')

            f.write("\n" + "="*100 + "\n")
            f.write(f"SAMPLE {idx + 1} of {len(results)}: {note_id}\n")
            f.write("="*100 + "\n\n")

            # Input (truncated)
            f.write("─"*100 + "\n")
            f.write("INPUT NOTE (truncated):\n")
            f.write("─"*100 + "\n")
            input_text = sample['input']
            f.write(input_text[:400] + "...\n\n" if len(input_text) > 400 else input_text + "\n\n")

            # Target Summary
            f.write("─"*100 + "\n")
            f.write("TARGET NARRATIVE SUMMARY:\n")
            f.write("─"*100 + "\n")

            # Format target summary with paragraph wrapping (80 chars per line)
            import textwrap
            target_text = sample['target_summary']
            # Split by existing paragraph breaks first
            target_paragraphs = target_text.split('\n\n')
            formatted_target_paragraphs = []

            for para in target_paragraphs:
                # Wrap each paragraph to 80 characters
                wrapped = textwrap.fill(para, width=100, break_long_words=False, break_on_hyphens=False)
                formatted_target_paragraphs.append(wrapped)

            # Join paragraphs with double newlines
            formatted_target = '\n\n'.join(formatted_target_paragraphs)
            f.write(formatted_target + "\n\n")

            # Generated Summary
            f.write("─"*100 + "\n")
            f.write("GENERATED NARRATIVE SUMMARY:\n")
            f.write("─"*100 + "\n")

            # Format summary with paragraph wrapping (100 chars per line)
            generated_text = sample['generated_summary']
            # Split by existing paragraph breaks first
            paragraphs = generated_text.split('\n\n')
            formatted_paragraphs = []

            for para in paragraphs:
                # Wrap each paragraph to 100 characters
                wrapped = textwrap.fill(para, width=100, break_long_words=False, break_on_hyphens=False)
                formatted_paragraphs.append(wrapped)

            # Join paragraphs with double newlines
            formatted_summary = '\n\n'.join(formatted_paragraphs)
            f.write(formatted_summary + "\n\n")

            # Metrics
            f.write("─"*100 + "\n")
            f.write("EVALUATION METRICS:\n")
            f.write("─"*100 + "\n\n")

            metrics = sample.get('metrics', {})

            # Narrative Quality
            if 'narrative_quality' in metrics:
                nq = metrics['narrative_quality']
                f.write("Narrative Quality:\n")
                f.write(f"  Word Count:                  {nq.get('word_count', 0)}\n")
                f.write(f"  Paragraph Count:             {nq.get('paragraph_count', 0)}\n")
                f.write(f"  Avg Words/Paragraph:         {nq.get('avg_words_per_paragraph', 0):.1f}\n")
                f.write(f"  Pure Narrative (no bullets): {'Yes' if nq.get('is_pure_narrative', False) else 'No'}\n\n")

            # Standard NLP Metrics
            f.write("Standard NLP Metrics:\n")
            f.write(f"  ROUGE-1:           {metrics.get('rouge1', 0):.4f}\n")
            f.write(f"  ROUGE-2:           {metrics.get('rouge2', 0):.4f}\n")
            f.write(f"  ROUGE-L:           {metrics.get('rougeL', 0):.4f}\n")
            f.write(f"  METEOR:            {metrics.get('meteor', 0):.4f}\n\n")

            if 'bertscore' in metrics:
                bert = metrics['bertscore']
                f.write("BERTScore (Semantic Similarity):\n")
                f.write(f"  Precision:         {bert.get('precision', 0):.4f}\n")
                f.write(f"  Recall:            {bert.get('recall', 0):.4f}\n")
                f.write(f"  F1:                {bert.get('f1', 0):.4f}\n\n")

            if 'medical' in metrics:
                med = metrics['medical']
                f.write("Medical-Specific Metrics:\n")
                f.write(f"  Entity F1:                   {med.get('overall_entity_f1', 0):.4f}\n")
                f.write(f"  Medication F1:               {med.get('medication_f1', 0):.4f}\n")
                f.write(f"  Entity Coverage:             {med.get('overall_entity_coverage', 0):.4f}\n")
                f.write(f"  ClinicalBERT Similarity:     {med.get('clinical_bert_similarity', 0):.4f}\n")
                f.write(f"  Factual Consistency (NLI):   {med.get('factual_consistency', 0):.4f}\n\n")

                f.write("Readability:\n")
                f.write(f"  Flesch Reading Ease:         {med.get('flesch_reading_ease', 0):.1f}\n")
                f.write(f"  Flesch-Kincaid Grade:        {med.get('flesch_kincaid_grade', 0):.1f}\n\n")

            f.write(f"Generation Time: {sample.get('generation_time_sec', 0):.1f} seconds\n")
            f.write("\n" + "="*100 + "\n\n")


def evaluate_test_set(num_samples: int = None):
    """
    Evaluate narrative fine-tuned Llama on test set

    Args:
        num_samples: Number of samples to evaluate (None = all 200 samples)
    """

    print("="*80)
    print("LLAMA NARRATIVE TEST SET EVALUATION")
    print("="*80)

    output_dir = "./Final/Llama/Narrative/Finetuning/outputs"

    # Load generator
    generator = LlamaNarrativeGenerator()

    # Load test set
    test_df = pd.read_csv(f'{output_dir}/test_set.csv')

    # Limit samples if requested
    if num_samples is not None:
        test_df = test_df.head(num_samples)
        print(f"\n📊 Test set loaded: {len(test_df)} samples (LIMITED FOR TESTING)")
    else:
        print(f"\n📊 Test set loaded: {len(test_df)} samples (FULL TEST SET)")

    # Initialize medical metrics calculator
    print("\n🔬 Initializing medical metrics calculator...")
    med_calculator = MedicalMetricsCalculator()
    print("✓ Medical metrics ready\n")

    # Process test samples
    results = []
    overall_start = time.time()

    print(f"\n{'='*80}")
    print(f"GENERATING NARRATIVE SUMMARIES FOR TEST SET")
    print('='*80)

    for i, (idx, row) in enumerate(test_df.iterrows()):
        print(f"\n[{i+1}/{len(test_df)}] Processing Note ID: {row.get('note_id', idx)}")

        start_time = time.time()

        # Generate narrative summary
        generated = generator.generate_narrative_summary(row['input'])

        gen_time = time.time() - start_time

        # Check narrative quality
        word_count = len(generated.split())
        has_bullets = '•' in generated
        is_narrative = not has_bullets

        # Calculate metrics
        metrics = calculate_narrative_metrics(generated, row['target'], row['input'], med_calculator)

        print(f"  ✓ Generated in {gen_time:.1f}s | Words: {word_count} | {'✅ Pure Narrative' if is_narrative else '❌ Has Structure'}")

        result = {
            'note_id': row.get('note_id', f'test_{idx}'),
            'input': row['input'],
            'target_summary': row['target'],
            'generated_summary': generated,
            'word_count': word_count,
            'generation_time_sec': gen_time,
            'is_pure_narrative': metrics['narrative_quality']['is_pure_narrative'],
            'metrics': metrics
        }

        results.append(result)

        # Progress update
        if (i + 1) % 10 == 0:
            elapsed = time.time() - overall_start
            avg_time = elapsed / (i + 1)
            remaining = len(test_df) - (i + 1)
            eta_minutes = (remaining * avg_time) / 60
            print(f"\n  Progress: {i+1}/{len(test_df)} ({(i+1)/len(test_df)*100:.1f}%) | ETA: {eta_minutes:.1f} min")

    total_time = time.time() - overall_start

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print('='*80)

    results_path = f'{output_dir}/llama_narrative_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: llama_narrative_results.json")

    # Calculate BERTScore
    print(f"\n{'='*80}")
    print("CALCULATING BERTSCORE")
    print('='*80)

    generated_texts = [r['generated_summary'] for r in results]
    target_texts = [r['target_summary'] for r in results]
    bert_scores = calculate_bertscore_batch(generated_texts, target_texts)

    # Add BERTScore to results
    for i, result in enumerate(results):
        result['metrics']['bertscore'] = {
            'precision': bert_scores['precision'][i],
            'recall': bert_scores['recall'][i],
            'f1': bert_scores['f1'][i]
        }

    # Calculate aggregate metrics
    print(f"\n{'='*80}")
    print("CALCULATING AGGREGATE METRICS")
    print('='*80)

    test_metrics = {
        'num_samples': len(results),

        # Narrative quality
        'avg_word_count': np.mean([r['metrics']['narrative_quality']['word_count'] for r in results]),
        'avg_paragraph_count': np.mean([r['metrics']['narrative_quality']['paragraph_count'] for r in results]),
        'pure_narrative_pct': np.mean([r['metrics']['narrative_quality']['is_pure_narrative'] for r in results]) * 100,

        # ROUGE metrics
        'rouge1': np.mean([r['metrics']['rouge1'] for r in results]),
        'rouge2': np.mean([r['metrics']['rouge2'] for r in results]),
        'rougeL': np.mean([r['metrics']['rougeL'] for r in results]),

        # BLEU metrics
        'bleu1': np.mean([r['metrics'].get('bleu1', 0) for r in results]),
        'bleu2': np.mean([r['metrics'].get('bleu2', 0) for r in results]),
        'bleu3': np.mean([r['metrics'].get('bleu3', 0) for r in results]),
        'bleu4': np.mean([r['metrics'].get('bleu4', 0) for r in results]),

        # METEOR
        'meteor': np.mean([r['metrics']['meteor'] for r in results]),

        # BERTScore
        'bertscore_precision': np.mean(bert_scores['precision']),
        'bertscore_recall': np.mean(bert_scores['recall']),
        'bertscore_f1': np.mean(bert_scores['f1']),

        # Medical metrics
        'entity_f1': np.mean([r['metrics']['medical'].get('overall_entity_f1', 0) for r in results]),
        'entity_coverage': np.mean([r['metrics']['medical'].get('overall_entity_coverage', 0) for r in results]),
        'medication_f1': np.mean([r['metrics']['medical'].get('medication_f1', 0) for r in results]),
        'clinical_bert_similarity': np.mean([r['metrics']['medical'].get('clinical_bert_similarity', 0) for r in results]),
        'factual_consistency': np.mean([r['metrics']['medical'].get('factual_consistency', 0) for r in results]),

        # Hallucination metrics
        'hallucination_coverage': np.mean([r['metrics']['medical'].get('hallucination_coverage', 0) for r in results]),
        'hallucination_rate': np.mean([r['metrics']['medical'].get('hallucination_rate', 0) for r in results]),

        # Readability
        'flesch_reading_ease': np.mean([r['metrics']['medical'].get('flesch_reading_ease', 0) for r in results]),
        'flesch_kincaid_grade': np.mean([r['metrics']['medical'].get('flesch_kincaid_grade', 0) for r in results]),

        # Performance
        'avg_generation_time': np.mean([r['generation_time_sec'] for r in results]),
        'total_time': total_time,
        'throughput': len(results) / (total_time / 60),
    }

    # Save metrics
    metrics_path = f'{output_dir}/llama_narrative_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  ✓ Saved: llama_narrative_metrics.json")

    # Print summary
    print(f"\n{'='*80}")
    print("TEST SET EVALUATION SUMMARY")
    print('='*80)

    print(f"\nTest Set ({test_metrics['num_samples']} samples):")
    print(f"\n  NARRATIVE QUALITY:")
    print(f"    Pure Narrative: {test_metrics['pure_narrative_pct']:.1f}%")
    print(f"    Avg Word Count: {test_metrics['avg_word_count']:.0f}")
    print(f"    Avg Paragraphs: {test_metrics['avg_paragraph_count']:.1f}")

    print(f"\n  STANDARD NLP METRICS:")
    print(f"    ROUGE-1: {test_metrics['rouge1']:.4f}")
    print(f"    ROUGE-2: {test_metrics['rouge2']:.4f}")
    print(f"    ROUGE-L: {test_metrics['rougeL']:.4f}")
    print(f"    METEOR: {test_metrics['meteor']:.4f}")

    print(f"\n  BERTSCORE METRICS:")
    print(f"    Precision: {test_metrics['bertscore_precision']:.4f}")
    print(f"    Recall: {test_metrics['bertscore_recall']:.4f}")
    print(f"    F1: {test_metrics['bertscore_f1']:.4f}")

    print(f"\n  MEDICAL METRICS:")
    print(f"    Entity F1: {test_metrics['entity_f1']:.4f}")
    print(f"    Entity Coverage: {test_metrics['entity_coverage']:.4f}")
    print(f"    Medication F1: {test_metrics['medication_f1']:.4f}")
    print(f"    ClinicalBERT Similarity: {test_metrics['clinical_bert_similarity']:.4f}")
    print(f"    Factual Consistency: {test_metrics['factual_consistency']:.4f}")

    print(f"\n  PERFORMANCE:")
    print(f"    Avg Time: {test_metrics['avg_generation_time']:.1f}s")
    print(f"    Total Time: {total_time / 60:.1f} minutes")
    print(f"    Throughput: {test_metrics['throughput']:.2f} samples/minute")

    # AUTO-EXPORT: Create Word-friendly text file
    print(f"\n📄 Creating Word-friendly export...")
    word_export_path = os.path.join(output_dir, 'llama_narrative_results_for_word.txt')
    export_for_word_format(results, word_export_path)
    print(f"  ✓ {word_export_path}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE - READY FOR PHASE 2")
    print("="*80)
    print("\n✓ Narrative test set evaluation complete")
    print("✓ Results ready for RAG integration")
    print("✓ Metrics ready for explainability analysis")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import sys
    import re

    # Allow command-line argument for number of samples
    num_samples = None
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
            print(f"\n🔬 TESTING MODE: Running on {num_samples} samples only\n")
        except ValueError:
            print(f"⚠️  Invalid argument: {sys.argv[1]}. Using full test set.")

    evaluate_test_set(num_samples=num_samples)
