#!/usr/bin/env python3
"""
Display Comprehensive Evaluation Metrics from gemma_test_metrics.json

Organizes and displays all evaluation metrics in a clear, publication-ready format.
"""

import json
import sys

def display_all_metrics(metrics_file: str):
    """
    Read and display all evaluation metrics in organized sections

    Args:
        metrics_file: Path to the metrics JSON file
    """

    # Read metrics
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Metrics file not found: {metrics_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in metrics file: {metrics_file}")
        return

    print("=" * 80)
    print(" " * 15 + "COMPREHENSIVE EVALUATION METRICS")
    print(" " * 20 + "Gemma-2-9B Narrative Model")
    print("=" * 80)

    # 1. DATASET SUMMARY
    print("\n" + "─" * 80)
    print("1. DATASET SUMMARY")
    print("─" * 80)
    print(f"  Number of Test Samples:        {metrics.get('num_samples', 'N/A')}")
    print(f"  Average Word Count:            {metrics.get('avg_word_count', 0):.1f} words")
    print(f"  Average Paragraph Count:       {metrics.get('avg_paragraph_count', 0):.2f}")
    print(f"  Pure Narrative Percentage:     {metrics.get('pure_narrative_pct', 0):.1f}%")

    # 2. STANDARD NLP METRICS
    print("\n" + "─" * 80)
    print("2. STANDARD NLP METRICS (Text Similarity)")
    print("─" * 80)

    print("\n  ROUGE Scores (Overlap-based):")
    print(f"    ROUGE-1 (Unigram):           {metrics.get('rouge1', 0):.4f}")
    print(f"    ROUGE-2 (Bigram):            {metrics.get('rouge2', 0):.4f}")
    print(f"    ROUGE-L (Longest Seq):       {metrics.get('rougeL', 0):.4f}")

    print("\n  BLEU Scores (N-gram Precision):")
    print(f"    BLEU (Overall):              {metrics.get('bleu', 0):.4f}")
    print(f"    BLEU-1 (Unigram):            {metrics.get('bleu1', 0):.4f}")
    print(f"    BLEU-2 (Bigram):             {metrics.get('bleu2', 0):.4f}")
    print(f"    BLEU-3 (Trigram):            {metrics.get('bleu3', 0):.4f}")
    print(f"    BLEU-4 (4-gram):             {metrics.get('bleu4', 0):.4f}")

    print("\n  Other Metrics:")
    print(f"    METEOR:                      {metrics.get('meteor', 0):.4f}")

    # 3. SEMANTIC SIMILARITY
    print("\n" + "─" * 80)
    print("3. SEMANTIC SIMILARITY (Deep Learning-based)")
    print("─" * 80)

    bertscore_f1 = metrics.get('bertscore_f1', None)
    if bertscore_f1 is not None and bertscore_f1 > 0:
        print("\n  BERTScore (Contextual Embeddings):")
        print(f"    Precision:                   {metrics.get('bertscore_precision', 0):.4f}")
        print(f"    Recall:                      {metrics.get('bertscore_recall', 0):.4f}")
        print(f"    F1 Score:                    {bertscore_f1:.4f}")
    else:
        print("\n  BERTScore:                     Not available")

    print("\n  Clinical Domain Similarity:")
    print(f"    ClinicalBERT Similarity:     {metrics.get('clinical_bert_similarity', 0):.4f}")

    # 4. MEDICAL ENTITY METRICS
    print("\n" + "─" * 80)
    print("4. MEDICAL ENTITY EXTRACTION METRICS")
    print("─" * 80)

    print("\n  Entity F1 Scores:")
    print(f"    Overall Entity F1:           {metrics.get('entity_f1', 0):.4f}")
    print(f"    Medication F1:               {metrics.get('medication_f1', 0):.4f}")

    print("\n  Entity Coverage (Input → Generated):")
    print(f"    Overall Entity Coverage:     {metrics.get('entity_coverage', 0):.4f}")

    # 5. FACTUAL CONSISTENCY & HALLUCINATION
    print("\n" + "─" * 80)
    print("5. FACTUAL CONSISTENCY & HALLUCINATION DETECTION")
    print("─" * 80)

    print("\n  Factual Consistency (NLI-based):")
    print(f"    Factual Consistency Score:   {metrics.get('factual_consistency', 0):.4f}")
    print(f"    (Higher = More facts supported by input)")

    print("\n  Hallucination Analysis:")
    print(f"    Hallucination Coverage:      {metrics.get('hallucination_coverage', 0):.4f}")
    print(f"    (% of sentences grounded in input)")
    print(f"    Hallucination Rate:          {metrics.get('hallucination_rate', 0):.4f}")
    print(f"    (% of sentences potentially hallucinated)")

    # 6. READABILITY METRICS
    print("\n" + "─" * 80)
    print("6. READABILITY METRICS")
    print("─" * 80)

    fre = metrics.get('flesch_reading_ease', 0)
    fkg = metrics.get('flesch_kincaid_grade', 0)

    print("\n  Flesch Reading Ease:")
    print(f"    Score:                       {fre:.2f}")
    if fre >= 60:
        print(f"    Level:                       Easy to read (8th-9th grade)")
    elif fre >= 50:
        print(f"    Level:                       Fairly difficult (10th-12th grade)")
    else:
        print(f"    Level:                       Difficult (College level)")

    print(f"\n  Flesch-Kincaid Grade Level:  {fkg:.2f}")
    print(f"    (Requires ~{int(fkg)} years of education)")

    # 7. PERFORMANCE METRICS
    print("\n" + "─" * 80)
    print("7. GENERATION PERFORMANCE")
    print("─" * 80)

    print(f"\n  Average Generation Time:     {metrics.get('avg_generation_time', 0):.2f} seconds/sample")
    print(f"  Total Generation Time:       {metrics.get('total_time', 0):.2f} seconds")
    print(f"  Throughput:                  {metrics.get('throughput', 0):.2f} samples/minute")

    # 8. SUMMARY INTERPRETATION
    print("\n" + "=" * 80)
    print("SUMMARY INTERPRETATION")
    print("=" * 80)

    # Calculate overall quality score (weighted average)
    rouge1 = metrics.get('rouge1', 0)
    meteor = metrics.get('meteor', 0)
    bertscore = metrics.get('bertscore_f1', 0) if metrics.get('bertscore_f1') else 0
    clinical_sim = metrics.get('clinical_bert_similarity', 0)
    factual = metrics.get('factual_consistency', 0)

    if bertscore > 0:
        quality_score = (rouge1 * 0.15 + meteor * 0.15 + bertscore * 0.25 +
                        clinical_sim * 0.25 + factual * 0.20)
    else:
        quality_score = (rouge1 * 0.20 + meteor * 0.20 + clinical_sim * 0.35 + factual * 0.25)

    print(f"\n  Overall Quality Score:       {quality_score:.4f}")
    print(f"  (Weighted combination of key metrics)")

    # Strengths and Areas for Improvement
    print("\n  Key Strengths:")
    if clinical_sim >= 0.90:
        print(f"    ✓ Excellent semantic similarity (ClinicalBERT: {clinical_sim:.4f})")
    if bertscore >= 0.80:
        print(f"    ✓ Strong contextual alignment (BERTScore: {bertscore:.4f})")
    if metrics.get('pure_narrative_pct', 0) >= 95:
        print(f"    ✓ Excellent narrative format compliance ({metrics.get('pure_narrative_pct', 0):.1f}%)")
    if fre >= 50 and fre <= 70:
        print(f"    ✓ Appropriate readability for medical professionals")

    print("\n  Areas for Improvement:")
    if rouge1 < 0.40:
        print(f"    ⚠ Moderate lexical overlap (ROUGE-1: {rouge1:.4f})")
    if metrics.get('bleu', 0) < 0.10:
        print(f"    ⚠ Low n-gram precision (BLEU: {metrics.get('bleu', 0):.4f})")
    if metrics.get('hallucination_rate', 0) > 0.40:
        print(f"    ⚠ Notable hallucination rate ({metrics.get('hallucination_rate', 0):.2%})")
    if factual < 0.40:
        print(f"    ⚠ Factual consistency could be improved ({factual:.4f})")

    print("\n" + "=" * 80)
    print("END OF METRICS REPORT")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Default metrics file path
    default_path = "./Final/Gemma/Narrative/Finetuning/outputs/llama_narrative_metrics.json"

    # Allow custom path as command line argument
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else default_path

    print(f"\n📊 Reading metrics from: {metrics_file}\n")
    display_all_metrics(metrics_file)
