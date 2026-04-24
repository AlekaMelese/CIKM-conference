#!/usr/bin/env python3
"""
Compute inter-annotator agreement and correlation with automated scores.

Supports 2 or 3 annotators:
- 2 annotators: Cohen's kappa
- 3 annotators: Fleiss' kappa + pairwise Cohen's kappa

Run after all annotators complete their annotations:
    python compute_agreement.py

Outputs:
- Inter-annotator agreement (kappa)
- Spearman correlation between automated score and human consensus
- Confusion matrix: automated label vs. human consensus
- Summary statistics for the paper
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
from itertools import combinations
import json

BASE_DIR = Path(__file__).parent
ANNOTATED_DIR = BASE_DIR / 'annotated'
KEY_PATH = BASE_DIR / 'annotation_key.csv'
OUTPUT_PATH = BASE_DIR / 'agreement_results.json'

GROUNDING_LABELS = ['Fully Supported', 'Partially Supported', 'Unsupported', 'Cannot Determine']
SEVERITY_LABELS = ['None', 'Minor', 'Major', 'Critical']
ERROR_LABELS = ['None', 'Fabrication', 'Inaccuracy', 'Omission']


def load_all_annotators():
    """Discover and load all annotator files."""
    annotators = {}
    for f in sorted(ANNOTATED_DIR.glob('annotated_sheet_*.csv')):
        name = f.stem.replace('annotated_sheet_', '')
        df = pd.read_csv(f)
        done = df['factual_grounding'].notna() & (df['factual_grounding'] != '')
        if done.sum() > 0:
            annotators[name] = df
            print(f"  Loaded {name}: {done.sum()}/{len(df)} completed")
    return annotators


def fleiss_kappa(ratings_matrix):
    """
    Compute Fleiss' kappa for multiple annotators.
    ratings_matrix: (n_subjects x n_categories) counts matrix
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # assumes same # raters per subject

    # Proportion of assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Expected agreement by chance
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def build_fleiss_matrix(annotations_list, labels):
    """
    Build Fleiss' kappa ratings matrix from list of annotation Series.
    annotations_list: list of pd.Series (one per annotator)
    labels: list of category labels
    """
    n = len(annotations_list[0])
    n_cats = len(labels)
    matrix = np.zeros((n, n_cats), dtype=int)

    for ann in annotations_list:
        for i, val in enumerate(ann):
            if val in labels:
                matrix[i, labels.index(val)] += 1
    return matrix


def majority_consensus(labels_list):
    """Compute majority vote consensus from multiple annotators."""
    from collections import Counter
    counts = Counter(l for l in labels_list if pd.notna(l) and l != '')
    if not counts:
        return 'Cannot Determine'
    winner = counts.most_common(1)[0][0]
    return winner


def map_grounding_to_numeric(label):
    mapping = {
        'Fully Supported': 3,
        'Partially Supported': 2,
        'Unsupported': 1,
        'Cannot Determine': np.nan,
    }
    return mapping.get(label, np.nan)


def map_auto_to_human_label(label):
    mapping = {
        'SUPPORTED': 'Fully Supported',
        'PARTIAL': 'Partially Supported',
        'UNSUPPORTED': 'Unsupported',
    }
    return mapping.get(label, label)


def main():
    print("=== Clinical Validation Agreement Analysis ===\n")

    # Load annotators
    annotators = load_all_annotators()
    if len(annotators) < 2:
        print(f"\nERROR: Need at least 2 annotators, found {len(annotators)}")
        print("Run: streamlit run annotation_ui.py")
        return

    key = pd.read_csv(KEY_PATH)
    names = list(annotators.keys())
    n_annotators = len(names)
    print(f"\nAnnotators: {', '.join(names)} ({n_annotators} total)")

    # Find samples completed by ALL annotators
    completed_mask = pd.Series([True] * len(key))
    for name, df in annotators.items():
        completed_mask &= df['factual_grounding'].notna() & (df['factual_grounding'] != '')

    n = completed_mask.sum()
    print(f"Samples completed by all annotators: {n}\n")

    if n < 10:
        print("Too few completed samples for meaningful analysis.")
        return

    results = {'n_samples': int(n), 'n_annotators': n_annotators, 'annotators': names}

    # Get valid data
    valid_data = {name: df[completed_mask].reset_index(drop=True) for name, df in annotators.items()}
    key_valid = key[completed_mask].reset_index(drop=True)

    # === 1. Pairwise Cohen's Kappa ===
    print("1. Pairwise Cohen's Kappa (Factual Grounding):")
    pairwise_kappas = []
    for a1_name, a2_name in combinations(names, 2):
        kappa = cohen_kappa_score(
            valid_data[a1_name]['factual_grounding'],
            valid_data[a2_name]['factual_grounding']
        )
        pairwise_kappas.append(kappa)
        print(f"   {a1_name} vs {a2_name}: {kappa:.3f}")

    avg_kappa = np.mean(pairwise_kappas)
    print(f"   Average pairwise kappa: {avg_kappa:.3f}")
    results['pairwise_kappa_grounding'] = {
        f"{a1} vs {a2}": round(k, 3)
        for (a1, a2), k in zip(combinations(names, 2), pairwise_kappas)
    }
    results['avg_pairwise_kappa_grounding'] = round(avg_kappa, 3)

    # === 2. Fleiss' Kappa (if 3+ annotators) ===
    if n_annotators >= 3:
        grounding_series = [valid_data[name]['factual_grounding'] for name in names]
        matrix = build_fleiss_matrix(grounding_series, GROUNDING_LABELS)
        fk = fleiss_kappa(matrix)
        print(f"\n2. Fleiss' Kappa (Factual Grounding, {n_annotators} annotators): {fk:.3f}")
        results['fleiss_kappa_grounding'] = round(fk, 3)

        # Fleiss for severity
        severity_series = [valid_data[name]['clinical_severity'] for name in names]
        sev_matrix = build_fleiss_matrix(severity_series, SEVERITY_LABELS)
        fk_sev = fleiss_kappa(sev_matrix)
        print(f"   Fleiss' Kappa (Clinical Severity): {fk_sev:.3f}")
        results['fleiss_kappa_severity'] = round(fk_sev, 3)
    else:
        # Cohen's kappa for severity (2 annotators)
        kappa_sev = cohen_kappa_score(
            valid_data[names[0]]['clinical_severity'],
            valid_data[names[1]]['clinical_severity']
        )
        print(f"\n2. Cohen's Kappa (Clinical Severity): {kappa_sev:.3f}")
        results['kappa_severity'] = round(kappa_sev, 3)

    # Interpretation
    ref_kappa = results.get('fleiss_kappa_grounding', avg_kappa)
    if ref_kappa >= 0.81:
        interp = "Almost perfect"
    elif ref_kappa >= 0.61:
        interp = "Substantial"
    elif ref_kappa >= 0.41:
        interp = "Moderate"
    elif ref_kappa >= 0.21:
        interp = "Fair"
    else:
        interp = "Slight"
    print(f"   Interpretation: {interp} agreement")
    results['kappa_interpretation'] = interp

    # === 3. Raw Agreement ===
    all_agree = sum(
        1 for i in range(n)
        if len(set(valid_data[name].iloc[i]['factual_grounding'] for name in names)) == 1
    )
    print(f"\n3. Raw Agreement (all annotators agree): {all_agree}/{n} ({all_agree/n:.1%})")
    results['raw_agreement_all'] = round(all_agree / n, 3)

    # === 4. Majority Consensus vs Automated ===
    consensus_grounding = []
    for i in range(n):
        labels = [valid_data[name].iloc[i]['factual_grounding'] for name in names]
        consensus_grounding.append(majority_consensus(labels))

    # Correlation: auto score vs human consensus
    human_numeric = [map_grounding_to_numeric(c) for c in consensus_grounding]
    auto_scores = key_valid['_auto_score'].tolist()

    valid_pairs = [
        (h, a) for h, a in zip(human_numeric, auto_scores)
        if not np.isnan(h) and not np.isnan(a)
    ]
    if valid_pairs:
        h_vals, a_vals = zip(*valid_pairs)
        spearman_r, spearman_p = spearmanr(a_vals, h_vals)
        pearson_r, pearson_p = pearsonr(a_vals, h_vals)
        print(f"\n4. Correlation (Auto Score vs Human Consensus):")
        print(f"   Spearman rho: {spearman_r:.3f} (p={spearman_p:.4f})")
        print(f"   Pearson r:    {pearson_r:.3f} (p={pearson_p:.4f})")
        results['spearman_rho'] = round(spearman_r, 3)
        results['spearman_p'] = round(spearman_p, 4)
        results['pearson_r'] = round(pearson_r, 3)
        results['pearson_p'] = round(pearson_p, 4)

    # === 5. Confusion Matrix ===
    auto_labels_mapped = [map_auto_to_human_label(l) for l in key_valid['_auto_label']]
    label_order = ['Fully Supported', 'Partially Supported', 'Unsupported']

    valid_idx = [i for i, c in enumerate(consensus_grounding) if c in label_order]
    if valid_idx:
        auto_filtered = [auto_labels_mapped[i] for i in valid_idx]
        human_filtered = [consensus_grounding[i] for i in valid_idx]

        cm = confusion_matrix(human_filtered, auto_filtered, labels=label_order)
        print(f"\n5. Confusion Matrix (Human Consensus vs Automated):")
        print(f"   {'':>25} {'Auto-Supported':>15} {'Auto-Partial':>15} {'Auto-Unsupported':>17}")
        for i, label in enumerate(label_order):
            print(f"   {label:>25} {cm[i][0]:>15} {cm[i][1]:>15} {cm[i][2]:>17}")

        print(f"\n6. Classification Report (Automated vs Human Consensus):")
        print(classification_report(
            human_filtered, auto_filtered,
            labels=label_order, target_names=['Supported', 'Partial', 'Unsupported']
        ))

        accuracy = sum(1 for h, a in zip(human_filtered, auto_filtered) if h == a) / len(human_filtered)
        print(f"   Overall accuracy: {accuracy:.1%}")
        results['accuracy_auto_vs_human'] = round(accuracy, 3)

    # === 7. Severity Distribution by Auto Category ===
    # Use majority consensus for severity too
    consensus_severity = []
    for i in range(n):
        sevs = [valid_data[name].iloc[i]['clinical_severity'] for name in names]
        consensus_severity.append(majority_consensus(sevs))

    print(f"\n7. Severity Distribution by Automated Category:")
    severity_by_auto = {}
    for auto_label in ['SUPPORTED', 'PARTIAL', 'UNSUPPORTED']:
        mask_label = key_valid['_auto_label'] == auto_label
        if mask_label.sum() == 0:
            continue
        indices = mask_label[mask_label].index
        sevs = pd.Series([consensus_severity[i] for i in indices])
        dist = sevs.value_counts()
        severity_by_auto[auto_label] = dist.to_dict()
        print(f"   {auto_label}:")
        for sev, count in dist.items():
            print(f"     {sev}: {count} ({count/len(indices):.0%})")

    results['severity_by_auto_category'] = severity_by_auto

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
