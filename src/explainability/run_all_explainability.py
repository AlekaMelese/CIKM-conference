"""
Master Script - Run All Explainability Analyses
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_analysis(script_name, medical=True, samples=None):
    """Run a single explainability analysis"""
    print("\n" + "=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80)

    cmd = [sys.executable, script_name]
    if medical:
        cmd.append('--medical')
    if samples:
        cmd.extend(['--samples', str(samples)])

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all explainability analyses")
    parser.add_argument('--medical', action='store_true', default=True,
                       help='Analyze Medical RAG (default: True)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to analyze (default: all)')
    parser.add_argument('--skip', nargs='+', default=[],
                       help='Skip specific analyses (1-7)')

    args = parser.parse_args()

    explainability_dir = Path(__file__).parent

    analyses = [
        ('1_evidence_attribution.py', 'Evidence Attribution'),
        ('2_confidence_scoring.py', 'Confidence Scoring'),
        ('3_clinical_rationale.py', 'Clinical Rationale'),
        ('4_factual_alignment.py', 'Factual Alignment'),
        ('5_attention_visualization.py', 'Attention Visualization'),
        ('6_token_attribution.py', 'Token-level Attribution'),
        ('7_counterfactual_explanations.py', 'Counterfactual Explanations'),
    ]

    print("=" * 80)
    print("PHASE 3 - EXPLAINABILITY LAYER")
    print(f"Model: {'Medical' if args.medical else 'Standard'} RAG")
    print(f"Samples: {args.samples if args.samples else 'All'}")
    print("=" * 80)

    results = {}
    for i, (script, name) in enumerate(analyses, 1):
        if str(i) in args.skip:
            print(f"\nSkipping {i}. {name}")
            continue

        script_path = explainability_dir / script
        success = run_analysis(script_path, medical=args.medical, samples=args.samples)
        results[name] = "✓ Success" if success else "✗ Failed"

    print("\n" + "=" * 80)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nResults:")
    for name, status in results.items():
        print(f"  {status}: {name}")

    output_dir = explainability_dir / "outputs" / ("medical_rag" if args.medical else "standard_rag")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
