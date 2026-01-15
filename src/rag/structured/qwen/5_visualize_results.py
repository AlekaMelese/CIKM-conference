#!/usr/bin/env python3
"""
Phase 2 - Step 5: Visualize RAG Results
Creates publication-quality SVG visualizations of evaluation results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import RAGConfig

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Configure matplotlib for SVG output
plt.rcParams['svg.fonttype'] = 'none'  # Editable text in SVG
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300


class RAGVisualizer:
    """Create publication-quality SVG visualizations"""

    def __init__(self):
        self.config = RAGConfig
        self.eval_results = None
        self.viz_dir = self.config.RAG_OUTPUTS_DIR / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

    def load_results(self):
        """Load evaluation results"""
        print("=" * 80)
        print("STEP 5: VISUALIZE RAG RESULTS")
        print("=" * 80)
        print(f"\nLoading results from: {self.config.EVALUATION_RESULTS_PATH}")

        with open(self.config.EVALUATION_RESULTS_PATH, 'r') as f:
            self.eval_results = json.load(f)

        print(f"✓ Loaded evaluation results")
        print(f"  Total samples: {self.eval_results['evaluation_metadata']['total_samples']}")

    def create_metrics_overview(self):
        """Create overview of key metrics (SVG)"""
        print("\n📊 Creating metrics overview...")

        aggregate = self.eval_results['aggregate_metrics']

        # Prepare data
        metrics = {
            'ROUGE-1': aggregate.get('rouge1_f_mean', 0),
            'ROUGE-2': aggregate.get('rouge2_f_mean', 0),
            'ROUGE-L': aggregate.get('rougeL_f_mean', 0),
            'BLEU': aggregate.get('bleu_mean', 0),
            'Coverage': aggregate.get('coverage_mean', 0),
            'Completeness': aggregate.get('completeness_score_mean', 0)
        }

        errors = {
            'ROUGE-1': aggregate.get('rouge1_f_std', 0),
            'ROUGE-2': aggregate.get('rouge2_f_std', 0),
            'ROUGE-L': aggregate.get('rougeL_f_std', 0),
            'BLEU': aggregate.get('bleu_std', 0),
            'Coverage': aggregate.get('coverage_std', 0),
            'Completeness': aggregate.get('completeness_score_std', 0)
        }

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        x_pos = np.arange(len(metrics))
        bars = ax.bar(x_pos, metrics.values(), yerr=errors.values(),
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Color bars by performance
        colors = ['#2ecc71' if v > 0.5 else '#f39c12' if v > 0.3 else '#e74c3c'
                  for v in metrics.values()]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('RAG Performance: Key Evaluation Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics.keys(), rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Target: 0.5')
        ax.legend()

        plt.tight_layout()
        output_path = self.viz_dir / "metrics_overview.svg"
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def create_structure_completeness(self):
        """Visualize 11-section structure completeness (SVG)"""
        print("\n📊 Creating structure completeness visualization...")

        per_sample = self.eval_results['per_sample_metrics']

        # Count section presence across samples
        section_counts = {}
        for sample in per_sample:
            if 'section_presence' in sample:
                for section, present in sample['section_presence'].items():
                    if section not in section_counts:
                        section_counts[section] = 0
                    section_counts[section] += present

        if not section_counts:
            print("  ⚠️  No section presence data available")
            return

        # Calculate percentages
        total_samples = len(per_sample)
        section_percentages = {k: (v / total_samples) * 100 for k, v in section_counts.items()}

        # Clean section names
        clean_names = {
            r'Case Type:': 'Case Type',
            r'•\s*Patient & Service:': 'Patient & Service',
            r'•\s*Chief Complaint': 'Chief Complaint',
            r'•\s*History of Present Illness': 'HPI',
            r'•\s*Past Medical': 'Past Medical Hx',
            r'•\s*Medications': 'Medications',
            r'•\s*Physical Examination': 'Physical Exam',
            r'•\s*Investigations': 'Investigations',
            r'•\s*Assessment': 'Assessment',
            r'•\s*Discharge Condition': 'Discharge Condition',
            r'•\s*Follow-Up': 'Follow-Up'
        }

        sections = list(section_percentages.keys())
        percentages = list(section_percentages.values())
        labels = [clean_names.get(s, s) for s in sections]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, percentages, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Color by completeness
        colors = ['#2ecc71' if p >= 95 else '#f39c12' if p >= 80 else '#e74c3c'
                  for p in percentages]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=9)

        ax.set_xlabel('Presence (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Section', fontsize=12, fontweight='bold')
        ax.set_title('11-Section Structure Completeness\n(Percentage of Samples with Each Section)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 105)
        ax.axvline(x=95, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target: 95%')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.legend()

        plt.tight_layout()
        output_path = self.viz_dir / "structure_completeness.svg"
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def create_rouge_comparison(self):
        """Compare ROUGE scores with error bars (SVG)"""
        print("\n📊 Creating ROUGE comparison...")

        aggregate = self.eval_results['aggregate_metrics']

        rouge_metrics = {
            'ROUGE-1': {
                'Precision': (aggregate.get('rouge1_p_mean', 0), aggregate.get('rouge1_p_std', 0)),
                'Recall': (aggregate.get('rouge1_r_mean', 0), aggregate.get('rouge1_r_std', 0)),
                'F1': (aggregate.get('rouge1_f_mean', 0), aggregate.get('rouge1_f_std', 0))
            },
            'ROUGE-2': {
                'Precision': (aggregate.get('rouge2_p_mean', 0), aggregate.get('rouge2_p_std', 0)),
                'Recall': (aggregate.get('rouge2_r_mean', 0), aggregate.get('rouge2_r_std', 0)),
                'F1': (aggregate.get('rouge2_f_mean', 0), aggregate.get('rouge2_f_std', 0))
            },
            'ROUGE-L': {
                'Precision': (aggregate.get('rougeL_p_mean', 0), aggregate.get('rougeL_p_std', 0)),
                'Recall': (aggregate.get('rougeL_r_mean', 0), aggregate.get('rougeL_r_std', 0)),
                'F1': (aggregate.get('rougeL_f_mean', 0), aggregate.get('rougeL_f_std', 0))
            }
        }

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(rouge_metrics))
        width = 0.25

        for i, metric_type in enumerate(['Precision', 'Recall', 'F1']):
            means = [rouge_metrics[rouge][metric_type][0] for rouge in rouge_metrics]
            stds = [rouge_metrics[rouge][metric_type][1] for rouge in rouge_metrics]

            ax.bar(x + i*width, means, width, label=metric_type,
                   yerr=stds, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_xlabel('ROUGE Variant', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('ROUGE Scores: Precision, Recall, and F1',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(rouge_metrics.keys())
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.viz_dir / "rouge_comparison.svg"
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def create_quality_radar(self):
        """Create radar chart of quality dimensions (SVG)"""
        print("\n📊 Creating quality radar chart...")

        aggregate = self.eval_results['aggregate_metrics']

        # Quality dimensions
        categories = ['ROUGE-L', 'BLEU', 'Coverage', 'Completeness',
                     'Entity F1', 'Low Hallucination']

        values = [
            aggregate.get('rougeL_f_mean', 0),
            aggregate.get('bleu_mean', 0),
            aggregate.get('coverage_mean', 0),
            aggregate.get('completeness_score_mean', 0),
            aggregate.get('entity_f1_mean', 0),
            1.0 - aggregate.get('hallucination_rate_mean', 0)  # Invert hallucination
        ]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=2, label='RAG Performance', color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('RAG Quality Dimensions\n(Multi-Dimensional Performance)',
                    fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        output_path = self.viz_dir / "quality_radar.svg"
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def create_distribution_plots(self):
        """Create distribution plots for key metrics (SVG)"""
        print("\n📊 Creating metric distributions...")

        per_sample = self.eval_results['per_sample_metrics']

        # Extract metrics
        rouge_l = [s.get('rougeL_f', 0) for s in per_sample if 'rougeL_f' in s]
        completeness = [s.get('completeness_score', 0) for s in per_sample if 'completeness_score' in s]
        hallucination = [s.get('hallucination_rate', 0) for s in per_sample if 'hallucination_rate' in s]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Distribution of Key Metrics Across Samples',
                    fontsize=16, fontweight='bold')

        # ROUGE-L distribution
        if rouge_l:
            axes[0, 0].hist(rouge_l, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            axes[0, 0].axvline(np.mean(rouge_l), color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(rouge_l):.3f}')
            axes[0, 0].set_xlabel('ROUGE-L F1', fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontweight='bold')
            axes[0, 0].set_title('ROUGE-L Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

        # Completeness distribution
        if completeness:
            axes[0, 1].hist(completeness, bins=20, alpha=0.7, color='#2ecc71', edgecolor='black')
            axes[0, 1].axvline(np.mean(completeness), color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(completeness):.3f}')
            axes[0, 1].set_xlabel('Completeness Score', fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontweight='bold')
            axes[0, 1].set_title('Structure Completeness Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

        # Hallucination distribution
        if hallucination:
            axes[1, 0].hist(hallucination, bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
            axes[1, 0].axvline(np.mean(hallucination), color='blue', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(hallucination):.3f}')
            axes[1, 0].set_xlabel('Hallucination Rate', fontweight='bold')
            axes[1, 0].set_ylabel('Frequency', fontweight='bold')
            axes[1, 0].set_title('Hallucination Rate Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        # Box plot comparison
        data_to_plot = []
        labels = []
        if rouge_l:
            data_to_plot.append(rouge_l)
            labels.append('ROUGE-L')
        if completeness:
            data_to_plot.append(completeness)
            labels.append('Completeness')
        if hallucination:
            data_to_plot.append(hallucination)
            labels.append('Hallucination')

        if data_to_plot:
            bp = axes[1, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1, 1].set_ylabel('Score', fontweight='bold')
            axes[1, 1].set_title('Metric Distributions (Box Plot)')
            axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.viz_dir / "metric_distributions.svg"
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print(f"\nGenerating SVG visualizations...")
        print(f"Output directory: {self.viz_dir}")

        self.create_metrics_overview()
        self.create_structure_completeness()
        self.create_rouge_comparison()
        self.create_quality_radar()
        self.create_distribution_plots()

        print(f"\n✅ All visualizations created!")
        print(f"   Location: {self.viz_dir}")


def main():
    """Main execution"""
    # Create visualizer
    viz = RAGVisualizer()

    # Load results
    viz.load_results()

    # Generate all visualizations
    viz.generate_all_visualizations()

    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 SVG files created in: {viz.viz_dir}")
    print(f"\nGenerated visualizations:")
    for svg_file in sorted(viz.viz_dir.glob("*.svg")):
        print(f"  • {svg_file.name}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
