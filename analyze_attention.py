#!/usr/bin/env python3
"""
Standalone Attention Analysis Script

Analyze and visualize TabNet attention scores for model interpretability.
Can load saved attention data or run inference with attention tracing.
"""

import argparse
import sys
from pathlib import Path

from attention_analysis import (
    AttentionTracer,
    plot_attention_heatmap,
    plot_attention_per_step,
    plot_attention_evolution,
    plot_attention_sparsity,
    create_attention_report,
    compare_attention_across_samples,
    export_attention_summary,
)


def analyze_from_file(args):
    """Analyze attention from saved file."""
    print(f"Loading attention data from {args.input_file}...")

    tracer = AttentionTracer()
    tracer.load_from_file(args.input_file)

    n_samples = len(tracer.get_all_attention())
    print(f"Loaded {n_samples} samples")

    # Print summary statistics
    if args.summary:
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)

        stats = tracer.summary_statistics()
        print(f"\nTotal samples: {stats['n_samples']}")
        print(f"Decision steps: {stats['n_steps']}")
        print(f"Feature dimension: {stats['feature_dim']}")
        print(f"\nOverall attention:")
        print(f"  Mean: {stats['overall']['mean_attention']:.4f}")
        print(f"  Std:  {stats['overall']['std_attention']:.4f}")
        print(f"  Sparsity: {stats['overall']['sparsity']:.2%}")

        print(f"\nPer-step statistics:")
        for step in range(stats['n_steps']):
            step_stats = stats[f'step_{step}']
            print(f"\n  Step {step}:")
            print(f"    Mean: {step_stats['mean_attention']:.4f}")
            print(f"    Sparsity: {step_stats['sparsity']:.2%}")
            print(f"    Top features: {step_stats['top_k_features'][:5]}")

    # Export summary table
    if args.export_summary:
        print(f"\nExporting summary to {args.export_summary}...")
        export_attention_summary(tracer, args.export_summary)

    # Generate report for specific sample
    if args.sample_report is not None:
        print(f"\nGenerating detailed report for sample {args.sample_report}...")
        output_dir = args.output_dir or f"./attention_reports/sample_{args.sample_report}"
        create_attention_report(
            tracer,
            args.sample_report,
            output_dir,
            include_plots=args.visualize,
        )

    # Compare multiple samples
    if args.compare_samples:
        sample_indices = [int(idx.strip()) for idx in args.compare_samples.split(',')]
        print(f"\nComparing samples: {sample_indices}...")

        save_path = args.output_dir + "/comparison.png" if args.output_dir else "attention_comparison.png"
        compare_attention_across_samples(
            tracer,
            sample_indices,
            save_path=save_path if args.save_plots else None,
            show=args.visualize,
        )

    # Visualize specific sample
    if args.visualize_sample is not None:
        print(f"\nVisualizing sample {args.visualize_sample}...")

        record = tracer.get_attention(args.visualize_sample)
        if record is None:
            print(f"Error: Sample {args.visualize_sample} not found")
            return

        masks = record['masks']

        # Generate all visualizations
        output_dir = Path(args.output_dir) if args.output_dir else Path(".")

        plot_attention_heatmap(
            masks,
            title=f"Attention Heatmap - Sample {args.visualize_sample}",
            save_path=output_dir / f"heatmap_{args.visualize_sample}.png" if args.save_plots else None,
            show=args.visualize,
        )

        plot_attention_per_step(
            masks,
            title=f"Top Features per Step - Sample {args.visualize_sample}",
            save_path=output_dir / f"top_features_{args.visualize_sample}.png" if args.save_plots else None,
            show=args.visualize,
        )

        plot_attention_evolution(
            masks,
            title=f"Attention Evolution - Sample {args.visualize_sample}",
            save_path=output_dir / f"evolution_{args.visualize_sample}.png" if args.save_plots else None,
            show=args.visualize,
        )

        plot_attention_sparsity(
            masks,
            title=f"Attention Sparsity - Sample {args.visualize_sample}",
            save_path=output_dir / f"sparsity_{args.visualize_sample}.png" if args.save_plots else None,
            show=args.visualize,
        )


def run_inference_with_tracing(args):
    """Run inference with attention tracing."""
    print("Running inference with attention tracing...")
    print("Use inference.py with --trace-attention flag instead:")
    print()
    print(f"  python inference.py \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --task {args.task} \\")
    print(f"    --input-file {args.data_file} \\")
    print(f"    --trace-attention \\")
    print(f"    --attention-output attention_data.json")
    print()
    print("Then use this script to analyze the saved attention data.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TabNet attention scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to saved attention data (.json, .npz)",
    )

    # Analysis options
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics",
    )
    parser.add_argument(
        "--sample-report",
        type=int,
        help="Generate detailed report for specific sample",
    )
    parser.add_argument(
        "--visualize-sample",
        type=int,
        help="Visualize attention for specific sample",
    )
    parser.add_argument(
        "--compare-samples",
        type=str,
        help="Compare attention across samples (comma-separated indices)",
    )
    parser.add_argument(
        "--export-summary",
        type=str,
        help="Export summary table to file (.csv or .parquet)",
    )

    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualizations interactively",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./attention_analysis",
        help="Output directory for reports and plots",
    )

    # Alternative: run inference with tracing
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Model checkpoint (for inference mode)",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Task type (for inference mode)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Input data file (for inference mode)",
    )

    args = parser.parse_args()

    # Check mode
    if args.input_file:
        # Analysis mode
        analyze_from_file(args)
    elif args.checkpoint and args.task and args.data_file:
        # Inference mode
        run_inference_with_tracing(args)
    else:
        parser.error("Either --input-file or (--checkpoint, --task, --data-file) required")


if __name__ == "__main__":
    main()
