"""
ByteTabNet Attention Analysis

Utilities for extracting, analyzing, and visualizing TabNet attention scores
at each decision step for model interpretability.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Visualization functions will not work.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ============================================================================
# Attention Score Extraction
# ============================================================================

class AttentionTracer:
    """
    Extract and track attention scores from ByteTabNet models.

    Captures attention masks at each decision step for interpretability.
    """

    def __init__(self):
        self.attention_history = []
        self.current_sample_idx = 0

    def record_attention(
        self,
        masks: Array,
        input_text: Optional[str] = None,
        prediction: Optional[Any] = None,
        sample_idx: Optional[int] = None,
    ):
        """
        Record attention masks for a sample.

        Args:
            masks: Attention masks of shape (n_steps, feature_dim) or (batch, n_steps, feature_dim)
            input_text: Optional input text for reference
            prediction: Optional model prediction
            sample_idx: Optional sample index
        """
        if masks.ndim == 3:
            # Batch of masks, process each
            for i in range(masks.shape[0]):
                self._record_single(
                    masks[i],
                    input_text=input_text if isinstance(input_text, str) else (input_text[i] if input_text else None),
                    prediction=prediction[i] if isinstance(prediction, (list, np.ndarray)) else prediction,
                    sample_idx=sample_idx + i if sample_idx is not None else self.current_sample_idx + i,
                )
            self.current_sample_idx += masks.shape[0]
        else:
            # Single sample
            self._record_single(masks, input_text, prediction, sample_idx)

    def _record_single(
        self,
        masks: Array,
        input_text: Optional[str] = None,
        prediction: Optional[Any] = None,
        sample_idx: Optional[int] = None,
    ):
        """Record attention for a single sample."""
        if sample_idx is None:
            sample_idx = self.current_sample_idx
            self.current_sample_idx += 1

        record = {
            'sample_idx': sample_idx,
            'masks': np.array(masks),  # Shape: (n_steps, feature_dim)
            'input_text': input_text,
            'prediction': prediction,
            'n_steps': masks.shape[0],
            'feature_dim': masks.shape[1],
        }

        self.attention_history.append(record)

    def get_attention(self, sample_idx: int) -> Optional[Dict]:
        """Get attention record for a specific sample."""
        for record in self.attention_history:
            if record['sample_idx'] == sample_idx:
                return record
        return None

    def get_all_attention(self) -> List[Dict]:
        """Get all attention records."""
        return self.attention_history

    def clear(self):
        """Clear attention history."""
        self.attention_history = []
        self.current_sample_idx = 0

    def summary_statistics(self, sample_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute summary statistics of attention scores.

        Args:
            sample_idx: If provided, compute stats for specific sample, else all samples

        Returns:
            Dictionary with statistics
        """
        if sample_idx is not None:
            record = self.get_attention(sample_idx)
            if record is None:
                raise ValueError(f"Sample {sample_idx} not found")
            records = [record]
        else:
            records = self.attention_history

        if not records:
            return {}

        # Aggregate statistics
        all_masks = [r['masks'] for r in records]

        stats = {
            'n_samples': len(records),
            'n_steps': records[0]['n_steps'],
            'feature_dim': records[0]['feature_dim'],
        }

        # Per-step statistics
        for step in range(stats['n_steps']):
            step_masks = np.stack([m[step] for m in all_masks])  # (n_samples, feature_dim)

            stats[f'step_{step}'] = {
                'mean_attention': float(np.mean(step_masks)),
                'std_attention': float(np.std(step_masks)),
                'max_attention': float(np.max(step_masks)),
                'min_attention': float(np.min(step_masks)),
                'sparsity': float(np.mean(step_masks > 0.01)),  # Fraction of features with attention > threshold
                'top_k_features': np.argsort(np.mean(step_masks, axis=0))[-10:][::-1].tolist(),
            }

        # Overall statistics
        all_masks_flat = np.concatenate([m.flatten() for m in all_masks])
        stats['overall'] = {
            'mean_attention': float(np.mean(all_masks_flat)),
            'std_attention': float(np.std(all_masks_flat)),
            'sparsity': float(np.mean(all_masks_flat > 0.01)),
        }

        return stats

    def save_to_file(self, output_path: str):
        """Save attention history to file."""
        output_path = Path(output_path)

        if output_path.suffix == '.json':
            # Convert numpy arrays to lists for JSON
            records = []
            for record in self.attention_history:
                r = record.copy()
                r['masks'] = r['masks'].tolist()
                records.append(r)

            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)

        elif output_path.suffix == '.npz':
            # Save as numpy compressed archive
            np.savez_compressed(
                output_path,
                **{f'sample_{r["sample_idx"]}_masks': r['masks'] for r in self.attention_history}
            )

            # Save metadata separately
            metadata = [{
                'sample_idx': r['sample_idx'],
                'input_text': r['input_text'],
                'prediction': str(r['prediction']),
                'n_steps': r['n_steps'],
                'feature_dim': r['feature_dim'],
            } for r in self.attention_history]

            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

        elif PANDAS_AVAILABLE and output_path.suffix in ['.csv', '.parquet']:
            # Save as table (flattened)
            rows = []
            for record in self.attention_history:
                for step in range(record['n_steps']):
                    for feat_idx in range(record['feature_dim']):
                        rows.append({
                            'sample_idx': record['sample_idx'],
                            'step': step,
                            'feature_idx': feat_idx,
                            'attention_score': record['masks'][step, feat_idx],
                            'input_text': record['input_text'],
                            'prediction': str(record['prediction']),
                        })

            df = pd.DataFrame(rows)
            if output_path.suffix == '.csv':
                df.to_csv(output_path, index=False)
            else:
                df.to_parquet(output_path, index=False)

        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        print(f"Attention history saved to {output_path}")

    def load_from_file(self, input_path: str):
        """Load attention history from file."""
        input_path = Path(input_path)

        if input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                records = json.load(f)

            for record in records:
                record['masks'] = np.array(record['masks'])

            self.attention_history = records

        elif input_path.suffix == '.npz':
            # Load numpy archive
            data = np.load(input_path)

            # Load metadata
            metadata_path = input_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = []

            # Reconstruct records
            self.attention_history = []
            for key in data.files:
                sample_idx = int(key.split('_')[1])
                masks = data[key]

                # Find metadata
                meta = next((m for m in metadata if m['sample_idx'] == sample_idx), {})

                record = {
                    'sample_idx': sample_idx,
                    'masks': masks,
                    'input_text': meta.get('input_text'),
                    'prediction': meta.get('prediction'),
                    'n_steps': masks.shape[0],
                    'feature_dim': masks.shape[1],
                }
                self.attention_history.append(record)

        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        print(f"Loaded {len(self.attention_history)} attention records from {input_path}")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_attention_heatmap(
    masks: Array,
    title: str = "TabNet Attention Scores",
    step_labels: Optional[List[str]] = None,
    feature_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot attention masks as a heatmap.

    Args:
        masks: Attention masks of shape (n_steps, feature_dim)
        title: Plot title
        step_labels: Optional labels for decision steps
        feature_labels: Optional labels for features
        figsize: Figure size
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    masks = np.array(masks)
    n_steps, feature_dim = masks.shape

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(masks, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set labels
    if step_labels is None:
        step_labels = [f'Step {i}' for i in range(n_steps)]
    if feature_labels is None:
        # Only show some feature labels if too many
        if feature_dim > 20:
            feature_labels = [str(i) if i % (feature_dim // 10) == 0 else '' for i in range(feature_dim)]
        else:
            feature_labels = [str(i) for i in range(feature_dim)]

    ax.set_yticks(range(n_steps))
    ax.set_yticklabels(step_labels)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Decision Step')
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Score', rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_per_step(
    masks: Array,
    title: str = "Attention Distribution per Step",
    figsize: Tuple[int, int] = (14, 8),
    top_k: int = 15,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot top-k features by attention score for each decision step.

    Args:
        masks: Attention masks of shape (n_steps, feature_dim)
        title: Plot title
        figsize: Figure size
        top_k: Number of top features to show per step
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    masks = np.array(masks)
    n_steps, feature_dim = masks.shape

    fig, axes = plt.subplots(1, n_steps, figsize=figsize, sharey=True)
    if n_steps == 1:
        axes = [axes]

    for step in range(n_steps):
        ax = axes[step]
        step_attention = masks[step]

        # Get top-k features
        top_indices = np.argsort(step_attention)[-top_k:]
        top_scores = step_attention[top_indices]

        # Plot
        y_pos = np.arange(len(top_indices))
        ax.barh(y_pos, top_scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'F{i}' for i in top_indices])
        ax.set_xlabel('Attention Score')
        ax.set_title(f'Step {step}')
        ax.grid(axis='x', alpha=0.3)

    axes[0].set_ylabel('Feature Index')
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_evolution(
    masks: Array,
    feature_indices: Optional[List[int]] = None,
    title: str = "Feature Attention Evolution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot how attention to specific features evolves across decision steps.

    Args:
        masks: Attention masks of shape (n_steps, feature_dim)
        feature_indices: Indices of features to track (if None, use top-k overall)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    masks = np.array(masks)
    n_steps, feature_dim = masks.shape

    if feature_indices is None:
        # Select top-k features by average attention
        avg_attention = np.mean(masks, axis=0)
        feature_indices = np.argsort(avg_attention)[-10:][::-1]

    fig, ax = plt.subplots(figsize=figsize)

    for feat_idx in feature_indices:
        attention_over_steps = masks[:, feat_idx]
        ax.plot(range(n_steps), attention_over_steps, marker='o', label=f'Feature {feat_idx}')

    ax.set_xlabel('Decision Step')
    ax.set_ylabel('Attention Score')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_sparsity(
    masks: Array,
    threshold: float = 0.01,
    title: str = "Attention Sparsity per Step",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot sparsity of attention (fraction of features above threshold) per step.

    Args:
        masks: Attention masks of shape (n_steps, feature_dim)
        threshold: Threshold for considering a feature as "attended"
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    masks = np.array(masks)
    n_steps, feature_dim = masks.shape

    # Compute sparsity per step
    sparsity = np.mean(masks > threshold, axis=1) * 100  # Percentage

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(range(n_steps), sparsity, color='steelblue', alpha=0.7)
    ax.axhline(y=100/feature_dim, color='r', linestyle='--', label='Random baseline')

    ax.set_xlabel('Decision Step')
    ax.set_ylabel(f'% Features with Attention > {threshold}')
    ax.set_title(title)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f'Step {i}' for i in range(n_steps)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_attention_report(
    tracer: AttentionTracer,
    sample_idx: int,
    output_dir: str,
    include_plots: bool = True,
):
    """
    Generate a comprehensive attention analysis report for a sample.

    Args:
        tracer: AttentionTracer with recorded attention
        sample_idx: Sample index to analyze
        output_dir: Directory to save report and visualizations
        include_plots: Whether to generate visualization plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get attention record
    record = tracer.get_attention(sample_idx)
    if record is None:
        raise ValueError(f"Sample {sample_idx} not found in tracer")

    masks = record['masks']

    # Generate statistics
    stats = tracer.summary_statistics(sample_idx)

    # Save statistics as JSON
    with open(output_dir / f'attention_stats_sample_{sample_idx}.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Save raw attention scores
    np.save(output_dir / f'attention_masks_sample_{sample_idx}.npy', masks)

    # Generate report text
    report_lines = [
        f"# Attention Analysis Report - Sample {sample_idx}",
        f"\n## Input",
        f"Text: {record['input_text'][:200] if record['input_text'] else 'N/A'}...",
        f"\n## Prediction",
        f"{record['prediction']}",
        f"\n## Attention Summary",
        f"- Number of decision steps: {record['n_steps']}",
        f"- Feature dimension: {record['feature_dim']}",
        f"- Overall mean attention: {stats['overall']['mean_attention']:.4f}",
        f"- Overall sparsity: {stats['overall']['sparsity']:.2%}",
        f"\n## Per-Step Statistics",
    ]

    for step in range(record['n_steps']):
        step_stats = stats[f'step_{step}']
        report_lines.extend([
            f"\n### Step {step}",
            f"- Mean: {step_stats['mean_attention']:.4f}",
            f"- Std: {step_stats['std_attention']:.4f}",
            f"- Max: {step_stats['max_attention']:.4f}",
            f"- Sparsity: {step_stats['sparsity']:.2%}",
            f"- Top features: {step_stats['top_k_features'][:5]}",
        ])

    # Save report
    with open(output_dir / f'attention_report_sample_{sample_idx}.md', 'w') as f:
        f.write('\n'.join(report_lines))

    # Generate plots
    if include_plots and MATPLOTLIB_AVAILABLE:
        plot_attention_heatmap(
            masks,
            title=f"Attention Heatmap - Sample {sample_idx}",
            save_path=output_dir / f'heatmap_sample_{sample_idx}.png',
            show=False,
        )

        plot_attention_per_step(
            masks,
            title=f"Top Features per Step - Sample {sample_idx}",
            save_path=output_dir / f'top_features_sample_{sample_idx}.png',
            show=False,
        )

        plot_attention_evolution(
            masks,
            title=f"Attention Evolution - Sample {sample_idx}",
            save_path=output_dir / f'evolution_sample_{sample_idx}.png',
            show=False,
        )

        plot_attention_sparsity(
            masks,
            title=f"Attention Sparsity - Sample {sample_idx}",
            save_path=output_dir / f'sparsity_sample_{sample_idx}.png',
            show=False,
        )

    print(f"\nAttention report generated in {output_dir}/")
    print(f"  - Statistics: attention_stats_sample_{sample_idx}.json")
    print(f"  - Report: attention_report_sample_{sample_idx}.md")
    if include_plots:
        print(f"  - Visualizations: 4 PNG files")


# ============================================================================
# Comparison and Analysis
# ============================================================================

def compare_attention_across_samples(
    tracer: AttentionTracer,
    sample_indices: List[int],
    title: str = "Attention Comparison Across Samples",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Compare attention patterns across multiple samples.

    Args:
        tracer: AttentionTracer with recorded attention
        sample_indices: List of sample indices to compare
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    n_samples = len(sample_indices)

    # Get all records
    records = [tracer.get_attention(idx) for idx in sample_indices]
    if any(r is None for r in records):
        raise ValueError("Some sample indices not found")

    n_steps = records[0]['n_steps']

    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)
    if n_samples == 1:
        axes = [axes]

    for i, (record, ax) in enumerate(zip(records, axes)):
        masks = record['masks']

        im = ax.imshow(masks, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_ylabel(f'Sample {record["sample_idx"]}\nStep')

        if i == n_samples - 1:
            ax.set_xlabel('Feature Index')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def export_attention_summary(
    tracer: AttentionTracer,
    output_path: str,
):
    """
    Export summary statistics for all samples to a table.

    Args:
        tracer: AttentionTracer with recorded attention
        output_path: Path to save summary table (CSV or Parquet)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for exporting tables")

    rows = []
    for record in tracer.get_all_attention():
        sample_idx = record['sample_idx']
        stats = tracer.summary_statistics(sample_idx)

        row = {
            'sample_idx': sample_idx,
            'input_text': record['input_text'][:100] if record['input_text'] else '',
            'prediction': str(record['prediction']),
            'n_steps': stats['n_steps'],
            'feature_dim': stats['feature_dim'],
            'overall_mean': stats['overall']['mean_attention'],
            'overall_std': stats['overall']['std_attention'],
            'overall_sparsity': stats['overall']['sparsity'],
        }

        # Add per-step stats
        for step in range(stats['n_steps']):
            step_stats = stats[f'step_{step}']
            row[f'step{step}_mean'] = step_stats['mean_attention']
            row[f'step{step}_sparsity'] = step_stats['sparsity']

        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path.with_suffix('.csv'), index=False)

    print(f"Attention summary exported to {output_path}")
