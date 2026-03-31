"""
ByteTabNet Inference Script

Comprehensive inference script supporting:
- Task types: Classification, Regression, Generation (seq2seq)
- Modes: Single prediction, batch inference, interactive, evaluation
- Data formats: ROOT files, Parquet files, Images, Text
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import numpy as np
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

# Import ByteTabNet components
from byte_tabnet import (
    ByteTabNet,
    ByteTabNetSeq2Seq,
    ByteEmbedding,
)

# Import configs and loaders from trainer
from trainer import (
    ModelConfig,
    DataConfig,
    ClassificationTask,
    RegressionTask,
    GenerationTask,
    ParquetDataLoader,
    TextDataLoader,
    ImageDataLoader,
    ROOTDataLoader,
)

# Import attention analysis utilities
from attention_analysis import (
    AttentionTracer,
    plot_attention_heatmap,
    plot_attention_per_step,
    plot_attention_evolution,
    plot_attention_sparsity,
    create_attention_report,
)


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    task_type: str,
    model_config: ModelConfig,
) -> eqx.Module:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.eqx)
        task_type: "classification", "regression", or "generation"
        model_config: Model configuration

    Returns:
        Loaded model
    """
    key = jrandom.PRNGKey(0)  # Seed doesn't matter for loading

    # Create template model
    if task_type == "classification":
        handler = ClassificationTask(model_config.output_dim)
    elif task_type == "regression":
        handler = RegressionTask()
    elif task_type == "generation":
        handler = GenerationTask()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    template = handler.create_model(model_config, key)

    # Load weights
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = eqx.tree_deserialise_leaves(path, template)

    return model


def load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load configuration metadata from checkpoint directory."""
    path = Path(checkpoint_path)
    metadata_path = path.with_suffix('.json')

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)

    # Try to find in same directory
    checkpoint_dir = path.parent
    for json_file in checkpoint_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            if 'model_config' in metadata:
                return metadata

    print("Warning: No metadata found, using defaults")
    return {}


# ============================================================================
# Inference Engines
# ============================================================================

class InferenceEngine(ABC):
    """Abstract base class for inference."""

    def __init__(self, model: eqx.Module, config: ModelConfig, trace_attention: bool = False):
        self.model = model
        self.config = config
        self.trace_attention = trace_attention
        self.attention_tracer = AttentionTracer() if trace_attention else None

    @abstractmethod
    def predict(self, texts: List[str]) -> Array:
        """Predict on batch of texts."""
        pass

    @abstractmethod
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict on single text."""
        pass

    def get_attention_tracer(self) -> Optional[AttentionTracer]:
        """Get the attention tracer if enabled."""
        return self.attention_tracer

    def save_attention(self, output_path: str):
        """Save traced attention to file."""
        if self.attention_tracer:
            self.attention_tracer.save_to_file(output_path)
        else:
            print("Attention tracing is not enabled")


class ClassificationInference(InferenceEngine):
    """Inference engine for classification tasks."""

    def predict(self, texts: List[str]) -> Tuple[Array, Array, Array]:
        """
        Predict classes and probabilities for batch of texts.

        Returns:
            (predicted_classes, probabilities, attention_masks)
        """
        input_ids, attention_mask = ByteEmbedding.encode_batch(
            texts, max_length=self.config.max_seq_length
        )

        logits, masks, _ = self.model(input_ids, attention_mask, inference=True)
        probs = jax.nn.softmax(logits, axis=-1)
        pred_classes = jnp.argmax(probs, axis=-1)

        # Record attention if tracing is enabled
        if self.trace_attention and self.attention_tracer:
            for i in range(len(texts)):
                self.attention_tracer.record_attention(
                    masks[i] if len(masks) > 0 else jnp.array([]),
                    input_text=texts[i],
                    prediction=int(pred_classes[i]),
                )

        return pred_classes, probs, masks

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict for a single text."""
        pred_classes, probs, masks = self.predict([text])

        result = {
            'predicted_class': int(pred_classes[0]),
            'probabilities': probs[0].tolist(),
            'confidence': float(jnp.max(probs[0])),
            'feature_importance': jnp.mean(masks[0], axis=0).tolist() if len(masks[0]) > 0 else [],
        }

        # Include detailed attention if tracing
        if self.trace_attention and len(masks[0]) > 0:
            result['attention_per_step'] = masks[0].tolist()

        return result

    def predict_with_explanation(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """Predict with feature importance explanation."""
        result = self.predict_single(text)

        # Get top-k important features
        if result['feature_importance']:
            importance = jnp.array(result['feature_importance'])
            actual_k = min(top_k, len(importance))
            top_indices = jnp.argsort(importance)[-actual_k:][::-1]

            result['top_features'] = {
                'indices': top_indices.tolist(),
                'importance': importance[top_indices].tolist(),
            }

        return result


class RegressionInference(InferenceEngine):
    """Inference engine for regression tasks."""

    def predict(self, texts: List[str]) -> Tuple[Array, Array]:
        """Predict values for batch of texts."""
        input_ids, attention_mask = ByteEmbedding.encode_batch(
            texts, max_length=self.config.max_seq_length
        )

        logits, masks, _ = self.model(input_ids, attention_mask, inference=True)
        predictions = logits.squeeze(-1)

        # Record attention if tracing is enabled
        if self.trace_attention and self.attention_tracer:
            for i in range(len(texts)):
                self.attention_tracer.record_attention(
                    masks[i] if len(masks) > 0 else jnp.array([]),
                    input_text=texts[i],
                    prediction=float(predictions[i]),
                )

        return predictions, masks

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict for a single text."""
        predictions, masks = self.predict([text])

        result = {
            'predicted_value': float(predictions[0]),
            'feature_importance': jnp.mean(masks[0], axis=0).tolist() if len(masks[0]) > 0 else [],
        }

        # Include detailed attention if tracing
        if self.trace_attention and len(masks[0]) > 0:
            result['attention_per_step'] = masks[0].tolist()

        return result


class GenerationInference(InferenceEngine):
    """Inference engine for generation tasks."""

    def __init__(self, model: ByteTabNetSeq2Seq, config: ModelConfig):
        super().__init__(model, config)
        self.bos_token_id = 1
        self.eos_token_id = 2

    def generate(
        self,
        texts: List[str],
        max_length: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> List[str]:
        """Generate text for batch of inputs."""
        input_ids, attention_mask = ByteEmbedding.encode_batch(
            texts, max_length=self.config.max_seq_length
        )

        if key is None:
            key = jrandom.PRNGKey(42)

        # Generate using model's generate method
        generated_ids = self.model.generate(
            input_ids,
            attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            key=key,
        )

        # Decode generated sequences
        outputs = []
        for seq in generated_ids:
            # Remove BOS/EOS and decode bytes
            tokens = seq[1:]  # Skip BOS

            # Find EOS position
            eos_positions = jnp.where(tokens == self.eos_token_id, jnp.arange(len(tokens)), len(tokens))
            eos_pos = int(eos_positions.min())
            tokens = tokens[:eos_pos]

            # Decode bytes (subtract offset of 64)
            byte_vals = [int(t) - 64 for t in tokens if 64 <= t < 320]
            try:
                text = bytes(byte_vals).decode('utf-8', errors='ignore')
            except Exception:
                text = ""

            outputs.append(text)

        return outputs

    def generate_single(
        self,
        text: str,
        max_length: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """Generate for a single input."""
        key = jrandom.PRNGKey(np.random.randint(0, 10000))
        outputs = self.generate(
            [text],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            key=key,
        )
        return outputs[0]

    def predict(self, texts: List[str]) -> List[str]:
        """Alias for generate with greedy decoding."""
        return self.generate(texts, do_sample=False)

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Generate for single input and return as dict."""
        output = self.generate_single(text, do_sample=False)
        return {
            'generated_text': output,
        }


# ============================================================================
# Batch Processing
# ============================================================================

def batch_inference(
    inference_engine: InferenceEngine,
    texts: List[str],
    labels: Optional[Array] = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Tuple[List, Optional[Dict]]:
    """
    Run inference on a large dataset in batches.

    Args:
        inference_engine: Inference engine to use
        texts: List of input texts
        labels: Optional ground truth labels for evaluation
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar

    Returns:
        (predictions, metrics)
    """
    all_predictions = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Running inference")

    for i in iterator:
        batch_texts = texts[i:i + batch_size]

        if isinstance(inference_engine, ClassificationInference):
            pred_classes, probs, _ = inference_engine.predict(batch_texts)
            all_predictions.extend([
                {
                    'class': int(pred_classes[j]),
                    'probabilities': probs[j].tolist(),
                    'confidence': float(jnp.max(probs[j])),
                }
                for j in range(len(batch_texts))
            ])

        elif isinstance(inference_engine, RegressionInference):
            predictions, _ = inference_engine.predict(batch_texts)
            all_predictions.extend([
                {'value': float(predictions[j])}
                for j in range(len(batch_texts))
            ])

        elif isinstance(inference_engine, GenerationInference):
            outputs = inference_engine.generate(batch_texts, do_sample=False)
            all_predictions.extend([
                {'generated_text': output}
                for output in outputs
            ])

    # Compute metrics if labels provided
    metrics = None
    if labels is not None:
        metrics = compute_metrics(inference_engine, all_predictions, labels)

    return all_predictions, metrics


def compute_metrics(
    inference_engine: InferenceEngine,
    predictions: List[Dict],
    labels: Array,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if isinstance(inference_engine, ClassificationInference):
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        except ImportError:
            print("Warning: sklearn not available, computing only accuracy")
            pred_classes = jnp.array([p['class'] for p in predictions])
            accuracy = jnp.mean(pred_classes == labels)
            return {'accuracy': float(accuracy)}

        pred_classes = np.array([p['class'] for p in predictions])
        accuracy = accuracy_score(labels, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_classes, average='weighted', zero_division=0
        )

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    elif isinstance(inference_engine, RegressionInference):
        pred_values = jnp.array([p['value'] for p in predictions])
        mse = jnp.mean((pred_values - labels) ** 2)
        rmse = jnp.sqrt(mse)
        mae = jnp.mean(jnp.abs(pred_values - labels))

        ss_res = jnp.sum((labels - pred_values) ** 2)
        ss_tot = jnp.sum((labels - jnp.mean(labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }

    return {}


def save_predictions(
    predictions: List[Dict],
    output_path: str,
    input_texts: Optional[List[str]] = None,
    labels: Optional[Array] = None,
):
    """Save predictions to file."""
    try:
        import pandas as pd
    except ImportError:
        # Fallback to JSON
        output = []
        for i, pred in enumerate(predictions):
            item = {'index': i, **pred}
            if input_texts:
                item['input_text'] = input_texts[i]
            if labels is not None:
                item['true_label'] = float(labels[i])
            output.append(item)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Predictions saved to {output_path}")
        return

    # Use pandas
    df = pd.DataFrame(predictions)
    if input_texts:
        df.insert(0, 'input_text', input_texts)
    if labels is not None:
        df['true_label'] = labels

    # Save based on extension
    output_path = Path(output_path)
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix == '.json':
        df.to_json(output_path, orient='records', indent=2)
    elif output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path.with_suffix('.csv'), index=False)

    print(f"Predictions saved to {output_path}")


# ============================================================================
# Data Loading for Inference
# ============================================================================

def load_inference_data(
    data_config: DataConfig,
) -> Tuple[List[str], Optional[Array]]:
    """Load data for inference."""
    # Create appropriate data loader
    format_map = {
        "parquet": ParquetDataLoader,
        "text": TextDataLoader,
        "csv": TextDataLoader,
        "image": ImageDataLoader,
        "root": ROOTDataLoader,
    }

    loader_class = format_map.get(data_config.data_format.lower())
    if loader_class is None:
        raise ValueError(f"Unknown data format: {data_config.data_format}")

    loader = loader_class(data_config)
    data = loader.load_data()

    # For generation, data is (sources, targets)
    # For classification/regression, data is (texts, labels)
    if isinstance(data[1], list):  # Generation
        return data[0], None
    else:
        return data


# ============================================================================
# Command-line Interface
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference with ByteTabNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model loading
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.eqx file)",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "generation"],
        required=True,
        help="Task type",
    )

    # Input data
    parser.add_argument("--input-file", type=str, help="Path to input file for batch inference")
    parser.add_argument("--input-text", type=str, help="Single text for inference")
    parser.add_argument("--data-format", type=str, help="Data format (parquet, text, csv, image, root)")
    parser.add_argument("--text-column", type=str, help="Column name for text input")
    parser.add_argument("--label-column", type=str, help="Column name for labels (evaluation mode)")

    # Inference mode
    parser.add_argument(
        "--mode",
        choices=["batch", "single", "interactive"],
        default="batch",
        help="Inference mode",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for batch mode")

    # Output
    parser.add_argument("--output-file", type=str, help="Path to save predictions")
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Output file format",
    )

    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on test data with ground truth labels",
    )

    # Generation-specific
    parser.add_argument("--max-length", type=int, default=128, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")

    # Model config overrides
    parser.add_argument("--num-classes", type=int, help="Number of classes (if not in checkpoint metadata)")
    parser.add_argument("--max-seq-length", type=int, help="Max sequence length")

    # Attention tracing options
    parser.add_argument(
        "--trace-attention",
        action="store_true",
        help="Enable attention tracing for interpretability",
    )
    parser.add_argument(
        "--attention-output",
        type=str,
        help="Path to save traced attention scores (.json, .npz, .csv, or .parquet)",
    )
    parser.add_argument(
        "--attention-report",
        type=str,
        help="Directory to save comprehensive attention report for first sample",
    )
    parser.add_argument(
        "--visualize-attention",
        action="store_true",
        help="Generate attention visualizations (requires matplotlib)",
    )
    parser.add_argument(
        "--attention-sample-idx",
        type=int,
        default=0,
        help="Sample index for attention visualization/report",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load checkpoint metadata
    print(f"Loading checkpoint from {args.checkpoint}...")
    metadata = load_config_from_checkpoint(args.checkpoint)

    # Extract model config
    if 'model_config' in metadata:
        model_config_dict = metadata['model_config']
        model_config = ModelConfig(**model_config_dict)
    else:
        # Use defaults
        print("Warning: No model config in metadata, using defaults")
        model_config = ModelConfig()

    # Override with command-line args if provided
    if args.max_seq_length:
        model_config.max_seq_length = args.max_seq_length
    if args.num_classes:
        model_config.output_dim = args.num_classes

    # For classification, ensure output_dim is set
    if args.task == "classification" and not hasattr(model_config, 'output_dim'):
        if args.num_classes:
            model_config.output_dim = args.num_classes
        else:
            raise ValueError("--num-classes required for classification task")

    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(args.checkpoint, args.task, model_config)
    print("Model loaded successfully!")

    # Create inference engine
    trace_attention = args.trace_attention or args.attention_report or args.attention_output

    if args.task == "classification":
        engine = ClassificationInference(model, model_config, trace_attention=trace_attention)
    elif args.task == "regression":
        engine = RegressionInference(model, model_config, trace_attention=trace_attention)
    elif args.task == "generation":
        engine = GenerationInference(model, model_config)

    # Run inference based on mode
    if args.mode == "single":
        # Single text inference
        if not args.input_text:
            args.input_text = input("Enter text: ")

        print(f"\nInput: {args.input_text}")
        result = engine.predict_single(args.input_text)

        print("\nPrediction:")
        print(json.dumps(result, indent=2))

    elif args.mode == "interactive":
        # Interactive mode
        print("\nInteractive mode. Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                text = input("Enter text: ")
                if text.lower() in ['quit', 'exit', 'q']:
                    break

                if isinstance(engine, GenerationInference):
                    output = engine.generate_single(
                        text,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        do_sample=not args.greedy,
                    )
                    print(f"Generated: {output}\n")
                else:
                    result = engine.predict_single(text)
                    print(json.dumps(result, indent=2))
                    print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")

    elif args.mode == "batch":
        # Batch inference
        if not args.input_file:
            parser.error("--input-file required for batch mode")

        # Load data
        print(f"Loading data from {args.input_file}...")

        data_config = DataConfig(
            data_path=args.input_file,
            data_format=args.data_format or "text",
            text_column=args.text_column,
            label_column=args.label_column,
        )

        texts, labels = load_inference_data(data_config)
        print(f"Loaded {len(texts)} examples")

        # Run batch inference
        print("Running inference...")
        predictions, metrics = batch_inference(
            engine,
            texts,
            labels=labels if args.evaluate else None,
            batch_size=args.batch_size,
            show_progress=True,
        )

        # Print metrics if evaluating
        if metrics:
            print("\nEvaluation Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

        # Save predictions
        if args.output_file:
            save_predictions(
                predictions,
                args.output_file,
                input_texts=texts,
                labels=labels if args.evaluate else None,
            )
        else:
            # Print first few predictions
            print("\nFirst 5 predictions:")
            for i, pred in enumerate(predictions[:5]):
                print(f"{i}: {pred}")

    # Handle attention tracing outputs
    if trace_attention and engine.attention_tracer:
        print(f"\nAttention tracing enabled: {len(engine.attention_tracer.get_all_attention())} samples recorded")

        # Save attention scores
        if args.attention_output:
            engine.save_attention(args.attention_output)

        # Generate attention report
        if args.attention_report:
            try:
                create_attention_report(
                    engine.attention_tracer,
                    args.attention_sample_idx,
                    args.attention_report,
                    include_plots=args.visualize_attention,
                )
            except Exception as e:
                print(f"Warning: Could not generate attention report: {e}")

        # Generate visualizations
        if args.visualize_attention and not args.attention_report:
            try:
                from attention_analysis import (
                    plot_attention_heatmap,
                    plot_attention_per_step,
                    plot_attention_evolution,
                    plot_attention_sparsity,
                )

                record = engine.attention_tracer.get_attention(args.attention_sample_idx)
                if record:
                    print(f"\nGenerating attention visualizations for sample {args.attention_sample_idx}...")

                    plot_attention_heatmap(
                        record['masks'],
                        title=f"Attention Heatmap - Sample {args.attention_sample_idx}",
                    )

                    plot_attention_per_step(
                        record['masks'],
                        title=f"Top Features per Step - Sample {args.attention_sample_idx}",
                    )

                    plot_attention_evolution(
                        record['masks'],
                        title=f"Attention Evolution - Sample {args.attention_sample_idx}",
                    )

                    plot_attention_sparsity(
                        record['masks'],
                        title=f"Attention Sparsity - Sample {args.attention_sample_idx}",
                    )
                else:
                    print(f"Warning: Sample {args.attention_sample_idx} not found in attention tracer")

            except ImportError:
                print("Warning: matplotlib not available, skipping visualizations")
            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
