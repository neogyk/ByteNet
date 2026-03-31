# TabNet Attention Tracing Guide

This guide explains how to trace, analyze, and visualize TabNet's attention scores at each decision step for model interpretability.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Tracing Attention During Inference](#tracing-attention-during-inference)
- [Analyzing Saved Attention Data](#analyzing-saved-attention-data)
- [Visualization Options](#visualization-options)
- [Understanding Attention Scores](#understanding-attention-scores)
- [Use Cases](#use-cases)

## Overview

TabNet uses a sequential attention mechanism across multiple decision steps to select important features. Each step produces an attention mask showing which features the model focuses on. This attention tracing capability allows you to:

- **Understand model decisions**: See which features contribute to predictions
- **Debug model behavior**: Identify unexpected attention patterns
- **Validate feature engineering**: Check if important features get attention
- **Compare samples**: Analyze attention differences between predictions
- **Generate interpretability reports**: Create visualizations for stakeholders

## Quick Start

### 1. Run Inference with Attention Tracing

```bash
python inference.py \
  --checkpoint checkpoints/best_model.eqx \
  --task classification \
  --input-file data/test.parquet \
  --trace-attention \
  --attention-output attention_data.json
```

### 2. Analyze the Traced Attention

```bash
python analyze_attention.py \
  --input-file attention_data.json \
  --summary \
  --visualize-sample 0 \
  --visualize \
  --save-plots
```

## Tracing Attention During Inference

### Enable Attention Tracing

Add the `--trace-attention` flag to any inference command:

```bash
python inference.py \
  --checkpoint model.eqx \
  --task classification \
  --input-file test_data.parquet \
  --trace-attention
```

### Save Attention to File

Use `--attention-output` to save traced attention:

```bash
# Save as JSON (human-readable)
python inference.py \
  --checkpoint model.eqx \
  --task classification \
  --input-file test.parquet \
  --trace-attention \
  --attention-output attention.json

# Save as NPZ (compressed, efficient)
python inference.py \
  --checkpoint model.eqx \
  --task classification \
  --input-file test.parquet \
  --trace-attention \
  --attention-output attention.npz

# Save as CSV (tabular format)
python inference.py \
  --checkpoint model.eqx \
  --task classification \
  --input-file test.parquet \
  --trace-attention \
  --attention-output attention.csv
```

### Generate Comprehensive Report

Create a full attention analysis report for a sample:

```bash
python inference.py \
  --checkpoint model.eqx \
  --task classification \
  --input-file test.parquet \
  --trace-attention \
  --attention-report ./reports/sample_0 \
  --attention-sample-idx 0 \
  --visualize-attention
```

This creates:
- `attention_stats_sample_0.json`: Statistical summary
- `attention_masks_sample_0.npy`: Raw attention masks
- `attention_report_sample_0.md`: Text report
- `heatmap_sample_0.png`: Attention heatmap
- `top_features_sample_0.png`: Top features per step
- `evolution_sample_0.png`: Feature attention evolution
- `sparsity_sample_0.png`: Sparsity analysis

### Interactive Visualization

For interactive plots during inference:

```bash
python inference.py \
  --checkpoint model.eqx \
  --task classification \
  --input-text "Your test input" \
  --mode single \
  --trace-attention \
  --visualize-attention
```

## Analyzing Saved Attention Data

### Load and Summarize

```bash
python analyze_attention.py \
  --input-file attention_data.json \
  --summary
```

Output:
```
SUMMARY STATISTICS
======================================================================

Total samples: 100
Decision steps: 5
Feature dimension: 128

Overall attention:
  Mean: 0.0234
  Std:  0.0145
  Sparsity: 32.5%

Per-step statistics:

  Step 0:
    Mean: 0.0289
    Sparsity: 45.2%
    Top features: [42, 78, 15, 91, 33]
  ...
```

### Visualize Specific Sample

```bash
python analyze_attention.py \
  --input-file attention_data.json \
  --visualize-sample 5 \
  --visualize \
  --save-plots \
  --output-dir ./visualizations
```

### Compare Multiple Samples

```bash
python analyze_attention.py \
  --input-file attention_data.json \
  --compare-samples "0,5,10,15,20" \
  --save-plots \
  --output-dir ./comparisons
```

### Export Summary Table

```bash
# Export to CSV
python analyze_attention.py \
  --input-file attention_data.json \
  --export-summary attention_summary.csv

# Export to Parquet
python analyze_attention.py \
  --input-file attention_data.json \
  --export-summary attention_summary.parquet
```

### Generate Detailed Report

```bash
python analyze_attention.py \
  --input-file attention_data.json \
  --sample-report 0 \
  --output-dir ./reports \
  --save-plots
```

## Visualization Options

### 1. Attention Heatmap

Shows attention scores across all decision steps and features.

- **X-axis**: Feature indices
- **Y-axis**: Decision steps
- **Color**: Attention intensity

**When to use**: Overview of attention patterns

### 2. Top Features per Step

Bar charts showing the most attended features at each decision step.

- Shows top-k features (default: 15)
- Separate chart for each step

**When to use**: Identify key features at each step

### 3. Attention Evolution

Line plot showing how attention to specific features changes across steps.

- Tracks selected features over time
- Shows feature importance trajectory

**When to use**: Understand feature selection dynamics

### 4. Attention Sparsity

Bar chart showing the percentage of features with significant attention at each step.

- Threshold: 0.01 (configurable)
- Compares to random baseline

**When to use**: Measure feature selection sparsity

## Understanding Attention Scores

### What are Attention Scores?

Attention scores are values between 0 and 1 that indicate how much the model "attends to" or "focuses on" each feature at each decision step.

- **High score (>0.1)**: Strong attention, feature is important
- **Medium score (0.01-0.1)**: Moderate attention
- **Low score (<0.01)**: Minimal attention, feature is ignored

### Interpreting the Masks

```python
# Attention masks shape: (n_steps, feature_dim)
# Example: (5, 128) for 5 decision steps and 128 features

masks[0]  # Attention at step 0
masks[0][42]  # Attention to feature 42 at step 0
```

### Sparsity

TabNet uses **sparsemax** activation instead of softmax, resulting in:
- Exactly zero attention for most features
- Sparse attention patterns (only a few features per step)
- Improved interpretability

Typical sparsity: 20-40% of features get non-zero attention per step.

### Feature Importance

**Aggregated importance** = Average attention across all steps:

```python
import numpy as np

# Load attention
tracer = AttentionTracer()
tracer.load_from_file('attention.json')

# Get masks for sample 0
record = tracer.get_attention(0)
masks = record['masks']

# Compute feature importance
importance = np.mean(masks, axis=0)  # Average across steps

# Get top-k features
top_k = 10
top_features = np.argsort(importance)[-top_k:][::-1]
```

## Use Cases

### 1. Model Debugging

**Problem**: Model makes wrong predictions on specific samples

**Solution**:
```bash
# Trace attention for misclassified samples
python inference.py \
  --checkpoint model.eqx \
  --task classification \
  --input-file misclassified.parquet \
  --trace-attention \
  --attention-report ./debug_reports

# Analyze what features the model focused on
python analyze_attention.py \
  --input-file attention.json \
  --visualize-sample 0 \
  --visualize
```

**Look for**:
- Unexpected features getting high attention
- Important features being ignored
- Attention patterns different from correct predictions

### 2. Feature Engineering Validation

**Problem**: Want to verify that engineered features are useful

**Solution**:
```bash
# Run inference and track attention
python inference.py \
  --checkpoint model.eqx \
  --task regression \
  --input-file validation.parquet \
  --trace-attention \
  --attention-output attention.json

# Export summary statistics
python analyze_attention.py \
  --input-file attention.json \
  --export-summary feature_importance.csv
```

**Check**:
- Are your engineered features in the top-k attended features?
- Do they get consistent attention across samples?
- How does their attention compare to raw features?

### 3. Model Comparison

**Problem**: Compare attention patterns between two models

**Solution**:
```bash
# Model 1
python inference.py \
  --checkpoint model1.eqx \
  --input-file test.parquet \
  --trace-attention \
  --attention-output model1_attention.json

# Model 2
python inference.py \
  --checkpoint model2.eqx \
  --input-file test.parquet \
  --trace-attention \
  --attention-output model2_attention.json

# Compare
python analyze_attention.py \
  --input-file model1_attention.json \
  --export-summary model1_summary.csv

python analyze_attention.py \
  --input-file model2_attention.json \
  --export-summary model2_summary.csv

# Manually compare the CSVs or write a comparison script
```

### 4. Explainable AI Reports

**Problem**: Need to explain model predictions to stakeholders

**Solution**:
```bash
# Generate comprehensive reports
for i in {0..9}; do
  python inference.py \
    --checkpoint model.eqx \
    --task classification \
    --input-file important_cases.parquet \
    --trace-attention \
    --attention-report ./reports/case_$i \
    --attention-sample-idx $i \
    --visualize-attention
done
```

This creates professional reports with:
- Visual attention heatmaps
- Feature importance rankings
- Step-by-step decision analysis
- Statistical summaries

### 5. Attention-Based Feature Selection

**Problem**: Want to select features based on model attention

**Solution**:
```python
from attention_analysis import AttentionTracer
import numpy as np

# Load attention data
tracer = AttentionTracer()
tracer.load_from_file('attention.json')

# Aggregate importance across all samples
all_masks = [r['masks'] for r in tracer.get_all_attention()]
avg_masks = np.mean(all_masks, axis=0)  # (n_steps, feature_dim)

# Compute overall feature importance
feature_importance = np.mean(avg_masks, axis=0)  # (feature_dim,)

# Select top-k features
k = 50
top_features = np.argsort(feature_importance)[-k:][::-1]

print(f"Selected {k} features based on attention: {top_features}")
```

## Programmatic Usage

### In Python Scripts

```python
from inference import ClassificationInference, load_model_from_checkpoint
from attention_analysis import AttentionTracer, plot_attention_heatmap

# Load model
model = load_model_from_checkpoint('model.eqx', 'classification', config)

# Create engine with attention tracing
engine = ClassificationInference(model, config, trace_attention=True)

# Run inference
predictions, probs, masks = engine.predict(['test input 1', 'test input 2'])

# Get attention tracer
tracer = engine.get_attention_tracer()

# Save attention
tracer.save_to_file('attention.json')

# Visualize
record = tracer.get_attention(0)
plot_attention_heatmap(record['masks'], title='My Analysis')

# Get statistics
stats = tracer.summary_statistics(0)
print(f"Mean attention: {stats['overall']['mean_attention']}")
```

## Best Practices

1. **Always trace on validation/test data**, not training data
2. **Use NPZ format** for large-scale attention tracking (more efficient than JSON)
3. **Generate reports** for representative samples (correct, incorrect, edge cases)
4. **Compare attention** across different model versions to track changes
5. **Combine with feature names** if available, map feature indices to meaningful names
6. **Check sparsity** to ensure TabNet's sparse attention is working
7. **Look for patterns** across samples, not just individual cases

## Advanced Tips

### Custom Feature Labels

```python
from attention_analysis import plot_attention_heatmap

# Define feature names
feature_names = ['age', 'income', 'credit_score', ...]  # Your actual features

# Plot with labels
plot_attention_heatmap(
    masks,
    feature_labels=feature_names,
    title='Attention with Feature Names'
)
```

### Attention Aggregation Strategies

```python
import numpy as np

# Strategy 1: Mean across steps
mean_importance = np.mean(masks, axis=0)

# Strategy 2: Max across steps (most attended at any step)
max_importance = np.max(masks, axis=0)

# Strategy 3: Weighted by step (later steps more important)
weights = np.array([1, 2, 3, 4, 5])  # 5 steps
weighted_importance = np.average(masks, axis=0, weights=weights)

# Strategy 4: First step only (initial feature selection)
first_step_importance = masks[0]
```

## Troubleshooting

**Q: Attention all zeros?**
- Check that model was trained properly
- Verify using `--trace-attention` flag
- Ensure model is in inference mode

**Q: Matplotlib errors?**
- Install: `pip install matplotlib`
- Or use `--no-visualize` and analyze JSON/CSV output

**Q: Memory issues with large attention files?**
- Use NPZ format instead of JSON
- Process samples in batches
- Use `--attention-sample-idx` for specific samples only

**Q: Can't interpret feature indices?**
- Map indices to your original feature names
- Use custom feature labels in plots
- Check your data preprocessing pipeline

## Examples

See [TRAINING.md](TRAINING.md) for complete examples of using attention tracing in end-to-end workflows.
