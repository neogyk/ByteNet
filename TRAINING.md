# ByteTabNet Training and Inference Guide

This guide explains how to use the comprehensive training and inference scripts for ByteTabNet.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Data Formats](#data-formats)
- [Configuration Files](#configuration-files)
- [Examples](#examples)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Training a Classification Model

```bash
python trainer.py \
  --task classification \
  --data-path data/train.parquet \
  --data-format parquet \
  --text-column text \
  --label-column label \
  --num-classes 5 \
  --batch-size 32 \
  --num-epochs 50
```

### 2. Running Inference

```bash
python inference.py \
  --checkpoint checkpoints/best_model.eqx \
  --task classification \
  --input-file data/test.parquet \
  --output-file predictions.csv
```

## Training

### Using Config Files (Recommended)

Create a config file (see `configs/` for examples) and run:

```bash
python trainer.py --config configs/classification_example.json
```

You can override config values with command-line arguments:

```bash
python trainer.py --config configs/classification_example.json --learning-rate 0.0005 --batch-size 64
```

### Command-Line Only

```bash
python trainer.py \
  --task <classification|regression|generation> \
  --data-path <path-to-data> \
  --data-format <parquet|root|text|csv|image> \
  [additional options...]
```

### Key Training Arguments

#### Task Configuration
- `--task`: Task type (`classification`, `regression`, `generation`)
- `--data-path`: Path to training data
- `--data-format`: Data format (`parquet`, `root`, `text`, `csv`, `image`)
- `--num-classes`: Number of classes (classification only)

#### Data Configuration
- `--text-column`: Column name for text input
- `--label-column`: Column name for labels
- `--target-column`: Column name for generation targets

#### Model Architecture
- `--max-seq-length`: Maximum sequence length (default: 512)
- `--embed-dim`: Embedding dimension (default: 64)
- `--hidden-dim`: Hidden dimension (default: 128)
- `--n-steps`: Number of TabNet decision steps (default: 5)
- `--n-d`: Decision embedding dimension (default: 64)
- `--n-a`: Attention embedding dimension (default: 64)
- `--pooling`: Sequence pooling method (`mean`, `max`, `attention`)

#### Training Hyperparameters
- `--batch-size`: Batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 100)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--optimizer`: Optimizer (`adam`, `adamw`)
- `--lr-schedule`: LR schedule (`constant`, `cosine`, `exponential`, `warmup_cosine`)
- `--patience`: Early stopping patience (default: 10)

#### Output
- `--checkpoint-dir`: Directory to save checkpoints (default: `./checkpoints`)

### Training Output

The trainer will:
1. Load and split your data into train/val/test sets
2. Initialize the model and optimizer
3. Train for the specified number of epochs
4. Save checkpoints (best model + periodic saves)
5. Print training progress and metrics
6. Evaluate on the test set at the end

Checkpoints are saved as:
- `best_model.eqx`: Best model based on validation loss
- `checkpoint_epoch_N.eqx`: Periodic checkpoints
- Corresponding `.json` files with metadata

## Inference

### Modes

#### 1. Single Text Inference

```bash
python inference.py \
  --checkpoint checkpoints/best_model.eqx \
  --task classification \
  --mode single \
  --input-text "Your text here"
```

#### 2. Batch Inference

```bash
python inference.py \
  --checkpoint checkpoints/best_model.eqx \
  --task classification \
  --mode batch \
  --input-file data/test.parquet \
  --data-format parquet \
  --text-column text \
  --output-file predictions.csv \
  --batch-size 64
```

#### 3. Interactive Mode

```bash
python inference.py \
  --checkpoint checkpoints/best_model.eqx \
  --task generation \
  --mode interactive \
  --temperature 0.7
```

Type your input and press Enter. Type `quit` or `exit` to stop.

### Evaluation Mode

Evaluate model performance on test data with ground truth labels:

```bash
python inference.py \
  --checkpoint checkpoints/best_model.eqx \
  --task classification \
  --input-file data/test.parquet \
  --label-column true_label \
  --evaluate \
  --output-file predictions_with_metrics.csv
```

This will compute and print:
- **Classification**: accuracy, precision, recall, F1
- **Regression**: MSE, RMSE, MAE, R²
- **Generation**: perplexity

### Generation-Specific Options

```bash
python inference.py \
  --checkpoint checkpoints/generation_best.eqx \
  --task generation \
  --input-file sources.txt \
  --max-length 128 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --greedy  # Use greedy decoding instead of sampling
```

### Key Inference Arguments

- `--checkpoint`: Path to model checkpoint (`.eqx` file)
- `--task`: Task type (must match training)
- `--mode`: Inference mode (`single`, `batch`, `interactive`)
- `--input-file`: Input file for batch mode
- `--input-text`: Single text for single mode
- `--output-file`: Path to save predictions
- `--output-format`: Output format (`csv`, `json`, `parquet`)
- `--evaluate`: Enable evaluation mode (requires `--label-column`)
- `--batch-size`: Batch size for processing

## Data Formats

### 1. Parquet Files

**Format:** Apache Parquet columnar format

**Training:**
```bash
python trainer.py \
  --data-format parquet \
  --data-path data.parquet \
  --text-column text \
  --label-column label
```

**Multiple columns** (concatenated):
```bash
python trainer.py \
  --data-format parquet \
  --data-path data.parquet \
  --parquet-columns "col1,col2,col3" \
  --label-column target
```

### 2. ROOT Files (HEP Format)

**Format:** CERN ROOT format for particle physics

**Training:**
```bash
python trainer.py \
  --data-format root \
  --data-path experiment.root \
  --root-tree-name Events \
  --root-branches "jet_pt,jet_eta,met,lepton_pt" \
  --label-column event_class
```

### 3. Text/CSV Files

**Format:** Plain text, CSV, or TSV

**Training:**
```bash
python trainer.py \
  --data-format text \
  --data-path data.csv \
  --text-column text \
  --label-column label
```

### 4. Images

**Format:** PNG, JPG, JPEG images

**Training:**
```bash
python trainer.py \
  --data-format image \
  --image-dir ./images \
  --image-label-file labels.csv
```

The label file should have format:
```
image_path,label
img1.png,0
img2.jpg,1
```

## Configuration Files

Configuration files use JSON or YAML format with three sections:

### Example Config Structure

```json
{
  "model": {
    "max_seq_length": 512,
    "embed_dim": 64,
    "hidden_dim": 128,
    ...
  },
  "training": {
    "task_type": "classification",
    "learning_rate": 0.001,
    "batch_size": 32,
    ...
  },
  "data": {
    "data_path": "data/train.parquet",
    "data_format": "parquet",
    "text_column": "text",
    "label_column": "label",
    ...
  }
}
```

See `configs/` directory for complete examples:
- `classification_example.json`: Text classification
- `regression_example.json`: Regression task
- `generation_example.json`: Seq2seq generation
- `root_example.json`: ROOT file processing

## Examples

### Example 1: Text Classification

**Data:** Parquet file with text and labels

```bash
# Training
python trainer.py \
  --config configs/classification_example.json \
  --data-path data/reviews.parquet \
  --text-column review_text \
  --label-column sentiment \
  --num-classes 3

# Inference
python inference.py \
  --checkpoint checkpoints/classification/best_model.eqx \
  --task classification \
  --input-file data/test_reviews.parquet \
  --text-column review_text \
  --output-file predictions.csv \
  --evaluate \
  --label-column sentiment
```

### Example 2: Regression on Scientific Data

**Data:** ROOT file from particle physics experiment

```bash
# Training
python trainer.py \
  --config configs/root_example.json \
  --task regression \
  --data-path experiment.root \
  --root-tree-name Events \
  --root-branches "jet_pt,jet_eta,jet_phi,met" \
  --label-column energy

# Inference
python inference.py \
  --checkpoint checkpoints/regression/best_model.eqx \
  --task regression \
  --data-format root \
  --input-file test_experiment.root \
  --output-file energy_predictions.parquet
```

### Example 3: Sequence-to-Sequence Generation

**Data:** Translation or text generation

```bash
# Training
python trainer.py \
  --config configs/generation_example.json \
  --data-path translations.parquet \
  --text-column source \
  --target-column target \
  --n-decoder-layers 6

# Interactive generation
python inference.py \
  --checkpoint checkpoints/generation/best_model.eqx \
  --task generation \
  --mode interactive \
  --temperature 0.7 \
  --top-p 0.9
```

### Example 4: Image Classification

**Data:** Images with labels

```bash
# Training
python trainer.py \
  --task classification \
  --data-format image \
  --image-dir ./dataset/images \
  --image-label-file ./dataset/labels.csv \
  --num-classes 10 \
  --max-seq-length 1024

# Inference
python inference.py \
  --checkpoint checkpoints/best_model.eqx \
  --task classification \
  --data-format image \
  --image-dir ./test_images \
  --image-label-file ./test_labels.csv \
  --output-file image_predictions.csv
```

## Tips and Best Practices

### 1. Choosing Hyperparameters

- **Batch size**: Start with 32, increase if you have more memory
- **Learning rate**: Start with 1e-3, decrease if training is unstable
- **LR schedule**: `warmup_cosine` works well for most tasks
- **Patience**: 10-15 for early stopping
- **n_steps**: 5-6 for TabNet decision steps (more = better but slower)

### 2. Data Preprocessing

- **Text**: ByteTabNet works directly on UTF-8 bytes, no tokenization needed
- **Numeric data**: Convert to string representation for byte encoding
- **Images**: Can use raw bytes or pixel values
- **Long sequences**: Truncated to `max_seq_length`

### 3. Checkpointing

- Best model is always saved as `best_model.eqx`
- Periodic checkpoints saved every `save_every` epochs
- Only `keep_best_k` checkpoints retained (sorted by validation loss)
- Metadata (configs, metrics) saved alongside as `.json`

### 4. Memory Management

- Reduce `batch_size` if out of memory
- Reduce `max_seq_length` for shorter sequences
- Use `virtual_batch_size` to control TabNet batch norm behavior

### 5. Evaluation

- Always use `--evaluate` flag with test data to get metrics
- Check feature importance via TabNet attention masks
- Compare multiple checkpoints on validation set

## Troubleshooting

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues

JAX will automatically use GPU if available. To force CPU:
```bash
export JAX_PLATFORMS=cpu
python trainer.py ...
```

### Memory Errors

- Reduce `--batch-size`
- Reduce `--max-seq-length`
- Reduce `--hidden-dim` or `--n-steps`

### Slow Training

- Increase `--batch-size` (if memory allows)
- Reduce `--eval-every` to evaluate less frequently
- Use `--lr-schedule constant` to skip LR schedule computation

## Advanced Usage

### Custom Data Loaders

You can extend the data loaders by subclassing `DataLoader` in `trainer.py`:

```python
class CustomDataLoader(DataLoader):
    def load_data(self):
        # Your custom loading logic
        texts = [...]
        labels = [...]
        return texts, labels
```

### Mixed Data Types

For ROOT/Parquet files with multiple branches/columns, the loader automatically concatenates them:

```bash
python trainer.py \
  --data-format parquet \
  --parquet-columns "feature1,feature2,feature3" \
  --label-column target
```

This creates text like: `feature1:value1 | feature2:value2 | feature3:value3`

### Feature Importance Analysis

TabNet provides interpretable feature selection. Access attention masks during inference:

```python
from inference import ClassificationInference

engine = ClassificationInference(model, config)
result = engine.predict_with_explanation(text, top_k=10)
print(result['top_features'])
```

## Citation

If you use ByteTabNet in your research, please cite the original TabNet paper:

```
@article{arik2019tabnet,
  title={TabNet: Attentive Interpretable Tabular Learning},
  author={Arik, Sercan O and Pfister, Tomas},
  journal={arXiv preprint arXiv:1908.07442},
  year={2019}
}
```

## Support

For issues, questions, or contributions, please check the repository documentation or open an issue on GitHub.
