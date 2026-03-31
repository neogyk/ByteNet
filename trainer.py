"""
ByteTabNet Training Script

Comprehensive training script supporting:
- Task types: Classification, Regression, Generation (seq2seq)
- Data formats: ROOT files, Parquet files, Images, Text
- Features: Checkpointing, metrics tracking, early stopping, LR scheduling
"""

import argparse
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import warnings

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import numpy as np
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

# Import ByteTabNet components
from byte_tabnet import (
    ByteTabNet,
    ByteTabNetSeq2Seq,
    ByteEmbedding,
    tabnet_loss,
    tabnet_sparsity_loss,
    seq2seq_loss_with_tabnet_sparsity,
    seq2seq_kl_div_loss,
    prepare_seq2seq_batch,
)
from tokenizer import Tokenizer, ByteTokenizer, create_tokenizer

# Import dataset components
from dataset import (
    DataConfig,
    DataLoader,
    ParquetDataLoader,
    TextDataLoader,
    ImageDataLoader,
    ROOTDataLoader,
    HuggingFaceDataLoader,
)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    max_seq_length: int = 2048
    embed_dim: int = 64
    hidden_dim: int = 512
    n_steps: int = 3
    n_d: int = 512
    n_a: int = 512
    gamma: float = 1.5
    n_shared: int = 2
    n_step: int = 4
    virtual_batch_size: int = 128
    pooling: str = "attention"  # "mean", "max", or "attention"
    use_positional: bool = True
    vocab_size: int = 320
    n_decoder_layers: int = 4  # For generation tasks
    max_target_length: int = 1024  # Decoder positional embedding length
    # Tokenizer settings
    tokenizer_type: str = "byte"  # "byte" or "bpe"
    bpe_encoding: str = "cl100k_base"  # tiktoken encoding name (pre-trained BPE)
    bpe_tokenizer_path: Optional[str] = None  # path to custom-trained BPE model
    bpe_vocab_size: int = 8192  # vocab size for training custom BPE
    train_bpe: bool = False  # train a new BPE tokenizer on the dataset


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    task_type: str = "generation"  # "classification", "regression", "generation"
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "adamw"  # "adam" or "adamw"

    # Regularization
    sparsity_weight: float = 1e-3
    label_smoothing: float = 0.0
    kl_weight: float = 0.0  # KL divergence weight for generation tasks (0.0 disables it)
    gradient_clip: Optional[float] = 1.0

    # Learning rate scheduling
    lr_schedule: str = "warmup_cosine"  # "constant", "cosine", "exponential", "warmup_cosine"
    warmup_steps: int = 500
    min_lr: float = 1e-6

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "./checkpoints"
    keep_best_k: int = 3

    # Logging
    log_every: int = 10
    eval_every: int = 100

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle_seed: int = 42


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints using Equinox serialization."""

    def __init__(self, output_dir: str, keep_best_k: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_k = keep_best_k
        self.checkpoints = []  # List of (epoch, val_loss, path)

    def save_checkpoint(
        self,
        model: eqx.Module,
        epoch: int,
        val_loss: float,
        metadata: Dict[str, Any],
        is_best: bool = False,
    ):
        """Save a model checkpoint."""
        if is_best:
            path = self.output_dir / "best_model.eqx"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.eqx"

        # Save model using Equinox
        eqx.tree_serialise_leaves(path, model)

        # Save metadata
        metadata_full = {
            'epoch': epoch,
            'val_loss': float(val_loss),
            **metadata
        }
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata_full, f, indent=2)

        if not is_best:
            self.checkpoints.append((epoch, val_loss, path))
            self._cleanup_old_checkpoints()

        print(f"  Checkpoint saved: {path}")

    def _cleanup_old_checkpoints(self):
        """Keep only the best K checkpoints."""
        if len(self.checkpoints) > self.keep_best_k:
            # Sort by validation loss
            self.checkpoints.sort(key=lambda x: x[1])
            # Remove worst checkpoints
            for _, _, path in self.checkpoints[self.keep_best_k:]:
                if path.exists() and "best" not in path.name:
                    path.unlink()
                    json_path = path.with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()
            self.checkpoints = self.checkpoints[:self.keep_best_k]

    def load_checkpoint(self, model_template: eqx.Module, checkpoint_path: str):
        """Load a checkpoint."""
        path = Path(checkpoint_path)
        model = eqx.tree_deserialise_leaves(path, model_template)

        # Load metadata
        json_path = path.with_suffix('.json')
        metadata = {}
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)

        return model, metadata


# ============================================================================
# Abstract Base Classes
# ============================================================================

class TaskHandler(ABC):
    """Abstract base class for task-specific logic."""

    @abstractmethod
    def create_model(self, config: ModelConfig, key: PRNGKeyArray) -> eqx.Module:
        """Create a model for this task."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: eqx.Module,
        batch: Tuple,
        sparsity_weight: float,
        label_smoothing: float = 0.0,
        kl_weight: float = 0.0,
    ) -> Tuple[Array, Tuple]:
        """Compute loss for this task."""
        pass

    @abstractmethod
    def compute_metrics(self, predictions: Array, labels: Array) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass


# ============================================================================
# Task Handlers
# ============================================================================

class ClassificationTask(TaskHandler):
    """Handler for classification tasks."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def create_model(self, config: ModelConfig, key: PRNGKeyArray) -> ByteTabNet:
        return ByteTabNet(
            output_dim=self.num_classes,
            max_seq_length=config.max_seq_length,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_steps=config.n_steps,
            n_d=config.n_d,
            n_a=config.n_a,
            gamma=config.gamma,
            n_shared=config.n_shared,
            n_step=config.n_step,
            virtual_batch_size=config.virtual_batch_size,
            use_positional=config.use_positional,
            vocab_size=config.vocab_size,
            key=key,
        )

    def compute_loss(
        self,
        model: ByteTabNet,
        batch: Tuple,
        sparsity_weight: float,
        label_smoothing: float = 0.0,
        kl_weight: float = 0.0,
    ) -> Tuple[Array, Tuple]:
        input_ids, attention_mask, labels = batch
        logits, masks, state = model(input_ids, attention_mask, inference=False)

        # Use existing tabnet_loss function
        loss = tabnet_loss(logits, labels, masks, sparsity_weight=sparsity_weight)

        return loss, (logits, masks, state)

    def compute_metrics(self, predictions: Array, labels: Array) -> Dict[str, float]:
        """Compute classification metrics."""
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        except ImportError:
            warnings.warn("sklearn not available, computing only accuracy")
            pred_classes = jnp.argmax(predictions, axis=-1)
            accuracy = jnp.mean(pred_classes == labels)
            return {'accuracy': float(accuracy)}

        pred_classes = jnp.argmax(predictions, axis=-1)
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


class RegressionTask(TaskHandler):
    """Handler for regression tasks."""

    def create_model(self, config: ModelConfig, key: PRNGKeyArray) -> ByteTabNet:
        return ByteTabNet(
            output_dim=1,  # Single output for regression
            max_seq_length=config.max_seq_length,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_steps=config.n_steps,
            n_d=config.n_d,
            n_a=config.n_a,
            gamma=config.gamma,
            n_shared=config.n_shared,
            n_step=config.n_step,
            virtual_batch_size=config.virtual_batch_size,
            use_positional=config.use_positional,
            vocab_size=config.vocab_size,
            key=key,
        )

    def compute_loss(
        self,
        model: ByteTabNet,
        batch: Tuple,
        sparsity_weight: float,
        label_smoothing: float = 0.0,
        kl_weight: float = 0.0,
    ) -> Tuple[Array, Tuple]:
        input_ids, attention_mask, labels = batch
        logits, masks, state = model(input_ids, attention_mask, inference=False)

        # MSE loss + sparsity regularization
        predictions = logits.squeeze(-1)
        mse = jnp.mean((predictions - labels) ** 2)
        sparse_loss = tabnet_sparsity_loss(masks)
        total_loss = mse + sparsity_weight * sparse_loss

        return total_loss, (logits, masks, state)

    def compute_metrics(self, predictions: Array, labels: Array) -> Dict[str, float]:
        """Compute regression metrics."""
        if predictions.ndim > 1:
            predictions = predictions.squeeze(-1)

        mse = jnp.mean((predictions - labels) ** 2)
        rmse = jnp.sqrt(mse)
        mae = jnp.mean(jnp.abs(predictions - labels))

        # R² score
        ss_res = jnp.sum((labels - predictions) ** 2)
        ss_tot = jnp.sum((labels - jnp.mean(labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }


class GenerationTask(TaskHandler):
    """Handler for sequence-to-sequence generation tasks."""

    def create_model(self, config: ModelConfig, key: PRNGKeyArray) -> ByteTabNetSeq2Seq:
        return ByteTabNetSeq2Seq(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_steps=config.n_steps,
            n_d=config.n_d,
            n_a=config.n_a,
            gamma=config.gamma,
            virtual_batch_size=config.virtual_batch_size,
            n_decoder_layers=config.n_decoder_layers,
            max_target_length=config.max_target_length,
            key=key,
        )

    def compute_loss(
        self,
        model: ByteTabNetSeq2Seq,
        batch: Tuple,
        sparsity_weight: float,
        label_smoothing: float = 0.0,
        kl_weight: float = 0.0,
    ) -> Tuple[Array, Tuple]:
        input_ids, input_mask, target_ids, target_mask = batch

        logits, masks, state = model(
            input_ids,
            target_ids,
            input_mask,
            target_mask,
            inference=False,
        )

        # Use existing seq2seq loss with optional KL divergence
        # logits[:, :-1] predicts target_ids[:, 1:] (teacher forcing shift)
        loss = seq2seq_loss_with_tabnet_sparsity(
            logits[:, :-1],
            target_ids[:, 1:],
            masks,
            target_mask[:, 1:],
            sparsity_weight=sparsity_weight,
            label_smoothing=label_smoothing,
            kl_weight=kl_weight,
        )

        return loss, (logits, masks, state)

    def compute_metrics(self, predictions: Array, labels: Array) -> Dict[str, float]:
        """Compute generation metrics."""
        # Compute perplexity
        log_probs = jax.nn.log_softmax(predictions, axis=-1)

        # Get log probabilities of true labels
        # predictions shape: (batch, seq_len, vocab_size)
        # labels shape: (batch, seq_len)
        batch_size, seq_len = labels.shape

        # Safely gather log probs
        labels_expanded = labels[..., None]  # (batch, seq_len, 1)
        selected_log_probs = jnp.take_along_axis(log_probs, labels_expanded, axis=-1).squeeze(-1)

        # Average cross-entropy
        ce = -jnp.mean(selected_log_probs)
        perplexity = jnp.exp(ce)

        # KL divergence between predicted and target distributions
        kl = float(seq2seq_kl_div_loss(predictions, labels))

        return {
            'perplexity': float(perplexity),
            'cross_entropy': float(ce),
            'kl_divergence': kl,
        }


# ============================================================================
# Main Trainer Class
# ============================================================================

class Trainer:
    """Main training orchestrator."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config

        # Tokenizer (may be overridden after prepare_data when train_bpe=True)
        self.tokenizer: Tokenizer = self._create_tokenizer()

        # Initialize components
        self.task_handler = self._create_task_handler()
        self.data_loader = self._create_data_loader()
        self.checkpoint_manager = CheckpointManager(
            training_config.checkpoint_dir,
            keep_best_k=training_config.keep_best_k,
        )

        # Training state
        self.model = None
        self.optimizer = None
        self.opt_state = None
        self.train_step_fn = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Data
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _create_task_handler(self) -> TaskHandler:
        """Create appropriate task handler."""
        if self.training_config.task_type == "classification":
            if self.data_config.num_classes is None:
                raise ValueError("num_classes must be specified for classification task")
            return ClassificationTask(self.data_config.num_classes)
        elif self.training_config.task_type == "regression":
            return RegressionTask()
        elif self.training_config.task_type == "generation":
            return GenerationTask()
        else:
            raise ValueError(f"Unknown task type: {self.training_config.task_type}")

    def _create_data_loader(self) -> DataLoader:
        """Create appropriate data loader."""
        format_map = {
            "parquet": ParquetDataLoader,
            "text": TextDataLoader,
            "csv": TextDataLoader,
            "image": ImageDataLoader,
            "root": ROOTDataLoader,
            "huggingface": HuggingFaceDataLoader,
        }

        loader_class = format_map.get(self.data_config.data_format.lower())
        if loader_class is None:
            raise ValueError(f"Unknown data format: {self.data_config.data_format}")

        return loader_class(self.data_config)

    def _create_tokenizer(self, train_texts: Optional[List[str]] = None) -> Tokenizer:
        """Create tokenizer from model config."""
        mc = self.model_config
        if mc.train_bpe and train_texts is None:
            # Defer BPE training until prepare_data provides the corpus
            return ByteTokenizer()  # placeholder
        tok = create_tokenizer(
            tokenizer_type=mc.tokenizer_type,
            bpe_encoding=mc.bpe_encoding,
            bpe_tokenizer_path=mc.bpe_tokenizer_path,
            train_bpe=mc.train_bpe,
            bpe_vocab_size=mc.bpe_vocab_size,
            train_texts=train_texts,
        )
        # Sync vocab_size in config so model gets the right size
        self.model_config.vocab_size = tok.vocab_size
        return tok

    def _create_optimizer(self, total_steps: int) -> optax.GradientTransformation:
        """Create optimizer with learning rate schedule."""
        # Learning rate schedule
        if self.training_config.lr_schedule == "constant":
            schedule = self.training_config.learning_rate
        elif self.training_config.lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=self.training_config.learning_rate,
                decay_steps=total_steps,
                alpha=self.training_config.min_lr / self.training_config.learning_rate,
            )
        elif self.training_config.lr_schedule == "exponential":
            schedule = optax.exponential_decay(
                init_value=self.training_config.learning_rate,
                transition_steps=total_steps // 10,
                decay_rate=0.9,
            )
        elif self.training_config.lr_schedule == "warmup_cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.training_config.learning_rate,
                warmup_steps=self.training_config.warmup_steps,
                decay_steps=total_steps,
                end_value=self.training_config.min_lr,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {self.training_config.lr_schedule}")

        # Optimizer
        if self.training_config.optimizer == "adam":
            optimizer = optax.adam(learning_rate=schedule)
        elif self.training_config.optimizer == "adamw":
            optimizer = optax.adamw(
                learning_rate=schedule,
                weight_decay=self.training_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")

        # Add gradient clipping if specified
        if self.training_config.gradient_clip:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.training_config.gradient_clip),
                optimizer,
            )

        return optimizer

    def _create_train_step(self):
        """Create JIT-compiled training step function."""

        @eqx.filter_jit
        def train_step(model, opt_state, batch):
            def loss_fn(model):
                loss, aux = self.task_handler.compute_loss(
                    model,
                    batch,
                    self.training_config.sparsity_weight,
                    self.training_config.label_smoothing,
                    self.training_config.kl_weight,
                )
                return loss, aux

            (loss, (logits, masks, state)), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(model)

            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)

            return model, opt_state, loss, logits, masks

        return train_step

    def prepare_data(self):
        """Load and split data into train/val/test sets."""
        print("Loading data...")
        data = self.data_loader.load_data()

        if self.training_config.task_type == "generation":
            source_texts, target_texts = data
            data_pairs = list(zip(source_texts, target_texts))
        else:
            texts, labels = data
            data_pairs = list(zip(texts, labels))

        # Shuffle
        rng = np.random.RandomState(self.training_config.shuffle_seed)
        rng.shuffle(data_pairs)

        # Split
        n = len(data_pairs)
        n_train = int(n * self.training_config.train_split)
        n_val = int(n * self.training_config.val_split)

        self.train_data = data_pairs[:n_train]
        self.val_data = data_pairs[n_train:n_train + n_val]
        self.test_data = data_pairs[n_train + n_val:]

        print(f"Data splits: train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}")

        # Train BPE tokenizer on the training corpus if requested
        if self.model_config.train_bpe and self.model_config.tokenizer_type == "bpe":
            if self.training_config.task_type == "generation":
                corpus = [s for s, _ in self.train_data] + [t for _, t in self.train_data]
            else:
                corpus = [s for s, _ in self.train_data]
            print(f"Training BPE tokenizer on {len(corpus)} texts (vocab_size={self.model_config.bpe_vocab_size})...")
            self.tokenizer = self._create_tokenizer(train_texts=corpus)
            print(f"BPE tokenizer ready — vocab_size={self.tokenizer.vocab_size}")

    def create_batches(self, data: List[Tuple], shuffle: bool = True):
        """Create batches from data."""
        data = list(data)  # Make a copy
        if shuffle:
            np.random.shuffle(data)

        batch_size = self.training_config.batch_size

        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            if self.training_config.task_type == "generation":
                sources, targets = zip(*batch_data)
                src_ids, src_mask = self.tokenizer.encode_batch(
                    list(sources),
                    max_length=self.model_config.max_seq_length,
                    add_bos=True, add_eos=False,
                )
                tgt_ids, tgt_mask = self.tokenizer.encode_batch(
                    list(targets),
                    max_length=self.data_config.max_target_length,
                    add_bos=True, add_eos=True,
                )
                yield (src_ids, src_mask, tgt_ids, tgt_mask)
            else:
                texts, labels = zip(*batch_data)
                input_ids, attention_mask = self.tokenizer.encode_batch(
                    list(texts),
                    max_length=self.model_config.max_seq_length,
                    add_bos=True, add_eos=False,
                )
                yield (input_ids, attention_mask, jnp.array(labels))

    def train(self):
        """Main training loop."""
        print("\nInitializing model and optimizer...")

        # Initialize model
        key = jrandom.PRNGKey(self.training_config.shuffle_seed)
        self.model = self.task_handler.create_model(self.model_config, key)
        # Calculate total steps
        steps_per_epoch = len(self.train_data) // self.training_config.batch_size
        total_steps = steps_per_epoch * self.training_config.num_epochs

        # Initialize optimizer
        self.optimizer = self._create_optimizer(total_steps)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        # Create training step function
        self.train_step_fn = self._create_train_step()

        print(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        print(f"Starting training for {self.training_config.num_epochs} epochs...\n")

        # Training loop
        global_step = 0

        for epoch in range(self.training_config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")

            # Training phase
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                self.create_batches(self.train_data, shuffle=True),
                total=steps_per_epoch,
                desc="Training",
            )

            for batch in pbar:

                self.model, self.opt_state, loss, logits, masks = self.train_step_fn(
                    self.model, self.opt_state, batch
                )

                epoch_loss += float(loss)
                num_batches += 1
                global_step += 1

                # Update progress bar
                pbar.set_postfix({'loss': f'{float(loss):.4f}'})
                # Logging
                if global_step % self.training_config.log_every == 0:
                    print(f"  Step {global_step}: loss={float(loss):.4f}")

                # Generate a sample every 10 steps (generation tasks only)
                if (self.training_config.task_type == "generation"
                        and global_step % 10 == 0):
                    self._generate_sample(epoch, step=global_step)

            # Epoch metrics
            avg_train_loss = epoch_loss / num_batches
            print(f"  Average train loss: {avg_train_loss:.4f}")

            # Validation
            val_metrics = self.evaluate(self.val_data, "Validation")
            val_loss = val_metrics.get('loss', avg_train_loss)

            print(f"  Validation metrics: {val_metrics}")

            # Early stopping and checkpointing
            if val_loss < self.best_val_loss - self.training_config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best checkpoint
                metadata = {
                    'model_config': asdict(self.model_config),
                    'training_config': asdict(self.training_config),
                    'data_config': asdict(self.data_config),
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, epoch, val_loss, metadata, is_best=True
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.training_config.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            # Periodic checkpoint
            if (epoch + 1) % self.training_config.save_every == 0:
                metadata = {
                    'model_config': asdict(self.model_config),
                    'training_config': asdict(self.training_config),
                    'data_config': asdict(self.data_config),
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, epoch, val_loss, metadata
                )

        # Final test evaluation
        if self.test_data:
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate(self.test_data, "Test")
            print(f"Final test metrics: {test_metrics}")

    def _generate_sample(self, epoch: int, step: Optional[int] = None):
        """Generate a sample output from the first test example."""
        # Pick the first available test sample, fall back to validation
        sample_data = self.test_data if self.test_data else self.val_data
        if not sample_data:
            return

        source_text, target_text = sample_data[0]

        # Encode source using tokenizer
        input_ids, attention_mask = self.tokenizer.encode_batch(
            [source_text],
            max_length=self.model_config.max_seq_length,
            add_bos=True, add_eos=False,
        )

        # Generate with temperature sampling
        seed = step if step is not None else epoch
        key = jrandom.PRNGKey(seed)
        generated_ids = self.model.generate(
            input_ids,
            attention_mask,
            max_length=min(256, self.model_config.max_target_length),
            temperature=0.8,
            top_k=50,
            do_sample=True,
            key=key,
        )

        # Decode generated tokens using tokenizer
        generated_text = self.tokenizer.decode(generated_ids[0])
        tokens = [int(t) for t in generated_ids[0]]

        # Print sample
        label = f"step {step}" if step is not None else f"epoch {epoch + 1}"
        print(f"\n  --- Generation Sample ({label}) ---")
        print(f"  INPUT:     {source_text[:150]}{'...' if len(source_text) > 150 else ''}")
        print(f"  REFERENCE: {target_text[:150]}{'...' if len(target_text) > 150 else ''}")
        print(f"  GENERATED: {generated_text[:150]}{'...' if len(generated_text) > 150 else ''}")
        print(f"  ({len(tokens)} tokens generated)")
        print()

    def evaluate(self, data: List[Tuple], split_name: str = "Validation") -> Dict[str, float]:
        """Evaluate model on a dataset."""
        total_loss = 0.0
        total_ce = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in self.create_batches(data, shuffle=False):
            # Compute loss
            loss, (logits, masks, state) = self.task_handler.compute_loss(
                self.model,
                batch,
                self.training_config.sparsity_weight,
                self.training_config.label_smoothing,
                self.training_config.kl_weight,
            )

            total_loss += float(loss)
            num_batches += 1

            # Accumulate per-token cross-entropy for generation
            if self.training_config.task_type == "generation":
                shifted_logits = logits[:, :-1]
                shifted_targets = batch[2][:, 1:]
                shifted_mask = batch[3][:, 1:]

                log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
                target_log_probs = jnp.take_along_axis(
                    log_probs, shifted_targets[..., None], axis=-1
                ).squeeze(-1)
                batch_tokens = float(shifted_mask.sum())
                total_ce += float((-target_log_probs * shifted_mask).sum())
                total_tokens += batch_tokens

        avg_loss = total_loss / max(num_batches, 1)

        if self.training_config.task_type == "generation":
            avg_ce = total_ce / max(total_tokens, 1)
            perplexity = float(jnp.exp(min(avg_ce, 100)))
            metrics = {
                'loss': avg_loss,
                'cross_entropy': avg_ce,
                'perplexity': perplexity,
            }
        else:
            # For classification/regression, concatenate and compute metrics
            all_predictions = []
            all_labels = []
            for batch in self.create_batches(data, shuffle=False):
                _, (logits, _, _) = self.task_handler.compute_loss(
                    self.model, batch,
                    self.training_config.sparsity_weight,
                    self.training_config.label_smoothing,
                )
                all_predictions.append(logits)
                all_labels.append(batch[2])
            predictions = jnp.concatenate(all_predictions, axis=0)
            labels = jnp.concatenate(all_labels, axis=0)
            metrics = self.task_handler.compute_metrics(predictions, labels)
            metrics['loss'] = avg_loss

        return metrics


# ============================================================================
# Configuration File Loading
# ============================================================================

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("pyyaml is required for YAML config files. Install with: pip install pyyaml")
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def merge_configs(file_config: Dict, args: argparse.Namespace) -> Tuple[ModelConfig, TrainingConfig, DataConfig]:
    """Merge config file with command-line arguments."""
    # Start with file config
    model_dict = file_config.get('model', {})
    training_dict = file_config.get('training', {})
    data_dict = file_config.get('data', {})

    # Override with command-line args (if provided and not default)
    parser = create_argument_parser()
    defaults = vars(parser.parse_args([]))

    for key, value in vars(args).items():
        if value is not None and value != defaults.get(key):
            # Determine which config this belongs to
            if key in ModelConfig.__annotations__:
                model_dict[key] = value
            elif key in TrainingConfig.__annotations__:
                training_dict[key] = value
            elif key in DataConfig.__annotations__:
                data_dict[key] = value

    # Create config objects
    model_config = ModelConfig(**model_dict)
    training_config = TrainingConfig(**training_dict)
    data_config = DataConfig(**data_dict)

    return model_config, training_config, data_config


# ============================================================================
# Command-line Interface
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Train ByteTabNet models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON/YAML config file (other args override config values)",
    )

    # Task and data
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "generation"],
        help="Task type",
    )
    parser.add_argument("--data-path", type=str, help="Path to data file/directory")
    parser.add_argument(
        "--data-format",
        choices=["root", "parquet", "image", "text", "csv", "huggingface"],
        help="Data format",
    )
    parser.add_argument("--text-column", type=str, help="Column name for text input")
    parser.add_argument("--label-column", type=str, help="Column name for labels")
    parser.add_argument("--target-column", type=str, help="Column name for generation targets")
    parser.add_argument("--num-classes", type=int, help="Number of classes (classification only)")

    # Model architecture
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--embed-dim", type=int, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension")
    parser.add_argument("--n-steps", type=int, help="Number of TabNet decision steps")
    parser.add_argument("--n-d", type=int, help="Decision embedding dimension")
    parser.add_argument("--n-a", type=int, help="Attention embedding dimension")
    parser.add_argument("--pooling", choices=["mean", "max", "attention"], help="Pooling method")
    parser.add_argument("--n-decoder-layers", type=int, help="Number of decoder layers (generation)")

    # Tokenizer
    parser.add_argument(
        "--tokenizer-type", choices=["byte", "bpe"], default=None,
        help="Tokenizer type: 'byte' (default, vocab=320) or 'bpe' (BPE via tiktoken/HF tokenizers)",
    )
    parser.add_argument("--bpe-encoding", type=str, help="tiktoken encoding name for pre-trained BPE (e.g. cl100k_base, o200k_base, gpt2)")
    parser.add_argument("--bpe-tokenizer-path", type=str, help="Path to a custom-trained BPE tokenizer directory")
    parser.add_argument("--bpe-vocab-size", type=int, help="Vocabulary size for training a custom BPE tokenizer")
    parser.add_argument("--train-bpe", action="store_true", default=None, help="Train a new BPE tokenizer on the dataset before model training")

    # Training
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine", "exponential", "warmup_cosine"],
        help="Learning rate schedule",
    )
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--kl-weight", type=float, help="KL divergence loss weight for generation tasks (0 disables)")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints")

    # ROOT-specific
    parser.add_argument("--root-tree-name", type=str, help="ROOT tree name")
    parser.add_argument("--root-branches", type=str, help="Comma-separated ROOT branches")

    # Parquet-specific
    parser.add_argument("--parquet-columns", type=str, help="Comma-separated Parquet columns")

    # Image-specific
    parser.add_argument("--image-dir", type=str, help="Image directory")
    parser.add_argument("--image-label-file", type=str, help="Image label file")

    # HuggingFace-specific
    parser.add_argument("--hf-dataset-name", type=str, help="HuggingFace dataset name (e.g. 'khaimaitien/leetcode_problem_solution')")
    parser.add_argument("--hf-subset", type=str, help="HuggingFace dataset config/subset name")
    parser.add_argument("--hf-split-train", type=str, default="train", help="HuggingFace train split name")
    parser.add_argument("--hf-split-test", type=str, default="test", help="HuggingFace test split name")

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load and merge configs
    if args.config:
        file_config = load_config_from_file(args.config)
        model_config, training_config, data_config = merge_configs(file_config, args)
    else:
        # Build from command-line args only
        data_format = args.data_format or "text"
        if not args.task:
            parser.error("--task is required when not using --config")
        if not args.data_path and data_format != "huggingface":
            parser.error("--data-path is required when not using --config (except for huggingface format)")

        # Convert comma-separated strings to lists
        root_branches = args.root_branches.split(',') if args.root_branches else []
        parquet_columns = args.parquet_columns.split(',') if args.parquet_columns else []

        # Build configs from args
        model_config = ModelConfig(
            max_seq_length=args.max_seq_length or 2048,
            embed_dim=args.embed_dim or 64,
            hidden_dim=args.hidden_dim or 512,
            n_steps=args.n_steps or 4,
            n_d=args.n_d or 64,
            n_a=args.n_a or 64,
            pooling=args.pooling or "attention",
            n_decoder_layers=args.n_decoder_layers or 4,
            tokenizer_type=args.tokenizer_type or "byte",
            bpe_encoding=args.bpe_encoding or "cl100k_base",
            bpe_tokenizer_path=args.bpe_tokenizer_path,
            bpe_vocab_size=args.bpe_vocab_size or 8192,
            train_bpe=args.train_bpe or False,
        )

        training_config = TrainingConfig(
            task_type=args.task,
            batch_size=args.batch_size or 32,
            num_epochs=args.num_epochs or 100,
            learning_rate=args.learning_rate or 1e-3,
            optimizer=args.optimizer or "adamw",
            lr_schedule=args.lr_schedule or "warmup_cosine",
            patience=args.patience or 10,
            checkpoint_dir=args.checkpoint_dir or "./checkpoints",
        )

        data_config = DataConfig(
            data_path=args.data_path or "",
            data_format=data_format,
            text_column=args.text_column,
            label_column=args.label_column or "label",
            target_column=args.target_column,
            num_classes=args.num_classes,
            root_tree_name=args.root_tree_name,
            root_branches=root_branches,
            parquet_columns=parquet_columns,
            image_dir=args.image_dir,
            image_label_file=args.image_label_file,
            hf_dataset_name=args.hf_dataset_name,
            hf_subset=args.hf_subset,
            hf_split_train=args.hf_split_train,
            hf_split_test=args.hf_split_test,
        )

    # Print configuration
    print("=" * 70)
    print("ByteTabNet Training")
    print("=" * 70)
    print(f"\nTask: {training_config.task_type}")
    print(f"Data format: {data_config.data_format}")
    print(f"Data path: {data_config.data_path}")
    print(f"\nModel configuration:")
    for key, value in asdict(model_config).items():
        print(f"  {key}: {value}")
    print(f"\nTraining configuration:")
    for key, value in asdict(training_config).items():
        print(f"  {key}: {value}")
    print("=" * 70)
    print()

    # Create and run trainer
    trainer = Trainer(model_config, training_config, data_config)
    trainer.prepare_data()
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
