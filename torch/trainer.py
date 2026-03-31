"""
ByteTabNet Training Script (PyTorch)

Comprehensive training script supporting:
- Task types: Classification, Regression, Generation (seq2seq)
- Data formats: ROOT files, Parquet files, Images, Text, HuggingFace
- Features: Checkpointing, metrics tracking, early stopping, LR scheduling
"""

import argparse
import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    seq2seq_reverse_kl_div_loss,
    prepare_seq2seq_batch,
)

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
    pooling: str = "attention"
    use_positional: bool = True
    vocab_size: int = 320
    n_decoder_layers: int = 4
    max_target_length: int = 1024


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    task_type: str = "generation"
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "adamw"

    # Regularization
    sparsity_weight: float = 1e-3
    label_smoothing: float = 0.0
    kl_weight: float = 0.0
    reverse_kl_weight: float = 0.0
    gradient_clip: Optional[float] = 1.0

    # Learning rate scheduling
    lr_schedule: str = "warmup_cosine"
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


@dataclass
class QuantizationConfig:
    """Quantization-Aware Training configuration."""
    enabled: bool = False
    pretrained_checkpoint: Optional[str] = None
    qat_epochs: int = 5
    qat_learning_rate: float = 1e-5
    backend: str = "fbgemm"
    convert_to_quantized: bool = True
    quantized_model_path: Optional[str] = None


# ============================================================================
# Device Helpers
# ============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints using torch.save."""

    def __init__(self, output_dir: str, keep_best_k: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_k = keep_best_k
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        val_loss: float,
        metadata: Dict[str, Any],
        is_best: bool = False,
    ):
        if is_best:
            path = self.output_dir / "best_model.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(model.state_dict(), path)

        metadata_full = {
            'epoch': epoch,
            'val_loss': float(val_loss),
            **metadata,
        }
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata_full, f, indent=2)

        if not is_best:
            self.checkpoints.append((epoch, val_loss, path))
            self._cleanup_old_checkpoints()

        print(f"  Checkpoint saved: {path}")

    def _cleanup_old_checkpoints(self):
        if len(self.checkpoints) > self.keep_best_k:
            self.checkpoints.sort(key=lambda x: x[1])
            for _, _, path in self.checkpoints[self.keep_best_k:]:
                if path.exists() and "best" not in path.name:
                    path.unlink()
                    json_path = path.with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()
            self.checkpoints = self.checkpoints[:self.keep_best_k]


# ============================================================================
# Abstract Base Classes
# ============================================================================

class TaskHandler(ABC):
    @abstractmethod
    def create_model(self, config: ModelConfig) -> nn.Module:
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: Tuple,
        sparsity_weight: float,
        label_smoothing: float = 0.0,
        kl_weight: float = 0.0,
        reverse_kl_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Tuple]:
        pass

    @abstractmethod
    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        pass


# ============================================================================
# Task Handlers
# ============================================================================

class ClassificationTask(TaskHandler):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def create_model(self, config: ModelConfig) -> ByteTabNet:
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
        )

    def compute_loss(self, model, batch, sparsity_weight, label_smoothing=0.0, kl_weight=0.0, reverse_kl_weight=0.0):
        input_ids, attention_mask, labels = batch
        logits, masks = model(input_ids, attention_mask)
        loss = tabnet_loss(logits, labels, masks, sparsity_weight=sparsity_weight)
        return loss, (logits, masks)

    def compute_metrics(self, predictions, labels):
        pred_classes = torch.argmax(predictions, dim=-1)
        accuracy = (pred_classes == labels).float().mean().item()
        return {'accuracy': accuracy}


class RegressionTask(TaskHandler):
    def create_model(self, config: ModelConfig) -> ByteTabNet:
        return ByteTabNet(
            output_dim=1,
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
        )

    def compute_loss(self, model, batch, sparsity_weight, label_smoothing=0.0, kl_weight=0.0, reverse_kl_weight=0.0):
        input_ids, attention_mask, labels = batch
        logits, masks = model(input_ids, attention_mask)
        predictions = logits.squeeze(-1)
        mse = F.mse_loss(predictions, labels)
        sparse_loss = tabnet_sparsity_loss(masks)
        total_loss = mse + sparsity_weight * sparse_loss
        return total_loss, (logits, masks)

    def compute_metrics(self, predictions, labels):
        if predictions.ndim > 1:
            predictions = predictions.squeeze(-1)
        mse = F.mse_loss(predictions, labels).item()
        mae = (predictions - labels).abs().mean().item()
        ss_res = ((labels - predictions) ** 2).sum()
        ss_tot = ((labels - labels.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        return {'mse': mse, 'rmse': math.sqrt(mse), 'mae': mae, 'r2': r2}


class GenerationTask(TaskHandler):
    def create_model(self, config: ModelConfig) -> ByteTabNetSeq2Seq:
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
        )

    def compute_loss(self, model, batch, sparsity_weight, label_smoothing=0.0, kl_weight=0.0, reverse_kl_weight=0.0):
        input_ids, input_mask, target_ids, target_mask = batch
        logits, masks = model(input_ids, target_ids, input_mask, target_mask)

        loss = seq2seq_loss_with_tabnet_sparsity(
            logits[:, :-1],
            target_ids[:, 1:],
            masks,
            target_mask[:, 1:],
            sparsity_weight=sparsity_weight,
            label_smoothing=label_smoothing,
            kl_weight=kl_weight,
            reverse_kl_weight=reverse_kl_weight,
        )
        return loss, (logits, masks)

    def compute_metrics(self, predictions, labels):
        log_probs = F.log_softmax(predictions, dim=-1)
        labels_expanded = labels.unsqueeze(-1)
        selected = torch.gather(log_probs, -1, labels_expanded).squeeze(-1)
        ce = -selected.mean().item()
        perplexity = math.exp(min(ce, 100))
        return {'perplexity': perplexity, 'cross_entropy': ce}


# ============================================================================
# Main Trainer Class
# ============================================================================

class Trainer:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.quantization_config = quantization_config or QuantizationConfig()
        self.device = get_device()

        self.task_handler = self._create_task_handler()
        self.data_loader = self._create_data_loader()
        self.checkpoint_manager = CheckpointManager(
            training_config.checkpoint_dir,
            keep_best_k=training_config.keep_best_k,
        )

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _create_task_handler(self) -> TaskHandler:
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

    def _create_optimizer_and_scheduler(self, total_steps: int):
        cfg = self.training_config
        if cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.learning_rate,
            )
        elif cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        if cfg.lr_schedule == "constant":
            self.scheduler = None
        elif cfg.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=cfg.min_lr,
            )
        elif cfg.lr_schedule == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.9,
            )
        elif cfg.lr_schedule == "warmup_cosine":
            def warmup_cosine_fn(step):
                if step < cfg.warmup_steps:
                    return step / max(cfg.warmup_steps, 1)
                progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
                return max(cfg.min_lr / cfg.learning_rate,
                           0.5 * (1.0 + math.cos(math.pi * progress)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=warmup_cosine_fn,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")

    def prepare_data(self):
        print("Loading data...")
        data = self.data_loader.load_data()

        if self.training_config.task_type == "generation":
            source_texts, target_texts = data
            data_pairs = list(zip(source_texts, target_texts))
        else:
            texts, labels = data
            data_pairs = list(zip(texts, labels))

        rng = np.random.RandomState(self.training_config.shuffle_seed)
        rng.shuffle(data_pairs)

        n = len(data_pairs)
        n_train = int(n * self.training_config.train_split)
        n_val = int(n * self.training_config.val_split)

        self.train_data = data_pairs[:n_train]
        self.val_data = data_pairs[n_train:n_train + n_val]
        self.test_data = data_pairs[n_train + n_val:]

        print(f"Data splits: train={len(self.train_data)}, "
              f"val={len(self.val_data)}, test={len(self.test_data)}")

    def create_batches(self, data: List[Tuple], shuffle: bool = True):
        data = list(data)
        if shuffle:
            np.random.shuffle(data)

        batch_size = self.training_config.batch_size

        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            if self.training_config.task_type == "generation":
                sources, targets = zip(*batch_data)
                src_ids, tgt_ids, src_mask, tgt_mask = prepare_seq2seq_batch(
                    list(sources),
                    list(targets),
                    self.model_config.max_seq_length,
                    self.data_config.max_target_length,
                )
                yield (
                    src_ids.to(self.device),
                    src_mask.to(self.device),
                    tgt_ids.to(self.device),
                    tgt_mask.to(self.device),
                )
            else:
                texts, labels = zip(*batch_data)
                input_ids, attention_mask = ByteEmbedding.encode_batch(
                    list(texts),
                    max_length=self.model_config.max_seq_length,
                )
                yield (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    torch.tensor(labels, dtype=torch.float32).to(self.device),
                )

    def train(self):
        print("\nInitializing model and optimizer...")

        torch.manual_seed(self.training_config.shuffle_seed)
        self.model = self.task_handler.create_model(self.model_config)
        self.model.to(self.device)

        steps_per_epoch = max(len(self.train_data) // self.training_config.batch_size, 1)
        total_steps = steps_per_epoch * self.training_config.num_epochs

        self._create_optimizer_and_scheduler(total_steps)

        print(f"Device: {self.device}")
        print(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        print(f"Starting training for {self.training_config.num_epochs} epochs...\n")

        global_step = 0

        for epoch in range(self.training_config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")

            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                self.create_batches(self.train_data, shuffle=True),
                total=steps_per_epoch,
                desc="Training",
            )

            for batch in pbar:
                self.optimizer.zero_grad()

                loss, (logits, masks) = self.task_handler.compute_loss(
                    self.model,
                    batch,
                    self.training_config.sparsity_weight,
                    self.training_config.label_smoothing,
                    self.training_config.kl_weight,
                    self.training_config.reverse_kl_weight,
                )

                loss.backward()

                if self.training_config.gradient_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip,
                    )

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                if global_step % self.training_config.log_every == 0:
                    print(f"  Step {global_step}: loss={loss.item():.4f}")

                # Generate a sample every 10 steps (generation tasks only)
                if (self.training_config.task_type == "generation"
                        and global_step % 10 == 0):
                    self._generate_sample(epoch, step=global_step)

            avg_train_loss = epoch_loss / max(num_batches, 1)
            print(f"  Average train loss: {avg_train_loss:.4f}")

            # Validation
            val_metrics = self.evaluate(self.val_data, "Validation")
            val_loss = val_metrics.get('loss', avg_train_loss)
            print(f"  Validation metrics: {val_metrics}")

            # Early stopping and checkpointing
            if val_loss < self.best_val_loss - self.training_config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                metadata = {
                    'model_config': asdict(self.model_config),
                    'training_config': asdict(self.training_config),
                    'data_config': asdict(self.data_config),
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, epoch, val_loss, metadata, is_best=True,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.training_config.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            if (epoch + 1) % self.training_config.save_every == 0:
                metadata = {
                    'model_config': asdict(self.model_config),
                    'training_config': asdict(self.training_config),
                    'data_config': asdict(self.data_config),
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, epoch, val_loss, metadata,
                )

        if self.test_data:
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate(self.test_data, "Test")
            print(f"Final test metrics: {test_metrics}")

        # QAT phase (if enabled)
        if (self.quantization_config.enabled
                and self.training_config.task_type == "generation"):
            self._train_qat()

    def _train_qat(self):
        """Quantization-Aware Training fine-tuning phase."""
        from qat_utils import (
            prepare_model_for_qat,
            convert_qat_to_quantized,
            get_model_size_mb,
        )

        print("\n" + "=" * 70)
        print("Quantization-Aware Training (QAT)")
        print("=" * 70)

        # Load pre-trained checkpoint if specified
        if self.quantization_config.pretrained_checkpoint:
            ckpt_path = self.quantization_config.pretrained_checkpoint
            print(f"Loading pre-trained model from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)

        fp32_size = get_model_size_mb(self.model)
        print(f"FP32 model size: {fp32_size:.1f} MB")

        # Prepare for QAT
        print(f"Preparing model for QAT (backend={self.quantization_config.backend})...")
        self.model = prepare_model_for_qat(
            self.model, backend=self.quantization_config.backend,
        )
        self.model.to(self.device)

        # Create optimizer with lower LR for fine-tuning
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.quantization_config.qat_learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        qat_epochs = self.quantization_config.qat_epochs
        steps_per_epoch = max(
            len(self.train_data) // self.training_config.batch_size, 1
        )

        print(f"QAT fine-tuning for {qat_epochs} epochs "
              f"(lr={self.quantization_config.qat_learning_rate})...\n")

        for epoch in range(qat_epochs):
            print(f"QAT Epoch {epoch + 1}/{qat_epochs}")
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                self.create_batches(self.train_data, shuffle=True),
                total=steps_per_epoch,
                desc="QAT Training",
            )

            for batch in pbar:
                optimizer.zero_grad()

                loss, _ = self.task_handler.compute_loss(
                    self.model,
                    batch,
                    self.training_config.sparsity_weight,
                    self.training_config.label_smoothing,
                    self.training_config.kl_weight,
                    self.training_config.reverse_kl_weight,
                )

                loss.backward()

                if self.training_config.gradient_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip,
                    )

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"  QAT avg train loss: {avg_loss:.4f}")

            val_metrics = self.evaluate(self.val_data, "QAT Validation")
            print(f"  QAT validation: {val_metrics}")

        # Save QAT checkpoint
        qat_path = Path(self.training_config.checkpoint_dir) / "qat_model.pt"
        qat_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), qat_path)
        print(f"\nQAT model saved to {qat_path}")

        # Convert to int8
        if self.quantization_config.convert_to_quantized:
            print("Converting to quantized int8 model...")
            quantized = convert_qat_to_quantized(self.model)

            q_path = self.quantization_config.quantized_model_path
            if not q_path:
                q_path = str(
                    Path(self.training_config.checkpoint_dir) / "quantized_model.pt"
                )
            Path(q_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(quantized.state_dict(), q_path)

            q_size = get_model_size_mb(quantized)
            print(f"Quantized model size: {q_size:.1f} MB "
                  f"({fp32_size / max(q_size, 0.01):.1f}x reduction)")
            print(f"Quantized model saved to {q_path}")

    @torch.no_grad()
    def _generate_sample(self, epoch: int, step: Optional[int] = None):
        sample_data = self.test_data if self.test_data else self.val_data
        if not sample_data:
            return

        self.model.eval()
        source_text, target_text = sample_data[0]

        input_ids = torch.tensor(
            [[1] + [b + 64 for b in source_text.encode("utf-8")]],
            dtype=torch.long, device=self.device,
        )
        if input_ids.shape[1] > self.model_config.max_seq_length:
            input_ids = input_ids[:, :self.model_config.max_seq_length]
        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)

        generated_ids = self.model.generate(
            input_ids,
            attention_mask,
            max_length=min(256, self.model_config.max_target_length),
            temperature=0.1,
            top_k=50,
            do_sample=True,
        )

        tokens = generated_ids[0].cpu().tolist()
        if tokens and tokens[0] == 1:
            tokens = tokens[1:]
        try:
            eos_idx = tokens.index(2)
            tokens = tokens[:eos_idx]
        except ValueError:
            pass
        raw_bytes = bytes(t - 64 for t in tokens if 64 <= t < 320)
        generated_text = raw_bytes.decode("utf-8", errors="replace")

        label = f"step {step}" if step is not None else f"epoch {epoch + 1}"
        print(f"\n  --- Generation Sample ({label}) ---")
        print(f"  INPUT:     {source_text[:150]}{'...' if len(source_text) > 150 else ''}")
        print(f"  REFERENCE: {target_text[:150]}{'...' if len(target_text) > 150 else ''}")
        print(f"  GENERATED: {generated_text[:150]}{'...' if len(generated_text) > 150 else ''}")
        print(f"  ({len(tokens)} tokens generated)")
        print()
        self.model.train()

    @torch.no_grad()
    def evaluate(self, data: List[Tuple], split_name: str = "Validation") -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in self.create_batches(data, shuffle=False):
            loss, (logits, masks) = self.task_handler.compute_loss(
                self.model, batch,
                self.training_config.sparsity_weight,
                self.training_config.label_smoothing,
                self.training_config.kl_weight,
            )

            total_loss += loss.item()
            num_batches += 1

            if self.training_config.task_type == "generation":
                shifted_logits = logits[:, :-1]
                shifted_targets = batch[2][:, 1:]
                shifted_mask = batch[3][:, 1:]

                log_probs = F.log_softmax(shifted_logits, dim=-1)
                target_log_probs = torch.gather(
                    log_probs, -1, shifted_targets.unsqueeze(-1),
                ).squeeze(-1)
                batch_tokens = shifted_mask.sum().item()
                total_ce += (-target_log_probs * shifted_mask).sum().item()
                total_tokens += batch_tokens

        avg_loss = total_loss / max(num_batches, 1)

        if self.training_config.task_type == "generation":
            avg_ce = total_ce / max(total_tokens, 1)
            perplexity = math.exp(min(avg_ce, 100))
            metrics = {
                'loss': avg_loss,
                'cross_entropy': avg_ce,
                'perplexity': perplexity,
            }
        else:
            all_preds, all_labels = [], []
            for batch in self.create_batches(data, shuffle=False):
                _, (logits, _) = self.task_handler.compute_loss(
                    self.model, batch,
                    self.training_config.sparsity_weight,
                    self.training_config.label_smoothing,
                )
                all_preds.append(logits.cpu())
                all_labels.append(batch[2].cpu())
            preds = torch.cat(all_preds, dim=0)
            labels = torch.cat(all_labels, dim=0)
            metrics = self.task_handler.compute_metrics(preds, labels)
            metrics['loss'] = avg_loss

        self.model.train()
        return metrics


# ============================================================================
# Configuration File Loading
# ============================================================================

def load_config_from_file(config_path: str) -> Dict[str, Any]:
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
            raise ImportError("pyyaml is required for YAML configs.")
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def merge_configs(file_config: Dict, args: argparse.Namespace) -> Tuple[ModelConfig, TrainingConfig, DataConfig, QuantizationConfig]:
    model_dict = file_config.get('model', {})
    training_dict = file_config.get('training', {})
    data_dict = file_config.get('data', {})
    quant_dict = file_config.get('quantization', {})

    parser = create_argument_parser()
    defaults = vars(parser.parse_args([]))

    for key, value in vars(args).items():
        if value is not None and value != defaults.get(key):
            if key in ModelConfig.__annotations__:
                model_dict[key] = value
            elif key in TrainingConfig.__annotations__:
                training_dict[key] = value
            elif key in DataConfig.__annotations__:
                data_dict[key] = value
            elif key in QuantizationConfig.__annotations__:
                quant_dict[key] = value

    return (
        ModelConfig(**model_dict),
        TrainingConfig(**training_dict),
        DataConfig(**data_dict),
        QuantizationConfig(**quant_dict),
    )


# ============================================================================
# Command-line Interface
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ByteTabNet models (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to JSON/YAML config file")
    parser.add_argument("--task", choices=["classification", "regression", "generation"], help="Task type")
    parser.add_argument("--data-path", type=str, help="Path to data file/directory")
    parser.add_argument("--data-format", choices=["root", "parquet", "image", "text", "csv", "huggingface"], help="Data format")
    parser.add_argument("--text-column", type=str, help="Column name for text input")
    parser.add_argument("--label-column", type=str, help="Column name for labels")
    parser.add_argument("--target-column", type=str, help="Column name for generation targets")
    parser.add_argument("--num-classes", type=int, help="Number of classes")
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--embed-dim", type=int, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension")
    parser.add_argument("--n-steps", type=int, help="Number of TabNet decision steps")
    parser.add_argument("--n-d", type=int, help="Decision embedding dimension")
    parser.add_argument("--n-a", type=int, help="Attention embedding dimension")
    parser.add_argument("--pooling", choices=["mean", "max", "attention"], help="Pooling method")
    parser.add_argument("--n-decoder-layers", type=int, help="Number of decoder layers")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine", "exponential", "warmup_cosine"], help="LR schedule")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--kl-weight", type=float, help="Forward KL divergence loss weight")
    parser.add_argument("--reverse-kl-weight", type=float, help="Reverse KL divergence loss weight")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--root-tree-name", type=str, help="ROOT tree name")
    parser.add_argument("--root-branches", type=str, help="Comma-separated ROOT branches")
    parser.add_argument("--parquet-columns", type=str, help="Comma-separated Parquet columns")
    parser.add_argument("--image-dir", type=str, help="Image directory")
    parser.add_argument("--image-label-file", type=str, help="Image label file")
    parser.add_argument("--hf-dataset-name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--hf-subset", type=str, help="HuggingFace subset")
    parser.add_argument("--hf-split-train", type=str, default="train")
    parser.add_argument("--hf-split-test", type=str, default="test")
    # Quantization-Aware Training
    parser.add_argument("--qat-enabled", dest="enabled", action="store_true", default=None, help="Enable QAT")
    parser.add_argument("--qat-pretrained-checkpoint", dest="pretrained_checkpoint", type=str, help="Pre-trained checkpoint for QAT")
    parser.add_argument("--qat-epochs", dest="qat_epochs", type=int, help="Number of QAT fine-tuning epochs")
    parser.add_argument("--qat-learning-rate", dest="qat_learning_rate", type=float, help="QAT learning rate")
    parser.add_argument("--qat-backend", dest="backend", choices=["fbgemm", "qnnpack"], help="Quantization backend")
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.config:
        file_config = load_config_from_file(args.config)
        model_config, training_config, data_config, quantization_config = merge_configs(file_config, args)
    else:
        data_format = args.data_format or "text"
        if not args.task:
            parser.error("--task is required when not using --config")
        if not args.data_path and data_format != "huggingface":
            parser.error("--data-path is required when not using --config (except for huggingface)")

        root_branches = args.root_branches.split(',') if args.root_branches else []
        parquet_columns = args.parquet_columns.split(',') if args.parquet_columns else []

        model_config = ModelConfig(
            max_seq_length=args.max_seq_length or 2048,
            embed_dim=args.embed_dim or 64,
            hidden_dim=args.hidden_dim or 512,
            n_steps=args.n_steps or 4,
            n_d=args.n_d or 64,
            n_a=args.n_a or 64,
            pooling=args.pooling or "attention",
            n_decoder_layers=args.n_decoder_layers or 4,
        )
        training_config = TrainingConfig(
            task_type=args.task,
            batch_size=args.batch_size or 512,
            num_epochs=args.num_epochs or 10,
            learning_rate=args.learning_rate or 1e-4,
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
        quantization_config = QuantizationConfig()

    print("=" * 70)
    print("ByteTabNet Training (PyTorch)")
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

    if quantization_config.enabled:
        print(f"\nQuantization-Aware Training:")
        for key, value in asdict(quantization_config).items():
            print(f"  {key}: {value}")

    trainer = Trainer(model_config, training_config, data_config, quantization_config)
    trainer.prepare_data()
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
