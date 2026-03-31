"""
ByteTabNet Masked Diffusion Model (MDM) Training Script (PyTorch)

Trains ByteTabNet using a masked diffusion objective instead of autoregressive
generation. Inspired by MDLM (Masked Diffusion Language Models) and MaskGIT.

Key differences from the autoregressive approach (trainer.py / ByteTabNetSeq2Seq):
  - Bidirectional: the model sees all unmasked tokens simultaneously, not left-to-right
  - Forward process: randomly mask tokens at a rate determined by a noise schedule
  - Training: predict original tokens at masked positions (cross-entropy)
  - Inference: start from all-masked, iteratively unmask most-confident tokens

Usage:
    python train_mdm.py --config configs/generation_example.json
    python train_mdm.py --task generation --data-format huggingface \
        --hf-dataset-name khaimaitien/leetcode_problem_solution \
        --text-column problem --target-column solution
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from byte_tabnet import (
    ByteEmbedding,
    ByteTabNet,
    TabNetEncoder,
    GhostBatchNorm,
    GLUBlock,
    sparsemax,
    tabnet_sparsity_loss,
    VOCAB_SIZE,
)

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
# Special token IDs (matches byte_tabnet.py vocabulary)
# ============================================================================

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MASK_TOKEN_ID = 3  # Reserved slot 3 used as the diffusion [MASK] token


# ============================================================================
# Noise Schedule
# ============================================================================

def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine schedule for the masking rate.

    At t=0 (clean data) the mask rate is 0.
    At t=1 (fully noised) the mask rate is 1.

    Uses: mask_rate = 1 - cos(pi/2 * t), which is monotonically
    increasing from 0 to 1 as t goes from 0 to 1.
    """
    return 1.0 - torch.cos(torch.tensor(math.pi / 2.0) * t)


def linear_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """Linear schedule: mask_rate = t."""
    return t


# ============================================================================
# Masking Utilities
# ============================================================================

def mask_tokens(
    input_ids: torch.Tensor,
    mask_rate: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly replace tokens with [MASK] at the given rate.

    Special tokens (PAD, BOS, EOS) are never masked.

    Args:
        input_ids: Token IDs (batch, seq_len).
        mask_rate: Scalar masking probability in [0, 1].

    Returns:
        (masked_ids, mask_positions)
        - masked_ids: input_ids with some tokens replaced by MASK_TOKEN_ID.
        - mask_positions: boolean tensor, True where tokens were masked.
    """
    rand = torch.rand_like(input_ids, dtype=torch.float32)
    should_mask = rand < mask_rate.item()

    # Never mask special tokens
    is_special = (
        (input_ids == PAD_TOKEN_ID) |
        (input_ids == BOS_TOKEN_ID) |
        (input_ids == EOS_TOKEN_ID)
    )
    should_mask = should_mask & ~is_special

    masked_ids = torch.where(should_mask, MASK_TOKEN_ID, input_ids)
    return masked_ids, should_mask


# ============================================================================
# Timestep Embedding
# ============================================================================

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal + learned timestep embedding.

    Maps a scalar timestep t in [0, 1] to a dense vector, using sinusoidal
    positional encoding followed by a two-layer MLP (like DDPM).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding of scalar t."""
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Scalar timestep in [0, 1].

        Returns:
            Embedding vector of shape (embed_dim,).
        """
        emb = self._sinusoidal(t)
        emb = F.silu(self.linear1(emb))
        emb = self.linear2(emb)
        return emb


# ============================================================================
# Bidirectional Self-Attention (no causal mask)
# ============================================================================

class BidirectionalSelfAttention(nn.Module):
    """
    Multi-head self-attention without causal masking.

    All positions can attend to all other positions, which is the key
    architectural difference from the autoregressive decoder.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional padding mask (batch, seq_len), 1 = keep, 0 = ignore.

        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention (no causal mask!)
        scale = math.sqrt(self.head_dim)
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / scale

        # Apply padding mask
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len) for broadcasting
            pad_mask = mask[:, None, None, :]
            scores = scores.masked_fill(pad_mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        attended = torch.einsum('bhqk,bhkd->bhqd', weights, v)

        # Reshape back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, hidden_dim)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_dim)

        return self.out_proj(attended)


# ============================================================================
# Bidirectional Transformer Block
# ============================================================================

class BidirectionalBlock(nn.Module):
    """
    A single bidirectional transformer block: self-attention + GLU feed-forward.

    Pre-norm style (LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        virtual_batch_size: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = BidirectionalSelfAttention(hidden_dim, n_heads)
        self.glu = GLUBlock(hidden_dim, hidden_dim, virtual_batch_size)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional padding mask (batch, seq_len).

        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape

        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        x = residual + self.attention(x_norm, mask)

        # GLU feed-forward with pre-norm
        residual = x
        x_norm = self.norm2(x)
        x_flat = x_norm.reshape(-1, self.hidden_dim)
        x_flat = self.glu(x_flat)
        x = residual + x_flat.reshape(batch, seq_len, self.hidden_dim)

        return x


# ============================================================================
# ByteTabNet Masked Diffusion Model
# ============================================================================

class ByteTabNetMDM(nn.Module):
    """
    ByteTabNet for Masked Diffusion Modeling.

    Architecture:
        1. ByteEmbedding (input: partially masked sequence)
        2. Positional embedding
        3. Timestep embedding (broadcast to all positions)
        4. Stack of bidirectional transformer blocks (no causal mask)
        5. TabNet encoder on globally-pooled features (for interpretability)
        6. Combine per-position hidden states with global TabNet context
        7. Output projection -> per-position vocab logits

    The model is trained to predict the original tokens at masked positions,
    conditioned on the unmasked context and the noise level (timestep).
    """

    def __init__(
        self,
        max_seq_length: int = 512,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_blocks: int = 6,
        n_heads: int = 4,
        n_steps: int = 5,
        n_d: int = 64,
        n_a: int = 64,
        gamma: float = 1.5,
        n_shared: int = 2,
        n_step: int = 2,
        virtual_batch_size: int = 128,
        use_positional: bool = True,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.n_d = n_d

        # Byte embedding
        self.byte_embedding = ByteEmbedding(vocab_size, embed_dim)

        # Positional embedding
        self.use_positional = use_positional
        if use_positional:
            self.position_embedding = nn.Parameter(
                torch.randn(max_seq_length, embed_dim) * 0.02
            )
        else:
            self.position_embedding = None

        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(embed_dim)

        # Project embed_dim -> hidden_dim
        self.input_projection = nn.Linear(embed_dim, hidden_dim)

        # Bidirectional transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                BidirectionalBlock(hidden_dim, n_heads, virtual_batch_size)
            )

        # Sparsemax attention for global pooling (same approach as ByteTabNet)
        self.sparsemax_attention = nn.Linear(hidden_dim, 1)

        # TabNet input projection
        self.tabnet_projection = nn.Linear(hidden_dim, hidden_dim)

        # TabNet encoder for global context / interpretability
        self.tabnet_encoder = TabNetEncoder(
            input_dim=hidden_dim,
            n_steps=n_steps,
            n_d=n_d,
            n_a=n_a,
            gamma=gamma,
            n_shared=n_shared,
            n_step=n_step,
            virtual_batch_size=virtual_batch_size,
        )

        # Combine per-position (hidden_dim) + global (n_d) -> hidden_dim
        self.combine_projection = nn.Linear(hidden_dim + n_d, hidden_dim)

        # Output layer norm
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Output projection -> vocab logits
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (Possibly masked) token IDs (batch, seq_len).
            timestep: Scalar timestep in [0, 1].
            attention_mask: Padding mask (batch, seq_len), 1 = real, 0 = pad.

        Returns:
            (logits, tabnet_masks)
            - logits: Per-position vocab logits (batch, seq_len, vocab_size).
            - tabnet_masks: TabNet attention masks for interpretability.
        """
        batch_size, seq_len = input_ids.shape

        # 1. Embed tokens
        embeddings = self.byte_embedding(input_ids)  # (batch, seq, embed_dim)

        # 2. Add positional embeddings
        if self.use_positional and self.position_embedding is not None:
            pos_emb = self.position_embedding[:seq_len]
            embeddings = embeddings + pos_emb

        # 3. Add timestep embedding (broadcast to all positions)
        t_emb = self.timestep_embedding(timestep)  # (embed_dim,)
        embeddings = embeddings + t_emb[None, None, :]  # broadcast

        # 4. Project to hidden dimension
        hidden = self.input_projection(embeddings)  # (batch, seq, hidden)

        # 5. Bidirectional transformer blocks
        for block in self.blocks:
            hidden = block(hidden, attention_mask)

        # 6. Global TabNet context via sparsemax pooling
        scores = self.sparsemax_attention(hidden)  # (batch, seq, 1)
        if attention_mask is not None:
            scores = torch.where(
                attention_mask.unsqueeze(-1) > 0, scores,
                torch.tensor(-1e9, device=scores.device),
            )
        weights = sparsemax(scores.squeeze(-1), dim=-1)  # (batch, seq)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)  # (batch, hidden)

        # Project and run TabNet
        tabnet_input = self.tabnet_projection(pooled)
        global_ctx, tabnet_masks = self.tabnet_encoder(tabnet_input)

        # 7. Broadcast global context and combine with per-position features
        global_broadcast = global_ctx[:, None, :].expand(
            batch_size, seq_len, self.n_d
        )
        combined = torch.cat([hidden, global_broadcast], dim=-1)
        combined = self.combine_projection(combined)

        # 8. Output layer norm + projection
        combined = self.output_norm(combined)
        logits = self.output_projection(combined)

        return logits, tabnet_masks

    @torch.no_grad()
    def get_feature_importance(
        self,
        input_ids: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return aggregated TabNet feature importance masks."""
        self.eval()
        _, masks = self(input_ids, timestep, attention_mask)
        return torch.mean(masks, dim=1)


# ============================================================================
# Loss Functions
# ============================================================================

def mdm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask_positions: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy loss computed only at masked positions.

    Args:
        logits: Model output (batch, seq_len, vocab_size).
        targets: Ground-truth token IDs (batch, seq_len).
        mask_positions: Boolean mask, True at positions that were masked.
        attention_mask: Padding mask (batch, seq_len), 1 = real token.
        label_smoothing: Label smoothing factor.

    Returns:
        Scalar loss.
    """
    vocab_size = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)

    # One-hot targets
    one_hot = F.one_hot(targets.long(), num_classes=vocab_size).float()
    if label_smoothing > 0:
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab_size

    # Per-position cross-entropy
    ce = -torch.sum(one_hot * log_probs, dim=-1)  # (batch, seq_len)

    # Only count masked positions (and non-padding)
    valid = mask_positions.float()
    if attention_mask is not None:
        valid = valid * attention_mask

    return torch.sum(ce * valid) / torch.clamp(torch.sum(valid), min=1.0)


def mdm_loss_with_sparsity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask_positions: torch.Tensor,
    tabnet_masks: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sparsity_weight: float = 1e-3,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """MDM loss + TabNet sparsity regularization."""
    ce = mdm_loss(logits, targets, mask_positions, attention_mask, label_smoothing)
    sparsity = tabnet_sparsity_loss(tabnet_masks)
    return ce + sparsity_weight * sparsity


# ============================================================================
# Iterative Demasking Inference
# ============================================================================

@torch.no_grad()
def iterative_demask(
    model: ByteTabNetMDM,
    seq_length: int,
    num_steps: int = 25,
    temperature: float = 1.0,
    noise_schedule_fn=cosine_noise_schedule,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a sequence by iterative demasking.

    Starts with all [MASK] tokens (plus BOS) and gradually reveals tokens
    from highest confidence to lowest over ``num_steps`` demasking rounds.

    Args:
        model: Trained ByteTabNetMDM.
        seq_length: Desired output length (including BOS).
        num_steps: Number of demasking steps.
        temperature: Sampling temperature.
        noise_schedule_fn: Schedule function mapping t -> mask_rate.
        device: Torch device.

    Returns:
        Generated token IDs (1, seq_length).
    """
    model.eval()

    # Start fully masked (except BOS)
    ids = torch.full((1, seq_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    ids[0, 0] = BOS_TOKEN_ID
    attention_mask = torch.ones((1, seq_length), dtype=torch.float32, device=device)

    for step in range(num_steps):
        # Current and next timestep
        t_current = 1.0 - step / num_steps
        t_next = max(1.0 - (step + 1) / num_steps, 0.0)

        t_arr = torch.tensor(t_current, device=device)

        # Get predictions
        logits, _ = model(ids, t_arr, attention_mask)
        logits = logits / temperature  # (1, seq_len, vocab_size)

        # Sample from the predicted distribution
        probs = F.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (seq_len,)

        # Confidence = max probability at each position
        confidence = probs.max(dim=-1).values  # (seq_len,)

        # Only consider currently masked positions
        is_masked = (ids[0] == MASK_TOKEN_ID)
        confidence = torch.where(is_masked, confidence, torch.tensor(-1.0, device=device))

        # How many tokens to unmask this step
        current_mask_rate = noise_schedule_fn(torch.tensor(t_current))
        next_mask_rate = noise_schedule_fn(torch.tensor(t_next))
        n_currently_masked = is_masked.sum()
        n_to_unmask = max(
            int(round((current_mask_rate - next_mask_rate).item() * seq_length)),
            1,
        )
        n_to_unmask = min(n_to_unmask, n_currently_masked.item())

        # Select top-confidence masked positions to unmask
        sorted_indices = torch.argsort(confidence, descending=True)

        # Create unmask indicator
        ranks = torch.zeros(seq_length, dtype=torch.long, device=device)
        ranks[sorted_indices] = torch.arange(seq_length, device=device)
        should_unmask = (ranks < n_to_unmask) & is_masked

        # Update IDs: unmask selected positions with sampled tokens
        new_ids = torch.where(should_unmask, sampled, ids[0])
        ids = new_ids.unsqueeze(0)

    return ids


@torch.no_grad()
def iterative_demask_seq2seq(
    model: ByteTabNetMDM,
    source_ids: torch.Tensor,
    source_mask: torch.Tensor,
    target_length: int,
    num_steps: int = 25,
    temperature: float = 1.0,
    noise_schedule_fn=cosine_noise_schedule,
) -> torch.Tensor:
    """
    Generate a target sequence conditioned on a source using iterative demasking.

    The source is prepended to the target: [BOS, source_tokens..., MASK...MASK].
    After generation the target portion is extracted.

    Args:
        model: Trained ByteTabNetMDM.
        source_ids: Encoded source (1, src_len) including BOS.
        source_mask: Source attention mask (1, src_len).
        target_length: Length of target to generate.
        num_steps: Number of demasking iterations.
        temperature: Sampling temperature.
        noise_schedule_fn: Masking schedule function.

    Returns:
        Generated target token IDs (1, target_length).
    """
    model.eval()
    device = source_ids.device
    src_len = source_ids.shape[1]

    # Build initial sequence: [source_tokens, MASK * target_length]
    target_init = torch.full(
        (1, target_length), MASK_TOKEN_ID, dtype=torch.long, device=device
    )
    full_ids = torch.cat([source_ids, target_init], dim=1)

    target_mask_init = torch.ones(
        (1, target_length), dtype=torch.float32, device=device
    )
    full_mask = torch.cat([source_mask, target_mask_init], dim=1)

    total_len = full_ids.shape[1]

    # Mark which positions are in the target region
    is_target_region = torch.zeros(total_len, dtype=torch.bool, device=device)
    is_target_region[src_len:] = True

    for step in range(num_steps):
        t_current = 1.0 - step / num_steps
        t_next = max(1.0 - (step + 1) / num_steps, 0.0)
        t_arr = torch.tensor(t_current, device=device)

        logits, _ = model(full_ids, t_arr, full_mask)
        logits = logits / temperature

        probs = F.softmax(logits[0], dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        confidence = probs.max(dim=-1).values
        is_masked = (full_ids[0] == MASK_TOKEN_ID) & is_target_region
        confidence = torch.where(is_masked, confidence, torch.tensor(-1.0, device=device))

        current_mask_rate = noise_schedule_fn(torch.tensor(t_current))
        next_mask_rate = noise_schedule_fn(torch.tensor(t_next))
        n_to_unmask = max(
            int(round((current_mask_rate - next_mask_rate).item() * target_length)),
            1,
        )
        n_currently_masked = is_masked.sum().item()
        n_to_unmask = min(n_to_unmask, n_currently_masked)

        sorted_indices = torch.argsort(confidence, descending=True)
        ranks = torch.zeros(total_len, dtype=torch.long, device=device)
        ranks[sorted_indices] = torch.arange(total_len, device=device)
        should_unmask = (ranks < n_to_unmask) & is_masked

        new_ids = torch.where(should_unmask, sampled, full_ids[0])
        full_ids = new_ids.unsqueeze(0)

    # Extract target portion
    return full_ids[:, src_len:]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MDMModelConfig:
    """MDM model architecture configuration."""
    max_seq_length: int = 512
    embed_dim: int = 64
    hidden_dim: int = 128
    n_blocks: int = 6
    n_heads: int = 4
    n_steps: int = 5  # TabNet decision steps
    n_d: int = 64
    n_a: int = 64
    gamma: float = 1.5
    n_shared: int = 2
    n_step: int = 2
    virtual_batch_size: int = 128
    use_positional: bool = True
    vocab_size: int = 320


@dataclass
class MDMTrainingConfig:
    """MDM training hyperparameters."""
    task_type: str = "generation"
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "adamw"

    # MDM-specific
    noise_schedule: str = "cosine"  # "cosine" or "linear"
    min_mask_rate: float = 0.0
    max_mask_rate: float = 1.0

    # Regularization
    sparsity_weight: float = 1e-3
    label_smoothing: float = 0.0
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
    checkpoint_dir: str = "./checkpoints_mdm"
    keep_best_k: int = 3

    # Logging
    log_every: int = 10

    # Inference
    num_denoise_steps: int = 25
    sample_temperature: float = 0.8

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle_seed: int = 42


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

        metadata_full = {'epoch': epoch, 'val_loss': float(val_loss), **metadata}
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata_full, f, indent=2)

        if not is_best:
            self.checkpoints.append((epoch, val_loss, path))
            self._cleanup_old()

        print(f"  Checkpoint saved: {path}")

    def _cleanup_old(self):
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
# MDM Trainer
# ============================================================================

class MDMTrainer:
    """
    Trainer for the ByteTabNet Masked Diffusion Model.

    Supports two modes:
      - Unconditional generation: train on target sequences only.
      - Seq2Seq generation: source is kept unmasked, target is masked.
    """

    def __init__(
        self,
        model_config: MDMModelConfig,
        training_config: MDMTrainingConfig,
        data_config: DataConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.device = get_device()

        self.data_loader = self._create_data_loader()
        self.checkpoint_manager = CheckpointManager(
            training_config.checkpoint_dir,
            keep_best_k=training_config.keep_best_k,
        )

        self.noise_fn = (
            cosine_noise_schedule
            if training_config.noise_schedule == "cosine"
            else linear_noise_schedule
        )

        # Will be set during train()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.train_data = None
        self.val_data = None
        self.test_data = None

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
                return max(
                    cfg.min_lr / cfg.learning_rate,
                    0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=warmup_cosine_fn,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Load and split data."""
        print("Loading data...")
        data = self.data_loader.load_data()

        if self.training_config.task_type == "generation":
            source_texts, target_texts = data
            data_pairs = list(zip(source_texts, target_texts))
        else:
            texts, labels = data
            # For non-generation tasks, target = input (self-supervised denoising)
            data_pairs = list(zip(texts, texts))

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

    def _encode_for_mdm(
        self,
        source_texts: List[str],
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode source + target into a single concatenated sequence for MDM.

        Returns:
            (full_ids, attention_mask, target_ids, target_region_mask)
        """
        max_src = self.model_config.max_seq_length // 2
        max_tgt = self.model_config.max_seq_length - max_src
        total_max = self.model_config.max_seq_length

        all_ids = []
        all_masks = []
        all_target_ids = []
        all_target_region = []

        for src, tgt in zip(source_texts, target_texts):
            # Encode source: [BOS] + bytes
            src_bytes = [b + 64 for b in src.encode("utf-8")]
            src_tokens = [BOS_TOKEN_ID] + src_bytes
            if len(src_tokens) > max_src:
                src_tokens = src_tokens[:max_src]

            # Encode target: bytes + [EOS]
            tgt_bytes = [b + 64 for b in tgt.encode("utf-8")]
            tgt_tokens = tgt_bytes + [EOS_TOKEN_ID]
            if len(tgt_tokens) > max_tgt:
                tgt_tokens = tgt_tokens[:max_tgt - 1] + [EOS_TOKEN_ID]

            # Concatenate
            full = src_tokens + tgt_tokens
            src_len = len(src_tokens)
            tgt_len = len(tgt_tokens)
            real_len = len(full)

            # Pad to total_max
            pad_len = total_max - real_len
            if pad_len < 0:
                full = full[:total_max]
                real_len = total_max
                pad_len = 0
                tgt_len = real_len - src_len

            ids = full + [PAD_TOKEN_ID] * pad_len
            mask = [1.0] * real_len + [0.0] * pad_len

            # target_ids: the ground truth for ALL positions
            target = ids[:]

            # target_region: 1 for target tokens, 0 elsewhere
            region = [0.0] * src_len + [1.0] * tgt_len + [0.0] * pad_len

            all_ids.append(ids)
            all_masks.append(mask)
            all_target_ids.append(target)
            all_target_region.append(region)

        return (
            torch.tensor(all_ids, dtype=torch.long, device=self.device),
            torch.tensor(all_masks, dtype=torch.float32, device=self.device),
            torch.tensor(all_target_ids, dtype=torch.long, device=self.device),
            torch.tensor(all_target_region, dtype=torch.float32, device=self.device),
        )

    def create_batches(self, data: List[Tuple], shuffle: bool = True):
        """Yield batches of (full_ids, attention_mask, target_ids, target_region_mask)."""
        data = list(data)
        if shuffle:
            np.random.shuffle(data)

        bs = self.training_config.batch_size

        for i in range(0, len(data), bs):
            batch_data = data[i:i + bs]
            sources, targets = zip(*batch_data)
            yield self._encode_for_mdm(list(sources), list(targets))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        """Main training loop."""
        print("\nInitializing MDM model and optimizer...")

        torch.manual_seed(self.training_config.shuffle_seed)

        # Create model
        mc = self.model_config
        self.model = ByteTabNetMDM(
            max_seq_length=mc.max_seq_length,
            embed_dim=mc.embed_dim,
            hidden_dim=mc.hidden_dim,
            n_blocks=mc.n_blocks,
            n_heads=mc.n_heads,
            n_steps=mc.n_steps,
            n_d=mc.n_d,
            n_a=mc.n_a,
            gamma=mc.gamma,
            n_shared=mc.n_shared,
            n_step=mc.n_step,
            virtual_batch_size=mc.virtual_batch_size,
            use_positional=mc.use_positional,
            vocab_size=mc.vocab_size,
        )
        self.model.to(self.device)

        # Optimizer
        steps_per_epoch = max(len(self.train_data) // self.training_config.batch_size, 1)
        total_steps = steps_per_epoch * self.training_config.num_epochs

        self._create_optimizer_and_scheduler(total_steps)

        print(f"Device: {self.device}")
        print(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        print(f"Noise schedule: {self.training_config.noise_schedule}")
        print(f"Starting MDM training for {self.training_config.num_epochs} epochs...\n")

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
                full_ids, attention_mask, target_ids, target_region_mask = batch

                # Sample random timestep
                t = torch.empty((), device=self.device).uniform_(0.01, 1.0)

                # Apply masking to target region only
                mask_rate = self.noise_fn(t)
                rand = torch.rand_like(full_ids, dtype=torch.float32)
                should_mask = (rand < mask_rate.item()) & (target_region_mask > 0)
                is_special = (
                    (full_ids == PAD_TOKEN_ID) |
                    (full_ids == BOS_TOKEN_ID) |
                    (full_ids == EOS_TOKEN_ID)
                )
                should_mask = should_mask & ~is_special
                masked_ids = torch.where(
                    should_mask,
                    torch.tensor(MASK_TOKEN_ID, device=self.device),
                    full_ids,
                )

                # Forward
                self.optimizer.zero_grad()
                logits, tabnet_masks = self.model(masked_ids, t, attention_mask)

                # Loss at masked positions
                loss = mdm_loss_with_sparsity(
                    logits, target_ids, should_mask, tabnet_masks,
                    attention_mask,
                    self.training_config.sparsity_weight,
                    self.training_config.label_smoothing,
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

                # Generate a sample every 10 steps
                if global_step % 10 == 0:
                    self._generate_sample(epoch, step=global_step)

            avg_train_loss = epoch_loss / max(num_batches, 1)
            print(f"  Average train loss: {avg_train_loss:.4f}")

            # Validation
            val_loss = self._evaluate(self.val_data)
            print(f"  Validation loss: {val_loss:.4f}")

            # Early stopping / checkpointing
            if val_loss < self.best_val_loss - self.training_config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                metadata = {
                    'model_config': asdict(self.model_config),
                    'training_config': asdict(self.training_config),
                    'data_config': asdict(self.data_config),
                    'model_type': 'mdm',
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
                    'model_type': 'mdm',
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, epoch, val_loss, metadata,
                )

        # Final test
        if self.test_data:
            print("\nEvaluating on test set...")
            test_loss = self._evaluate(self.test_data)
            print(f"Final test loss: {test_loss:.4f}")

    @torch.no_grad()
    def _evaluate(self, data: List[Tuple]) -> float:
        """Compute average loss on a dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.create_batches(data, shuffle=False):
            full_ids, attention_mask, target_ids, target_region_mask = batch

            # Evaluate at multiple timesteps and average for a more stable estimate
            losses = []
            for t_val in [0.25, 0.5, 0.75]:
                t = torch.tensor(t_val, device=self.device)
                mask_rate = self.noise_fn(t)
                rand = torch.rand_like(full_ids, dtype=torch.float32)
                should_mask = (rand < mask_rate.item()) & (target_region_mask > 0)
                is_special = (
                    (full_ids == PAD_TOKEN_ID) |
                    (full_ids == BOS_TOKEN_ID) |
                    (full_ids == EOS_TOKEN_ID)
                )
                should_mask = should_mask & ~is_special
                masked_ids = torch.where(
                    should_mask,
                    torch.tensor(MASK_TOKEN_ID, device=self.device),
                    full_ids,
                )

                logits, tabnet_masks = self.model(masked_ids, t, attention_mask)
                loss = mdm_loss_with_sparsity(
                    logits, target_ids, should_mask, tabnet_masks,
                    attention_mask, self.training_config.sparsity_weight,
                )
                losses.append(loss.item())

            total_loss += np.mean(losses)
            num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _generate_sample(self, epoch: int, step: Optional[int] = None):
        """Generate and print a sample from the first test example."""
        sample_data = self.test_data if self.test_data else self.val_data
        if not sample_data:
            return

        self.model.eval()

        source_text, target_text = sample_data[0]

        # Encode source
        src_bytes = [b + 64 for b in source_text.encode("utf-8")]
        src_tokens = [BOS_TOKEN_ID] + src_bytes
        max_src = self.model_config.max_seq_length // 2
        if len(src_tokens) > max_src:
            src_tokens = src_tokens[:max_src]

        source_ids = torch.tensor(
            [src_tokens], dtype=torch.long, device=self.device
        )
        source_mask = torch.ones_like(source_ids, dtype=torch.float32)

        target_length = min(128, self.model_config.max_seq_length - len(src_tokens))

        generated = iterative_demask_seq2seq(
            self.model,
            source_ids,
            source_mask,
            target_length=target_length,
            num_steps=self.training_config.num_denoise_steps,
            temperature=self.training_config.sample_temperature,
            noise_schedule_fn=self.noise_fn,
        )

        # Decode
        tokens = generated[0].cpu().tolist()
        # Stop at EOS
        try:
            eos_idx = tokens.index(EOS_TOKEN_ID)
            tokens = tokens[:eos_idx]
        except ValueError:
            pass
        # Remove any remaining MASK tokens
        tokens = [t for t in tokens if t != MASK_TOKEN_ID and t != PAD_TOKEN_ID]
        raw_bytes = bytes(t - 64 for t in tokens if 64 <= t < 320)
        generated_text = raw_bytes.decode("utf-8", errors="replace")

        label = f"step {step}" if step is not None else f"epoch {epoch + 1}"
        print(f"\n  --- MDM Generation Sample ({label}) ---")
        print(f"  INPUT:     {source_text[:150]}{'...' if len(source_text) > 150 else ''}")
        print(f"  REFERENCE: {target_text[:150]}{'...' if len(target_text) > 150 else ''}")
        print(f"  GENERATED: {generated_text[:150]}{'...' if len(generated_text) > 150 else ''}")
        print(f"  ({len(tokens)} tokens, {self.training_config.num_denoise_steps} demasking steps)")
        print()

        self.model.train()


# ============================================================================
# CLI
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ByteTabNet Masked Diffusion Model (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, help="Path to JSON/YAML config file")

    # Task and data
    parser.add_argument("--task", choices=["generation"], default="generation", help="Task type")
    parser.add_argument("--data-path", type=str, help="Path to data file/directory")
    parser.add_argument(
        "--data-format",
        choices=["root", "parquet", "image", "text", "csv", "huggingface"],
        help="Data format",
    )
    parser.add_argument("--text-column", type=str, help="Column name for text/source input")
    parser.add_argument("--label-column", type=str, help="Column name for labels")
    parser.add_argument("--target-column", type=str, help="Column name for generation targets")
    parser.add_argument("--num-classes", type=int, help="Number of classes")

    # Model
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--embed-dim", type=int, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension")
    parser.add_argument("--n-blocks", type=int, help="Number of bidirectional transformer blocks")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads")
    parser.add_argument("--n-steps", type=int, help="Number of TabNet decision steps")
    parser.add_argument("--n-d", type=int, help="TabNet decision embedding dim")
    parser.add_argument("--n-a", type=int, help="TabNet attention embedding dim")

    # MDM training
    parser.add_argument("--noise-schedule", choices=["cosine", "linear"], help="Noise schedule")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine", "exponential", "warmup_cosine"],
        help="LR schedule",
    )
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--num-denoise-steps", type=int, help="Demasking steps for inference")
    parser.add_argument("--sample-temperature", type=float, help="Sampling temperature")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")

    # HuggingFace
    parser.add_argument("--hf-dataset-name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--hf-subset", type=str, help="HuggingFace subset")
    parser.add_argument("--hf-split-train", type=str, default="train")
    parser.add_argument("--hf-split-test", type=str, default="test")

    # ROOT-specific
    parser.add_argument("--root-tree-name", type=str, help="ROOT tree name")
    parser.add_argument("--root-branches", type=str, help="Comma-separated ROOT branches")

    return parser


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.config:
        file_config = load_config_from_file(args.config)
        model_dict = file_config.get('model', {})
        training_dict = file_config.get('training', {})
        data_dict = file_config.get('data', {})
    else:
        model_dict = {}
        training_dict = {}
        data_dict = {}

    # Override from CLI args
    cli = vars(args)
    for cli_key, cli_val in cli.items():
        if cli_val is None or cli_key == 'config':
            continue
        field_name = cli_key.replace('-', '_')
        if field_name == 'task':
            field_name = 'task_type'
        if field_name in MDMModelConfig.__annotations__:
            model_dict[field_name] = cli_val
        elif field_name in MDMTrainingConfig.__annotations__:
            training_dict[field_name] = cli_val
        elif field_name in DataConfig.__annotations__:
            data_dict[field_name] = cli_val

    # Handle comma-separated lists
    if 'root_branches' in data_dict and isinstance(data_dict['root_branches'], str):
        data_dict['root_branches'] = data_dict['root_branches'].split(',')

    model_config = MDMModelConfig(**{
        k: v for k, v in model_dict.items() if k in MDMModelConfig.__annotations__
    })
    training_config = MDMTrainingConfig(**{
        k: v for k, v in training_dict.items() if k in MDMTrainingConfig.__annotations__
    })
    data_config = DataConfig(**{
        k: v for k, v in data_dict.items() if k in DataConfig.__annotations__
    })

    # Print config
    print("=" * 70)
    print("ByteTabNet Masked Diffusion Model Training (PyTorch)")
    print("=" * 70)
    print(f"\nTask: {training_config.task_type}")
    print(f"Data format: {data_config.data_format}")
    print(f"Noise schedule: {training_config.noise_schedule}")
    print(f"\nModel configuration:")
    for key, value in asdict(model_config).items():
        print(f"  {key}: {value}")
    print(f"\nTraining configuration:")
    for key, value in asdict(training_config).items():
        print(f"  {key}: {value}")
    print("=" * 70)
    print()

    trainer = MDMTrainer(model_config, training_config, data_config)
    trainer.prepare_data()
    trainer.train()

    print("\nMDM Training complete!")


if __name__ == "__main__":
    main()
