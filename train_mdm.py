"""
ByteTabNet Masked Diffusion Model (MDM) Training Script

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
import warnings

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import numpy as np
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from byte_tabnet import (
    ByteEmbedding,
    ByteTabNet,
    TabNetEncoder,
    GhostBatchNorm,
    GLUBlock,
    FeatureTransformer,
    sparsemax,
    tabnet_sparsity_loss,
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
from tokenizer import Tokenizer, ByteTokenizer, create_tokenizer


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

def cosine_noise_schedule(t: Array) -> Array:
    """
    Cosine schedule for the masking rate.

    At t=0 (clean data) the mask rate is 0.
    At t=1 (fully noised) the mask rate is 1.

    Uses: mask_rate = 1 - cos(pi/2 * t), which is monotonically
    increasing from 0 to 1 as t goes from 0 to 1.

    Args:
        t: Timestep(s) in [0, 1].

    Returns:
        Masking probability for each timestep.
    """
    return 1.0 - jnp.cos(jnp.pi / 2.0 * t)


def linear_noise_schedule(t: Array) -> Array:
    """Linear schedule: mask_rate = t."""
    return t


# ============================================================================
# Masking Utilities
# ============================================================================

def mask_tokens(
    input_ids: Array,
    mask_rate: Array,
    key: PRNGKeyArray,
) -> Tuple[Array, Array]:
    """
    Randomly replace tokens with [MASK] at the given rate.

    Special tokens (PAD, BOS, EOS) are never masked.

    Args:
        input_ids: Token IDs (batch, seq_len).
        mask_rate: Scalar or per-batch masking probability in [0, 1].
        key: PRNG key.

    Returns:
        (masked_ids, mask_positions)
        - masked_ids: input_ids with some tokens replaced by MASK_TOKEN_ID.
        - mask_positions: boolean array, True where tokens were masked.
    """
    # Draw uniform random values
    rand = jrandom.uniform(key, input_ids.shape)

    # Determine which positions to mask
    should_mask = rand < mask_rate

    # Never mask special tokens
    is_special = (input_ids == PAD_TOKEN_ID) | (input_ids == BOS_TOKEN_ID) | (input_ids == EOS_TOKEN_ID)
    should_mask = should_mask & ~is_special

    masked_ids = jnp.where(should_mask, MASK_TOKEN_ID, input_ids)
    return masked_ids, should_mask


# ============================================================================
# Timestep Embedding
# ============================================================================

class TimestepEmbedding(eqx.Module):
    """
    Sinusoidal + learned timestep embedding.

    Maps a scalar timestep t in [0, 1] to a dense vector, using sinusoidal
    positional encoding followed by a two-layer MLP (like DDPM).
    """
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, *, key: PRNGKeyArray):
        self.embed_dim = embed_dim
        k1, k2 = jrandom.split(key)
        self.linear1 = eqx.nn.Linear(embed_dim, embed_dim * 4, key=k1)
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=k2)

    def _sinusoidal(self, t: Array) -> Array:
        """Compute sinusoidal embedding of scalar t."""
        half = self.embed_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
        args = t * freqs
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)])

    def __call__(self, t: Array) -> Array:
        """
        Args:
            t: Scalar timestep in [0, 1].

        Returns:
            Embedding vector of shape (embed_dim,).
        """
        emb = self._sinusoidal(t)
        emb = jax.nn.silu(self.linear1(emb))
        emb = self.linear2(emb)
        return emb


# ============================================================================
# Bidirectional Self-Attention (no causal mask)
# ============================================================================

class BidirectionalSelfAttention(eqx.Module):
    """
    Multi-head self-attention without causal masking.

    All positions can attend to all other positions, which is the key
    architectural difference from the autoregressive decoder.
    """
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, hidden_dim: int, n_heads: int = 4, *, key: PRNGKeyArray):
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        k1, k2, k3, k4 = jrandom.split(key, 4)
        self.q_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k1)
        self.k_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.v_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k3)
        self.out_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k4)

    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
    ) -> Array:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional padding mask (batch, seq_len), 1 = keep, 0 = ignore.

        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = jax.vmap(jax.vmap(self.q_proj))(x)
        k = jax.vmap(jax.vmap(self.k_proj))(x)
        v = jax.vmap(jax.vmap(self.v_proj))(x)

        # Reshape to multi-head: (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention (no causal mask!)
        scale = jnp.sqrt(self.head_dim).astype(q.dtype)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale

        # Apply padding mask
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len) for broadcasting
            pad_mask = mask[:, None, None, :]
            scores = jnp.where(pad_mask == 0, -1e9, scores)

        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum('bhqk,bhkd->bhqd', weights, v)

        # Reshape back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, hidden_dim)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_dim)

        return jax.vmap(jax.vmap(self.out_proj))(attended)


# ============================================================================
# Bidirectional Transformer Block
# ============================================================================

class BidirectionalBlock(eqx.Module):
    """
    A single bidirectional transformer block: self-attention + GLU feed-forward.

    Pre-norm style (LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual).
    """
    attention: BidirectionalSelfAttention
    glu: GLUBlock
    norm1_weight: Array
    norm1_bias: Array
    norm2_weight: Array
    norm2_bias: Array
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        virtual_batch_size: int = 128,
        *,
        key: PRNGKeyArray,
    ):
        self.hidden_dim = hidden_dim
        k1, k2 = jrandom.split(key)
        self.attention = BidirectionalSelfAttention(hidden_dim, n_heads, key=k1)
        self.glu = GLUBlock(hidden_dim, hidden_dim, virtual_batch_size, key=k2)
        self.norm1_weight = jnp.ones(hidden_dim)
        self.norm1_bias = jnp.zeros(hidden_dim)
        self.norm2_weight = jnp.ones(hidden_dim)
        self.norm2_bias = jnp.zeros(hidden_dim)

    def _layer_norm(self, x: Array, weight: Array, bias: Array) -> Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + 1e-5) * weight + bias

    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional padding mask (batch, seq_len).
            inference: Whether in inference mode.
            state: Batch norm state for GLU.

        Returns:
            (output, new_state)
        """
        batch, seq_len, _ = x.shape

        # Self-attention with pre-norm
        residual = x
        x_norm = self._layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = residual + self.attention(x_norm, mask)

        # GLU feed-forward with pre-norm
        residual = x
        x_norm = self._layer_norm(x, self.norm2_weight, self.norm2_bias)
        x_flat = x_norm.reshape(-1, self.hidden_dim)
        x_flat, new_state = self.glu(x_flat, inference, state)
        x = residual + x_flat.reshape(batch, seq_len, self.hidden_dim)

        return x, new_state


# ============================================================================
# ByteTabNet Masked Diffusion Model
# ============================================================================

class ByteTabNetMDM(eqx.Module):
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
    byte_embedding: ByteEmbedding
    position_embedding: Optional[Array]
    timestep_embedding: TimestepEmbedding
    input_projection: eqx.nn.Linear
    blocks: List[BidirectionalBlock]
    sparsemax_attention: eqx.nn.Linear
    tabnet_projection: eqx.nn.Linear
    tabnet_encoder: TabNetEncoder
    combine_projection: eqx.nn.Linear
    output_norm_weight: Array
    output_norm_bias: Array
    output_projection: eqx.nn.Linear

    max_seq_length: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)
    n_d: int = eqx.field(static=True)

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
        vocab_size: int = 320,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            max_seq_length: Maximum sequence length.
            embed_dim: Byte embedding dimension.
            hidden_dim: Hidden dimension for transformer blocks.
            n_blocks: Number of bidirectional transformer blocks.
            n_heads: Number of attention heads per block.
            n_steps: Number of TabNet decision steps.
            n_d: TabNet decision embedding dimension.
            n_a: TabNet attention embedding dimension.
            gamma: TabNet relaxation factor.
            n_shared: Shared layers in TabNet feature transformer.
            n_step: Step-specific layers in TabNet feature transformer.
            virtual_batch_size: Ghost batch norm virtual batch size.
            use_positional: Whether to use positional embeddings.
            vocab_size: Vocabulary size (320 = 256 bytes + 64 special).
            key: PRNG key.
        """
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.n_d = n_d

        keys = jrandom.split(key, 10 + n_blocks)

        # Byte embedding
        self.byte_embedding = ByteEmbedding(vocab_size, embed_dim, key=keys[0])

        # Positional embedding
        if use_positional:
            self.position_embedding = jrandom.normal(keys[1], (max_seq_length, embed_dim)) * 0.02
        else:
            self.position_embedding = None

        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(embed_dim, key=keys[2])

        # Project embed_dim -> hidden_dim
        self.input_projection = eqx.nn.Linear(embed_dim, hidden_dim, key=keys[3])

        # Bidirectional transformer blocks
        self.blocks = []
        for i in range(n_blocks):
            block = BidirectionalBlock(
                hidden_dim, n_heads, virtual_batch_size, key=keys[10 + i]
            )
            self.blocks.append(block)

        # Sparsemax attention for global pooling (same approach as ByteTabNet)
        self.sparsemax_attention = eqx.nn.Linear(hidden_dim, 1, key=keys[4])

        # TabNet input projection
        self.tabnet_projection = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[5])

        # TabNet encoder for global context / interpretability
        self.tabnet_encoder = TabNetEncoder(
            input_dim=hidden_dim,
            output_dim=n_d,
            n_steps=n_steps,
            n_d=n_d,
            n_a=n_a,
            gamma=gamma,
            n_shared=n_shared,
            n_step=n_step,
            virtual_batch_size=virtual_batch_size,
            key=keys[6],
        )

        # Combine per-position (hidden_dim) + global (n_d) -> hidden_dim
        self.combine_projection = eqx.nn.Linear(hidden_dim + n_d, hidden_dim, key=keys[7])

        # Output layer norm
        self.output_norm_weight = jnp.ones(hidden_dim)
        self.output_norm_bias = jnp.zeros(hidden_dim)

        # Output projection -> vocab logits
        self.output_projection = eqx.nn.Linear(hidden_dim, vocab_size, key=keys[8])

    def __call__(
        self,
        input_ids: Array,
        timestep: Array,
        attention_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Array, Optional[dict]]:
        """
        Forward pass.

        Args:
            input_ids: (Possibly masked) token IDs (batch, seq_len).
            timestep: Scalar timestep in [0, 1].
            attention_mask: Padding mask (batch, seq_len), 1 = real, 0 = pad.
            inference: Whether in inference mode.
            state: Optional batch norm state dict.

        Returns:
            (logits, tabnet_masks, new_state)
            - logits: Per-position vocab logits (batch, seq_len, vocab_size).
            - tabnet_masks: TabNet attention masks for interpretability.
            - new_state: Updated batch norm state.
        """
        batch_size, seq_len = input_ids.shape
        all_states = {}

        # 1. Embed tokens
        embeddings = self.byte_embedding(input_ids)  # (batch, seq, embed_dim)

        # 2. Add positional embeddings
        if self.position_embedding is not None:
            pos_emb = self.position_embedding[:seq_len]
            embeddings = embeddings + pos_emb

        # 3. Add timestep embedding (broadcast to all positions)
        t_emb = self.timestep_embedding(timestep)  # (embed_dim,)
        embeddings = embeddings + t_emb[None, None, :]  # broadcast

        # 4. Project to hidden dimension
        hidden = jax.vmap(jax.vmap(self.input_projection))(embeddings)  # (batch, seq, hidden)

        # 5. Bidirectional transformer blocks
        for i, block in enumerate(self.blocks):
            block_state = state.get(f"block_{i}") if state else None
            hidden, new_block_state = block(hidden, attention_mask, inference, block_state)
            if new_block_state:
                all_states[f"block_{i}"] = new_block_state

        # 6. Global TabNet context via sparsemax pooling
        scores = jax.vmap(jax.vmap(self.sparsemax_attention))(hidden)  # (batch, seq, 1)
        if attention_mask is not None:
            scores = jnp.where(attention_mask[..., None] > 0, scores, -1e9)
        weights = sparsemax(scores.squeeze(-1), axis=-1)  # (batch, seq)
        pooled = jnp.sum(hidden * weights[..., None], axis=1)  # (batch, hidden)

        # Project and run TabNet
        tabnet_input = jax.vmap(self.tabnet_projection)(pooled)
        tabnet_state = state.get("tabnet") if state else None
        global_ctx, tabnet_masks, tabnet_new_state = self.tabnet_encoder(
            tabnet_input, inference, tabnet_state
        )
        if tabnet_new_state:
            all_states["tabnet"] = tabnet_new_state

        # 7. Broadcast global context and combine with per-position features
        global_broadcast = jnp.broadcast_to(
            global_ctx[:, None, :], (batch_size, seq_len, self.n_d)
        )
        combined = jnp.concatenate([hidden, global_broadcast], axis=-1)
        combined = jax.vmap(jax.vmap(self.combine_projection))(combined)

        # 8. Output layer norm + projection
        mean = jnp.mean(combined, axis=-1, keepdims=True)
        var = jnp.var(combined, axis=-1, keepdims=True)
        combined = (combined - mean) / jnp.sqrt(var + 1e-5)
        combined = combined * self.output_norm_weight + self.output_norm_bias

        logits = jax.vmap(jax.vmap(self.output_projection))(combined)

        return logits, tabnet_masks, all_states if all_states else None

    def get_feature_importance(
        self,
        input_ids: Array,
        timestep: Array,
        attention_mask: Optional[Array] = None,
    ) -> Array:
        """Return aggregated TabNet feature importance masks."""
        _, masks, _ = self(input_ids, timestep, attention_mask, inference=True)
        return jnp.mean(masks, axis=1)


# ============================================================================
# Loss Functions
# ============================================================================

def mdm_loss(
    logits: Array,
    targets: Array,
    mask_positions: Array,
    attention_mask: Optional[Array] = None,
    label_smoothing: float = 0.0,
) -> Array:
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
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # One-hot targets
    one_hot = jax.nn.one_hot(targets, vocab_size)
    if label_smoothing > 0:
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab_size

    # Per-position cross-entropy
    ce = -jnp.sum(one_hot * log_probs, axis=-1)  # (batch, seq_len)

    # Only count masked positions (and non-padding)
    valid = mask_positions.astype(jnp.float32)
    if attention_mask is not None:
        valid = valid * attention_mask

    return jnp.sum(ce * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def mdm_loss_with_sparsity(
    logits: Array,
    targets: Array,
    mask_positions: Array,
    tabnet_masks: Array,
    attention_mask: Optional[Array] = None,
    sparsity_weight: float = 1e-3,
    label_smoothing: float = 0.0,
) -> Array:
    """MDM loss + TabNet sparsity regularization."""
    ce = mdm_loss(logits, targets, mask_positions, attention_mask, label_smoothing)
    sparsity = tabnet_sparsity_loss(tabnet_masks)
    return ce + sparsity_weight * sparsity


# ============================================================================
# Iterative Demasking Inference
# ============================================================================

def iterative_demask(
    model: ByteTabNetMDM,
    seq_length: int,
    num_steps: int = 25,
    temperature: float = 1.0,
    noise_schedule_fn=cosine_noise_schedule,
    *,
    key: PRNGKeyArray,
) -> Array:
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
        key: PRNG key.

    Returns:
        Generated token IDs (1, seq_length).
    """
    # Start fully masked (except BOS)
    ids = jnp.full((1, seq_length), MASK_TOKEN_ID, dtype=jnp.int32)
    ids = ids.at[0, 0].set(BOS_TOKEN_ID)
    attention_mask = jnp.ones((1, seq_length), dtype=jnp.float32)

    for step in range(num_steps):
        key, subkey_pred, subkey_sample = jrandom.split(key, 3)

        # Current and next timestep
        t_current = 1.0 - step / num_steps
        t_next = max(1.0 - (step + 1) / num_steps, 0.0)

        t_arr = jnp.array(t_current)

        # Get predictions
        logits, _, _ = model(ids, t_arr, attention_mask, inference=True)
        logits = logits / temperature  # (1, seq_len, vocab_size)

        # Sample from the predicted distribution
        probs = jax.nn.softmax(logits[0], axis=-1)  # (seq_len, vocab_size)
        sampled = jrandom.categorical(subkey_sample, logits[0], axis=-1)  # (seq_len,)

        # Confidence = max probability at each position
        confidence = jnp.max(probs, axis=-1)  # (seq_len,)

        # Only consider currently masked positions
        is_masked = (ids[0] == MASK_TOKEN_ID)
        confidence = jnp.where(is_masked, confidence, -1.0)

        # How many tokens to unmask this step
        current_mask_rate = noise_schedule_fn(jnp.array(t_current))
        next_mask_rate = noise_schedule_fn(jnp.array(t_next))
        n_currently_masked = jnp.sum(is_masked)
        n_to_unmask = jnp.maximum(
            jnp.round((current_mask_rate - next_mask_rate) * seq_length).astype(jnp.int32),
            1,
        )
        # Don't unmask more than what's masked
        n_to_unmask = jnp.minimum(n_to_unmask, n_currently_masked)

        # Select top-confidence masked positions to unmask
        # Argsort descending by confidence
        sorted_indices = jnp.argsort(-confidence)

        # Create unmask indicator
        ranks = jnp.zeros(seq_length, dtype=jnp.int32)
        ranks = ranks.at[sorted_indices].set(jnp.arange(seq_length))
        should_unmask = (ranks < n_to_unmask) & is_masked

        # Update IDs: unmask selected positions with sampled tokens
        new_ids = jnp.where(should_unmask, sampled, ids[0])
        ids = new_ids[None, :]

    return ids


def iterative_demask_seq2seq(
    model: ByteTabNetMDM,
    source_ids: Array,
    source_mask: Array,
    target_length: int,
    num_steps: int = 25,
    temperature: float = 1.0,
    noise_schedule_fn=cosine_noise_schedule,
    *,
    key: PRNGKeyArray,
) -> Array:
    """
    Generate a target sequence conditioned on a source using iterative demasking.

    The source is prepended to the target: [BOS, source_tokens..., SEP, MASK...MASK].
    After generation the target portion is extracted.

    For simplicity this implementation concatenates source + masked target into a
    single sequence and lets the bidirectional model attend to both.

    Args:
        model: Trained ByteTabNetMDM.
        source_ids: Encoded source (1, src_len) including BOS.
        source_mask: Source attention mask (1, src_len).
        target_length: Length of target to generate.
        num_steps: Number of demasking iterations.
        temperature: Sampling temperature.
        noise_schedule_fn: Masking schedule function.
        key: PRNG key.

    Returns:
        Generated target token IDs (1, target_length).
    """
    src_len = source_ids.shape[1]

    # Build initial sequence: [source_tokens, MASK * target_length]
    target_init = jnp.full((1, target_length), MASK_TOKEN_ID, dtype=jnp.int32)
    full_ids = jnp.concatenate([source_ids, target_init], axis=1)

    target_mask_init = jnp.ones((1, target_length), dtype=jnp.float32)
    full_mask = jnp.concatenate([source_mask, target_mask_init], axis=1)

    total_len = full_ids.shape[1]

    # Mark which positions are in the target region
    is_target_region = jnp.zeros(total_len, dtype=jnp.bool_)
    is_target_region = is_target_region.at[src_len:].set(True)

    for step in range(num_steps):
        key, subkey = jrandom.split(key)

        t_current = 1.0 - step / num_steps
        t_next = max(1.0 - (step + 1) / num_steps, 0.0)
        t_arr = jnp.array(t_current)

        logits, _, _ = model(full_ids, t_arr, full_mask, inference=True)
        logits = logits / temperature

        probs = jax.nn.softmax(logits[0], axis=-1)
        sampled = jrandom.categorical(subkey, logits[0], axis=-1)

        confidence = jnp.max(probs, axis=-1)
        is_masked = (full_ids[0] == MASK_TOKEN_ID) & is_target_region
        confidence = jnp.where(is_masked, confidence, -1.0)

        current_mask_rate = noise_schedule_fn(jnp.array(t_current))
        next_mask_rate = noise_schedule_fn(jnp.array(t_next))
        n_to_unmask = jnp.maximum(
            jnp.round((current_mask_rate - next_mask_rate) * target_length).astype(jnp.int32),
            1,
        )
        n_currently_masked = jnp.sum(is_masked)
        n_to_unmask = jnp.minimum(n_to_unmask, n_currently_masked)

        sorted_indices = jnp.argsort(-confidence)
        ranks = jnp.zeros(total_len, dtype=jnp.int32)
        ranks = ranks.at[sorted_indices].set(jnp.arange(total_len))
        should_unmask = (ranks < n_to_unmask) & is_masked

        new_ids = jnp.where(should_unmask, sampled, full_ids[0])
        full_ids = new_ids[None, :]

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
    # Tokenizer settings
    tokenizer_type: str = "byte"  # "byte" or "bpe"
    bpe_encoding: str = "cl100k_base"  # tiktoken encoding name (pre-trained BPE)
    bpe_tokenizer_path: Optional[str] = None  # path to custom-trained BPE model
    bpe_vocab_size: int = 8192  # vocab size for training custom BPE
    train_bpe: bool = False  # train a new BPE tokenizer on the dataset


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
# Checkpoint Manager (reused pattern from trainer.py)
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, output_dir: str, keep_best_k: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_k = keep_best_k
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: eqx.Module,
        epoch: int,
        val_loss: float,
        metadata: Dict[str, Any],
        is_best: bool = False,
    ):
        if is_best:
            path = self.output_dir / "best_model.eqx"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.eqx"

        eqx.tree_serialise_leaves(path, model)

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

    def load_checkpoint(self, model_template: eqx.Module, checkpoint_path: str):
        path = Path(checkpoint_path)
        model = eqx.tree_deserialise_leaves(path, model_template)
        metadata = {}
        json_path = path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        return model, metadata


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

        # Tokenizer (may be overridden after prepare_data when train_bpe=True)
        self.tokenizer: Tokenizer = self._create_tokenizer()

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
        self.opt_state = None
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

    def _create_tokenizer(self, train_texts: Optional[List[str]] = None) -> Tokenizer:
        """Create tokenizer from model config."""
        mc = self.model_config
        if mc.train_bpe and train_texts is None:
            return ByteTokenizer()  # placeholder until prepare_data
        tok = create_tokenizer(
            tokenizer_type=mc.tokenizer_type,
            bpe_encoding=mc.bpe_encoding,
            bpe_tokenizer_path=mc.bpe_tokenizer_path,
            train_bpe=mc.train_bpe,
            bpe_vocab_size=mc.bpe_vocab_size,
            train_texts=train_texts,
        )
        self.model_config.vocab_size = tok.vocab_size
        return tok

    def _create_optimizer(self, total_steps: int) -> optax.GradientTransformation:
        cfg = self.training_config

        if cfg.lr_schedule == "constant":
            schedule = cfg.learning_rate
        elif cfg.lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=cfg.learning_rate,
                decay_steps=total_steps,
                alpha=cfg.min_lr / cfg.learning_rate,
            )
        elif cfg.lr_schedule == "exponential":
            schedule = optax.exponential_decay(
                init_value=cfg.learning_rate,
                transition_steps=total_steps // 10,
                decay_rate=0.9,
            )
        elif cfg.lr_schedule == "warmup_cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=cfg.learning_rate,
                warmup_steps=cfg.warmup_steps,
                decay_steps=total_steps,
                end_value=cfg.min_lr,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")

        if cfg.optimizer == "adam":
            opt = optax.adam(learning_rate=schedule)
        elif cfg.optimizer == "adamw":
            opt = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        if cfg.gradient_clip:
            opt = optax.chain(optax.clip_by_global_norm(cfg.gradient_clip), opt)

        return opt

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

        # Train BPE tokenizer on the training corpus if requested
        if self.model_config.train_bpe and self.model_config.tokenizer_type == "bpe":
            corpus = [s for s, _ in self.train_data] + [t for _, t in self.train_data]
            print(f"Training BPE tokenizer on {len(corpus)} texts (vocab_size={self.model_config.bpe_vocab_size})...")
            self.tokenizer = self._create_tokenizer(train_texts=corpus)
            print(f"BPE tokenizer ready — vocab_size={self.tokenizer.vocab_size}")

    def _encode_for_mdm(
        self,
        source_texts: List[str],
        target_texts: List[str],
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Encode source + target into a single concatenated sequence for MDM.

        Returns:
            (full_ids, attention_mask, target_ids, target_region_mask)
            - full_ids: [src_BOS, src_bytes..., tgt_bytes..., EOS, PAD...]
            - attention_mask: 1 where real tokens, 0 for padding.
            - target_ids: ground-truth for the target portion (same shape as full_ids,
              valid only where target_region_mask == 1).
            - target_region_mask: 1 in the target region, 0 elsewhere.
        """
        max_src = self.model_config.max_seq_length // 2
        max_tgt = self.model_config.max_seq_length - max_src
        total_max = self.model_config.max_seq_length

        all_ids = []
        all_masks = []
        all_target_ids = []
        all_target_region = []

        for src, tgt in zip(source_texts, target_texts):
            # Encode source: [BOS] + tokens
            src_tokens = self.tokenizer.encode_with_special(
                src, add_bos=True, add_eos=False, max_length=max_src,
            )

            # Encode target: tokens + [EOS]
            tgt_raw = self.tokenizer.encode(tgt)
            tgt_tokens = tgt_raw + [EOS_TOKEN_ID]
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
                # Recalculate target region
                tgt_len = real_len - src_len

            ids = full + [PAD_TOKEN_ID] * pad_len
            mask = [1.0] * real_len + [0.0] * pad_len

            # target_ids: the ground truth for ALL positions (we'll only use target region)
            target = ids[:]  # copy

            # target_region: 1 for target tokens, 0 elsewhere
            region = [0.0] * src_len + [1.0] * tgt_len + [0.0] * pad_len

            all_ids.append(ids)
            all_masks.append(mask)
            all_target_ids.append(target)
            all_target_region.append(region)

        return (
            jnp.array(all_ids, dtype=jnp.int32),
            jnp.array(all_masks, dtype=jnp.float32),
            jnp.array(all_target_ids, dtype=jnp.int32),
            jnp.array(all_target_region, dtype=jnp.float32),
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

    def _create_train_step(self):
        cfg = self.training_config

        @eqx.filter_jit
        def train_step(model, opt_state, full_ids, attention_mask, target_ids,
                        target_region_mask, t, mask_key):
            def loss_fn(model):
                # Apply masking to target region only
                mask_rate = self.noise_fn(t)
                rand = jrandom.uniform(mask_key, full_ids.shape)
                should_mask = (rand < mask_rate) & (target_region_mask > 0)
                # Never mask special tokens
                is_special = (
                    (full_ids == PAD_TOKEN_ID) |
                    (full_ids == BOS_TOKEN_ID) |
                    (full_ids == EOS_TOKEN_ID)
                )
                should_mask = should_mask & ~is_special

                masked_ids = jnp.where(should_mask, MASK_TOKEN_ID, full_ids)

                # Forward
                logits, tabnet_masks, new_state = model(
                    masked_ids, t, attention_mask, inference=False,
                )

                # Loss at masked positions
                loss = mdm_loss_with_sparsity(
                    logits, target_ids, should_mask, tabnet_masks,
                    attention_mask, cfg.sparsity_weight, cfg.label_smoothing,
                )
                return loss, (logits, tabnet_masks, new_state)

            (loss, (logits, masks, state)), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(model)

            updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)

            return model, new_opt_state, loss

        return train_step

    def train(self):
        """Main training loop."""
        print("\nInitializing MDM model and optimizer...")

        key = jrandom.PRNGKey(self.training_config.shuffle_seed)
        key, model_key = jrandom.split(key)

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
            key=model_key,
        )

        # Optimizer
        steps_per_epoch = max(len(self.train_data) // self.training_config.batch_size, 1)
        total_steps = steps_per_epoch * self.training_config.num_epochs
        self.optimizer = self._create_optimizer(total_steps)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        train_step_fn = self._create_train_step()

        print(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        print(f"Noise schedule: {self.training_config.noise_schedule}")
        print(f"Starting MDM training for {self.training_config.num_epochs} epochs...\n")

        global_step = 0

        for epoch in range(self.training_config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")

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
                key, t_key, mask_key = jrandom.split(key, 3)
                t = jrandom.uniform(t_key, (), minval=0.01, maxval=1.0)

                self.model, self.opt_state, loss = train_step_fn(
                    self.model, self.opt_state,
                    full_ids, attention_mask, target_ids, target_region_mask,
                    t, mask_key,
                )

                epoch_loss += float(loss)
                num_batches += 1
                global_step += 1

                pbar.set_postfix({'loss': f'{float(loss):.4f}'})

                if global_step % self.training_config.log_every == 0:
                    print(f"  Step {global_step}: loss={float(loss):.4f}")

                # Generate a sample every 10 steps
                if global_step % 10 == 0:
                    self._generate_sample(epoch, key, step=global_step)

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

    def _evaluate(self, data: List[Tuple]) -> float:
        """Compute average loss on a dataset."""
        key = jrandom.PRNGKey(0)
        total_loss = 0.0
        num_batches = 0

        for batch in self.create_batches(data, shuffle=False):
            full_ids, attention_mask, target_ids, target_region_mask = batch

            key, t_key, mask_key = jrandom.split(key, 3)
            # Evaluate at multiple timesteps and average for a more stable estimate
            losses = []
            for t_val in [0.25, 0.5, 0.75]:
                t = jnp.array(t_val)
                mask_rate = self.noise_fn(t)
                rand = jrandom.uniform(mask_key, full_ids.shape)
                should_mask = (rand < mask_rate) & (target_region_mask > 0)
                is_special = (
                    (full_ids == PAD_TOKEN_ID) |
                    (full_ids == BOS_TOKEN_ID) |
                    (full_ids == EOS_TOKEN_ID)
                )
                should_mask = should_mask & ~is_special
                masked_ids = jnp.where(should_mask, MASK_TOKEN_ID, full_ids)

                logits, tabnet_masks, _ = self.model(
                    masked_ids, t, attention_mask, inference=True,
                )
                loss = mdm_loss_with_sparsity(
                    logits, target_ids, should_mask, tabnet_masks,
                    attention_mask, self.training_config.sparsity_weight,
                )
                losses.append(float(loss))

            total_loss += np.mean(losses)
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _generate_sample(self, epoch: int, key: PRNGKeyArray, step: Optional[int] = None):
        """Generate and print a sample from the first test example."""
        sample_data = self.test_data if self.test_data else self.val_data
        if not sample_data:
            return

        source_text, target_text = sample_data[0]
        key, gen_key = jrandom.split(key)

        # Encode source using tokenizer
        max_src = self.model_config.max_seq_length // 2
        src_tokens = self.tokenizer.encode_with_special(
            source_text, add_bos=True, add_eos=False, max_length=max_src,
        )

        source_ids = jnp.array([src_tokens], dtype=jnp.int32)
        source_mask = jnp.ones_like(source_ids, dtype=jnp.float32)

        target_length = min(128, self.model_config.max_seq_length - len(src_tokens))

        generated = iterative_demask_seq2seq(
            self.model,
            source_ids,
            source_mask,
            target_length=target_length,
            num_steps=self.training_config.num_denoise_steps,
            temperature=self.training_config.sample_temperature,
            noise_schedule_fn=self.noise_fn,
            key=gen_key,
        )

        # Decode using tokenizer
        tokens = [int(t) for t in generated[0]]
        # Remove MASK and PAD before decoding
        tokens = [t for t in tokens if t != MASK_TOKEN_ID and t != PAD_TOKEN_ID]
        generated_text = self.tokenizer.decode(tokens)

        label = f"step {step}" if step is not None else f"epoch {epoch + 1}"
        print(f"\n  --- MDM Generation Sample ({label}) ---")
        print(f"  INPUT:     {source_text[:150]}{'...' if len(source_text) > 150 else ''}")
        print(f"  REFERENCE: {target_text[:150]}{'...' if len(target_text) > 150 else ''}")
        print(f"  GENERATED: {generated_text[:150]}{'...' if len(generated_text) > 150 else ''}")
        print(f"  ({len(tokens)} tokens, {self.training_config.num_denoise_steps} demasking steps)")
        print()


# ============================================================================
# CLI
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ByteTabNet Masked Diffusion Model",
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

    # Tokenizer
    parser.add_argument(
        "--tokenizer-type", choices=["byte", "bpe"], default=None,
        help="Tokenizer type: 'byte' (default, vocab=320) or 'bpe'",
    )
    parser.add_argument("--bpe-encoding", type=str, help="tiktoken encoding name for pre-trained BPE")
    parser.add_argument("--bpe-tokenizer-path", type=str, help="Path to custom-trained BPE tokenizer directory")
    parser.add_argument("--bpe-vocab-size", type=int, help="Vocabulary size for training custom BPE")
    parser.add_argument("--train-bpe", action="store_true", default=None, help="Train a new BPE tokenizer on the dataset")

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
    field_map_model = {k: k for k in MDMModelConfig.__annotations__}
    field_map_training = {k: k for k in MDMTrainingConfig.__annotations__}
    field_map_data = {k: k for k in DataConfig.__annotations__}

    # Map CLI arg names (with hyphens) to field names (with underscores)
    for cli_key, cli_val in cli.items():
        if cli_val is None or cli_key == 'config':
            continue
        field_name = cli_key.replace('-', '_')
        if field_name == 'task':
            field_name = 'task_type'
        if field_name in field_map_model:
            model_dict[field_name] = cli_val
        elif field_name in field_map_training:
            training_dict[field_name] = cli_val
        elif field_name in field_map_data:
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
    print("ByteTabNet Masked Diffusion Model Training")
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
