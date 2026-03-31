"""
ByteTabNet: TabNet implementation using JAX, Equinox, and Haliax.

This module implements TabNet (Attentive Interpretable Tabular Learning)
similar to the pytorch-tabnet (dream-quark) implementation, but using:
- JAX for numerical computation
- Equinox for neural network modules
- Haliax for named tensor axes

The model consumes byte-level encoded strings as input.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from jaxtyping import Array, Float, PRNGKeyArray


# ============================================================================
# Named Axes for Haliax
# ============================================================================

Batch = Axis("batch", 1)
SeqLen = Axis("seq_len", 1)
ByteVocab = Axis("byte_vocab", 320)  # 256 bytes + special tokens + padding
Embed = Axis("embed", 64)
Hidden = Axis("hidden", 128)
Feature = Axis("feature", 64)
Steps = Axis("steps", 5)


# ============================================================================
# Sparsemax Activation
# ============================================================================

def sparsemax(x: Array, axis: int = -1) -> Array:
    """
    Sparsemax activation function.

    Sparsemax is a sparse alternative to softmax that can output exact zeros.
    From: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"

    Args:
        x: Input array
        axis: Axis along which to compute sparsemax

    Returns:
        Sparsemax probabilities (sparse probability distribution)
    """
    # Sort input in descending order
    x_sorted = jnp.sort(x, axis=axis)[..., ::-1]

    # Compute cumulative sums
    cumsum = jnp.cumsum(x_sorted, axis=axis)

    # Find the support (number of non-zero elements)
    k = jnp.arange(1, x.shape[axis] + 1, dtype=x.dtype)

    # Reshape k for broadcasting
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]
    k = k.reshape(shape)

    # Check condition: 1 + k * z_k > cumsum
    support = (1 + k * x_sorted > cumsum).astype(x.dtype)

    # Find the threshold
    k_z = jnp.sum(support, axis=axis, keepdims=True)

    # Compute tau (threshold)
    tau_sum = jnp.sum(support * x_sorted, axis=axis, keepdims=True)
    tau = (tau_sum - 1) / jnp.maximum(k_z, 1)

    # Compute sparsemax output
    output = jnp.maximum(x - tau, 0)

    return output


def sparsemax_named(x: NamedArray, axis: Axis) -> NamedArray:
    """Sparsemax for NamedArrays using Haliax."""
    axis_idx = x.axes.index(axis)
    return hax.named(sparsemax(x.array, axis=axis_idx), x.axes)


# ============================================================================
# Ghost Batch Normalization
# ============================================================================

class GhostBatchNorm(eqx.Module):
    """
    Ghost Batch Normalization for TabNet.

    Applies batch normalization on virtual "ghost" batches to reduce
    the effect of large batch sizes and improve generalization.

    Args:
        feature_dim: Number of features to normalize
        virtual_batch_size: Size of ghost batches (default: 128)
        momentum: Momentum for running statistics (default: 0.01)
        eps: Small constant for numerical stability
    """
    weight: Array
    bias: Array
    running_mean: Array
    running_var: Array
    virtual_batch_size: int = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)

    def __init__(
        self,
        feature_dim: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.01,
        eps: float = 1e-5,
        *,
        key: PRNGKeyArray,
    ):
        self.feature_dim = feature_dim
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.eps = eps

        self.weight = jnp.ones(feature_dim)
        self.bias = jnp.zeros(feature_dim)
        self.running_mean = jnp.zeros(feature_dim)
        self.running_var = jnp.ones(feature_dim)

    def __call__(
        self,
        x: Array,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """
        Apply Ghost Batch Normalization.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)
            inference: Whether in inference mode
            state: Optional state dict for updating running statistics

        Returns:
            Normalized tensor and updated state
        """
        if inference:
            # Use running statistics during inference
            mean = self.running_mean
            var = self.running_var
            normalized = (x - mean) / jnp.sqrt(var + self.eps)
            return normalized * self.weight + self.bias, state

        batch_size = x.shape[0]

        if batch_size <= self.virtual_batch_size:
            # Standard batch norm if batch is small
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)
            normalized = (x - mean) / jnp.sqrt(var + self.eps)
        else:
            # Ghost batch normalization
            num_chunks = math.ceil(batch_size / self.virtual_batch_size)

            # Pad if necessary
            pad_size = num_chunks * self.virtual_batch_size - batch_size
            if pad_size > 0:
                x_padded = jnp.concatenate([x, jnp.zeros((pad_size, self.feature_dim))], axis=0)
            else:
                x_padded = x

            # Reshape into ghost batches
            x_reshaped = x_padded.reshape(num_chunks, self.virtual_batch_size, self.feature_dim)

            # Compute statistics per ghost batch
            mean = jnp.mean(x_reshaped, axis=1, keepdims=True)
            var = jnp.var(x_reshaped, axis=1, keepdims=True)

            # Normalize
            normalized = (x_reshaped - mean) / jnp.sqrt(var + self.eps)
            normalized = normalized.reshape(-1, self.feature_dim)[:batch_size]

            # Average statistics for running update
            mean = jnp.mean(mean.squeeze(1), axis=0)
            var = jnp.mean(var.squeeze(1), axis=0)

        output = normalized * self.weight + self.bias

        # Update running statistics
        new_running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        new_running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        new_state = {
            "running_mean": new_running_mean,
            "running_var": new_running_var,
        }

        return output, new_state


# ============================================================================
# GLU (Gated Linear Unit) Block
# ============================================================================

class GLUBlock(eqx.Module):
    """
    SwiGLU block used in TabNet's feature transformer.

    Applies: FC -> BN -> SwiGLU activation
    where SwiGLU(x, gate) = x * silu(gate).
    """
    fc: eqx.nn.Linear
    bn: GhostBatchNorm

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        virtual_batch_size: int = 128,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jrandom.split(key)
        # Double output for GLU gating
        self.fc = eqx.nn.Linear(input_dim, output_dim * 2, use_bias=False, key=key1)
        self.bn = GhostBatchNorm(output_dim * 2, virtual_batch_size, key=key2)

    def __call__(
        self,
        x: Array,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        x = jax.vmap(self.fc)(x)
        x, new_state = self.bn(x, inference, state)

        # SwiGLU activation: split and apply gating
        x, gate = jnp.split(x, 2, axis=-1)
        x = x * jax.nn.silu(gate)

        return x, new_state


# ============================================================================
# Feature Transformer
# ============================================================================

class FeatureTransformer(eqx.Module):
    """
    Feature Transformer block for TabNet.

    Consists of shared layers (across all decision steps) and
    step-specific layers. Uses GLU blocks with skip connections.
    """
    shared_layers: List[GLUBlock]
    step_layers: List[GLUBlock]
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    n_shared: int = eqx.field(static=True)
    n_step: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_shared: int = 2,
        n_step: int = 2,
        virtual_batch_size: int = 128,
        *,
        key: PRNGKeyArray,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_shared = n_shared
        self.n_step = n_step

        keys = jrandom.split(key, n_shared + n_step)

        # Shared layers
        self.shared_layers = []
        current_dim = input_dim
        for i in range(n_shared):
            layer = GLUBlock(current_dim, output_dim, virtual_batch_size, key=keys[i])
            self.shared_layers.append(layer)
            current_dim = output_dim

        # Step-specific layers
        self.step_layers = []
        for i in range(n_step):
            layer = GLUBlock(current_dim, output_dim, virtual_batch_size, key=keys[n_shared + i])
            self.step_layers.append(layer)
            current_dim = output_dim

    def __call__(
        self,
        x: Array,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """Forward pass through feature transformer."""
        all_states = {}

        # Shared layers with skip connections
        for i, layer in enumerate(self.shared_layers):
            layer_state = state.get(f"shared_{i}") if state else None
            out, new_state = layer(x, inference, layer_state)
            if new_state:
                all_states[f"shared_{i}"] = new_state

            # Skip connection (except first layer)
            if i > 0 or self.input_dim == self.output_dim:
                x = (x + out) * jnp.sqrt(0.5)
            else:
                x = out

        # Step-specific layers with skip connections
        for i, layer in enumerate(self.step_layers):
            layer_state = state.get(f"step_{i}") if state else None
            out, new_state = layer(x, inference, layer_state)
            if new_state:
                all_states[f"step_{i}"] = new_state
            x = (x + out) * jnp.sqrt(0.5)

        return x, all_states if all_states else None


# ============================================================================
# Attentive Transformer
# ============================================================================

class AttentiveTransformer(eqx.Module):
    """
    Attentive Transformer for TabNet.

    Computes attention masks for feature selection at each decision step.
    Uses sparsemax activation for sparse feature selection.
    """
    fc: eqx.nn.Linear
    bn: GhostBatchNorm

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        virtual_batch_size: int = 128,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jrandom.split(key)
        self.fc = eqx.nn.Linear(input_dim, output_dim, use_bias=False, key=key1)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size, key=key2)

    def __call__(
        self,
        x: Array,
        prior_scales: Array,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """
        Compute attention mask.

        Args:
            x: Processed features from feature transformer
            prior_scales: Aggregated attention from previous steps (for regularization)
            inference: Whether in inference mode
            state: Optional state dict

        Returns:
            Attention mask and updated state
        """
        x = jax.vmap(self.fc)(x)
        x, new_state = self.bn(x, inference, state)

        # Multiply by prior scales to encourage diversity
        x = x * prior_scales

        # Apply sparsemax for sparse attention
        mask = sparsemax(x, axis=-1)

        return mask, new_state


# ============================================================================
# Byte Embedding Layer
# ============================================================================

class ByteEmbedding(eqx.Module):
    """
    Byte-level embedding layer for string inputs.

    Converts byte sequences to embeddings, similar to:
    input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]])

    The vocabulary includes:
    - 0: Padding token
    - 1: BOS (beginning of sequence) token
    - 2: EOS (end of sequence) token
    - 3-63: Reserved special tokens
    - 64-319: Byte values (0-255) offset by 64
    """
    embedding: Array
    vocab_size: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int = 320,
        embed_dim: int = 64,
        *,
        key: PRNGKeyArray,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Initialize embeddings with small random values
        self.embedding = jrandom.normal(key, (vocab_size, embed_dim)) * 0.02

    def __call__(self, input_ids: Array) -> Array:
        """
        Embed byte sequences.

        Args:
            input_ids: Integer tensor of shape (batch_size, seq_len)
                       Values should be in range [0, vocab_size)

        Returns:
            Embedded tensor of shape (batch_size, seq_len, embed_dim)
        """
        return self.embedding[input_ids]

    @staticmethod
    def encode_string(text: str, max_length: Optional[int] = None) -> Array:
        """
        Encode a string to byte-level token IDs.

        Args:
            text: Input string
            max_length: Optional maximum sequence length

        Returns:
            Array of token IDs with BOS prefix
        """
        # Encode bytes with offset of 64
        byte_ids = [b + 64 for b in text.encode("utf-8")]

        # Add BOS token at the start
        token_ids = [1] + byte_ids

        # Truncate if necessary
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return jnp.array(token_ids, dtype=jnp.int32)

    @staticmethod
    def encode_batch(
        texts: List[str],
        max_length: Optional[int] = None,
        pad_to_max: bool = True,
    ) -> Tuple[Array, Array]:
        """
        Encode a batch of strings.

        Args:
            texts: List of input strings
            max_length: Maximum sequence length
            pad_to_max: Whether to pad all sequences to max_length

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        encoded = [ByteEmbedding.encode_string(t, max_length) for t in texts]

        if pad_to_max and max_length is not None:
            # Pad sequences
            padded = []
            masks = []
            for seq in encoded:
                pad_len = max_length - len(seq)
                if pad_len > 0:
                    padded.append(jnp.concatenate([seq, jnp.zeros(pad_len, dtype=jnp.int32)]))
                    masks.append(jnp.concatenate([jnp.ones(len(seq)), jnp.zeros(pad_len)]))
                else:
                    padded.append(seq)
                    masks.append(jnp.ones(len(seq)))

            return jnp.stack(padded), jnp.stack(masks)
        else:
            # Find max length in batch
            batch_max_len = max(len(seq) for seq in encoded)
            padded = []
            masks = []
            for seq in encoded:
                pad_len = batch_max_len - len(seq)
                if pad_len > 0:
                    padded.append(jnp.concatenate([seq, jnp.zeros(pad_len, dtype=jnp.int32)]))
                    masks.append(jnp.concatenate([jnp.ones(len(seq)), jnp.zeros(pad_len)]))
                else:
                    padded.append(seq)
                    masks.append(jnp.ones(len(seq)))

            return jnp.stack(padded), jnp.stack(masks)


# ============================================================================
# TabNet Encoder
# ============================================================================

class TabNetEncoder(eqx.Module):
    """
    TabNet Encoder module.

    Implements the core TabNet architecture with:
    - Initial batch normalization
    - Feature transformer (shared + step-specific layers)
    - Attentive transformer for feature selection
    - Multiple decision steps with sparse attention
    """
    initial_bn: GhostBatchNorm
    initial_fc: eqx.nn.Linear
    feature_transformers: List[FeatureTransformer]
    attention_transformers: List[AttentiveTransformer]

    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_d: int = eqx.field(static=True)  # Decision embedding dimension
    n_a: int = eqx.field(static=True)  # Attention embedding dimension
    gamma: float = eqx.field(static=True)  # Relaxation factor for attention

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_steps: int = 5,
        n_d: int = 64,
        n_a: int = 64,
        gamma: float = 1.5,
        n_shared: int = 2,
        n_step: int = 2,
        virtual_batch_size: int = 128,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize TabNet Encoder.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            n_steps: Number of decision steps
            n_d: Decision embedding dimension
            n_a: Attention embedding dimension
            gamma: Relaxation factor for feature reuse (higher = more reuse)
            n_shared: Number of shared layers in feature transformer
            n_step: Number of step-specific layers in feature transformer
            virtual_batch_size: Virtual batch size for Ghost BN
            key: PRNG key
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma

        keys = jrandom.split(key, 2 + 2 * n_steps)

        # Initial processing
        self.initial_bn = GhostBatchNorm(input_dim, virtual_batch_size, key=keys[0])
        self.initial_fc = eqx.nn.Linear(input_dim, n_d + n_a, use_bias=False, key=keys[1])

        # Feature transformers (one per step)
        self.feature_transformers = []
        for i in range(n_steps):
            ft = FeatureTransformer(
                input_dim=input_dim,
                output_dim=n_d + n_a,
                n_shared=n_shared,
                n_step=n_step,
                virtual_batch_size=virtual_batch_size,
                key=keys[2 + i],
            )
            self.feature_transformers.append(ft)

        # Attention transformers (one per step, except last)
        self.attention_transformers = []
        for i in range(n_steps - 1):
            at = AttentiveTransformer(
                input_dim=n_a,
                output_dim=input_dim,
                virtual_batch_size=virtual_batch_size,
                key=keys[2 + n_steps + i],
            )
            self.attention_transformers.append(at)

    def __call__(
        self,
        x: Array,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Array, Optional[dict]]:
        """
        Forward pass through TabNet encoder.

        Args:
            x: Input features of shape (batch_size, input_dim)
            inference: Whether in inference mode
            state: Optional state dict for batch norm

        Returns:
            Tuple of:
            - Output embeddings (batch_size, n_d)
            - Feature importance masks (batch_size, n_steps, input_dim)
            - Updated state dict
        """
        batch_size = x.shape[0]
        all_states = {}

        # Initial batch normalization
        x_bn, bn_state = self.initial_bn(x, inference, state.get("initial_bn") if state else None)
        if bn_state:
            all_states["initial_bn"] = bn_state

        # Initial transformation for first step
        processed = jax.vmap(self.initial_fc)(x_bn)

        # Initialize prior scales (for attention regularization)
        prior_scales = jnp.ones((batch_size, self.input_dim))

        # Aggregate outputs and masks
        output_aggregate = jnp.zeros((batch_size, self.n_d))
        masks = []

        for step in range(self.n_steps):
            # Split into decision and attention parts
            decision_out = processed[:, :self.n_d]
            attention_out = processed[:, self.n_d:]

            # Accumulate decision outputs with SiLU
            output_aggregate = output_aggregate + jax.nn.silu(decision_out)

            # Compute attention mask (except for last step)
            if step < self.n_steps - 1:
                mask, att_state = self.attention_transformers[step](
                    attention_out,
                    prior_scales,
                    inference,
                    state.get(f"attention_{step}") if state else None,
                )
                if att_state:
                    all_states[f"attention_{step}"] = att_state

                masks.append(mask)

                # Update prior scales (decrease weights of previously attended features)
                prior_scales = prior_scales * (self.gamma - mask)

                # Mask the input and process
                masked_x = mask * x_bn
                processed, ft_state = self.feature_transformers[step](
                    masked_x,
                    inference,
                    state.get(f"feature_{step}") if state else None,
                )
                if ft_state:
                    all_states[f"feature_{step}"] = ft_state

        # Stack masks
        masks = jnp.stack(masks, axis=1) if masks else jnp.zeros((batch_size, 0, self.input_dim))

        return output_aggregate, masks, all_states if all_states else None


# ============================================================================
# TabNet Decoder (for pretraining)
# ============================================================================

class TabNetDecoder(eqx.Module):
    """
    TabNet Decoder for reconstruction (used in pretraining).

    After the final decision step, the decoder can optionally loop back
    to the first decision step for ``n_recursive`` additional passes.
    Each recursive pass reuses the same feature transformers (weight
    sharing) and accumulates into the reconstruction output, allowing
    the decoder to iteratively refine its reconstruction.
    """
    feature_transformers: List[FeatureTransformer]
    reconstruction_layer: eqx.nn.Linear

    n_steps: int = eqx.field(static=True)
    n_recursive: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        n_d: int = 64,
        n_steps: int = 5,
        n_shared: int = 2,
        n_step: int = 2,
        n_recursive: int = 0,
        virtual_batch_size: int = 128,
        *,
        key: PRNGKeyArray,
    ):
        self.n_steps = n_steps
        self.n_recursive = n_recursive

        keys = jrandom.split(key, n_steps + 1)

        self.feature_transformers = []
        for i in range(n_steps):
            ft = FeatureTransformer(
                input_dim=n_d,
                output_dim=n_d,
                n_shared=n_shared,
                n_step=n_step,
                virtual_batch_size=virtual_batch_size,
                key=keys[i],
            )
            self.feature_transformers.append(ft)

        self.reconstruction_layer = eqx.nn.Linear(n_d, input_dim, key=keys[-1])

    def _run_decision_steps(
        self,
        x: Array,
        output: Array,
        inference: bool,
        state: Optional[dict],
        all_states: dict,
        pass_idx: int,
    ) -> Tuple[Array, Array, dict]:
        """Run one full pass through all decision steps.

        Args:
            x: Current hidden state (batch, n_d).
            output: Accumulated reconstruction output (batch, input_dim).
            inference: Whether in inference mode.
            state: Batch norm state dict.
            all_states: Dict collecting updated BN states.
            pass_idx: 0 for the initial pass, 1.. for recursive passes.

        Returns:
            (x, output, all_states) after this pass.
        """
        for step in range(self.n_steps):
            state_key = f"decoder_ft_{step}_r{pass_idx}"
            ft_state = state.get(state_key) if state else None
            x, new_state = self.feature_transformers[step](x, inference, ft_state)
            if new_state:
                all_states[state_key] = new_state

            output = output + jax.vmap(self.reconstruction_layer)(x)

        return x, output, all_states

    def __call__(
        self,
        x: Array,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """Decode embeddings back to input space.

        Runs the decision steps once, then loops back ``n_recursive``
        additional times — each time feeding the output of the last
        decision step back into the first.
        """
        all_states = {}
        batch_size = x.shape[0]
        output = jnp.zeros((batch_size, self.reconstruction_layer.out_features))

        # Initial pass (pass 0)
        x, output, all_states = self._run_decision_steps(
            x, output, inference, state, all_states, pass_idx=0,
        )

        # Recursive passes: feed x back through the decision steps
        for r in range(self.n_recursive):
            x, output, all_states = self._run_decision_steps(
                x, output, inference, state, all_states, pass_idx=r + 1,
            )

        return output, all_states if all_states else None


# ============================================================================
# ByteTabNet: Main Model
# ============================================================================

class ByteTabNet(eqx.Module):
    """
    ByteTabNet: TabNet for byte-level string inputs.

    This model combines:
    - Byte-level embedding for string inputs
    - Optional positional encoding
    - Sequence pooling (mean/max/attention)
    - TabNet encoder for feature processing
    - Output projection head

    Suitable for:
    - Text classification
    - Sequence labeling
    - Tabular data with string features
    """
    byte_embedding: ByteEmbedding
    position_embedding: Optional[Array]
    sparsemax_attention: eqx.nn.Linear
    input_projection: eqx.nn.Linear
    tabnet_encoder: TabNetEncoder
    output_head: eqx.nn.Linear

    max_seq_length: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    use_positional: bool = eqx.field(static=True)

    def __init__(
        self,
        output_dim: int,
        max_seq_length: int = 512,
        embed_dim: int = 64,
        hidden_dim: int = 128,
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
        Initialize ByteTabNet.

        Args:
            output_dim: Output dimension (num classes for classification)
            max_seq_length: Maximum sequence length
            embed_dim: Byte embedding dimension
            hidden_dim: Hidden dimension after projection
            n_steps: Number of TabNet decision steps
            n_d: Decision embedding dimension
            n_a: Attention embedding dimension
            gamma: Relaxation factor for attention
            n_shared: Number of shared layers
            n_step: Number of step-specific layers
            virtual_batch_size: Virtual batch size for Ghost BN
            use_positional: Whether to use positional embeddings
            vocab_size: Vocabulary size for byte embedding
            key: PRNG key
        """
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_positional = use_positional

        keys = jrandom.split(key, 5)

        # Byte embedding
        self.byte_embedding = ByteEmbedding(vocab_size, embed_dim, key=keys[0])

        # Positional embedding
        if use_positional:
            self.position_embedding = jrandom.normal(keys[1], (max_seq_length, embed_dim)) * 0.02
        else:
            self.position_embedding = None

        # Sparsemax attention for sequence aggregation
        self.sparsemax_attention = eqx.nn.Linear(embed_dim, 1, key=keys[2])

        # Project to hidden dim for TabNet
        self.input_projection = eqx.nn.Linear(embed_dim, hidden_dim, key=keys[3])

        # TabNet encoder
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
            key=keys[4],
        )

        # Output head
        key5 = jrandom.split(keys[4])[0]
        self.output_head = eqx.nn.Linear(n_d, output_dim, key=key5)

    def _sparsemax_pool(
        self,
        embeddings: Array,
        mask: Optional[Array] = None,
    ) -> Array:
        """
        Aggregate sequence embeddings using sparsemax attention.

        Sparsemax produces sparse weights, attending to only a few key
        positions rather than spreading weight across the full sequence.

        Args:
            embeddings: Shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Aggregated embeddings (batch_size, embed_dim)
        """
        # Compute attention scores
        scores = jax.vmap(jax.vmap(self.sparsemax_attention))(embeddings)  # (batch, seq, 1)

        if mask is not None:
            scores = jnp.where(mask[..., None] > 0, scores, -1e9)

        # Sparsemax: only a few positions get non-zero weight
        weights = sparsemax(scores.squeeze(-1), axis=-1)  # (batch, seq)
        weights = weights[..., None]  # (batch, seq, 1)

        return jnp.sum(embeddings * weights, axis=1)

    def _pool_sequence_windowed(
        self,
        embeddings: Array,
        num_windows: int = 8,
    ) -> Array:
        """
        Windowed pooling: split the sequence into windows, mean-pool each,
        then aggregate with sparsemax attention.

        Args:
            embeddings: Shape (batch_size, seq_len, embed_dim)
            num_windows: Number of windows to split the sequence into.

        Returns:
            Aggregated embeddings (batch_size, embed_dim)
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        win_size = max(seq_len // num_windows, 1)

        # Mean-pool each window
        window_embeds = []
        for w in range(num_windows):
            start = w * win_size
            end = min(start + win_size, seq_len)
            if start >= seq_len:
                window_embeds.append(jnp.zeros((batch_size, embed_dim)))
            else:
                window_embeds.append(jnp.mean(embeddings[:, start:end, :], axis=1))

        windows = jnp.stack(window_embeds, axis=1)  # (batch, num_windows, embed)

        # Sparsemax attention over windows
        scores = jax.vmap(jax.vmap(self.sparsemax_attention))(windows)  # (batch, num_windows, 1)
        weights = sparsemax(scores.squeeze(-1), axis=-1)[..., None]  # (batch, num_windows, 1)

        return jnp.sum(windows * weights, axis=1)

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Array, Optional[dict]]:
        """
        Forward pass through ByteTabNet.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)
            inference: Whether in inference mode
            state: Optional state dict for batch norm

        Returns:
            Tuple of:
            - Output logits (batch_size, output_dim)
            - Feature importance masks from TabNet
            - Updated state dict
        """
        batch_size, seq_len = input_ids.shape

        # Embed bytes
        embeddings = self.byte_embedding(input_ids)  # (batch, seq, embed)

        # Add positional embeddings
        if self.use_positional and self.position_embedding is not None:
            pos_emb = self.position_embedding[:seq_len]
            embeddings = embeddings + pos_emb

        # Sparsemax attention over sequence positions
        pooled = self._sparsemax_pool(embeddings, attention_mask)  # (batch, embed)

        # Project to hidden dimension
        hidden = jax.vmap(self.input_projection)(pooled)  # (batch, hidden)

        # Pass through TabNet
        encoded, masks, new_state = self.tabnet_encoder(hidden, inference, state)

        # Output projection
        logits = jax.vmap(self.output_head)(encoded)

        return logits, masks, new_state

    def encode_features(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Array, Optional[dict]]:
        """Return raw TabNet encoded features (n_d dim) before the output head.

        Used by ByteTabNetSeq2Seq to get hidden context for the decoder.
        """
        batch_size, seq_len = input_ids.shape
        embeddings = self.byte_embedding(input_ids)
        if self.use_positional and self.position_embedding is not None:
            pos_emb = self.position_embedding[:seq_len]
            embeddings = embeddings + pos_emb
        pooled = self._sparsemax_pool(embeddings, attention_mask)
        hidden = jax.vmap(self.input_projection)(pooled)
        encoded, masks, new_state = self.tabnet_encoder(hidden, inference, state)
        return encoded, masks, new_state

    def encode_sequence(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Array, Optional[dict]]:
        """Return per-position features without pooling.

        Projects each position to hidden_dim independently, giving the decoder
        the full sequence for cross-attention.  The pooled path is still run
        through TabNet so that sparsity masks (used for regularisation) are
        available.

        Returns:
            Tuple of:
            - sequence_features: (batch, seq_len, hidden_dim)
            - TabNet attention masks (from pooled path)
            - Updated state dict
        """
        batch_size, seq_len = input_ids.shape
        embeddings = self.byte_embedding(input_ids)  # (batch, seq_len, embed_dim)
        if self.use_positional and self.position_embedding is not None:
            pos_emb = self.position_embedding[:seq_len]
            embeddings = embeddings + pos_emb

        # Per-position projection: (batch, seq_len, embed_dim) → (batch, seq_len, hidden_dim)
        sequence_features = jax.vmap(jax.vmap(self.input_projection))(embeddings)

        # Pooled path through TabNet for masks / sparsity regularisation
        pooled = self._sparsemax_pool(embeddings, attention_mask)
        hidden = jax.vmap(self.input_projection)(pooled)
        _, masks, new_state = self.tabnet_encoder(hidden, inference, state)

        return sequence_features, masks, new_state

    def predict(self, input_ids: Array, attention_mask: Optional[Array] = None) -> Array:
        """Make predictions (convenience method for inference)."""
        logits, _, _ = self(input_ids, attention_mask, inference=True)
        return jax.nn.softmax(logits, axis=-1)

    def get_feature_importance(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
    ) -> Array:
        """Get aggregated feature importance from attention masks."""
        _, masks, _ = self(input_ids, attention_mask, inference=True)
        # Average across decision steps
        return jnp.mean(masks, axis=1)

    def generate_step(
        self,
        input_ids: Array,
        window_size: int = 128,
        inference: bool = True,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """
        Single step of autoregressive generation.

        Focuses on a trailing window of the most recent bytes so TabNet
        always sees a fixed-size, fresh context regardless of how long
        the generated sequence has grown.

        Args:
            input_ids: Token IDs generated so far (batch, current_seq_len).
            window_size: Number of trailing tokens to keep as context.
            inference: Whether in inference mode.
            state: Optional batch-norm state dict.

        Returns:
            (logits, new_state)
            - logits: next-token logits (batch, output_dim)
            - new_state: updated batch-norm state
        """
        # 1. Take only the last window_size tokens
        context_ids = input_ids[:, -window_size:]

        # 2. Embed
        embeddings = self.byte_embedding(context_ids)

        # 3. Positional encoding aligned to the window
        if self.use_positional and self.position_embedding is not None:
            seq_len = context_ids.shape[1]
            pos_emb = self.position_embedding[:seq_len]
            embeddings = embeddings + pos_emb

        # 4. Windowed sparsemax pooling
        pooled = self._pool_sequence_windowed(embeddings, num_windows=8)

        # 5. Project and pass through TabNet
        hidden = jax.vmap(self.input_projection)(pooled)
        encoded, _, new_state = self.tabnet_encoder(hidden, inference, state)

        # 6. Predict next-byte logits
        logits = jax.vmap(self.output_head)(encoded)

        return logits, new_state

    def chat(
        self,
        prompt: str,
        max_gen: int = 50,
        temperature: float = 0.8,
        seed: int = 0,
    ) -> Array:
        """
        Autoregressive generation loop using generate_step.

        Encodes the prompt, then repeatedly predicts the next byte token
        using temperature-scaled categorical sampling to avoid repetitive
        loops (e.g. "rryyyy").

        Args:
            prompt: Input string to continue from.
            max_gen: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more creative).
            seed: Random seed.

        Returns:
            Generated token IDs (1, seq_len) including the prompt tokens.
        """
        key = jrandom.PRNGKey(seed)

        # Encode prompt: [BOS] + byte tokens
        input_ids = ByteEmbedding.encode_string(prompt)[None, :]  # (1, prompt_len)
        generated_ids = input_ids
        state = None

        for _ in range(max_gen):
            key, subkey = jrandom.split(key)

            # Get next-token logits via windowed generate_step
            logits, state = self.generate_step(generated_ids, state=state)

            # Temperature-scaled sampling
            logits = logits / temperature
            next_token = jrandom.categorical(subkey, logits, axis=-1)

            # Append to sequence
            generated_ids = jnp.concatenate(
                [generated_ids, next_token[:, None]], axis=1,
            )

            # Stop on EOS (token 2)
            if int(next_token[0]) == 2:
                break

        return generated_ids


# ============================================================================
# Autoregressive Decoder for Seq2Seq
# ============================================================================

class ByteTabNetSeq2SeqDecoder(eqx.Module):
    """
    Autoregressive decoder for seq2seq generation.

    Generates output tokens one at a time, conditioning on:
    - Previously generated tokens
    - Encoder context

    Uses TabNet-style feature processing with causal masking.
    """
    byte_embedding: ByteEmbedding
    position_embedding: Optional[Array]
    context_projection: eqx.nn.Linear
    decoder_layers: List[GLUBlock]
    cross_attention_fc: eqx.nn.Linear
    cross_attention_proj: eqx.nn.Linear
    output_projection: eqx.nn.Linear

    vocab_size: int = eqx.field(static=True)
    max_seq_length: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int = 320,
        max_seq_length: int = 512,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        context_dim: int = 64,
        n_layers: int = 4,
        virtual_batch_size: int = 128,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize the decoder.

        Args:
            vocab_size: Output vocabulary size (320 for bytes + special tokens)
            max_seq_length: Maximum sequence length for generation
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            context_dim: Encoder context dimension (n_d from TabNet)
            n_layers: Number of decoder layers
            virtual_batch_size: Virtual batch size for Ghost BN
            key: PRNG key
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        keys = jrandom.split(key, 4 + n_layers)

        # Byte embedding (shared with encoder or separate)
        self.byte_embedding = ByteEmbedding(vocab_size, embed_dim, key=keys[0])

        # Positional embedding
        self.position_embedding = jrandom.normal(keys[1], (max_seq_length, embed_dim)) * 0.02

        # Context projection (from encoder n_d to decoder hidden)
        self.context_projection = eqx.nn.Linear(context_dim, hidden_dim, key=keys[2])

        # Decoder GLU layers
        self.decoder_layers = []
        current_dim = embed_dim + hidden_dim  # Concatenate token embedding with context
        for i in range(n_layers):
            layer = GLUBlock(current_dim, hidden_dim, virtual_batch_size, key=keys[3 + i])
            self.decoder_layers.append(layer)
            current_dim = hidden_dim

        # Cross-attention components
        self.cross_attention_fc = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[-2])
        self.cross_attention_proj = eqx.nn.Linear(hidden_dim * 2, hidden_dim, key=keys[-1])

        # Output projection to vocabulary
        key_out = jrandom.split(keys[-1])[0]
        self.output_projection = eqx.nn.Linear(hidden_dim, vocab_size, key=key_out)

    def _causal_attention(
        self,
        query: Array,
        key_value: Array,
        mask: Optional[Array] = None,
    ) -> Array:
        """
        Simple causal self-attention mechanism.

        Args:
            query: Query tensor (batch, seq_len, hidden)
            key_value: Key/Value tensor (batch, seq_len, hidden)
            mask: Optional attention mask

        Returns:
            Attended output (batch, seq_len, hidden)
        """
        # Compute attention scores
        scores = jnp.einsum('bqh,bkh->bqk', query, key_value)
        scores = scores / jnp.sqrt(query.shape[-1])

        # Apply causal mask
        seq_len = query.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(causal_mask == 0, -1e9, scores)

        # Apply optional padding mask
        if mask is not None:
            scores = jnp.where(mask[:, None, :] == 0, -1e9, scores)

        # Softmax and weighted sum
        weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bqk,bkh->bqh', weights, key_value)

        return output

    def _cross_attention(
        self,
        decoder_hidden: Array,
        encoder_context: Array,
        encoder_mask: Optional[Array] = None,
    ) -> Array:
        """
        Cross-attention between decoder and encoder.

        Args:
            decoder_hidden: Decoder hidden states (batch, dec_len, hidden)
            encoder_context: Encoder outputs (batch, enc_len, hidden)
            encoder_mask: Optional encoder attention mask

        Returns:
            Context-enhanced decoder hidden (batch, dec_len, hidden)
        """
        # Project decoder hidden for attention query
        query = jax.vmap(jax.vmap(self.cross_attention_fc))(decoder_hidden)

        # Compute attention scores
        scores = jnp.einsum('bqh,bkh->bqk', query, encoder_context)
        scores = scores / jnp.sqrt(decoder_hidden.shape[-1])

        # Apply encoder mask if provided
        if encoder_mask is not None:
            scores = jnp.where(encoder_mask[:, None, :] == 0, -1e9, scores)

        # Softmax and weighted sum
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum('bqk,bkh->bqh', weights, encoder_context)

        # Combine with decoder hidden
        combined = jnp.concatenate([decoder_hidden, context], axis=-1)
        output = jax.vmap(jax.vmap(self.cross_attention_proj))(combined)

        return output

    def __call__(
        self,
        input_ids: Array,
        encoder_context: Array,
        encoder_mask: Optional[Array] = None,
        decoder_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """
        Forward pass through the decoder.

        Args:
            input_ids: Decoder input token IDs (batch, dec_len) - shifted right during training
            encoder_context: Encoder hidden states (batch, enc_len, hidden) or (batch, hidden)
            encoder_mask: Optional encoder attention mask
            decoder_mask: Optional decoder attention mask
            inference: Whether in inference mode
            state: Optional batch norm state

        Returns:
            Tuple of (logits, state)
            - logits: (batch, dec_len, vocab_size)
            - state: Updated batch norm state
        """
        batch_size, dec_len = input_ids.shape
        all_states = {}

        # Embed decoder inputs
        embeddings = self.byte_embedding(input_ids)  # (batch, dec_len, embed)

        # Add positional embeddings
        pos_emb = self.position_embedding[:dec_len]
        embeddings = embeddings + pos_emb

        # Handle encoder context shape
        if encoder_context.ndim == 2:
            # Context is pooled: (batch, hidden) -> expand for cross-attention
            encoder_context = encoder_context[:, None, :]  # (batch, 1, hidden)

        # Project context
        context_proj = jax.vmap(jax.vmap(self.context_projection))(encoder_context)

        # Broadcast context to each decoder position
        if context_proj.shape[1] == 1:
            context_broadcast = jnp.broadcast_to(
                context_proj,
                (batch_size, dec_len, self.hidden_dim)
            )
        else:
            # Use cross-attention if encoder has sequence output
            context_broadcast = None

        # Concatenate embeddings with context for initial input
        if context_broadcast is not None:
            hidden = jnp.concatenate([embeddings, context_broadcast], axis=-1)
        else:
            # Will use cross-attention instead
            # Pad embeddings to expected dimension
            hidden = jnp.concatenate([
                embeddings,
                jnp.zeros((batch_size, dec_len, self.hidden_dim))
            ], axis=-1)

        # Process through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            # Reshape for GLUBlock: (batch * dec_len, hidden)
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])

            layer_state = state.get(f"decoder_layer_{i}") if state else None
            hidden_flat, new_state = layer(hidden_flat, inference, layer_state)

            if new_state:
                all_states[f"decoder_layer_{i}"] = new_state

            hidden = hidden_flat.reshape(batch_size, dec_len, -1)

            # Apply cross-attention if encoder has sequence
            if context_broadcast is None and i == len(self.decoder_layers) // 2:
                hidden = self._cross_attention(hidden, context_proj, encoder_mask)

        # Apply causal self-attention
        hidden = self._causal_attention(hidden, hidden, decoder_mask)

        # Project to vocabulary
        logits = jax.vmap(jax.vmap(self.output_projection))(hidden)

        return logits, all_states if all_states else None

    def generate_step(
        self,
        prev_token: Array,
        encoder_context: Array,
        position: int,
        cache: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Generate a single token (for autoregressive generation).

        Args:
            prev_token: Previous token ID (batch,) or (batch, 1)
            encoder_context: Encoder context (batch, hidden)
            position: Current position in sequence
            cache: Optional KV cache for efficiency

        Returns:
            Tuple of (next_token_logits, updated_cache)
        """
        if prev_token.ndim == 1:
            prev_token = prev_token[:, None]

        # Get logits for the single token
        logits, _ = self(
            prev_token,
            encoder_context,
            inference=True,
        )

        # Return logits for the last position
        return logits[:, -1, :], cache


# ============================================================================
# ByteTabNet Seq2Seq Model
# ============================================================================

class ByteTabNetSeq2Seq(eqx.Module):
    """
    ByteTabNet Sequence-to-Sequence model.

    Combines a TabNet-based encoder with an autoregressive decoder
    for tasks like:
    - Text generation
    - Translation
    - Summarization
    - Any sequence-to-sequence task with byte-level tokenization

    The encoder uses TabNet's attention mechanism for feature selection,
    while the decoder generates output tokens autoregressively.
    """
    encoder: ByteTabNet
    decoder: ByteTabNetSeq2SeqDecoder
    generation_head: eqx.nn.Linear

    vocab_size: int = eqx.field(static=True)
    max_seq_length: int = eqx.field(static=True)
    bos_token_id: int = eqx.field(static=True)
    eos_token_id: int = eqx.field(static=True)
    pad_token_id: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int = 320,
        max_seq_length: int = 512,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_steps: int = 5,
        n_d: int = 64,
        n_a: int = 64,
        gamma: float = 1.5,
        n_decoder_layers: int = 4,
        virtual_batch_size: int = 128,
        max_target_length: int = 1024,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize the Seq2Seq model.

        Args:
            vocab_size: Vocabulary size (320 for bytes + special tokens)
            max_seq_length: Maximum encoder sequence length
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            n_steps: Number of TabNet decision steps in encoder
            n_d: Decision embedding dimension
            n_a: Attention embedding dimension
            gamma: TabNet relaxation factor
            n_decoder_layers: Number of decoder layers
            virtual_batch_size: Virtual batch size for Ghost BN
            max_target_length: Maximum decoder sequence length
            key: PRNG key
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

        key1, key2, key3 = jrandom.split(key, 3)

        # Encoder uses hidden_dim as output_dim (for classification/features)
        self.encoder = ByteTabNet(
            output_dim=hidden_dim,
            max_seq_length=max_seq_length,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            n_d=n_d,
            n_a=n_a,
            gamma=gamma,
            virtual_batch_size=virtual_batch_size,
            vocab_size=vocab_size,
            key=key1,
        )

        # Separate head for generation: maps n_d features to vocab logits
        self.generation_head = eqx.nn.Linear(n_d, vocab_size, key=key3)

        # Decoder uses max_target_length for its positional embeddings
        self.decoder = ByteTabNetSeq2SeqDecoder(
            vocab_size=vocab_size,
            max_seq_length=max_target_length,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            n_layers=n_decoder_layers,
            virtual_batch_size=virtual_batch_size,
            key=key2,
        )

    def encode(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Array, Optional[dict]]:
        """
        Encode input sequence, returning per-position features for the
        decoder to cross-attend over.

        Args:
            input_ids: Input token IDs (batch, src_len)
            attention_mask: Optional attention mask
            inference: Whether in inference mode
            state: Optional batch norm state

        Returns:
            Tuple of (encoded_features, attention_masks, state)
            encoded_features shape: (batch, src_len, hidden_dim)
        """
        return self.encoder.encode_sequence(
            input_ids,
            attention_mask,
            inference=inference,
            state=state,
        )

    def decode(
        self,
        decoder_input_ids: Array,
        encoder_context: Array,
        encoder_mask: Optional[Array] = None,
        decoder_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Optional[dict]]:
        """
        Decode with teacher forcing.

        Args:
            decoder_input_ids: Decoder input token IDs (batch, tgt_len)
            encoder_context: Encoder output (batch, hidden)
            encoder_mask: Optional encoder attention mask
            decoder_mask: Optional decoder attention mask
            inference: Whether in inference mode
            state: Optional batch norm state

        Returns:
            Tuple of (logits, state)
        """
        return self.decoder(
            decoder_input_ids,
            encoder_context,
            encoder_mask,
            decoder_mask,
            inference,
            state,
        )

    def __call__(
        self,
        input_ids: Array,
        decoder_input_ids: Array,
        attention_mask: Optional[Array] = None,
        decoder_attention_mask: Optional[Array] = None,
        inference: bool = False,
        state: Optional[dict] = None,
    ) -> Tuple[Array, Array, Optional[dict]]:
        """
        Forward pass for training with teacher forcing.

        Args:
            input_ids: Encoder input token IDs (batch, src_len)
            decoder_input_ids: Decoder input token IDs (batch, tgt_len) - should be shifted right
            attention_mask: Optional encoder attention mask
            decoder_attention_mask: Optional decoder attention mask
            inference: Whether in inference mode
            state: Optional batch norm state

        Returns:
            Tuple of (decoder_logits, encoder_masks, state)
        """
        # Encode
        encoder_context, encoder_masks, enc_state = self.encode(
            input_ids,
            attention_mask,
            inference,
            state,
        )

        # Decode
        decoder_logits, dec_state = self.decode(
            decoder_input_ids,
            encoder_context,
            attention_mask,
            decoder_attention_mask,
            inference,
            state,
        )

        # Merge states
        new_state = None
        if enc_state or dec_state:
            new_state = {**(enc_state or {}), **(dec_state or {})}

        return decoder_logits, encoder_masks, new_state

    def generate(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        window_size: int = 128,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
        """
        Generate output sequence autoregressively using the encoder-decoder
        architecture.

        1. Encode the input with the TabNet encoder to get context features.
        2. Start the decoder with a BOS token.
        3. At each step, feed the growing decoder sequence through the decoder
           (conditioned on the encoder context) and sample the next token.

        Args:
            input_ids: Encoder input token IDs (batch, src_len)
            attention_mask: Optional encoder attention mask
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus sampling with this threshold
            do_sample: Whether to sample (True) or greedy (False)
            window_size: Unused, kept for API compat
            key: PRNG key for sampling

        Returns:
            Generated token IDs (batch, gen_len) — decoder output only
        """
        batch_size = input_ids.shape[0]

        # 1. Encode source input
        encoder_context, _, _ = self.encode(
            input_ids, attention_mask, inference=True,
        )  # (batch, hidden_dim)

        # 2. Start decoder with BOS
        generated = jnp.full((batch_size, 1), self.bos_token_id, dtype=jnp.int32)
        finished = jnp.zeros(batch_size, dtype=jnp.bool_)

        if key is None:
            key = jrandom.PRNGKey(0)

        # 3. Autoregressive decoding
        for step in range(max_length):
            # Run decoder on the full generated sequence so far
            decoder_logits, _ = self.decoder(
                generated, encoder_context, inference=True,
            )  # (batch, current_len, vocab_size)

            # Take logits for the last position
            logits = decoder_logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            next_token_logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
                mask = jnp.full_like(next_token_logits, -1e9)
                mask = mask.at[jnp.arange(batch_size)[:, None], top_k_indices].set(top_k_logits)
                next_token_logits = mask

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits = jnp.sort(next_token_logits, axis=-1)[:, ::-1]
                sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
                cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)

                mask = cumsum_probs > top_p
                mask = jnp.concatenate([
                    jnp.zeros((batch_size, 1), dtype=jnp.bool_),
                    mask[:, :-1]
                ], axis=-1)

                sorted_logits = jnp.where(mask, -1e9, sorted_logits)
                next_token_logits = sorted_logits[
                    jnp.arange(batch_size)[:, None],
                    jnp.argsort(jnp.argsort(next_token_logits, axis=-1)[:, ::-1], axis=-1)
                ]

            # Sample next token
            key, subkey = jrandom.split(key)
            next_token = jrandom.categorical(subkey, next_token_logits, axis=-1)

            # Check for EOS
            is_eos = next_token == self.eos_token_id
            finished = finished | is_eos

            # Replace finished sequences' tokens with padding
            next_token = jnp.where(finished, self.pad_token_id, next_token)

            # Append to generated sequence
            generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

            # Stop if all sequences have finished
            if jnp.all(finished):
                break

        return generated


# ============================================================================
# Seq2Seq Loss Functions
# ============================================================================

def seq2seq_cross_entropy_loss(
    logits: Array,
    targets: Array,
    mask: Optional[Array] = None,
    label_smoothing: float = 0.0,
) -> Array:
    """
    Cross-entropy loss for seq2seq models.

    Args:
        logits: Predicted logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        mask: Optional mask for padding (batch, seq_len)
        label_smoothing: Label smoothing factor

    Returns:
        Scalar loss value
    """
    vocab_size = logits.shape[-1]

    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # One-hot encode targets
    one_hot = jax.nn.one_hot(targets, vocab_size)

    # Apply label smoothing
    if label_smoothing > 0:
        one_hot = one_hot * (1 - label_smoothing) + label_smoothing / vocab_size

    # Compute cross-entropy
    ce = -jnp.sum(one_hot * log_probs, axis=-1)  # (batch, seq_len)

    # Apply mask
    if mask is not None:
        ce = ce * mask
        return jnp.sum(ce) / jnp.maximum(jnp.sum(mask), 1)
    else:
        return jnp.mean(ce)


def seq2seq_kl_div_loss(
    logits: Array,
    targets: Array,
    mask: Optional[Array] = None,
    eps: float = 1e-8,
) -> Array:
    """
    KL divergence loss between the generated probability distribution and target distribution.

    Computes KL(target || predicted) where the target distribution is derived from
    one-hot encoded target token IDs and the predicted distribution is the softmax
    of the model logits.

    Args:
        logits: Predicted logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        mask: Optional mask for padding (batch, seq_len)
        eps: Small constant for numerical stability

    Returns:
        Scalar KL divergence loss value
    """
    vocab_size = logits.shape[-1]

    # Target distribution: one-hot from ground truth token IDs
    target_dist = jax.nn.one_hot(targets, vocab_size)

    # Predicted distribution: softmax over logits
    pred_dist = jax.nn.softmax(logits, axis=-1)

    # KL(target || pred) = sum(target * log(target / pred))
    # For one-hot targets this simplifies to -log(pred) at the target index,
    # but we use the general form so it extends to soft/smoothed targets.
    kl = target_dist * (jnp.log(target_dist + eps) - jnp.log(pred_dist + eps))
    kl = jnp.sum(kl, axis=-1)  # (batch, seq_len)

    if mask is not None:
        kl = kl * mask
        return jnp.sum(kl) / jnp.maximum(jnp.sum(mask), 1)
    else:
        return jnp.mean(kl)


def seq2seq_loss_with_tabnet_sparsity(
    logits: Array,
    targets: Array,
    encoder_masks: Array,
    target_mask: Optional[Array] = None,
    sparsity_weight: float = 1e-3,
    label_smoothing: float = 0.0,
    kl_weight: float = 0.0,
) -> Array:
    """
    Combined seq2seq loss with TabNet sparsity regularization and optional KL divergence.

    Args:
        logits: Predicted logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        encoder_masks: TabNet attention masks from encoder
        target_mask: Optional mask for padding
        sparsity_weight: Weight for sparsity loss
        label_smoothing: Label smoothing factor
        kl_weight: Weight for KL divergence loss (0.0 disables it)

    Returns:
        Combined loss value
    """
    ce_loss = seq2seq_cross_entropy_loss(logits, targets, target_mask, label_smoothing)
    sparse_loss = tabnet_sparsity_loss(encoder_masks)
    total_loss = ce_loss + sparsity_weight * sparse_loss

    if kl_weight > 0.0:
        kl_loss = seq2seq_kl_div_loss(logits, targets, target_mask)
        total_loss = total_loss + kl_weight * kl_loss

    return total_loss


# ============================================================================
# Seq2Seq Training Utilities
# ============================================================================

def create_seq2seq_train_step(model: ByteTabNetSeq2Seq, optimizer):
    """Create a JIT-compiled training step for seq2seq."""

    @eqx.filter_jit
    def train_step(
        model,
        opt_state,
        input_ids,
        target_ids,
        input_mask,
        target_mask,
        state,
    ):
        # Shift targets for teacher forcing: input is [BOS, tok1, tok2], target is [tok1, tok2, EOS]
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]
        decoder_mask = target_mask[:, 1:] if target_mask is not None else None

        def loss_fn(model):
            logits, encoder_masks, new_state = model(
                input_ids,
                decoder_input,
                attention_mask=input_mask,
                decoder_attention_mask=decoder_mask,
                inference=False,
                state=state,
            )
            loss = seq2seq_loss_with_tabnet_sparsity(
                logits,
                decoder_target,
                encoder_masks,
                decoder_mask,
            )
            return loss, (logits, encoder_masks, new_state)

        (loss, (logits, masks, new_state)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss, new_state

    return train_step


def prepare_seq2seq_batch(
    source_texts: List[str],
    target_texts: List[str],
    max_src_length: int = 256,
    max_tgt_length: int = 128,
) -> Tuple[Array, Array, Array, Array]:
    """
    Prepare a batch for seq2seq training.

    Args:
        source_texts: List of source strings
        target_texts: List of target strings
        max_src_length: Maximum source sequence length
        max_tgt_length: Maximum target sequence length

    Returns:
        Tuple of (source_ids, target_ids, source_mask, target_mask)
    """
    # Encode sources
    source_ids, source_mask = ByteEmbedding.encode_batch(source_texts, max_src_length)

    # Encode targets with EOS token
    target_encoded = []
    for text in target_texts:
        # Encode: [BOS, bytes..., EOS]
        byte_ids = [b + 64 for b in text.encode("utf-8")]
        token_ids = [1] + byte_ids + [2]  # BOS + bytes + EOS
        if len(token_ids) > max_tgt_length:
            token_ids = token_ids[:max_tgt_length-1] + [2]  # Ensure EOS
        target_encoded.append(jnp.array(token_ids, dtype=jnp.int32))

    # Pad targets
    max_len = max(len(seq) for seq in target_encoded)
    target_ids = []
    target_mask = []
    for seq in target_encoded:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            target_ids.append(jnp.concatenate([seq, jnp.zeros(pad_len, dtype=jnp.int32)]))
            target_mask.append(jnp.concatenate([jnp.ones(len(seq)), jnp.zeros(pad_len)]))
        else:
            target_ids.append(seq)
            target_mask.append(jnp.ones(len(seq)))

    return (
        source_ids,
        jnp.stack(target_ids),
        source_mask,
        jnp.stack(target_mask),
    )


# ============================================================================
# Haliax-based ByteTabNet (Named Tensors)
# ============================================================================

class ByteTabNetHax(eqx.Module):
    """
    ByteTabNet using Haliax named tensors for improved readability.

    This version uses Haliax's named axes for clearer tensor operations.
    """
    embedding: hnn.Embedding
    pos_embedding: Optional[NamedArray]
    attention_weight: Optional[NamedArray]
    input_proj: hnn.Linear
    tabnet: TabNetEncoder
    output_proj: hnn.Linear

    Batch: Axis = eqx.field(static=True)
    SeqLen: Axis = eqx.field(static=True)
    Embed: Axis = eqx.field(static=True)
    Hidden: Axis = eqx.field(static=True)
    Output: Axis = eqx.field(static=True)
    pooling: str = eqx.field(static=True)

    def __init__(
        self,
        Batch: Axis,
        SeqLen: Axis,
        Vocab: Axis,
        Embed: Axis,
        Hidden: Axis,
        Output: Axis,
        n_steps: int = 5,
        n_d: int = 64,
        n_a: int = 64,
        gamma: float = 1.5,
        pooling: str = "attention",
        use_positional: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.Batch = Batch
        self.SeqLen = SeqLen
        self.Embed = Embed
        self.Hidden = Hidden
        self.Output = Output
        self.pooling = pooling

        keys = jrandom.split(key, 5)

        # Embedding layer
        self.embedding = hnn.Embedding(Vocab, Embed, key=keys[0])

        # Positional embedding
        if use_positional:
            self.pos_embedding = hax.random.normal(keys[1], (SeqLen, Embed)) * 0.02
        else:
            self.pos_embedding = None

        # Attention for pooling
        if pooling == "attention":
            One = Axis("one", 1)
            self.attention_weight = hax.random.normal(keys[2], (Embed, One)) * 0.02
        else:
            self.attention_weight = None

        # Input projection
        self.input_proj = hnn.Linear(Embed, Hidden, key=keys[3])

        # TabNet encoder (using regular arrays internally)
        self.tabnet = TabNetEncoder(
            input_dim=Hidden.size,
            output_dim=n_d,
            n_steps=n_steps,
            n_d=n_d,
            n_a=n_a,
            gamma=gamma,
            key=keys[4],
        )

        # Output projection
        NDim = Axis("n_d", n_d)
        key5 = jrandom.split(keys[4])[0]
        self.output_proj = hnn.Linear(NDim, Output, key=key5)

    def __call__(
        self,
        input_ids: NamedArray,
        attention_mask: Optional[NamedArray] = None,
        inference: bool = False,
    ) -> Tuple[NamedArray, Array]:
        """Forward pass with named tensors."""
        # Embed
        embeddings = self.embedding(input_ids)

        # Add positional
        if self.pos_embedding is not None:
            seq_len = input_ids.axis_size(self.SeqLen)
            pos_emb = self.pos_embedding.slice(self.SeqLen, 0, seq_len)
            embeddings = embeddings + pos_emb

        # Pool (simplified for named arrays)
        if self.pooling == "mean":
            pooled = hax.mean(embeddings, self.SeqLen)
        elif self.pooling == "max":
            pooled = hax.max(embeddings, self.SeqLen)
        else:
            # Attention pooling
            scores = hax.dot(embeddings, self.attention_weight, axis=self.Embed)
            weights = hax.nn.softmax(scores, axis=self.SeqLen)
            pooled = hax.sum(embeddings * weights, self.SeqLen)

        # Project
        hidden = self.input_proj(pooled)

        # TabNet (convert to raw arrays)
        hidden_raw = hidden.array
        if hidden_raw.ndim == 1:
            hidden_raw = hidden_raw[None, :]

        encoded, masks, _ = self.tabnet(hidden_raw, inference)

        # Output
        NDim = Axis("n_d", encoded.shape[-1])
        encoded_named = hax.named(encoded.squeeze(0) if encoded.shape[0] == 1 else encoded, (NDim,))
        logits = self.output_proj(encoded_named)

        return logits, masks


# ============================================================================
# Loss Functions
# ============================================================================

def sparse_cross_entropy(logits: Array, targets: Array) -> Array:
    """Cross-entropy loss with sparse (integer) targets."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=-1))


def tabnet_sparsity_loss(masks: Array, eps: float = 1e-10) -> Array:
    """
    Sparsity regularization loss for TabNet attention masks.

    Encourages sparse feature selection by penalizing the entropy of masks.
    """
    # masks: (batch, n_steps, n_features)
    # Average entropy across masks
    mask_mean = jnp.mean(masks, axis=(0, 1))  # Average mask
    entropy = -jnp.sum(mask_mean * jnp.log(mask_mean + eps))
    return entropy


def tabnet_loss(
    logits: Array,
    targets: Array,
    masks: Array,
    sparsity_weight: float = 1e-3,
) -> Array:
    """
    Combined TabNet loss with classification and sparsity regularization.
    """
    ce_loss = sparse_cross_entropy(logits, targets)
    sparse_loss = tabnet_sparsity_loss(masks)
    return ce_loss + sparsity_weight * sparse_loss


# ============================================================================
# Training Utilities
# ============================================================================

def create_train_step(model: ByteTabNet, optimizer):
    """Create a JIT-compiled training step function."""

    @eqx.filter_jit
    def train_step(model, opt_state, input_ids, attention_mask, targets, state):
        def loss_fn(model):
            logits, masks, new_state = model(input_ids, attention_mask, inference=False, state=state)
            loss = tabnet_loss(logits, targets, masks)
            return loss, (logits, masks, new_state)

        (loss, (logits, masks, new_state)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss, new_state

    return train_step


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use ByteTabNet."""
    import optax

    # Initialize model
    key = jrandom.PRNGKey(42)
    model = ByteTabNet(
        output_dim=10,  # 10 classes
        max_seq_length=256,
        embed_dim=64,
        hidden_dim=128,
        n_steps=5,
        n_d=64,
        n_a=64,
        key=key,
    )

    # Encode some example strings (like the PyTorch example)
    texts = ["Hello, world!", "This is a test.", "TabNet with bytes!"]
    input_ids, attention_mask = ByteEmbedding.encode_batch(texts, max_length=64)

    print(f"Input shape: {input_ids.shape}")
    print(f"First sequence tokens: {input_ids[0, :20]}")

    # Forward pass
    logits, masks, _ = model(input_ids, attention_mask, inference=True)

    print(f"Output logits shape: {logits.shape}")
    print(f"Attention masks shape: {masks.shape}")

    # Get predictions
    predictions = model.predict(input_ids, attention_mask)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")

    # Training setup
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Dummy targets
    targets = jnp.array([0, 1, 2])

    # Training step
    train_step = create_train_step(model, optimizer)
    model, opt_state, loss, _ = train_step(model, opt_state, input_ids, attention_mask, targets, None)

    print(f"Training loss: {loss}")

    return model


if __name__ == "__main__":
    example_usage()
