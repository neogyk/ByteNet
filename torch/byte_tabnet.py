"""
ByteTabNet: TabNet implementation using PyTorch.

This module is a complete PyTorch port of the JAX/Equinox ByteTabNet model.
It implements TabNet (Attentive Interpretable Tabular Learning) for byte-level
encoded string inputs.

Components:
- sparsemax activation
- entmax15 activation (1.5-entmax, sparse alternative to sparsemax)
- GhostBatchNorm
- GLUBlock (SwiGLU)
- FeatureTransformer
- AttentiveTransformer
- ByteEmbedding
- PerceiverEncoder (replaces sparsemax pooling; Jaegle et al., 2021)
- TabNetEncoder
- ByteTabNet (main model)
- ByteTabNetSeq2SeqDecoder
- ByteTabNetSeq2Seq
- Loss functions (sparse_cross_entropy, tabnet_sparsity_loss, tabnet_loss,
  seq2seq_cross_entropy_loss, seq2seq_kl_div_loss, seq2seq_loss_with_tabnet_sparsity)
- prepare_seq2seq_batch utility
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional xformers import for memory-efficient / sparse attention
try:
    from xformers.ops import memory_efficient_attention, LowerTriangularMask
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


# ============================================================================
# Vocabulary Constants
# ============================================================================

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MASK_TOKEN_ID = 3  # for Masked Diffusion Model
BYTE_OFFSET = 64
VOCAB_SIZE = 320  # 256 bytes + 64 special/reserved tokens


# ============================================================================
# Sparsemax Activation
# ============================================================================

def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation function.

    Sparsemax is a sparse alternative to softmax that can output exact zeros.
    From: "From Softmax to Sparsemax: A Sparse Model of Attention and
    Multi-Label Classification"

    Args:
        x: Input tensor
        dim: Dimension along which to compute sparsemax

    Returns:
        Sparsemax probabilities (sparse probability distribution)
    """
    # Sort input in descending order
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)

    # Compute cumulative sums
    cumsum = torch.cumsum(x_sorted, dim=dim)

    # Find the support (number of non-zero elements)
    k = torch.arange(1, x.shape[dim] + 1, dtype=x.dtype, device=x.device)

    # Reshape k for broadcasting
    shape = [1] * x.ndim
    shape[dim] = x.shape[dim]
    k = k.reshape(shape)

    # Check condition: 1 + k * z_k > cumsum
    support = (1 + k * x_sorted > cumsum).to(x.dtype)

    # Find the threshold
    k_z = torch.sum(support, dim=dim, keepdim=True)

    # Compute tau (threshold)
    tau_sum = torch.sum(support * x_sorted, dim=dim, keepdim=True)
    tau = (tau_sum - 1) / torch.clamp(k_z, min=1)

    # Compute sparsemax output
    output = torch.clamp(x - tau, min=0)

    return output


def entmax15(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    1.5-entmax activation function.

    A sparse alternative to softmax from the entmax family (alpha=1.5).
    Sits between softmax (alpha=1, dense) and sparsemax (alpha=2, very sparse),
    producing moderately sparse outputs that are differentiable everywhere.

    From: "Sparse Sequence-to-Sequence Models" (Peters et al., 2019)

    Uses the closed-form bisection algorithm for alpha=1.5.

    Args:
        x: Input tensor
        dim: Dimension along which to compute entmax15

    Returns:
        Entmax15 probabilities (sparse probability distribution)
    """
    # Shift for numerical stability
    x = x - x.max(dim=dim, keepdim=True).values

    # For alpha=1.5, the mapping is p* = ReLU(z - tau)^(alpha-1)^{-1} = ReLU(z - tau)^2
    # We need to find tau such that sum(ReLU(z - tau)^2) = 1

    # Bisection to find the threshold tau
    x_shape = x.shape
    n = x_shape[dim]

    # Sort in descending order
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)

    # Compute cumulative sum of squared terms via bisection
    # For alpha=1.5: p_i = max(z_i - tau, 0)^2, sum(p_i) = 1
    # Use the analytical approach: iterate over possible support sizes

    # Cumulative sums for the sorted values
    cumsum = torch.cumsum(x_sorted, dim=dim)
    cumsum_sq = torch.cumsum(x_sorted ** 2, dim=dim)

    # For support size k, tau = (cumsum_k - 1/sqrt(k * cumsum_sq_k - cumsum_k^2 + k)) ...
    # Actually, use the Newton/bisection approach for the 1.5 case.

    # Closed-form for alpha=1.5:
    # The threshold tau satisfies: sum_i max(x_i - tau, 0)^2 = 1
    # For the top-k elements: sum_{i=1}^{k} (x_sorted_i - tau)^2 = 1
    # Expanding: k*tau^2 - 2*tau*S_k + Q_k = 1
    # where S_k = sum of top-k x values, Q_k = sum of top-k x^2 values
    # tau = (S_k - sqrt(k * (1 - Q_k) + S_k^2)) / k  ... using the quadratic formula

    k = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    shape = [1] * x.ndim
    shape[dim] = n
    k = k.reshape(shape)

    # S_k and Q_k
    s_k = cumsum
    q_k = cumsum_sq

    # Discriminant: k + k*s_k^2 - k*q_k  (from quadratic: k*tau^2 - 2*s_k*tau + (q_k - 1) = 0)
    # disc = 4*s_k^2 - 4*k*(q_k - 1) = 4*(s_k^2 - k*q_k + k)
    disc = s_k ** 2 - k * (q_k - 1)

    # tau = (s_k - sqrt(disc)) / k  (we want the smaller root so that x_k - tau >= 0)
    tau_candidate = (s_k - torch.sqrt(torch.clamp(disc, min=0))) / k

    # Check which support size is valid: x_sorted_k >= tau
    support = (x_sorted >= tau_candidate).to(x.dtype)

    # Find the largest valid k
    # The tau for the correct k is the last one where support holds
    k_max = torch.sum(support, dim=dim, keepdim=True)

    # Gather the correct tau
    # Recompute tau for the correct k
    k_max_clamped = torch.clamp(k_max, min=1)
    s_final = torch.gather(cumsum, dim, (k_max_clamped - 1).long())
    q_final = torch.gather(cumsum_sq, dim, (k_max_clamped - 1).long())
    disc_final = s_final ** 2 - k_max_clamped * (q_final - 1)
    tau = (s_final - torch.sqrt(torch.clamp(disc_final, min=0))) / k_max_clamped

    # Output: p_i = max(x_i - tau, 0)^2
    output = torch.clamp(x - tau, min=0) ** 2

    return output


# ============================================================================
# Ghost Batch Normalization
# ============================================================================

class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization for TabNet.

    Applies batch normalization on virtual "ghost" batches to reduce
    the effect of large batch sizes and improve generalization.

    During training, the input is split into chunks of virtual_batch_size and
    each chunk is normalized independently. During eval, standard BN with
    running statistics is used.

    Args:
        feature_dim: Number of features to normalize
        virtual_batch_size: Size of ghost batches (default: 128)
        momentum: Momentum for running statistics (default: 0.01)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        feature_dim: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.01,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.virtual_batch_size = virtual_batch_size

        self.bn = nn.BatchNorm1d(
            feature_dim,
            eps=eps,
            momentum=momentum,
            affine=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Ghost Batch Normalization.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Normalized tensor of shape (batch_size, feature_dim)
        """
        if not self.training:
            return self.bn(x)

        batch_size = x.shape[0]

        # BatchNorm1d requires batch_size > 1 during training;
        # fall back to running stats when we get a single sample.
        if batch_size <= 1:
            return F.batch_norm(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                False,  # training=False → use running stats
                0.0,
                self.bn.eps,
            )

        if batch_size <= self.virtual_batch_size:
            return self.bn(x)

        # Split into ghost batches
        num_chunks = math.ceil(batch_size / self.virtual_batch_size)

        # Pad if necessary so we can reshape evenly
        pad_size = num_chunks * self.virtual_batch_size - batch_size
        if pad_size > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_size))
        else:
            x_padded = x

        # Reshape into ghost batches and process each
        chunks = x_padded.reshape(num_chunks, self.virtual_batch_size, self.feature_dim)
        normalized_chunks = []
        for i in range(num_chunks):
            normalized_chunks.append(self.bn(chunks[i]))

        # Recombine and remove padding
        output = torch.cat(normalized_chunks, dim=0)[:batch_size]

        return output


# ============================================================================
# GLU (Gated Linear Unit) Block
# ============================================================================

class GLUBlock(nn.Module):
    """
    SwiGLU block used in TabNet's feature transformer.

    Applies: FC -> BN -> SwiGLU activation
    where SwiGLU(x, gate) = x * silu(gate).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        virtual_batch_size: int = 128,
    ):
        super().__init__()
        # Double output for GLU gating
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        self.bn = GhostBatchNorm(output_dim * 2, virtual_batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GLU block.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = self.fc(x)
        x = self.bn(x)

        # SwiGLU activation: split and apply gating
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)

        return x


# ============================================================================
# Feature Transformer
# ============================================================================

class FeatureTransformer(nn.Module):
    """
    Feature Transformer block for TabNet.

    Consists of shared layers (across all decision steps) and
    step-specific layers. Uses GLU blocks with skip connections.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_shared: int = 2,
        n_step: int = 2,
        virtual_batch_size: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_shared = n_shared
        self.n_step = n_step

        # Shared layers
        self.shared_layers = nn.ModuleList()
        current_dim = input_dim
        for _ in range(n_shared):
            self.shared_layers.append(
                GLUBlock(current_dim, output_dim, virtual_batch_size)
            )
            current_dim = output_dim

        # Step-specific layers
        self.step_layers = nn.ModuleList()
        for _ in range(n_step):
            self.step_layers.append(
                GLUBlock(current_dim, output_dim, virtual_batch_size)
            )
            current_dim = output_dim

        # FloatFunctional for QAT-compatible skip connections
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()
        self.skip_mul = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature transformer.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Shared layers with skip connections
        for i, layer in enumerate(self.shared_layers):
            out = layer(x)
            # Skip connection (except first layer if dims differ)
            if i > 0 or self.input_dim == self.output_dim:
                x = self.skip_mul.mul_scalar(
                    self.skip_add.add(x, out), math.sqrt(0.5)
                )
            else:
                x = out

        # Step-specific layers with skip connections
        for layer in self.step_layers:
            out = layer(x)
            x = self.skip_mul.mul_scalar(
                self.skip_add.add(x, out), math.sqrt(0.5)
            )

        return x


# ============================================================================
# Attentive Transformer
# ============================================================================

class AttentiveTransformer(nn.Module):
    """
    Attentive Transformer for TabNet.

    Computes attention masks for feature selection at each decision step.
    Supports multiple sparse activation functions:
    - "sparsemax": Sparsemax (alpha=2 entmax), very sparse, from Martins & Astudillo 2016
    - "entmax15": 1.5-entmax, moderately sparse and smooth, from Peters et al. 2019
    - "softmax": Standard softmax (dense, non-sparse baseline)
    - "topk": Top-k sparse softmax inspired by DeepSeek's top-k gating.
        Keeps only the k highest-scoring features, zeros the rest, and
        renormalizes via softmax over the survivors. Provides explicit
        control over sparsity level through the ``topk`` parameter.
    - "retnet": RetNet-style retention scoring with a learnable exponential
        decay. Treats the feature dimension as a pseudo-sequence and applies
        causal retention: ret[i] = sum_{j<=i} gamma^(i-j) * score[j], then
        normalizes via softmax. gamma is constrained to (0, 1) via a
        sigmoid-transformed parameter. Inspired by "Retentive Network: A
        Successor to Transformer for Large Language Models" (Sun et al., 2023).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        virtual_batch_size: int = 128,
        mask_type: str = "retnet",
        topk: int = 8,
        retnet_gamma: float = 0.9,
    ):
        """
        Args:
            input_dim: Input feature dimension (from attention part of feature transformer)
            output_dim: Output dimension (number of input features to mask)
            virtual_batch_size: Virtual batch size for Ghost BN
            mask_type: Activation for the mask
                ("sparsemax", "entmax15", "softmax", "topk", "retnet")
            topk: Number of features to keep when mask_type="topk". Clamped to
                output_dim at forward time so it is safe to set a value larger
                than the actual feature count.
            retnet_gamma: Initial decay factor for RetNet retention (0 < gamma < 1).
                Higher values allow longer-range feature interactions. Only used
                when mask_type="retnet". Stored as a learnable sigmoid-logit so
                gamma stays in (0, 1) throughout training.
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size)

        if mask_type not in ("sparsemax", "entmax15", "softmax", "topk", "retnet"):
            raise ValueError(
                f"Unknown mask_type '{mask_type}'. "
                "Choose from: 'sparsemax', 'entmax15', 'softmax', 'topk', 'retnet'."
            )
        self.mask_type = mask_type
        self.topk = topk

        if mask_type == "retnet":
            # Store gamma as a logit so sigmoid always keeps it in (0, 1).
            # Initialize logit so that sigmoid(logit) ≈ retnet_gamma.
            init_logit = math.log(retnet_gamma / (1.0 - retnet_gamma))
            self.retnet_gamma_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(
        self,
        x: torch.Tensor,
        prior_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention mask.

        Args:
            x: Processed features from feature transformer (batch_size, input_dim)
            prior_scales: Aggregated attention from previous steps (batch_size, output_dim)

        Returns:
            Attention mask (batch_size, output_dim)
        """
        x = self.fc(x)
        x = self.bn(x)

        # Multiply by prior scales to encourage diversity
        x = x * prior_scales

        # Apply selected sparse activation
        if self.mask_type == "sparsemax":
            mask = sparsemax(x, dim=-1)
        elif self.mask_type == "entmax15":
            mask = entmax15(x, dim=-1)
        elif self.mask_type == "topk":
            mask = self._topk_softmax(x)
        elif self.mask_type == "retnet":
            mask = self._retnet_mask(x)
        else:
            mask = F.softmax(x, dim=-1)

        return mask

    def _topk_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Top-k sparse softmax inspired by DeepSeek's top-k gating mechanism.

        Retains only the k largest logits per sample, masks the rest to -inf,
        then applies softmax so the surviving entries form a valid probability
        distribution while all others are exactly zero.

        Args:
            x: Raw scores (batch_size, n_features)

        Returns:
            Sparse attention mask (batch_size, n_features)
        """
        k = min(self.topk, x.shape[-1])
        top_vals, top_idx = torch.topk(x, k, dim=-1)
        # Fill everything with -inf, then scatter the top-k values back
        masked = torch.full_like(x, float("-inf"))
        masked.scatter_(-1, top_idx, top_vals)
        return F.softmax(masked, dim=-1)

    def _retnet_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        RetNet-inspired retention scoring for feature selection.

        Treats the feature dimension as a pseudo-sequence and computes causal
        retention scores using the parallel form of the retention operator:

            D[i, j] = gamma^(i - j)  for i >= j,  else 0
            ret[b, i] = sum_{j=0}^{i} D[i, j] * x[b, j]

        i.e. each feature position i is a weighted sum of all preceding
        positions (including itself) with exponentially decaying weights.
        D is row-normalised before application so that the scale of ret is
        independent of sequence length. The resulting scores are then passed
        through softmax to produce a valid probability distribution.

        gamma is constrained to (0, 1) via a sigmoid-transformed learnable
        parameter so it stays valid throughout training.

        Reference: "Retentive Network: A Successor to Transformer for Large
        Language Models" (Sun et al., 2023).

        Args:
            x: Raw scores after FC + BN + prior scaling (batch_size, n_features)

        Returns:
            Retention-normalised attention mask (batch_size, n_features)
        """
        n = x.shape[-1]
        gamma = torch.sigmoid(self.retnet_gamma_logit)  # scalar in (0, 1)

        # Build causal decay matrix D of shape (n, n).
        # D[i, j] = gamma^(i - j) for i >= j, else 0.
        i_idx = torch.arange(n, device=x.device, dtype=x.dtype)
        j_idx = torch.arange(n, device=x.device, dtype=x.dtype)
        dist = i_idx.unsqueeze(1) - j_idx.unsqueeze(0)  # (n, n), dist[i,j] = i-j
        D = torch.where(dist >= 0, gamma ** dist, torch.zeros_like(dist))  # (n, n)

        # Row-normalise: divide each row by its L1 norm so that the weighted
        # sum does not grow with the number of features in the support.
        D = D / (D.sum(dim=-1, keepdim=True) + 1e-8)  # (n, n)

        # Apply retention: ret[b, i] = sum_j D[i, j] * x[b, j]
        # Equivalent to x @ D.T  (D is already (n, n), rows = output positions)
        ret = x @ D.t()  # (batch, n)

        # Normalise to a probability distribution over features.
        return F.softmax(ret, dim=-1)


# ============================================================================
# Byte Embedding Layer
# ============================================================================

class ByteEmbedding(nn.Module):
    """
    Byte-level embedding layer for string inputs.

    Converts byte sequences to embeddings. The vocabulary includes:
    - 0: Padding token (PAD)
    - 1: BOS (beginning of sequence) token
    - 2: EOS (end of sequence) token
    - 3: MASK token (for Masked Diffusion Model)
    - 4-63: Reserved special tokens
    - 64-319: Byte values (0-255) offset by 64

    Encoding: token_id = byte_value + 64
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialize with small random values (Normal(0, 0.02))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed byte sequences.

        Args:
            input_ids: Integer tensor of shape (batch_size, seq_len)
                       Values should be in range [0, vocab_size)

        Returns:
            Embedded tensor of shape (batch_size, seq_len, embed_dim)
        """
        return self.embedding(input_ids)

    @staticmethod
    def encode_string(
        text: str, max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode a string to byte-level token IDs.

        Args:
            text: Input string
            max_length: Optional maximum sequence length

        Returns:
            Tensor of token IDs with BOS prefix: [BOS, byte_id_0, byte_id_1, ...]
        """
        # Encode bytes with offset of 64
        byte_ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]

        # Add BOS token at the start
        token_ids = [BOS_TOKEN_ID] + byte_ids

        # Truncate if necessary
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return torch.tensor(token_ids, dtype=torch.long)

    @staticmethod
    def encode_batch(
        texts: List[str],
        max_length: Optional[int] = None,
        pad_to_max: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of strings.

        Args:
            texts: List of input strings
            max_length: Maximum sequence length
            pad_to_max: Whether to pad all sequences to max_length

        Returns:
            Tuple of (token_ids, attention_mask)
            - token_ids: (batch_size, padded_len) int64 tensor
            - attention_mask: (batch_size, padded_len) float32 tensor
        """
        encoded = [ByteEmbedding.encode_string(t, max_length) for t in texts]

        if pad_to_max and max_length is not None:
            target_len = max_length
        else:
            target_len = max(len(seq) for seq in encoded)

        padded = []
        masks = []
        for seq in encoded:
            seq_len = len(seq)
            pad_len = target_len - seq_len
            if pad_len > 0:
                padded.append(
                    torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
                )
                masks.append(
                    torch.cat(
                        [torch.ones(seq_len), torch.zeros(pad_len)]
                    )
                )
            else:
                padded.append(seq[:target_len])
                masks.append(torch.ones(target_len))

        return torch.stack(padded), torch.stack(masks)


class ByteUnembedding(nn.Module):
    """
    Inverse of ByteEmbedding: projects hidden states to vocabulary logits.

    Maps hidden representations of shape (..., hidden_dim) to logits of shape
    (..., vocab_size) suitable for cross-entropy loss or token sampling.

    Supports optional weight tying with a ByteEmbedding instance, which
    shares the embedding matrix and transposes it as the output projection
    (a common technique for language models that reduces parameters and
    often improves perplexity).

    Args:
        hidden_dim: Dimension of the input hidden states.
        vocab_size:  Number of output tokens (default: VOCAB_SIZE = 320).
        bias:        Whether to add a learnable bias to the projection.
        tie_weights: If True, ``set_embedding`` must be called to provide
                     the ByteEmbedding whose weight matrix will be reused.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = VOCAB_SIZE,
        bias: bool = True,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.tie_weights = tie_weights

        if tie_weights:
            # Will be set via set_embedding(); no owned parameter.
            self._tied_weight: Optional[nn.Parameter] = None
            self.bias = nn.Parameter(torch.zeros(vocab_size)) if bias else None
        else:
            self.projection = nn.Linear(hidden_dim, vocab_size, bias=bias)
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def set_embedding(self, embedding: "ByteEmbedding") -> None:
        """Tie this unembedding to the given ByteEmbedding weight matrix."""
        if not self.tie_weights:
            raise ValueError("set_embedding() called but tie_weights=False")
        self._tied_weight = embedding.embedding.weight  # (vocab_size, embed_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            hidden: Float tensor of shape (..., hidden_dim).

        Returns:
            Logits of shape (..., vocab_size).
        """
        if self.tie_weights:
            if self._tied_weight is None:
                raise RuntimeError(
                    "Tied-weight ByteUnembedding: call set_embedding() first."
                )
            # F.linear computes  hidden @ weight.T + bias
            return F.linear(hidden, self._tied_weight, self.bias)
        return self.projection(hidden)


# ============================================================================
# Perceiver Encoder
# ============================================================================

class _PerceiverSwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block used inside Perceiver layers."""

    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()
        hidden = int(dim * ff_mult)
        self.fc   = nn.Linear(dim, hidden * 2, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc(x).chunk(2, dim=-1)
        return self.proj(x * F.silu(gate))


class PerceiverCrossAttentionBlock(nn.Module):
    """
    Cross-attention block: latents (Q) attend over byte-embedding inputs (K, V).

    Pre-norm on both sides; residual on latents only.
    Followed by a SwiGLU feed-forward with residual.
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim:  int,
        n_heads:    int,
        ff_mult:    int   = 4,
        dropout:    float = 0.0,
    ):
        super().__init__()
        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = latent_dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm_latents = nn.LayerNorm(latent_dim)
        self.norm_inputs  = nn.LayerNorm(input_dim)

        self.q_proj   = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_proj   = nn.Linear(input_dim,  latent_dim, bias=False)
        self.v_proj   = nn.Linear(input_dim,  latent_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)

        self.norm_ffn = nn.LayerNorm(latent_dim)
        self.ffn      = _PerceiverSwiGLUFFN(latent_dim, ff_mult)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        latents:    torch.Tensor,
        inputs:     torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents:    (batch, n_latents, latent_dim)
            inputs:     (batch, seq_len,   input_dim)
            input_mask: (batch, seq_len) — 1 = real token, 0 = padding

        Returns:
            latents:    (batch, n_latents, latent_dim)
        """
        B, N, _ = latents.shape
        S        = inputs.shape[1]
        H, D     = self.n_heads, self.head_dim

        q = self.q_proj(self.norm_latents(latents))  # (B, N, latent_dim)
        k = self.k_proj(self.norm_inputs(inputs))    # (B, S, latent_dim)
        v = self.v_proj(inputs)                      # (B, S, latent_dim)

        # Reshape to multi-head
        q = q.view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        v = v.view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, S)

        if input_mask is not None:
            scores = scores.masked_fill(
                input_mask[:, None, None, :] == 0, -1e9
            )

        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)               # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, H * D)
        out = self.out_proj(out)

        latents = latents + out
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


class PerceiverSelfAttentionBlock(nn.Module):
    """
    Self-attention block among latents with SwiGLU feed-forward.

    Pre-norm; both attention and FFN use residual connections.
    """

    def __init__(
        self,
        latent_dim: int,
        n_heads:    int,
        ff_mult:    int   = 4,
        dropout:    float = 0.0,
    ):
        super().__init__()
        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = latent_dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm_attn = nn.LayerNorm(latent_dim)
        self.q_proj    = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_proj    = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_proj    = nn.Linear(latent_dim, latent_dim, bias=False)
        self.out_proj  = nn.Linear(latent_dim, latent_dim, bias=False)

        self.norm_ffn = nn.LayerNorm(latent_dim)
        self.ffn      = _PerceiverSwiGLUFFN(latent_dim, ff_mult)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (batch, n_latents, latent_dim)

        Returns:
            latents: (batch, n_latents, latent_dim)
        """
        B, N, _ = latents.shape
        H, D     = self.n_heads, self.head_dim

        h = self.norm_attn(latents)
        q = self.q_proj(h).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(h).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(h).view(B, N, H, D).transpose(1, 2)

        scores  = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = self.dropout(F.softmax(scores, dim=-1))
        out     = torch.matmul(weights, v)
        out     = out.transpose(1, 2).contiguous().view(B, N, H * D)
        out     = self.out_proj(out)

        latents = latents + out
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


class PerceiverEncoder(nn.Module):
    """
    Perceiver Encoder: maps variable-length byte sequences to a fixed-size
    latent array via iterative cross-attention (Jaegle et al., 2021).

    Architecture per layer:
        1. Cross-attention: n_latents learned queries attend over all input
           positions  →  compresses arbitrary-length sequence to n_latents.
        2. Self-attention: latents refine representations among themselves.

    Replaces the sparsemax-pooling → single-vector bottleneck with a richer
    (batch, n_latents, latent_dim) output, preserving more sequence structure
    for TabNet and downstream decoders.

    Args:
        input_dim:  Embedding dimension of the byte input (embed_dim).
        latent_dim: Dimension of each latent vector.
        n_latents:  Number of latent tokens (fixed output sequence length).
        n_layers:   Number of cross-attn + self-attn iterations.
        n_heads:    Number of attention heads (must divide latent_dim).
        ff_mult:    Feed-forward hidden multiplier (default 4).
        dropout:    Dropout probability applied inside attention.
    """

    def __init__(
        self,
        input_dim:  int,
        latent_dim: int,
        n_latents:  int   = 64,
        n_layers:   int   = 3,
        n_heads:    int   = 4,
        ff_mult:    int   = 4,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.n_latents  = n_latents
        self.latent_dim = latent_dim

        # Learned latent array — initialised like embeddings
        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.02)

        self.cross_attn_blocks = nn.ModuleList([
            PerceiverCrossAttentionBlock(latent_dim, input_dim, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.self_attn_blocks = nn.ModuleList([
            PerceiverSelfAttentionBlock(latent_dim, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        inputs:     torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs:     Byte embeddings  (batch, seq_len, input_dim)
            input_mask: Padding mask     (batch, seq_len); 1=real, 0=pad

        Returns:
            Latent representations       (batch, n_latents, latent_dim)
        """
        B       = inputs.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        for cross_blk, self_blk in zip(self.cross_attn_blocks, self.self_attn_blocks):
            latents = cross_blk(latents, inputs, input_mask)
            latents = self_blk(latents)

        return self.norm(latents)  # (B, n_latents, latent_dim)


# ============================================================================
# TabNet Encoder
# ============================================================================

class TabNetEncoder(nn.Module):
    """
    TabNet Encoder module.

    Implements the core TabNet architecture with:
    - Initial batch normalization
    - Feature transformer (shared + step-specific layers)
    - Attentive transformer for feature selection
    - Multiple decision steps with sparse attention

    Args:
        input_dim: Input feature dimension
        n_steps: Number of decision steps
        n_d: Decision embedding dimension
        n_a: Attention embedding dimension
        gamma: Relaxation factor for feature reuse (higher = more reuse)
        n_shared: Number of shared layers in feature transformer
        n_step: Number of step-specific layers in feature transformer
        virtual_batch_size: Virtual batch size for Ghost BN
    """

    def __init__(
        self,
        input_dim: int,
        n_steps: int = 5,
        n_d: int = 64,
        n_a: int = 64,
        gamma: float = 1.5,
        n_shared: int = 2,
        n_step: int = 2,
        virtual_batch_size: int = 128,
        mask_type: str = "sparsemax",
        topk: int = 8,
        retnet_gamma: float = 0.9,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma

        # Initial processing
        self.initial_bn = GhostBatchNorm(input_dim, virtual_batch_size)
        self.initial_fc = nn.Linear(input_dim, n_d + n_a, bias=False)

        # Feature transformers (one per step)
        self.feature_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.feature_transformers.append(
                FeatureTransformer(
                    input_dim=input_dim,
                    output_dim=n_d + n_a,
                    n_shared=n_shared,
                    n_step=n_step,
                    virtual_batch_size=virtual_batch_size,
                )
            )

        # Attention transformers (one per step, except last)
        self.attention_transformers = nn.ModuleList()
        for _ in range(n_steps - 1):
            self.attention_transformers.append(
                AttentiveTransformer(
                    input_dim=n_a,
                    output_dim=input_dim,
                    virtual_batch_size=virtual_batch_size,
                    mask_type=mask_type,
                    topk=topk,
                    retnet_gamma=retnet_gamma,
                )
            )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through TabNet encoder.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Tuple of:
            - Output embeddings (batch_size, n_d)
            - Feature importance masks (batch_size, n_steps-1, input_dim)
        """
        batch_size = x.shape[0]

        # Initial batch normalization
        x_bn = self.initial_bn(x)

        # Initial transformation for first step
        processed = self.initial_fc(x_bn)

        # Initialize prior scales (for attention regularization)
        prior_scales = torch.ones(
            batch_size, self.input_dim, device=x.device, dtype=x.dtype
        )

        # Aggregate outputs and masks
        output_aggregate = torch.zeros(
            batch_size, self.n_d, device=x.device, dtype=x.dtype
        )
        masks = []

        for step in range(self.n_steps):
            # Split into decision and attention parts
            decision_out = processed[:, : self.n_d]
            attention_out = processed[:, self.n_d :]

            # Accumulate decision outputs with SiLU
            output_aggregate = output_aggregate + F.silu(decision_out)

            # Compute attention mask (except for last step)
            if step < self.n_steps - 1:
                mask = self.attention_transformers[step](
                    attention_out, prior_scales
                )
                masks.append(mask)

                # Update prior scales (decrease weights of previously attended features)
                prior_scales = prior_scales * (self.gamma - mask)

                # Mask the input and process
                masked_x = mask * x_bn
                processed = self.feature_transformers[step](masked_x)

        # Stack masks
        if masks:
            masks = torch.stack(masks, dim=1)  # (batch, n_steps-1, input_dim)
        else:
            masks = torch.zeros(
                batch_size, 0, self.input_dim, device=x.device, dtype=x.dtype
            )

        return output_aggregate, masks


# ============================================================================
# ByteTabNet: Main Model
# ============================================================================

class ByteTabNet(nn.Module):
    """
    ByteTabNet: TabNet for byte-level string inputs.

    Combines:
    - Byte-level embedding for string inputs
    - Optional sinusoidal positional encoding
    - PerceiverEncoder: maps the variable-length sequence to a fixed set of
      latent vectors via iterative cross-attention + self-attention
    - Mean-pooling over latents → input projection → TabNet encoder
    - Output projection head

    Suitable for:
    - Text classification
    - Sequence labeling
    - Tabular data with string features
    """

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
        vocab_size: int = VOCAB_SIZE,
        mask_type: str = "retnet",
        topk: int = 8,
        retnet_gamma: float = 0.9,
        # Perceiver hyper-parameters
        n_latents: int = 64,
        n_perceiver_layers: int = 3,
        n_perceiver_heads: int = 4,
        perceiver_ff_mult: int = 4,
        perceiver_dropout: float = 0.0,
    ):
        """
        Initialize ByteTabNet.

        Args:
            output_dim:          Output dimension (num classes for classification)
            max_seq_length:      Maximum sequence length
            embed_dim:           Byte embedding dimension
            hidden_dim:          Latent/hidden dimension (shared by Perceiver and TabNet)
            n_steps:             Number of TabNet decision steps
            n_d:                 Decision embedding dimension
            n_a:                 Attention embedding dimension
            gamma:               Relaxation factor for attention
            n_shared:            Number of shared layers in FeatureTransformer
            n_step:              Number of step-specific layers in FeatureTransformer
            virtual_batch_size:  Virtual batch size for Ghost BN
            use_positional:      Whether to add positional embeddings before Perceiver
            vocab_size:          Vocabulary size for byte embedding
            mask_type:           Sparse attention type for TabNet masks
            topk:                Top-k features when mask_type="topk"
            retnet_gamma:        RetNet decay when mask_type="retnet"
            n_latents:           Number of Perceiver latent tokens
            n_perceiver_layers:  Cross+self-attention iterations in Perceiver
            n_perceiver_heads:   Attention heads in Perceiver (must divide hidden_dim)
            perceiver_ff_mult:   Feed-forward multiplier inside Perceiver
            perceiver_dropout:   Dropout probability inside Perceiver attention
        """
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim      = embed_dim
        self.hidden_dim     = hidden_dim
        self.output_dim     = output_dim
        self.use_positional = use_positional
        self.n_d            = n_d
        self.n_latents      = n_latents

        # Byte embedding
        self.byte_embedding = ByteEmbedding(vocab_size, embed_dim)

        # Positional embedding (added to byte embeddings before Perceiver)
        if use_positional:
            self.position_embedding = nn.Parameter(
                torch.randn(max_seq_length, embed_dim) * 0.02
            )
        else:
            self.position_embedding = None

        # Perceiver encoder: seq → (n_latents, hidden_dim)
        self.perceiver = PerceiverEncoder(
            input_dim  = embed_dim,
            latent_dim = hidden_dim,
            n_latents  = n_latents,
            n_layers   = n_perceiver_layers,
            n_heads    = n_perceiver_heads,
            ff_mult    = perceiver_ff_mult,
            dropout    = perceiver_dropout,
        )

        # TabNet encoder operates on a single vector: mean-pool latents → hidden_dim
        self.tabnet_encoder = TabNetEncoder(
            input_dim=hidden_dim,
            n_steps=n_steps,
            n_d=n_d,
            n_a=n_a,
            gamma=gamma,
            n_shared=n_shared,
            n_step=n_step,
            virtual_batch_size=virtual_batch_size,
            mask_type=mask_type,
            topk=topk,
            retnet_gamma=retnet_gamma,
        )

        # Output head
        self.output_head = nn.Linear(n_d, output_dim)

    def _embed(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Byte embedding + optional positional encoding."""
        embeddings = self.byte_embedding(input_ids)           # (B, S, embed_dim)
        if self.use_positional and self.position_embedding is not None:
            embeddings = embeddings + self.position_embedding[:input_ids.shape[1]]
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ByteTabNet.

        Args:
            input_ids:      Token IDs       (batch, seq_len)
            attention_mask: Padding mask    (batch, seq_len); 1=real, 0=pad

        Returns:
            logits: (batch, output_dim)
            masks:  TabNet feature-importance masks
        """
        embeddings = self._embed(input_ids)                   # (B, S, embed_dim)

        # Perceiver: variable-length → fixed latents
        latents = self.perceiver(embeddings, attention_mask)  # (B, n_latents, hidden_dim)

        # Pool latents to a single vector for TabNet
        hidden = latents.mean(dim=1)                          # (B, hidden_dim)

        encoded, masks = self.tabnet_encoder(hidden)
        logits = self.output_head(encoded)
        return logits, masks

    def encode_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return raw TabNet encoded features (n_d dim) before the output head.

        Returns:
            encoded: (batch, n_d)
            masks:   TabNet attention masks
        """
        embeddings      = self._embed(input_ids)
        latents         = self.perceiver(embeddings, attention_mask)
        hidden          = latents.mean(dim=1)
        encoded, masks  = self.tabnet_encoder(hidden)
        return encoded, masks

    def encode_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the full latent sequence from the Perceiver for cross-attention.

        Unlike the old per-position projection, this returns the Perceiver's
        compressed latent array — a fixed-size sequence regardless of input
        length — which is naturally suited for decoder cross-attention.

        Returns:
            latents: (batch, n_latents, hidden_dim)
            masks:   TabNet attention masks (from mean-pooled path)
        """
        embeddings     = self._embed(input_ids)
        latents        = self.perceiver(embeddings, attention_mask)
        hidden         = latents.mean(dim=1)
        _, masks       = self.tabnet_encoder(hidden)
        return latents, masks

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Make predictions (convenience method for inference)."""
        self.eval()
        logits, _ = self(input_ids, attention_mask)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def get_feature_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get aggregated feature importance from attention masks."""
        self.eval()
        _, masks = self(input_ids, attention_mask)
        # Average across decision steps
        return torch.mean(masks, dim=1)


# ============================================================================
# Autoregressive Decoder for Seq2Seq
# ============================================================================

class ByteTabNetSeq2SeqDecoder(nn.Module):
    """
    Autoregressive decoder for seq2seq generation.

    Generates output tokens one at a time, conditioning on:
    - Previously generated tokens
    - Encoder context

    Uses GLU layers with causal self-attention and cross-attention.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_length: int = 512,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        context_dim: int = 64,
        n_layers: int = 4,
        virtual_batch_size: int = 128,
        attention_type: str = "default",
    ):
        """
        Initialize the decoder.

        Args:
            vocab_size: Output vocabulary size (320 for bytes + special tokens)
            max_seq_length: Maximum sequence length for generation
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            context_dim: Encoder context dimension (n_d from TabNet or hidden_dim)
            n_layers: Number of decoder layers
            virtual_batch_size: Virtual batch size for Ghost BN
            attention_type: "default" for manual attention or "xformers" for
                memory-efficient sparse attention from xformers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attention_type = attention_type

        if attention_type == "xformers" and not XFORMERS_AVAILABLE:
            raise ImportError(
                "xformers is required for attention_type='xformers'. "
                "Install it with: pip install xformers"
            )

        # Byte embedding (separate from encoder)
        self.byte_embedding = ByteEmbedding(vocab_size, embed_dim)

        # Positional embedding
        self.position_embedding = nn.Parameter(
            torch.randn(max_seq_length, embed_dim) * 0.02
        )

        # Context projection (from encoder context_dim to decoder hidden)
        self.context_projection = nn.Linear(context_dim, hidden_dim)

        # Decoder GLU layers
        self.decoder_layers = nn.ModuleList()
        current_dim = embed_dim + hidden_dim  # Concatenate token embedding with context
        for _ in range(n_layers):
            self.decoder_layers.append(
                GLUBlock(current_dim, hidden_dim, virtual_batch_size)
            )
            current_dim = hidden_dim

        # Cross-attention components
        self.cross_attention_fc = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def _causal_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Causal self-attention mechanism.

        Uses xformers memory-efficient attention when attention_type="xformers",
        otherwise falls back to the manual implementation.

        Args:
            query: Query tensor (batch, seq_len, hidden)
            key_value: Key/Value tensor (batch, seq_len, hidden)
            mask: Optional attention mask

        Returns:
            Attended output (batch, seq_len, hidden)
        """
        if self.attention_type == "xformers":
            return self._xformers_causal_attention(query, key_value, mask)

        # Manual implementation
        scores = torch.einsum("bqh,bkh->bqk", query, key_value)
        scores = scores / math.sqrt(query.shape[-1])

        seq_len = query.shape[1]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=query.device)
        )
        scores = scores.masked_fill(causal_mask == 0, -1e9)

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :] == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bqk,bkh->bqh", weights, key_value)

        return output

    def _xformers_causal_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Causal self-attention using xformers memory_efficient_attention.

        Args:
            query: (batch, seq_len, hidden)
            key_value: (batch, seq_len, hidden)
            mask: Optional padding mask (batch, seq_len)

        Returns:
            (batch, seq_len, hidden)
        """
        # xformers expects (B, M, H, K) — use H=1 for single-head
        q = query.unsqueeze(2)      # (B, M, 1, K)
        k = key_value.unsqueeze(2)  # (B, M, 1, K)
        v = key_value.unsqueeze(2)  # (B, M, 1, K)

        attn_bias = LowerTriangularMask()

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        return out.squeeze(2)  # (B, M, K)

    def _cross_attention(
        self,
        decoder_hidden: torch.Tensor,
        encoder_context: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention between decoder and encoder.

        Uses xformers memory-efficient attention when attention_type="xformers",
        otherwise falls back to the manual implementation.

        Args:
            decoder_hidden: Decoder hidden states (batch, dec_len, hidden)
            encoder_context: Encoder outputs (batch, enc_len, hidden)
            encoder_mask: Optional encoder attention mask

        Returns:
            Context-enhanced decoder hidden (batch, dec_len, hidden)
        """
        query = self.cross_attention_fc(decoder_hidden)

        if self.attention_type == "xformers":
            context = self._xformers_cross_attention(
                query, encoder_context, encoder_mask
            )
        else:
            # Manual implementation
            scores = torch.einsum("bqh,bkh->bqk", query, encoder_context)
            scores = scores / math.sqrt(decoder_hidden.shape[-1])

            if encoder_mask is not None:
                scores = scores.masked_fill(encoder_mask[:, None, :] == 0, -1e9)

            weights = F.softmax(scores, dim=-1)
            context = torch.einsum("bqk,bkh->bqh", weights, encoder_context)

        combined = torch.cat([decoder_hidden, context], dim=-1)
        output = self.cross_attention_proj(combined)

        return output

    def _xformers_cross_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention using xformers memory_efficient_attention.

        Args:
            query: (batch, dec_len, hidden)
            key_value: (batch, enc_len, hidden)
            mask: Optional encoder padding mask (batch, enc_len)

        Returns:
            (batch, dec_len, hidden)
        """
        q = query.unsqueeze(2)      # (B, M, 1, K)
        k = key_value.unsqueeze(2)  # (B, N, 1, K)
        v = key_value.unsqueeze(2)  # (B, N, 1, K)

        # No causal mask for cross-attention (decoder attends to all encoder positions)
        out = memory_efficient_attention(q, k, v)
        return out.squeeze(2)  # (B, M, K)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_context: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            input_ids: Decoder input token IDs (batch, dec_len) - shifted right during training
            encoder_context: Encoder hidden states (batch, enc_len, hidden) or (batch, hidden)
            encoder_mask: Optional encoder attention mask
            decoder_mask: Optional decoder attention mask

        Returns:
            Logits of shape (batch, dec_len, vocab_size)
        """
        batch_size, dec_len = input_ids.shape

        # Embed decoder inputs
        embeddings = self.byte_embedding(input_ids)  # (batch, dec_len, embed)

        # Add positional embeddings
        pos_emb = self.position_embedding[:dec_len]
        embeddings = embeddings + pos_emb

        # Handle encoder context shape
        if encoder_context.ndim == 2:
            # Context is pooled: (batch, hidden) -> expand for cross-attention
            encoder_context = encoder_context.unsqueeze(1)  # (batch, 1, hidden)

        # Project context
        context_proj = self.context_projection(encoder_context)

        # Broadcast context to each decoder position or prepare for cross-attention
        if context_proj.shape[1] == 1:
            context_broadcast = context_proj.expand(
                batch_size, dec_len, self.hidden_dim
            )
            use_cross_attention = False
        else:
            # Use cross-attention if encoder has sequence output
            context_broadcast = None
            use_cross_attention = True

        # Concatenate embeddings with context for initial input
        if context_broadcast is not None:
            hidden = torch.cat([embeddings, context_broadcast], dim=-1)
        else:
            # Will use cross-attention instead; pad with zeros initially
            hidden = torch.cat(
                [
                    embeddings,
                    torch.zeros(
                        batch_size, dec_len, self.hidden_dim,
                        device=input_ids.device, dtype=embeddings.dtype,
                    ),
                ],
                dim=-1,
            )

        # Process through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            # Reshape for GLUBlock: (batch * dec_len, dim)
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            hidden_flat = layer(hidden_flat)
            hidden = hidden_flat.reshape(batch_size, dec_len, -1)

            # Apply cross-attention at middle layer if encoder has sequence
            if use_cross_attention and i == len(self.decoder_layers) // 2:
                hidden = self._cross_attention(
                    hidden, context_proj, encoder_mask
                )

        # Apply causal self-attention
        hidden = self._causal_attention(hidden, hidden, decoder_mask)

        # Project to vocabulary
        logits = self.output_projection(hidden)

        return logits


# ============================================================================
# ByteTabNet Seq2Seq Model
# ============================================================================

class ByteTabNetSeq2Seq(nn.Module):
    """
    ByteTabNet Sequence-to-Sequence model.

    Combines a TabNet-based encoder with an autoregressive decoder for tasks
    like text generation, translation, summarization, or any sequence-to-sequence
    task with byte-level tokenization.

    The encoder uses TabNet's attention mechanism for feature selection,
    while the decoder generates output tokens autoregressively.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
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
        attention_type: str = "default",
        mask_type: str = "sparsemax",
        topk: int = 8,
        retnet_gamma: float = 0.9,
        # Perceiver params forwarded to ByteTabNet encoder
        n_latents: int = 64,
        n_perceiver_layers: int = 3,
        n_perceiver_heads: int = 4,
        perceiver_ff_mult: int = 4,
        perceiver_dropout: float = 0.0,
    ):
        """
        Initialize the Seq2Seq model.

        Args:
            vocab_size:          Vocabulary size (320 for bytes + special tokens)
            max_seq_length:      Maximum encoder sequence length
            embed_dim:           Embedding dimension
            hidden_dim:          Hidden/latent dimension (shared by Perceiver and TabNet)
            n_steps:             Number of TabNet decision steps in encoder
            n_d:                 Decision embedding dimension
            n_a:                 Attention embedding dimension
            gamma:               TabNet relaxation factor
            n_decoder_layers:    Number of decoder layers
            virtual_batch_size:  Virtual batch size for Ghost BN
            max_target_length:   Maximum decoder sequence length
            attention_type:      "default" or "xformers" for decoder attention
            mask_type:           Sparse attention type for TabNet masks
            topk:                Top-k features when mask_type="topk"
            retnet_gamma:        RetNet decay when mask_type="retnet"
            n_latents:           Perceiver latent tokens (encoder output seq length)
            n_perceiver_layers:  Perceiver cross+self-attention iterations
            n_perceiver_heads:   Perceiver attention heads
            perceiver_ff_mult:   Perceiver FFN multiplier
            perceiver_dropout:   Perceiver attention dropout
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.bos_token_id = BOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self.n_d = n_d

        # Encoder: ByteTabNet with PerceiverEncoder backbone
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
            mask_type=mask_type,
            topk=topk,
            retnet_gamma=retnet_gamma,
            n_latents=n_latents,
            n_perceiver_layers=n_perceiver_layers,
            n_perceiver_heads=n_perceiver_heads,
            perceiver_ff_mult=perceiver_ff_mult,
            perceiver_dropout=perceiver_dropout,
        )

        # Separate head for generation: maps n_d features to vocab logits
        # (kept for compatibility, unused in new code)
        self.generation_head = nn.Linear(n_d, vocab_size)

        # Decoder uses max_target_length for its positional embeddings
        self.decoder = ByteTabNetSeq2SeqDecoder(
            vocab_size=vocab_size,
            max_seq_length=max_target_length,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            n_layers=n_decoder_layers,
            virtual_batch_size=virtual_batch_size,
            attention_type=attention_type,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence, returning per-position features for the
        decoder to cross-attend over.

        Args:
            input_ids: Input token IDs (batch, src_len)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (encoded_features, attention_masks)
            - encoded_features: (batch, src_len, hidden_dim)
            - attention_masks: TabNet masks
        """
        return self.encoder.encode_sequence(input_ids, attention_mask)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_context: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode with teacher forcing.

        Args:
            decoder_input_ids: Decoder input token IDs (batch, tgt_len)
            encoder_context: Encoder output (batch, src_len, hidden) or (batch, hidden)
            encoder_mask: Optional encoder attention mask
            decoder_mask: Optional decoder attention mask

        Returns:
            Logits of shape (batch, tgt_len, vocab_size)
        """
        return self.decoder(
            decoder_input_ids,
            encoder_context,
            encoder_mask,
            decoder_mask,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training with teacher forcing.

        Args:
            input_ids: Encoder input token IDs (batch, src_len)
            decoder_input_ids: Decoder input token IDs (batch, tgt_len) - should be shifted right
            attention_mask: Optional encoder attention mask
            decoder_attention_mask: Optional decoder attention mask

        Returns:
            Tuple of (decoder_logits, encoder_masks)
            - decoder_logits: (batch, tgt_len, vocab_size)
            - encoder_masks: TabNet attention masks
        """
        # Encode
        encoder_context, encoder_masks = self.encode(
            input_ids, attention_mask
        )

        # Decode
        decoder_logits = self.decode(
            decoder_input_ids,
            encoder_context,
            attention_mask,
            decoder_attention_mask,
        )

        return decoder_logits, encoder_masks

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
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

        Returns:
            Generated token IDs (batch, gen_len) -- decoder output only
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. Encode source input
        encoder_context, _ = self.encode(input_ids, attention_mask)

        # 2. Start decoder with BOS
        generated = torch.full(
            (batch_size, 1), self.bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 3. Autoregressive decoding
        for step in range(max_length):
            # Run decoder on the full generated sequence so far
            decoder_logits = self.decoder(
                generated, encoder_context
            )  # (batch, current_len, vocab_size)

            # Take logits for the last position
            logits = decoder_logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            next_token_logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, top_k, dim=-1
                )
                fill_mask = torch.full_like(next_token_logits, -1e9)
                fill_mask.scatter_(1, top_k_indices, top_k_logits)
                next_token_logits = fill_mask

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, dim=-1, descending=True
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_mask = cumsum_probs > top_p
                # Shift so that first token above threshold is also kept
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                sorted_logits[sorted_mask] = -1e9
                # Unsort
                next_token_logits = sorted_logits.scatter(
                    1, sorted_indices.argsort(dim=-1), sorted_logits
                )

            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Check for EOS
            is_eos = next_token == self.eos_token_id
            finished = finished | is_eos

            # Replace finished sequences' tokens with padding
            next_token = torch.where(
                finished,
                torch.tensor(self.pad_token_id, device=device),
                next_token,
            )

            # Append to generated sequence
            generated = torch.cat(
                [generated, next_token.unsqueeze(1)], dim=1
            )

            # Stop if all sequences have finished
            if finished.all():
                break

        return generated


# ============================================================================
# Loss Functions
# ============================================================================

def sparse_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Cross-entropy loss with sparse (integer) targets.

    Args:
        logits: Predicted logits (batch_size, num_classes)
        targets: Target class indices (batch_size,)

    Returns:
        Scalar loss value
    """
    return F.cross_entropy(logits, targets)


def tabnet_sparsity_loss(
    masks: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Sparsity regularization loss for TabNet attention masks.

    Encourages sparse feature selection by penalizing the entropy
    of the average mask across the batch and decision steps.

    Args:
        masks: Attention masks (batch, n_steps, n_features)
        eps: Small constant for numerical stability

    Returns:
        Scalar sparsity loss
    """
    # Average mask across batch and steps
    mask_mean = torch.mean(masks, dim=(0, 1))
    entropy = -torch.sum(mask_mean * torch.log(mask_mean + eps))
    return entropy


def tabnet_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    sparsity_weight: float = 1e-3,
) -> torch.Tensor:
    """
    Combined TabNet loss with classification and sparsity regularization.

    Args:
        logits: Predicted logits (batch_size, num_classes)
        targets: Target class indices (batch_size,)
        masks: TabNet attention masks
        sparsity_weight: Weight for sparsity regularization

    Returns:
        Combined loss value
    """
    ce_loss = sparse_cross_entropy(logits, targets)
    sparse_loss = tabnet_sparsity_loss(masks)
    return ce_loss + sparsity_weight * sparse_loss


def seq2seq_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
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
    log_probs = F.log_softmax(logits, dim=-1)

    # One-hot encode targets
    one_hot = F.one_hot(targets.long(), num_classes=vocab_size).float()

    # Apply label smoothing
    if label_smoothing > 0:
        one_hot = one_hot * (1 - label_smoothing) + label_smoothing / vocab_size

    # Compute cross-entropy
    ce = -torch.sum(one_hot * log_probs, dim=-1)  # (batch, seq_len)

    # Apply mask
    if mask is not None:
        ce = ce * mask
        return torch.sum(ce) / torch.clamp(torch.sum(mask), min=1)
    else:
        return torch.mean(ce)


def seq2seq_kl_div_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL divergence loss between the generated probability distribution and
    the target distribution.

    Computes KL(target || predicted) where the target distribution is derived
    from one-hot encoded target token IDs and the predicted distribution is
    the softmax of the model logits.

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
    target_dist = F.one_hot(targets.long(), num_classes=vocab_size).float()

    # Predicted distribution: softmax over logits
    pred_dist = F.softmax(logits, dim=-1)

    # KL(target || pred) = sum(target * log(target / pred))
    kl = target_dist * (torch.log(target_dist + eps) - torch.log(pred_dist + eps))
    kl = torch.sum(kl, dim=-1)  # (batch, seq_len)

    if mask is not None:
        kl = kl * mask
        return torch.sum(kl) / torch.clamp(torch.sum(mask), min=1)
    else:
        return torch.mean(kl)


def seq2seq_reverse_kl_div_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Reverse KL divergence loss: KL(predicted || target).

    Unlike the forward KL (KL(target || pred)) which is mean-seeking and
    tries to cover all modes of the target, the reverse KL is mode-seeking:
    the model concentrates probability mass on the highest-confidence modes
    of the target distribution.  This encourages sharper, more confident
    predictions which is desirable for autoregressive generation.

    Args:
        logits: Predicted logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        mask: Optional mask for padding (batch, seq_len)
        eps: Small constant for numerical stability

    Returns:
        Scalar reverse KL divergence loss value
    """
    vocab_size = logits.shape[-1]

    # Target distribution: one-hot from ground truth token IDs
    target_dist = F.one_hot(targets.long(), num_classes=vocab_size).float()

    # Predicted distribution: softmax over logits
    pred_dist = F.softmax(logits, dim=-1)

    # KL(pred || target) = sum(pred * log(pred / target))
    # Only terms where pred > 0 contribute; target=0 terms are handled by eps
    rkl = pred_dist * (torch.log(pred_dist + eps) - torch.log(target_dist + eps))
    rkl = torch.sum(rkl, dim=-1)  # (batch, seq_len)

    if mask is not None:
        rkl = rkl * mask
        return torch.sum(rkl) / torch.clamp(torch.sum(mask), min=1)
    else:
        return torch.mean(rkl)


def seq2seq_loss_with_tabnet_sparsity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    encoder_masks: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    sparsity_weight: float = 1e-3,
    label_smoothing: float = 0.0,
    kl_weight: float = 0.0,
    reverse_kl_weight: float = 0.0,
) -> torch.Tensor:
    """
    Combined seq2seq loss with TabNet sparsity regularization and optional
    KL divergence terms.

    Args:
        logits: Predicted logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        encoder_masks: TabNet attention masks from encoder
        target_mask: Optional mask for padding
        sparsity_weight: Weight for sparsity loss
        label_smoothing: Label smoothing factor
        kl_weight: Weight for forward KL divergence loss (0.0 disables it)
        reverse_kl_weight: Weight for reverse KL divergence loss (0.0 disables it)

    Returns:
        Combined loss value
    """
    ce_loss = seq2seq_cross_entropy_loss(
        logits, targets, target_mask, label_smoothing
    )
    sparse_loss = tabnet_sparsity_loss(encoder_masks)
    total_loss = ce_loss + sparsity_weight * sparse_loss

    if kl_weight > 0.0:
        kl_loss = seq2seq_kl_div_loss(logits, targets, target_mask)
        total_loss = total_loss + kl_weight * kl_loss

    if reverse_kl_weight > 0.0:
        rkl_loss = seq2seq_reverse_kl_div_loss(logits, targets, target_mask)
        total_loss = total_loss + reverse_kl_weight * rkl_loss

    return total_loss


# ============================================================================
# Seq2Seq Training Utilities
# ============================================================================

def prepare_seq2seq_batch(
    source_texts: List[str],
    target_texts: List[str],
    max_src_length: int = 256,
    max_tgt_length: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch for seq2seq training.

    Args:
        source_texts: List of source strings
        target_texts: List of target strings
        max_src_length: Maximum source sequence length
        max_tgt_length: Maximum target sequence length

    Returns:
        Tuple of (source_ids, target_ids, source_mask, target_mask)
        All tensors have dtype long/float and are not on any particular device.
    """
    # Encode sources
    source_ids, source_mask = ByteEmbedding.encode_batch(
        source_texts, max_src_length
    )

    # Encode targets with EOS token
    target_encoded = []
    for text in target_texts:
        # Encode: [BOS, bytes..., EOS]
        byte_ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]
        token_ids = [BOS_TOKEN_ID] + byte_ids + [EOS_TOKEN_ID]
        if len(token_ids) > max_tgt_length:
            token_ids = token_ids[: max_tgt_length - 1] + [EOS_TOKEN_ID]
        target_encoded.append(torch.tensor(token_ids, dtype=torch.long))

    # Pad targets
    max_len = max(len(seq) for seq in target_encoded)
    target_ids_list = []
    target_mask_list = []
    for seq in target_encoded:
        seq_len = len(seq)
        pad_len = max_len - seq_len
        if pad_len > 0:
            target_ids_list.append(
                torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
            )
            target_mask_list.append(
                torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
            )
        else:
            target_ids_list.append(seq)
            target_mask_list.append(torch.ones(seq_len))

    return (
        source_ids,
        torch.stack(target_ids_list),
        source_mask,
        torch.stack(target_mask_list),
    )


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use ByteTabNet in PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # 1. Classification with ByteTabNet
    # -----------------------------------------------------------------
    print("=" * 60)
    print("ByteTabNet Classification Example")
    print("=" * 60)

    model = ByteTabNet(
        output_dim=10,  # 10 classes
        max_seq_length=256,
        embed_dim=64,
        hidden_dim=256,
        n_steps=5,
        n_d=256,
        n_a=256,
    ).to(device)

    # Encode some example strings
    texts = ["Hello, world!", "This is a test.", "TabNet with bytes!"]
    input_ids, attention_mask = ByteEmbedding.encode_batch(texts, max_length=64)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    print(f"Input shape: {input_ids.shape}")
    print(f"First sequence tokens: {input_ids[0, :20]}")

    # Forward pass (training mode)
    model.train()
    logits, masks = model(input_ids, attention_mask)
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention masks shape: {masks.shape}")

    # Predictions (eval mode)
    predictions = model.predict(input_ids, attention_mask)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")

    # Training step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    targets = torch.tensor([0, 1, 2], device=device)

    model.train()
    logits, masks = model(input_ids, attention_mask)
    loss = tabnet_loss(logits, targets, masks)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Training loss: {loss.item()}")

    # -----------------------------------------------------------------
    # 2. Seq2Seq with ByteTabNetSeq2Seq
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("ByteTabNetSeq2Seq Example")
    print("=" * 60)

    seq2seq_model = ByteTabNetSeq2Seq(
        vocab_size=VOCAB_SIZE,
        max_seq_length=128,
        embed_dim=64,
        hidden_dim=128,
        n_steps=3,
        n_d=64,
        n_a=64,
        n_decoder_layers=4,
        max_target_length=128,
    ).to(device)

    # Prepare batch
    source_texts = ["What is 2+2?", "Hello"]
    target_texts = ["4", "Hi there!"]
    print(source_texts, target_texts)
    src_ids, tgt_ids, src_mask, tgt_mask = prepare_seq2seq_batch(
        source_texts, target_texts, max_src_length=64, max_tgt_length=32
    )
    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    print(f"Source IDs shape: {src_ids.shape}")
    print(f"Target IDs shape: {tgt_ids.shape}")

    # Training forward pass (teacher forcing)
    seq2seq_model.train()
    decoder_input = tgt_ids[:, :-1]
    decoder_target = tgt_ids[:, 1:]
    decoder_mask = tgt_mask[:, 1:]

    decoder_logits, encoder_masks = seq2seq_model(
        src_ids, decoder_input, src_mask, decoder_mask
    )
    print(f"Decoder logits shape: {decoder_logits.shape}")

    loss = seq2seq_loss_with_tabnet_sparsity(
        decoder_logits, decoder_target, encoder_masks, decoder_mask
    )
    print(f"Seq2Seq training loss: {loss.item()}")

    # Generation
    generated = seq2seq_model.generate(
        src_ids, src_mask, max_length=20, temperature=0.1, top_k=50
    )
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    # Decode generated tokens back to text
    tokens = generated[0].tolist()
    byte_values = [t - BYTE_OFFSET for t in tokens if BYTE_OFFSET <= t < VOCAB_SIZE]
    decoded_text = bytes(b for b in byte_values if 0 <= b < 256).decode(
        "utf-8", errors="replace"
    )
    print(f"Decoded text: {decoded_text}")

    return model, seq2seq_model


if __name__ == "__main__":
    example_usage()
