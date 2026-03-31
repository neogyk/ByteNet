"""
Tokenizer abstraction for ByteTabNet.

Supports two tokenization modes:
- **byte**: Original byte-level tokenization (vocab=320, bytes offset by 64)
- **bpe**: Byte Pair Encoding via tiktoken (pre-trained) or HuggingFace tokenizers (custom-trained)

Usage:
    # Byte-level (default)
    tok = ByteTokenizer()

    # Pre-trained BPE (tiktoken)
    tok = BPETokenizer.from_pretrained("cl100k_base")

    # Train custom BPE on corpus
    tok = BPETokenizer.train(texts, vocab_size=8192, save_path="my_bpe")

    # Load custom BPE from disk
    tok = BPETokenizer.from_file("my_bpe")

    # Common interface
    ids = tok.encode("hello")          # List[int]
    text = tok.decode(ids)             # str
    batch_ids, mask = tok.encode_batch(["hello", "world"], max_length=64)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array


# ============================================================================
# Special token IDs (shared across all tokenizers)
# ============================================================================
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3  # Reserved for masked diffusion
NUM_SPECIAL = 4  # Number of reserved special token slots


# ============================================================================
# Abstract base
# ============================================================================
class Tokenizer(ABC):
    """Abstract tokenizer interface for ByteTabNet."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        ...

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def mask_id(self) -> int:
        return MASK_ID

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (without BOS/EOS)."""
        ...

    @abstractmethod
    def decode(self, token_ids) -> str:
        """Decode token IDs back to text (strips special tokens)."""
        ...

    def encode_with_special(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text with optional BOS/EOS and truncation."""
        ids = self.encode(text)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos and ids[-1] != EOS_ID:
                ids[-1] = EOS_ID
        return ids

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> Tuple[Array, Array]:
        """Encode a batch of strings with padding.

        Returns:
            (token_ids, attention_mask) as jnp arrays of shape (batch, seq_len).
        """
        encoded = [
            self.encode_with_special(t, add_bos=add_bos, add_eos=add_eos,
                                     max_length=max_length)
            for t in texts
        ]

        if max_length is not None:
            pad_to = max_length
        else:
            pad_to = max(len(seq) for seq in encoded)

        padded = []
        masks = []
        for seq in encoded:
            pad_len = pad_to - len(seq)
            if pad_len > 0:
                padded.append(seq + [PAD_ID] * pad_len)
                masks.append([1.0] * len(seq) + [0.0] * pad_len)
            else:
                padded.append(seq[:pad_to])
                masks.append([1.0] * pad_to)

        return (
            jnp.array(padded, dtype=jnp.int32),
            jnp.array(masks, dtype=jnp.float32),
        )


# ============================================================================
# Byte-level tokenizer (original ByteTabNet scheme)
# ============================================================================
BYTE_OFFSET = 64

class ByteTokenizer(Tokenizer):
    """Original byte-level tokenizer: vocab=320, byte values offset by 64.

    Token layout:
        0       = PAD
        1       = BOS
        2       = EOS
        3       = MASK (for MDM)
        4-63    = reserved
        64-319  = byte values 0-255
    """

    @property
    def vocab_size(self) -> int:
        return 320

    def encode(self, text: str) -> List[int]:
        return [b + BYTE_OFFSET for b in text.encode("utf-8")]

    def decode(self, token_ids) -> str:
        ids = [int(t) for t in token_ids]
        # Strip BOS prefix
        if ids and ids[0] == BOS_ID:
            ids = ids[1:]
        # Truncate at EOS
        try:
            eos_idx = ids.index(EOS_ID)
            ids = ids[:eos_idx]
        except ValueError:
            pass
        raw_bytes = bytes(
            t - BYTE_OFFSET for t in ids
            if BYTE_OFFSET <= t < BYTE_OFFSET + 256
        )
        return raw_bytes.decode("utf-8", errors="replace")


# ============================================================================
# BPE tokenizer (tiktoken pre-trained or custom-trained via HF tokenizers)
# ============================================================================
class BPETokenizer(Tokenizer):
    """BPE tokenizer backed by tiktoken (pre-trained) or HuggingFace tokenizers (custom).

    Special tokens are mapped to IDs 0-3. All BPE token IDs are shifted by
    NUM_SPECIAL (4) so they don't collide with special tokens.
    """

    def __init__(self, backend, backend_type: str, _vocab_size: int):
        self._backend = backend
        self._backend_type = backend_type  # "tiktoken" or "hf"
        self._vocab_size = _vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ------------------------------------------------------------------
    # Factory: pre-trained tiktoken encoding
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, encoding_name: str = "cl100k_base") -> "BPETokenizer":
        """Load a pre-trained tiktoken encoding.

        Args:
            encoding_name: One of "gpt2", "r50k_base", "p50k_base",
                           "cl100k_base", "o200k_base".
        """
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        total_vocab = enc.n_vocab + NUM_SPECIAL
        return cls(backend=enc, backend_type="tiktoken", _vocab_size=total_vocab)

    # ------------------------------------------------------------------
    # Factory: train custom BPE with HuggingFace tokenizers
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 8192,
        save_path: Optional[str] = None,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a byte-level BPE tokenizer on a corpus.

        Args:
            texts: Training corpus (list of strings).
            vocab_size: Desired BPE vocabulary size (excluding special tokens).
            save_path: If provided, save the trained tokenizer to this directory.
            min_frequency: Minimum frequency for a merge to be kept.
        """
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        import os

        hf_tok = HFTokenizer(BPE(unk_token="<UNK>"))
        hf_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        hf_tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"],
        )
        hf_tok.train_from_iterator(texts, trainer=trainer)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            hf_tok.save(os.path.join(save_path, "tokenizer.json"))

        total_vocab = hf_tok.get_vocab_size() + NUM_SPECIAL
        return cls(backend=hf_tok, backend_type="hf", _vocab_size=total_vocab)

    # ------------------------------------------------------------------
    # Factory: load custom BPE from disk
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        """Load a custom-trained BPE tokenizer from disk.

        Args:
            path: Directory containing tokenizer.json (from train()).
        """
        from tokenizers import Tokenizer as HFTokenizer
        import os

        tok_path = path if path.endswith(".json") else os.path.join(path, "tokenizer.json")
        hf_tok = HFTokenizer.from_file(tok_path)
        total_vocab = hf_tok.get_vocab_size() + NUM_SPECIAL
        return cls(backend=hf_tok, backend_type="hf", _vocab_size=total_vocab)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        if self._backend_type == "tiktoken":
            raw_ids = self._backend.encode(text, allowed_special=set())
        else:
            raw_ids = self._backend.encode(text).ids
        # Shift all BPE IDs by NUM_SPECIAL to reserve 0-3 for special tokens
        return [tid + NUM_SPECIAL for tid in raw_ids]

    def decode(self, token_ids) -> str:
        ids = [int(t) for t in token_ids]
        # Strip BOS prefix
        if ids and ids[0] == BOS_ID:
            ids = ids[1:]
        # Truncate at EOS
        try:
            eos_idx = ids.index(EOS_ID)
            ids = ids[:eos_idx]
        except ValueError:
            pass
        # Remove special tokens and shift back
        raw_ids = [t - NUM_SPECIAL for t in ids if t >= NUM_SPECIAL]
        if self._backend_type == "tiktoken":
            return self._backend.decode(raw_ids)
        else:
            return self._backend.decode(raw_ids)


# ============================================================================
# Factory helper
# ============================================================================
def create_tokenizer(
    tokenizer_type: str = "byte",
    bpe_encoding: str = "cl100k_base",
    bpe_tokenizer_path: Optional[str] = None,
    train_bpe: bool = False,
    bpe_vocab_size: int = 8192,
    train_texts: Optional[List[str]] = None,
) -> Tokenizer:
    """Create a tokenizer from configuration.

    Args:
        tokenizer_type: "byte" or "bpe".
        bpe_encoding: tiktoken encoding name (used when bpe_tokenizer_path is None).
        bpe_tokenizer_path: Path to a custom-trained BPE tokenizer directory.
        train_bpe: If True, train a new BPE tokenizer on train_texts.
        bpe_vocab_size: Vocabulary size for training custom BPE.
        train_texts: Corpus for training BPE (required if train_bpe=True).

    Returns:
        A Tokenizer instance.
    """
    if tokenizer_type == "byte":
        return ByteTokenizer()

    if tokenizer_type != "bpe":
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type!r}. Use 'byte' or 'bpe'.")

    # BPE mode
    if train_bpe:
        if train_texts is None or len(train_texts) == 0:
            raise ValueError("train_bpe=True requires non-empty train_texts.")
        tok = BPETokenizer.train(
            texts=train_texts,
            vocab_size=bpe_vocab_size,
            save_path=bpe_tokenizer_path,
        )
        return tok

    if bpe_tokenizer_path is not None:
        return BPETokenizer.from_file(bpe_tokenizer_path)

    return BPETokenizer.from_pretrained(bpe_encoding)
