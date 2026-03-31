"""
ByteTabNet Evaluation Script

Evaluate a trained ByteTabNet model on the test split of a HuggingFace dataset.

Supports:
- Teacher-forcing metrics: loss, perplexity, cross-entropy
- Generation metrics: BLEU score, exact match, character-level accuracy
- Sample-level inspection with side-by-side comparison
- Results export to JSON

Usage:
    # Evaluate using checkpoint metadata (auto-detects config)
    python evaluate.py --checkpoint ./checkpoints/best_model.eqx

    # Override dataset / split
    python evaluate.py --checkpoint ./checkpoints/best_model.eqx \
        --dataset khaimaitien/leetcode_problem_solution --split test

    # Control generation parameters
    python evaluate.py --checkpoint ./checkpoints/best_model.eqx \
        --max-gen-length 512 --temperature 0.8 --top-k 50

    # Save results
    python evaluate.py --checkpoint ./checkpoints/best_model.eqx \
        --output results.json --num-samples 20
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import numpy as np
from tqdm import tqdm
import pdb
from byte_tabnet import (
    ByteTabNet,
    ByteTabNetSeq2Seq,
    ByteTabNetSeq2SeqDecoder,
    prepare_seq2seq_batch,
    seq2seq_loss_with_tabnet_sparsity,
)
from trainer import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    GenerationTask,
)
from tokenizer import Tokenizer, ByteTokenizer, create_tokenizer


# ============================================================================
# Model Loading
# ============================================================================

class _OldSeq2SeqShim(eqx.Module):
    """Shim matching the old ByteTabNetSeq2Seq tree (no generation_head)."""
    encoder: ByteTabNet
    decoder: ByteTabNetSeq2SeqDecoder
    vocab_size: int = eqx.field(static=True)
    max_seq_length: int = eqx.field(static=True)
    bos_token_id: int = eqx.field(static=True)
    eos_token_id: int = eqx.field(static=True)
    pad_token_id: int = eqx.field(static=True)

    @staticmethod
    def from_new(model: ByteTabNetSeq2Seq) -> "_OldSeq2SeqShim":
        return _OldSeq2SeqShim(
            encoder=model.encoder,
            decoder=model.decoder,
            vocab_size=model.vocab_size,
            max_seq_length=model.max_seq_length,
            bos_token_id=model.bos_token_id,
            eos_token_id=model.eos_token_id,
            pad_token_id=model.pad_token_id,
        )


def load_checkpoint(checkpoint_path: str) -> Tuple[eqx.Module, Dict]:
    """Load model and metadata from a checkpoint.

    Args:
        checkpoint_path: Path to .eqx checkpoint file.

    Returns:
        (model, metadata) tuple.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load metadata
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {json_path}. "
            "The .json file must accompany the .eqx checkpoint."
        )

    with open(json_path, "r") as f:
        metadata = json.load(f)

    model_cfg = metadata.get("model_config", {})
    model_config = ModelConfig(**model_cfg)

    # Build template model and deserialise weights
    key = jrandom.PRNGKey(0)
    handler = GenerationTask()
    template = handler.create_model(model_config, key)

    try:
        model = eqx.tree_deserialise_leaves(path, template)
    except (RuntimeError, ValueError, EOFError):
        # Older checkpoints lack generation_head. Load encoder+decoder via a
        # shim module that matches the old tree structure, then graft them
        # into the new template (which keeps its freshly-initialised head).
        old_template = _OldSeq2SeqShim.from_new(template)
        old_loaded = eqx.tree_deserialise_leaves(path, old_template)
        model = eqx.tree_at(
            lambda m: (m.encoder, m.decoder),
            template,
            (old_loaded.encoder, old_loaded.decoder),
        )

    return model, metadata


# ============================================================================
# Dataset Loading (test split only)
# ============================================================================

def load_test_data(
    dataset_name: str,
    split: str = "test",
    text_column: str = "algo_input",
    target_column: str = "solution_py",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Load the test split from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Which split to load (default: "test").
        text_column: Column name for input text.
        target_column: Column name for reference output.
        subset: Optional dataset config / subset name.
        max_samples: Limit number of samples (None = all).

    Returns:
        (inputs, references) lists of strings.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    load_kwargs = {}
    if subset:
        load_kwargs["name"] = subset

    print(f"Loading dataset: {dataset_name} (split={split}) ...")
    ds = load_dataset(dataset_name, split=split, **load_kwargs)

    inputs = [str(row[text_column]) for row in ds]
    references = [str(row[target_column]) for row in ds]

    if max_samples is not None:
        inputs = inputs[:max_samples]
        references = references[:max_samples]

    print(f"Loaded {len(inputs)} examples from '{split}' split")
    return inputs, references


def load_test_data_from_split(
    dataset_name: str,
    text_column: str,
    target_column: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    shuffle_seed: int = 42,
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
    hf_split_train: str = "train",
    hf_split_test: str = "test",
) -> Tuple[List[str], List[str]]:
    """Reconstruct the held-out test portion from training data.

    Uses the same shuffle seed and split ratios as the trainer to extract
    exactly the test portion that was NOT seen during training.  This is
    useful for datasets that have no dedicated test split (e.g. Quora QA).

    Args:
        dataset_name: HuggingFace dataset identifier.
        text_column: Column name for input text.
        target_column: Column name for reference output.
        train_split: Fraction used for training (must match trainer config).
        val_split: Fraction used for validation (must match trainer config).
        shuffle_seed: Seed used by the trainer to shuffle the data.
        subset: Optional dataset config / subset name.
        max_samples: Limit number of samples (None = all).
        hf_split_train: Name of the train split in HuggingFace.
        hf_split_test: Name of the test split in HuggingFace.

    Returns:
        (inputs, references) lists of strings for the held-out test portion.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    load_kwargs = {}
    if subset:
        load_kwargs["name"] = subset

    print(f"Loading dataset: {dataset_name} (recreating test split) ...")
    ds = load_dataset(dataset_name, **load_kwargs)

    # Combine train and test splits the same way the HuggingFaceDataLoader does
    all_rows = []
    for split_name in [hf_split_train, hf_split_test]:
        if split_name and split_name in ds:
            all_rows.extend(ds[split_name])

    if not all_rows:
        raise ValueError(
            f"No data found. Available splits: {list(ds.keys())}"
        )

    # Build (source, target) pairs
    pairs = [(str(row[text_column]), str(row[target_column])) for row in all_rows]

    # Shuffle with the same seed as the trainer
    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(pairs)

    # Extract the test portion
    n = len(pairs)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    test_pairs = pairs[n_train + n_val:]

    if not test_pairs:
        raise ValueError(
            f"No test data after splitting {n} examples with "
            f"train_split={train_split}, val_split={val_split}. "
            f"Ensure test_split > 0 in the training config."
        )

    inputs = [p[0] for p in test_pairs]
    references = [p[1] for p in test_pairs]

    if max_samples is not None:
        inputs = inputs[:max_samples]
        references = references[:max_samples]

    print(f"Reconstructed {len(inputs)} test examples "
          f"(from {n} total, seed={shuffle_seed}, "
          f"train={train_split}, val={val_split})")
    return inputs, references


# ============================================================================
# Teacher-Forcing Evaluation
# ============================================================================

def evaluate_teacher_forcing(
    model: ByteTabNetSeq2Seq,
    inputs: List[str],
    references: List[str],
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate using teacher forcing (loss, perplexity, cross-entropy).

    Mirrors the Trainer.evaluate() logic used during training.
    """
    total_loss = 0.0
    total_ce = 0.0
    total_tokens = 0
    num_batches = 0

    max_src_len = model_config.max_seq_length
    max_tgt_len = data_config.max_target_length

    num_examples = len(inputs)
    pbar = tqdm(range(0, num_examples, batch_size), desc="Teacher-forcing eval")

    for start in pbar:
        end = min(start + batch_size, num_examples)
        batch_inputs = inputs[start:end]
        batch_refs = references[start:end]

        # Prepare batch (same as training)
        src_ids, tgt_ids, src_mask, tgt_mask = prepare_seq2seq_batch(
            batch_inputs, batch_refs, max_src_len, max_tgt_len,
        )

        # Forward pass with teacher forcing
        logits, masks, _ = model(
            src_ids, tgt_ids, src_mask, tgt_mask, inference=True,
        )

        # Compute loss (shifted: logits[:, :-1] predicts tgt_ids[:, 1:])
        loss = seq2seq_loss_with_tabnet_sparsity(
            logits[:, :-1],
            tgt_ids[:, 1:],
            masks,
            tgt_mask[:, 1:],
            sparsity_weight=training_config.sparsity_weight,
            label_smoothing=training_config.label_smoothing,
        )

        # Per-token cross-entropy (without sparsity regularisation)
        shifted_logits = logits[:, :-1]
        shifted_targets = tgt_ids[:, 1:]
        shifted_mask = tgt_mask[:, 1:]

        log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs, shifted_targets[..., None], axis=-1
        ).squeeze(-1)
        token_ce = -target_log_probs * shifted_mask
        batch_tokens = float(shifted_mask.sum())

        total_loss += float(loss)
        total_ce += float(token_ce.sum())
        total_tokens += batch_tokens
        num_batches += 1

        pbar.set_postfix({"loss": f"{float(loss):.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    avg_ce = total_ce / max(total_tokens, 1)
    perplexity = float(np.exp(min(avg_ce, 100)))  # clamp to avoid overflow

    return {
        "loss": avg_loss,
        "cross_entropy": avg_ce,
        "perplexity": perplexity,
        "total_tokens": int(total_tokens),
        "num_batches": num_batches,
    }


# ============================================================================
# Byte Token Mapping & UTF-8 Decoding
# ============================================================================

# ByteTabNet token layout:
#   0       = PAD
#   1       = BOS  (beginning of sequence)
#   2       = EOS  (end of sequence)
#   3 .. 63 = reserved special tokens
#  64 ..319 = byte values 0..255  (token_id - 64 = byte)

BYTE_OFFSET = 64
SPECIAL_TOKENS = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
for _i in range(3, 64):
    SPECIAL_TOKENS[_i] = f"<SPECIAL_{_i}>"


def encode_text_to_tokens(
    texts: List[str],
    max_length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Encode a batch of strings to byte-level token IDs with BOS prefix.

    Encoding: token_id = byte_value + 64, with BOS (1) prepended.

    Args:
        texts: List of input strings.
        max_length: Maximum sequence length (truncate/pad to this).

    Returns:
        (input_ids, attention_mask) as jnp arrays of shape (batch, max_length).
    """
    batch_ids = []
    for text in texts:
        ids = [1] + [b + BYTE_OFFSET for b in text.encode("utf-8")]
        if len(ids) > max_length:
            ids = ids[:max_length]
        batch_ids.append(ids)

    # Pad to max_length and build attention masks
    padded = []
    masks = []
    for ids in batch_ids:
        pad_len = max_length - len(ids)
        padded.append(ids + [0] * pad_len)
        masks.append([1.0] * len(ids) + [0.0] * pad_len)

    return jnp.array(padded, dtype=jnp.int32), jnp.array(masks, dtype=jnp.float32)


def map_token_to_char(token_id: int) -> str:
    """Decode a single byte-level token ID back to its character.

    Inverse of the encoding: byte_value = token_id - 64.
    For multi-byte UTF-8 characters, use map_tokens_to_utf8_chars() instead.

    Args:
        token_id: A byte-level token ID (64..319 maps to byte 0..255).

    Returns:
        The decoded ASCII character, a hex representation for non-printable
        bytes, or the special token name.
    """
    tid = int(token_id)
    if tid in SPECIAL_TOKENS:
        return SPECIAL_TOKENS[tid]
    if 64 <= tid < 320:
        byte_val = tid - BYTE_OFFSET
        if 32 <= byte_val < 127:
            return chr(byte_val)
        return f"\\x{byte_val:02x}"
    return f"<UNK_{tid}>"


def token_to_label(token_id: int) -> str:
    """Human-readable label for a single token ID."""
    tid = int(token_id)
    if tid in SPECIAL_TOKENS:
        return SPECIAL_TOKENS[tid]
    if 64 <= tid < 320:
        byte_val = tid - BYTE_OFFSET
        # printable ASCII range
        if 32 <= byte_val < 127:
            return f"0x{byte_val:02X} '{chr(byte_val)}'"
        return f"0x{byte_val:02X}"
    return f"<UNK_{tid}>"


def _strip_special(token_ids) -> List[int]:
    """Remove BOS prefix and truncate at EOS."""
    tokens = [int(t) for t in token_ids]
    if tokens and tokens[0] == 1:
        tokens = tokens[1:]
    try:
        eos_idx = tokens.index(2)
        tokens = tokens[:eos_idx]
    except ValueError:
        pass
    return tokens


def decode_token_ids(token_ids) -> str:
    """Decode a sequence of byte-level token IDs back to a UTF-8 string.

    Iterates over each token, applies map_token_to_char to get the raw byte,
    collects all bytes, and decodes the result as UTF-8.
    Special tokens (PAD, BOS, EOS) are stripped before decoding.
    """
    tokens = _strip_special(token_ids)
    raw_bytes = bytes(int(t) - BYTE_OFFSET for t in tokens if 64 <= int(t) < 320)
    return raw_bytes.decode("utf-8", errors="replace")


def map_tokens_to_bytes(token_ids) -> List[Dict]:
    """Map each token to its byte value and metadata.

    Returns a list of dicts:
        {"pos": int, "token_id": int, "byte": int|None, "label": str}
    """
    tokens = _strip_special(token_ids)
    result = []
    for pos, tid in enumerate(tokens):
        entry = {"pos": pos, "token_id": tid, "label": token_to_label(tid)}
        if 64 <= tid < 320:
            entry["byte"] = tid - BYTE_OFFSET
        else:
            entry["byte"] = None
        result.append(entry)
    return result


def map_tokens_to_utf8_chars(token_ids) -> List[Dict]:
    """Group byte-tokens into UTF-8 characters.

    UTF-8 byte patterns:
      0xxxxxxx                           → 1 byte  (ASCII)
      110xxxxx 10xxxxxx                  → 2 bytes
      1110xxxx 10xxxxxx 10xxxxxx         → 3 bytes
      11110xxx 10xxxxxx 10xxxxxx 10xxxxxx → 4 bytes

    Returns a list of dicts per decoded character:
        {
            "char": str,           -- the decoded character (or "?" on error)
            "byte_values": [int],  -- raw byte values
            "token_ids": [int],    -- original token IDs
            "positions": [int],    -- positions in the stripped sequence
            "hex": str,            -- hex representation of the bytes
        }
    """
    tokens = _strip_special(token_ids)
    # Extract (pos, byte_value) pairs, skipping non-byte tokens
    byte_seq = []
    for pos, tid in enumerate(tokens):
        if 64 <= tid < 320:
            byte_seq.append((pos, tid, tid - BYTE_OFFSET))

    chars = []
    i = 0
    while i < len(byte_seq):
        pos0, tid0, b0 = byte_seq[i]

        # Determine how many bytes this UTF-8 character needs
        if b0 < 0x80:
            n_bytes = 1
        elif b0 < 0xC0:
            # Continuation byte appearing as lead → decode error
            chars.append({
                "char": "\uFFFD",
                "byte_values": [b0],
                "token_ids": [tid0],
                "positions": [pos0],
                "hex": f"0x{b0:02X}",
            })
            i += 1
            continue
        elif b0 < 0xE0:
            n_bytes = 2
        elif b0 < 0xF0:
            n_bytes = 3
        else:
            n_bytes = 4

        # Gather the required continuation bytes
        positions = [pos0]
        tids = [tid0]
        bvals = [b0]

        ok = True
        for j in range(1, n_bytes):
            if i + j >= len(byte_seq):
                ok = False
                break
            _, tid_j, b_j = byte_seq[i + j]
            if not (0x80 <= b_j < 0xC0):
                ok = False
                break
            positions.append(byte_seq[i + j][0])
            tids.append(tid_j)
            bvals.append(b_j)

        hex_str = " ".join(f"0x{b:02X}" for b in bvals)

        if ok and len(bvals) == n_bytes:
            try:
                decoded = bytes(bvals).decode("utf-8")
            except UnicodeDecodeError:
                decoded = "\uFFFD"
        else:
            decoded = "\uFFFD"

        chars.append({
            "char": decoded,
            "byte_values": bvals,
            "token_ids": tids,
            "positions": positions,
            "hex": hex_str,
        })

        i += max(len(bvals), 1)

    return chars


def format_token_mapping(token_ids, max_chars: int = 60) -> str:
    """Pretty-print the token → byte → UTF-8 char mapping.

    Example output:
      Pos  Token  Byte  Hex   Char
        0    164    100  0x64  'd'   ─┐
        1    165    101  0x65  'e'   ─┤ "def"
        2    166    102  0x66  'f'   ─┘
        3    096     32  0x20  ' '      " "
    """
    char_groups = map_tokens_to_utf8_chars(token_ids)
    if not char_groups:
        return "(empty sequence)"

    lines = []
    lines.append(f"  {'Pos':>4}  {'Token':>5}  {'Byte':>4}  {'Hex':>6}  → Char")
    lines.append("  " + "-" * 42)

    shown = 0
    for group in char_groups:
        if shown >= max_chars:
            lines.append(f"  ... ({len(char_groups) - shown} more characters)")
            break

        n = len(group["byte_values"])
        char_repr = repr(group["char"])

        for k in range(n):
            pos = group["positions"][k]
            tid = group["token_ids"][k]
            bval = group["byte_values"][k]

            if n == 1:
                connector = f"  → {char_repr}"
            elif k == 0:
                connector = f"  ─┐ {char_repr}" if n > 1 else f"  → {char_repr}"
            elif k == n - 1:
                connector = "  ─┘"
            else:
                connector = "  ─┤"

            lines.append(f"  {pos:4d}  {tid:5d}  {bval:4d}  0x{bval:02X}{connector}")

        shown += 1

    return "\n".join(lines)


def decode_with_tokenizer(token_ids, tokenizer: Optional[Tokenizer] = None) -> str:
    """Decode token IDs using a tokenizer, falling back to byte-level decoding.

    Args:
        token_ids: Sequence of token IDs.
        tokenizer: If provided, use this tokenizer's decode method.
                   If None, use the default byte-level decode_token_ids.
    """
    if tokenizer is not None:
        return tokenizer.decode(token_ids)
    return decode_token_ids(token_ids)


def generate_predictions(
    model: ByteTabNetSeq2Seq,
    inputs: List[str],
    model_config: ModelConfig,
    max_gen_length: int = 512,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    batch_size: int = 1,
    seed: int = 42,
    num_samples: Optional[int] = None,
) -> Tuple[List[str], List[List[int]]]:
    """Generate outputs for each input using the encoder's chat method.

    Uses autoregressive byte-level generation via windowed sparsemax
    pooling on the encoder (TabNet).

    Returns:
        (decoded_texts, raw_token_id_lists)
        - decoded_texts: UTF-8 decoded strings
        - raw_token_id_lists: raw token ID sequences (including BOS/EOS)
    """
    predictions = []
    all_token_ids = []
    gen_inputs = inputs[:num_samples] if num_samples is not None else inputs
    n_inputs = len(gen_inputs)
    pbar = tqdm(range(n_inputs), desc="Generating")
    for i in pbar:
        generated_ids = model.encoder.chat(
            prompt=gen_inputs[i],
            max_gen=max_gen_length,
            temperature=temperature,
            seed=seed + i,
        )

        seq = generated_ids[0]  # chat returns (1, seq_len)
        all_token_ids.append([int(t) for t in seq])
        predictions.append(decode_token_ids(seq))
        if i==2:break
    return predictions, all_token_ids


def compute_generation_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute text-level metrics between generated and reference strings.

    Metrics:
    - exact_match: fraction of predictions that exactly match the reference
    - bleu (1-4): corpus-level BLEU scores (character-level tokenisation)
    - avg_pred_length / avg_ref_length: average character counts
    - char_precision / char_recall / char_f1: character overlap metrics
    """
    assert len(predictions) == len(references)
    n = len(predictions)
    if n == 0:
        return {}

    # Exact match
    exact = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())

    # Character-level precision / recall / F1
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_chars = set(enumerate(pred))  # positional chars
        ref_chars = set(enumerate(ref))
        # Use bag-of-characters (non-positional) for softer metric
        pred_bag = {}
        for c in pred:
            pred_bag[c] = pred_bag.get(c, 0) + 1
        ref_bag = {}
        for c in ref:
            ref_bag[c] = ref_bag.get(c, 0) + 1

        overlap = sum(min(pred_bag.get(c, 0), ref_bag.get(c, 0)) for c in ref_bag)
        p = overlap / max(len(pred), 1)
        r = overlap / max(len(ref), 1)
        f = 2 * p * r / max(p + r, 1e-8)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    # BLEU score (character-level n-grams)
    bleu_scores = _compute_bleu(predictions, references)

    avg_pred_len = np.mean([len(p) for p in predictions])
    avg_ref_len = np.mean([len(r) for r in references])

    metrics = {
        "exact_match": exact / n,
        "char_precision": float(np.mean(precisions)),
        "char_recall": float(np.mean(recalls)),
        "char_f1": float(np.mean(f1s)),
        "avg_pred_length": float(avg_pred_len),
        "avg_ref_length": float(avg_ref_len),
        "num_examples": n,
    }
    metrics.update(bleu_scores)

    return metrics


def _compute_bleu(
    predictions: List[str], references: List[str], max_n: int = 4
) -> Dict[str, float]:
    """Compute corpus-level BLEU scores using character-level n-grams."""

    def _ngrams(text: str, n: int) -> Dict[str, int]:
        counts = {}
        for i in range(len(text) - n + 1):
            ng = text[i : i + n]
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    pred_length = 0
    ref_length = 0

    for pred, ref in zip(predictions, references):
        pred_length += len(pred)
        ref_length += len(ref)

        for n in range(1, max_n + 1):
            pred_ng = _ngrams(pred, n)
            ref_ng = _ngrams(ref, n)

            for ng, count in pred_ng.items():
                clipped_counts[n - 1] += min(count, ref_ng.get(ng, 0))
                total_counts[n - 1] += count

    # Brevity penalty
    if pred_length == 0:
        bp = 0.0
    elif pred_length >= ref_length:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_length / max(pred_length, 1))

    # Compute individual n-gram precisions and combined BLEU
    bleu_scores = {}
    log_avg = 0.0
    valid_n = 0

    for n in range(1, max_n + 1):
        if total_counts[n - 1] > 0:
            p = clipped_counts[n - 1] / total_counts[n - 1]
        else:
            p = 0.0
        bleu_scores[f"bleu_{n}"] = float(bp * p) if p > 0 else 0.0

        if p > 0:
            log_avg += np.log(p)
            valid_n += 1

    if valid_n == max_n and valid_n > 0:
        bleu_scores["bleu"] = float(bp * np.exp(log_avg / max_n))
    else:
        bleu_scores["bleu"] = 0.0

    return bleu_scores


# ============================================================================
# Results Display
# ============================================================================

def print_separator(char: str = "=", width: int = 70):
    print(char * width)


def print_metrics(title: str, metrics: Dict[str, float]):
    print_separator()
    print(f"  {title}")
    print_separator()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s} : {value:.6f}")
        else:
            print(f"  {key:25s} : {value}")
    print_separator()


def print_samples(
    inputs: List[str],
    references: List[str],
    predictions: List[str],
    raw_token_ids: Optional[List[List[int]]] = None,
    num_samples: int = 5,
    show_token_map: bool = True,
):
    """Print side-by-side comparison of reference vs generated outputs."""
    print_separator()
    print("  Sample Predictions")
    print_separator()

    n = min(num_samples, len(inputs))
    for i in range(n):
        print(f"\n--- Example {i + 1}/{n} ---")
        # Truncate long inputs for display
        inp = inputs[i][:] #+ ("..." if len(inputs[i]) > 300 else "")
        ref = references[i][:] #+ ("..." if len(references[i]) > 500 else "")
        pred = predictions[i][:] #+ ("..." if len(predictions[i]) > 500 else "")

        print(f"INPUT:\n{inp}\n")
        print(f"REFERENCE:\n{ref}\n")
        print(f"GENERATED:\n{pred}\n")
        print("=======================")
        # Quick match indicator
        if predictions[i].strip() == references[i].strip():
            print("[EXACT MATCH]")
        else:
            overlap = sum(
                1 for a, b in zip(predictions[i], references[i]) if a == b
            )
            max_len = max(len(predictions[i]), len(references[i]), 1)
            print(f"[Character overlap: {overlap}/{max_len} = {overlap/max_len:.1%}]")

        # Token → byte → UTF-8 character mapping
        if show_token_map and raw_token_ids is not None:
            tids = raw_token_ids[i]
            stripped = _strip_special(tids)
            print(f"\nTOKEN MAPPING ({len(stripped)} byte-tokens → "
                  f"{len(map_tokens_to_utf8_chars(tids))} UTF-8 chars):")
            print(format_token_mapping(tids))


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ByteTabNet model on a HuggingFace dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.eqx file)",
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="HuggingFace dataset name (default: from checkpoint metadata)",
    )
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--text-column", type=str, default=None, help="Input text column")
    parser.add_argument("--target-column", type=str, default=None, help="Reference target column")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of evaluation samples (default: all)",
    )
    parser.add_argument(
        "--recreate-split", action="store_true",
        help="Reconstruct the held-out test portion from training data using "
             "the same shuffle seed and split ratios (for datasets without a "
             "dedicated test split, e.g. Quora QA)",
    )

    # Evaluation modes
    parser.add_argument(
        "--skip-teacher-forcing", action="store_true",
        help="Skip teacher-forcing metrics (loss / perplexity)",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip generation-based metrics (BLEU / exact match)",
    )

    # Generation parameters
    parser.add_argument("--max-gen-length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling (top-p)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")

    # Batch size
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for teacher-forcing eval")
    parser.add_argument(
        "--gen-batch-size", type=int, default=1,
        help="Batch size for generation (1 recommended for long sequences)",
    )

    # Output
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to display")
    parser.add_argument(
        "--show-token-map", action="store_true", default=True,
        help="Show token-to-byte-to-UTF8 mapping for each sample (default: on)",
    )
    parser.add_argument(
        "--no-token-map", action="store_true",
        help="Disable token-to-byte-to-UTF8 mapping in sample display",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    print_separator()
    print("  ByteTabNet Evaluation")
    print_separator()

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, metadata = load_checkpoint(args.checkpoint)

    model_cfg = metadata.get("model_config", {})
    training_cfg = metadata.get("training_config", {})
    data_cfg = metadata.get("data_config", {})

    model_config = ModelConfig(**model_cfg)
    training_config = TrainingConfig(**training_cfg)
    data_config = DataConfig(**data_cfg)

    print(f"  Epoch:     {metadata.get('epoch', '?')}")
    print(f"  Val loss:  {metadata.get('val_loss', '?')}")
    print(f"  Task:      {training_config.task_type}")
    print(f"  Model:     embed={model_config.embed_dim}, hidden={model_config.hidden_dim}, "
          f"steps={model_config.n_steps}, decoder_layers={model_config.n_decoder_layers}")

    if training_config.task_type != "generation":
        print(f"\nWarning: checkpoint task_type is '{training_config.task_type}', "
              "but this script is designed for generation evaluation.")

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    dataset_name = args.dataset or data_config.hf_dataset_name
    text_column = args.text_column or data_config.text_column or "algo_input"
    target_column = args.target_column or data_config.target_column or "solution_py"

    if not dataset_name:
        parser.error(
            "--dataset is required (no hf_dataset_name found in checkpoint metadata)"
        )

    if args.recreate_split:
        # Reconstruct held-out test portion using the same seed / split
        # ratios as the trainer.  Useful for datasets without a dedicated
        # test split (e.g. Quora QA).
        inputs, references = load_test_data_from_split(
            dataset_name=dataset_name,
            text_column=text_column,
            target_column=target_column,
            train_split=training_config.train_split,
            val_split=training_config.val_split,
            shuffle_seed=training_config.shuffle_seed,
            subset=args.subset or data_config.hf_subset,
            max_samples=args.max_samples,
            hf_split_train=data_config.hf_split_train,
            hf_split_test=data_config.hf_split_test,
        )
    else:
        inputs, references = load_test_data(
            dataset_name=dataset_name,
            split=args.split,
            text_column=text_column,
            target_column=target_column,
            subset=args.subset or data_config.hf_subset,
            max_samples=args.max_samples,
        )

    all_metrics = {}
    predictions = None
    raw_token_ids = None
    start_time = time.time()

    # ------------------------------------------------------------------
    # 3. Teacher-forcing evaluation
    # ------------------------------------------------------------------
    if not args.skip_teacher_forcing:
        print("\n--- Teacher-Forcing Evaluation ---")
        tf_metrics = evaluate_teacher_forcing(
            model, inputs, references,
            model_config, data_config, training_config,
            batch_size=args.batch_size,
        )
        all_metrics["teacher_forcing"] = tf_metrics
        print_metrics("Teacher-Forcing Metrics", tf_metrics)

    # ------------------------------------------------------------------
    # 4. Generation evaluation
    # ------------------------------------------------------------------
    if not args.skip_generation:
        print("\n--- Generation Evaluation ---")
        print(f"  Decoding: sample(T={args.temperature}, k={args.top_k}, p={args.top_p})")
        print(f"  Max generation length: {args.max_gen_length}")
        predictions, raw_token_ids = generate_predictions(
            model, inputs, model_config,
            max_gen_length=args.max_gen_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            batch_size=args.gen_batch_size,
            seed=args.seed,
            num_samples=args.num_samples,
        )
        # Show samples with token mapping

        show_map = args.show_token_map and not args.no_token_map


        # Slice inputs/references to match generated count
        gen_inputs = inputs[:len(predictions)]
        gen_references = references[:len(predictions)]

        print_samples(
            gen_inputs, gen_references, predictions,
            raw_token_ids=raw_token_ids,
            num_samples=args.num_samples,
            show_token_map=show_map,
        )

        gen_metrics = compute_generation_metrics(predictions, gen_references)
        all_metrics["generation"] = gen_metrics
        print_metrics("Generation Metrics", gen_metrics)

        

    elapsed = time.time() - start_time
    all_metrics["eval_time_seconds"] = round(elapsed, 2)
    all_metrics["num_examples"] = len(inputs)
    all_metrics["checkpoint"] = str(args.checkpoint)
    all_metrics["dataset"] = dataset_name
    all_metrics["split"] = args.split

    print(f"\nTotal evaluation time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        results = {
            "metrics": all_metrics,
            "config": {
                "model": model_cfg,
                "training": training_cfg,
                "data": data_cfg,
            },
        }

        # Optionally include per-sample predictions with token mapping
        if predictions is not None:
            samples_out = []
            for i in range(len(inputs)):
                sample = {
                    "input": inputs[i][:1000],
                    "reference": references[i][:2000],
                    "prediction": predictions[i][:2000],
                }
                if raw_token_ids is not None:
                    sample["token_ids"] = _strip_special(raw_token_ids[i])
                    sample["utf8_char_map"] = map_tokens_to_utf8_chars(raw_token_ids[i])
                samples_out.append(sample)
            results["samples"] = samples_out

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
