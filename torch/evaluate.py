"""
ByteTabNet Evaluation Script (PyTorch)

Evaluate a trained ByteTabNet model on the test split of a HuggingFace dataset.

Supports:
- Teacher-forcing metrics: loss, perplexity, cross-entropy
- Generation metrics: BLEU score, exact match, character-level accuracy
- Sample-level inspection with side-by-side comparison
- Results export to JSON

Usage:
    # Evaluate using checkpoint metadata (auto-detects config)
    python evaluate.py --checkpoint ./checkpoints/best_model.pt

    # Override dataset / split
    python evaluate.py --checkpoint ./checkpoints/best_model.pt \
        --dataset khaimaitien/leetcode_problem_solution --split test

    # Control generation parameters
    python evaluate.py --checkpoint ./checkpoints/best_model.pt \
        --max-gen-length 512 --temperature 0.8 --top-k 50

    # Save results
    python evaluate.py --checkpoint ./checkpoints/best_model.pt \
        --output results.json --num-samples 20
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from byte_tabnet import (
    ByteTabNet,
    ByteTabNetSeq2Seq,
    ByteEmbedding,
    prepare_seq2seq_batch,
    seq2seq_loss_with_tabnet_sparsity,
    BYTE_OFFSET,
    VOCAB_SIZE,
)
from trainer import (
    ModelConfig,
    TrainingConfig,
    GenerationTask,
    get_device,
)
from dataset import DataConfig


# ============================================================================
# Model Loading
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict]:
    """Load model and metadata from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Torch device to load model onto.

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
            "The .json file must accompany the .pt checkpoint."
        )

    with open(json_path, "r") as f:
        metadata = json.load(f)

    model_cfg = metadata.get("model_config", {})
    model_config = ModelConfig(**model_cfg)

    # Build template model and load weights
    handler = GenerationTask()
    model = handler.create_model(model_config)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

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
    """Load the test split from a HuggingFace dataset."""
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
    exactly the test portion that was NOT seen during training.
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

    all_rows = []
    for split_name in [hf_split_train, hf_split_test]:
        if split_name and split_name in ds:
            all_rows.extend(ds[split_name])

    if not all_rows:
        raise ValueError(f"No data found. Available splits: {list(ds.keys())}")

    pairs = [(str(row[text_column]), str(row[target_column])) for row in all_rows]

    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    test_pairs = pairs[n_train + n_val:]

    if not test_pairs:
        raise ValueError(
            f"No test data after splitting {n} examples with "
            f"train_split={train_split}, val_split={val_split}."
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

@torch.no_grad()
def evaluate_teacher_forcing(
    model: ByteTabNetSeq2Seq,
    inputs: List[str],
    references: List[str],
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluate using teacher forcing (loss, perplexity, cross-entropy)."""
    model.eval()
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

        src_ids, tgt_ids, src_mask, tgt_mask = prepare_seq2seq_batch(
            batch_inputs, batch_refs, max_src_len, max_tgt_len,
        )
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        logits, masks = model(src_ids, tgt_ids, src_mask, tgt_mask)

        loss = seq2seq_loss_with_tabnet_sparsity(
            logits[:, :-1],
            tgt_ids[:, 1:],
            masks,
            tgt_mask[:, 1:],
            sparsity_weight=training_config.sparsity_weight,
            label_smoothing=training_config.label_smoothing,
        )

        shifted_logits = logits[:, :-1]
        shifted_targets = tgt_ids[:, 1:]
        shifted_mask = tgt_mask[:, 1:]

        log_probs = F.log_softmax(shifted_logits, dim=-1)
        target_log_probs = torch.gather(
            log_probs, -1, shifted_targets.unsqueeze(-1),
        ).squeeze(-1)
        token_ce = -target_log_probs * shifted_mask
        batch_tokens = shifted_mask.sum().item()

        total_loss += loss.item()
        total_ce += token_ce.sum().item()
        total_tokens += batch_tokens
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    avg_ce = total_ce / max(total_tokens, 1)
    perplexity = float(np.exp(min(avg_ce, 100)))

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

SPECIAL_TOKENS = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
for _i in range(3, 64):
    SPECIAL_TOKENS[_i] = f"<SPECIAL_{_i}>"


def token_to_label(token_id: int) -> str:
    tid = int(token_id)
    if tid in SPECIAL_TOKENS:
        return SPECIAL_TOKENS[tid]
    if 64 <= tid < 320:
        byte_val = tid - BYTE_OFFSET
        if 32 <= byte_val < 127:
            return f"0x{byte_val:02X} '{chr(byte_val)}'"
        return f"0x{byte_val:02X}"
    return f"<UNK_{tid}>"


def _strip_special(token_ids) -> List[int]:
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
    tokens = _strip_special(token_ids)
    raw_bytes = bytes(int(t) - BYTE_OFFSET for t in tokens if 64 <= int(t) < 320)
    return raw_bytes.decode("utf-8", errors="replace")


def map_tokens_to_utf8_chars(token_ids) -> List[Dict]:
    """Group byte-tokens into UTF-8 characters."""
    tokens = _strip_special(token_ids)
    byte_seq = []
    for pos, tid in enumerate(tokens):
        if 64 <= tid < 320:
            byte_seq.append((pos, tid, tid - BYTE_OFFSET))

    chars = []
    i = 0
    while i < len(byte_seq):
        pos0, tid0, b0 = byte_seq[i]

        if b0 < 0x80:
            n_bytes = 1
        elif b0 < 0xC0:
            chars.append({
                "char": "\uFFFD", "byte_values": [b0], "token_ids": [tid0],
                "positions": [pos0], "hex": f"0x{b0:02X}",
            })
            i += 1
            continue
        elif b0 < 0xE0:
            n_bytes = 2
        elif b0 < 0xF0:
            n_bytes = 3
        else:
            n_bytes = 4

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
            "char": decoded, "byte_values": bvals, "token_ids": tids,
            "positions": positions, "hex": hex_str,
        })
        i += max(len(bvals), 1)

    return chars


def format_token_mapping(token_ids, max_chars: int = 60) -> str:
    char_groups = map_tokens_to_utf8_chars(token_ids)
    if not char_groups:
        return "(empty sequence)"

    lines = []
    lines.append(f"  {'Pos':>4}  {'Token':>5}  {'Byte':>4}  {'Hex':>6}  -> Char")
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
                connector = f"  -> {char_repr}"
            elif k == 0:
                connector = f"  -+ {char_repr}" if n > 1 else f"  -> {char_repr}"
            elif k == n - 1:
                connector = "  -'"
            else:
                connector = "  -|"

            lines.append(f"  {pos:4d}  {tid:5d}  {bval:4d}  0x{bval:02X}{connector}")

        shown += 1

    return "\n".join(lines)


# ============================================================================
# Generation
# ============================================================================

@torch.no_grad()
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
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[str], List[List[int]]]:
    """Generate outputs for each input using the seq2seq model."""
    model.eval()
    predictions = []
    all_token_ids = []
    gen_inputs = inputs[:num_samples] if num_samples is not None else inputs

    pbar = tqdm(range(len(gen_inputs)), desc="Generating")
    for i in pbar:
        src_bytes = [b + BYTE_OFFSET for b in gen_inputs[i].encode("utf-8")]
        src_tokens = [1] + src_bytes  # BOS + bytes
        if len(src_tokens) > model_config.max_seq_length:
            src_tokens = src_tokens[:model_config.max_seq_length]

        input_ids = torch.tensor([src_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)

        generated_ids = model.generate(
            input_ids,
            attention_mask,
            max_length=max_gen_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )

        seq = generated_ids[0].cpu().tolist()
        all_token_ids.append(seq)
        predictions.append(decode_token_ids(seq))

    return predictions, all_token_ids


# ============================================================================
# Generation Metrics
# ============================================================================

def compute_generation_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute text-level metrics between generated and reference strings."""
    assert len(predictions) == len(references)
    n = len(predictions)
    if n == 0:
        return {}

    exact = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())

    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_bag: Dict[str, int] = {}
        for c in pred:
            pred_bag[c] = pred_bag.get(c, 0) + 1
        ref_bag: Dict[str, int] = {}
        for c in ref:
            ref_bag[c] = ref_bag.get(c, 0) + 1

        overlap = sum(min(pred_bag.get(c, 0), ref_bag.get(c, 0)) for c in ref_bag)
        p = overlap / max(len(pred), 1)
        r = overlap / max(len(ref), 1)
        f = 2 * p * r / max(p + r, 1e-8)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

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
        counts: Dict[str, int] = {}
        for i in range(len(text) - n + 1):
            ng = text[i:i + n]
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    pred_length = 0
    ref_length = 0

    for pred, ref in zip(predictions, references):
        pred_length += len(pred)
        ref_length += len(ref)
        for n_val in range(1, max_n + 1):
            pred_ng = _ngrams(pred, n_val)
            ref_ng = _ngrams(ref, n_val)
            for ng, count in pred_ng.items():
                clipped_counts[n_val - 1] += min(count, ref_ng.get(ng, 0))
                total_counts[n_val - 1] += count

    if pred_length == 0:
        bp = 0.0
    elif pred_length >= ref_length:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_length / max(pred_length, 1))

    bleu_scores = {}
    log_avg = 0.0
    valid_n = 0

    for n_val in range(1, max_n + 1):
        if total_counts[n_val - 1] > 0:
            p = clipped_counts[n_val - 1] / total_counts[n_val - 1]
        else:
            p = 0.0
        bleu_scores[f"bleu_{n_val}"] = float(bp * p) if p > 0 else 0.0
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
    print_separator()
    print("  Sample Predictions")
    print_separator()

    n = min(num_samples, len(inputs))
    for i in range(n):
        print(f"\n--- Example {i + 1}/{n} ---")
        inp = inputs[i]
        ref = references[i]
        pred = predictions[i]

        print(f"INPUT:\n{inp}\n")
        print(f"REFERENCE:\n{ref}\n")
        print(f"GENERATED:\n{pred}\n")
        print("=======================")
        if predictions[i].strip() == references[i].strip():
            print("[EXACT MATCH]")
        else:
            overlap = sum(
                1 for a, b in zip(predictions[i], references[i]) if a == b
            )
            max_len = max(len(predictions[i]), len(references[i]), 1)
            print(f"[Character overlap: {overlap}/{max_len} = {overlap / max_len:.1%}]")

        if show_token_map and raw_token_ids is not None:
            tids = raw_token_ids[i]
            stripped = _strip_special(tids)
            print(f"\nTOKEN MAPPING ({len(stripped)} byte-tokens -> "
                  f"{len(map_tokens_to_utf8_chars(tids))} UTF-8 chars):")
            print(format_token_mapping(tids))


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ByteTabNet model (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--text-column", type=str, default=None, help="Input text column")
    parser.add_argument("--target-column", type=str, default=None, help="Reference target column")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument(
        "--recreate-split", action="store_true",
        help="Reconstruct held-out test portion from training data",
    )

    parser.add_argument("--skip-teacher-forcing", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")

    parser.add_argument("--max-gen-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=1)

    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--show-token-map", action="store_true", default=True)
    parser.add_argument("--no-token-map", action="store_true")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    device = get_device()

    print_separator()
    print("  ByteTabNet Evaluation (PyTorch)")
    print_separator()

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, metadata = load_checkpoint(args.checkpoint, device)

    model_cfg = metadata.get("model_config", {})
    training_cfg = metadata.get("training_config", {})
    data_cfg = metadata.get("data_config", {})

    model_config = ModelConfig(**model_cfg)
    training_config = TrainingConfig(**training_cfg)
    data_config = DataConfig(**data_cfg)

    print(f"  Device:    {device}")
    print(f"  Epoch:     {metadata.get('epoch', '?')}")
    print(f"  Val loss:  {metadata.get('val_loss', '?')}")
    print(f"  Task:      {training_config.task_type}")
    print(f"  Model:     embed={model_config.embed_dim}, hidden={model_config.hidden_dim}, "
          f"steps={model_config.n_steps}, decoder_layers={model_config.n_decoder_layers}")

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    dataset_name = args.dataset or data_config.hf_dataset_name
    text_column = args.text_column or data_config.text_column or "algo_input"
    target_column = args.target_column or data_config.target_column or "solution_py"

    if not dataset_name:
        parser.error("--dataset is required (no hf_dataset_name found in metadata)")

    if args.recreate_split:
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

    all_metrics: Dict[str, Any] = {}
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
            device=device,
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
            device=device,
        )

        show_map = args.show_token_map and not args.no_token_map

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

        if predictions is not None:
            samples_out = []
            for i in range(len(predictions)):
                sample: Dict[str, Any] = {
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
