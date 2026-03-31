"""
Evaluate a trained ByteTabNet MD4 checkpoint on XythicK/code-generation-dataset.

Evaluation pipeline:
  1. Load model and config from checkpoint.
  2. Recreate the test split (same seed as training).
  3. Compute the MD4 loss on the test set.
  4. Generate code completions for a sample of test prompts.
  5. Write a summary + all generated samples to the output directory.

Usage:
    cd torch/
    python evaluate_code_mdm.py
    python evaluate_code_mdm.py --checkpoint checkpoints_code_mdm/best_model.pt
    python evaluate_code_mdm.py --checkpoint ... --n-generate 20 --n-eval-batches 50
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import torch

from train_mdm import (
    ByteTabNetMDM,
    get_device,
)
from byte_tabnet import VOCAB_SIZE
from train_code_mdm import (
    CodeGenMDMConfig,
    CodeGenMDMDataset,
    load_code_dataset,
    _evaluate,
    save_text_samples,
)


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ByteTabNetMDM, CodeGenMDMConfig, dict]:
    """
    Load a ByteTabNetMDM model from a .pt checkpoint file.

    The companion .json file (same stem, same directory) stores the
    CodeGenMDMConfig so the model architecture is reconstructed exactly.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:          Target device.

    Returns:
        (model, cfg, meta)  where meta is the raw JSON dict.
    """
    ckpt_path = Path(checkpoint_path)
    json_path = ckpt_path.with_suffix(".json")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load metadata -------------------------------------------------------
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        saved_cfg = meta.get("cfg", {})
    else:
        print(f"[warn] No metadata JSON at {json_path}; using default config.")
        meta, saved_cfg = {}, {}

    # Reconstruct config (ignore unknown keys for forward-compat) ---------
    cfg_fields = {f.name for f in fields(CodeGenMDMConfig)}
    cfg_kwargs = {k: v for k, v in saved_cfg.items() if k in cfg_fields}
    cfg = CodeGenMDMConfig(**cfg_kwargs)

    print("Loaded config:")
    for k, v in cfg_kwargs.items():
        print(f"  {k}: {v}")
    print()

    # Build model ---------------------------------------------------------
    model = ByteTabNetMDM(
        max_seq_length     = cfg.max_seq_length,
        embed_dim          = cfg.embed_dim,
        hidden_dim         = cfg.hidden_dim,
        n_blocks           = cfg.n_blocks,
        n_heads            = cfg.n_heads,
        n_steps            = cfg.n_steps,
        n_d                = cfg.n_d,
        n_a                = cfg.n_a,
        gamma              = cfg.gamma,
        n_shared           = cfg.n_shared,
        n_step             = cfg.n_step,
        virtual_batch_size = cfg.virtual_batch_size,
        use_positional     = True,
        vocab_size         = VOCAB_SIZE,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    epoch    = meta.get("epoch", "?")
    val_loss = meta.get("val_loss", "?")
    print(f"Model loaded  : {ckpt_path}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Saved epoch : {epoch}   val_loss: {val_loss}\n")

    return model, cfg, meta


# ============================================================================
# Main evaluation routine
# ============================================================================

def evaluate(
    checkpoint_path: str = "checkpoints_code_mdm/best_model.pt",
    out_dir:         str = "eval_output_code",
    n_generate:      int = 10,
    n_eval_batches:  int = 50,
):
    """
    Full evaluation pipeline:

      1. Load model from checkpoint.
      2. Recreate the test split using the same dataset name + seed as training.
      3. Compute MD4 loss on the test set.
      4. Generate code completions for n_generate test prompts (conditioned
         on the prompt; code region starts fully masked and is iteratively
         unmasked with the MD4 reverse process).
      5. Write all outputs to out_dir/.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        out_dir:         Directory for evaluation outputs.
        n_generate:      Number of prompts to generate code for.
        n_eval_batches:  Number of test batches to average the MD4 loss over.
    """
    device = get_device()
    print(f"Device: {device}\n")

    model, cfg, meta = load_checkpoint(checkpoint_path, device)

    # Override sample_dir so generated files go to out_dir
    cfg.sample_dir = out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Test data --------------------------------------------------------------
    print("Loading dataset (test split) ...")
    _, _, _, _, te_p, te_c = load_code_dataset(
        cfg.dataset_name, cfg.train_fraction, cfg.val_fraction, cfg.seed
    )
    print(f"Test set: {len(te_p):,} examples\n")

    test_ds = CodeGenMDMDataset(te_p, te_c, cfg.max_seq_length, device)

    # ---- Loss evaluation --------------------------------------------------------
    print(f"Evaluating MD4 loss on test set ({n_eval_batches} batches) ...")
    test_loss = _evaluate(model, test_ds, cfg, device, num_batches=n_eval_batches)
    print(f"  Test MD4 loss: {test_loss:.6f}\n")

    # ---- Code generation --------------------------------------------------------
    n_gen     = min(n_generate, len(te_p))
    gen_path  = os.path.join(out_dir, "generated_code.txt")
    print(f"Generating {n_gen} code completions → {gen_path} ...")
    save_text_samples(model, te_p, cfg, device, gen_path, n=n_gen)

    # ---- Summary ----------------------------------------------------------------
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== ByteTabNet MD4  —  Code Generation Evaluation ===\n\n")
        f.write(f"Checkpoint    : {checkpoint_path}\n")
        f.write(f"Dataset       : {cfg.dataset_name}\n")
        f.write(f"Test examples : {len(te_p):,}\n")
        f.write(f"Eval batches  : {n_eval_batches}\n")
        f.write(f"Test MD4 loss : {test_loss:.6f}\n\n")
        f.write(f"Generated     : {n_gen} completions  →  {gen_path}\n")
        f.write(f"Output dir    : {out_dir}/\n\n")
        f.write("Model config:\n")
        for k, v in cfg.__dict__.items():
            f.write(f"  {k}: {v}\n")

    print("\n=== Evaluation complete ===")
    print(f"  Checkpoint    : {checkpoint_path}")
    print(f"  Test MD4 loss : {test_loss:.6f}")
    print(f"  Generated code: {gen_path}")
    print(f"  Summary       : {summary_path}")

    return test_loss


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a ByteTabNet MD4 checkpoint on code-generation-dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str,
        default="checkpoints_code_mdm/best_model.pt",
        help="Path to the .pt checkpoint file.",
    )
    p.add_argument(
        "--out-dir", type=str, default="eval_output_code",
        help="Directory to save evaluation outputs.",
    )
    p.add_argument(
        "--n-generate", type=int, default=10,
        help="Number of test prompts to generate code for.",
    )
    p.add_argument(
        "--n-eval-batches", type=int, default=50,
        help="Number of test batches to average the MD4 loss over.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path = args.checkpoint,
        out_dir         = args.out_dir,
        n_generate      = args.n_generate,
        n_eval_batches  = args.n_eval_batches,
    )
