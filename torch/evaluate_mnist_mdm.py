"""
Evaluate a trained ByteTabNet MD4 checkpoint on MNIST.

Usage:
    cd torch/
    python evaluate_mnist_mdm.py
    python evaluate_mnist_mdm.py --checkpoint checkpoints_mnist_mdm/best_model.pt
    python evaluate_mnist_mdm.py --checkpoint checkpoints_mnist_mdm/best_model.pt --n-showcase 100
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from train_mdm import (
    ByteTabNetMDM,
    get_device,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
)
from byte_tabnet import BYTE_OFFSET, VOCAB_SIZE
from train_mnist_mdm import (
    MNISTMDMConfig,
    MNISTMDMDataset,
    load_mnist,
    _evaluate,
    plot_comparison,
    generate_final_showcase,
    SEQ_LEN_COND,
    SEQ_LEN_UNCOND,
)


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ByteTabNetMDM, MNISTMDMConfig, dict]:
    """
    Load a ByteTabNetMDM model from a checkpoint .pt file.

    The accompanying .json metadata file (same stem, same directory) is used to
    reconstruct the MNISTMDMConfig so the model architecture matches exactly.

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

    # Load metadata ----------------------------------------------------------------
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        saved_cfg = meta.get("cfg", {})
    else:
        print(f"[warn] No metadata JSON at {json_path}; using default config.")
        meta, saved_cfg = {}, {}

    # Build config from saved values (fall back to defaults for missing keys) -------
    cfg_fields = {f.name for f in fields(MNISTMDMConfig)}
    cfg_kwargs = {k: v for k, v in saved_cfg.items() if k in cfg_fields}
    cfg = MNISTMDMConfig(**cfg_kwargs)

    print("Loaded config:")
    for k, v in cfg_kwargs.items():
        print(f"  {k}: {v}")
    print()

    # Build model ------------------------------------------------------------------
    seq_len = SEQ_LEN_COND if cfg.conditional else SEQ_LEN_UNCOND
    model = ByteTabNetMDM(
        max_seq_length     = seq_len,
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
    print(f"Model loaded from {ckpt_path}")
    print(f"  Parameters : {n_params:,}")
    print(f"  Saved epoch: {epoch}   val_loss: {val_loss}\n")

    return model, cfg, meta


# ============================================================================
# Main evaluation routine
# ============================================================================

def evaluate(
    checkpoint_path: str = "checkpoints_mnist_mdm/best_model.pt",
    out_dir: str = "eval_output",
    n_compare: int = 8,
    n_showcase: int = 100,
    eval_batches: int = 50,
):
    """
    Full evaluation pipeline:

      1. Load model from checkpoint.
      2. Load MNIST test set.
      3. Compute MD4 loss on the test set.
      4. Save 3-panel comparison figure (real / generated / |diff|).
      5. Generate and save a final ~n_showcase image showcase grid.
    """
    device = get_device()
    print(f"Device: {device}\n")

    model, cfg, meta = load_checkpoint(checkpoint_path, device)

    # Override sample dir to eval_output so we don't pollute training dirs
    cfg.sample_dir = out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- MNIST test set ----------------------------------------------------------
    print("Loading MNIST test set...")
    _, _, te_imgs, te_lbl = load_mnist()
    print(f"Test set: {len(te_imgs):,} images\n")

    test_ds = MNISTMDMDataset(te_imgs, te_lbl, cfg.conditional, device)

    # ---- Loss evaluation ---------------------------------------------------------
    print(f"Evaluating MD4 loss on test set ({eval_batches} batches) ...")
    test_loss = _evaluate(model, test_ds, cfg, device, num_batches=eval_batches)
    print(f"  Test MD4 loss: {test_loss:.6f}\n")

    # ---- Comparison plot ---------------------------------------------------------
    compare_path = os.path.join(out_dir, "comparison.png")
    print(f"Generating comparison plot ({n_compare} pairs) ...")
    plot_comparison(
        model,
        te_imgs.reshape(-1, 28, 28),
        cfg, device,
        n=n_compare,
        path=compare_path,
    )

    # ---- Final showcase ----------------------------------------------------------
    print(f"\nGenerating showcase ({n_showcase} images) ...")
    generate_final_showcase(model, cfg, device, n_samples=n_showcase)

    # ---- Summary -----------------------------------------------------------------
    print("\n=== Evaluation complete ===")
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  Test MD4 loss: {test_loss:.6f}")
    print(f"  Output dir   : {out_dir}/")
    print(f"    comparison.png, final_showcase.png")
    if cfg.conditional:
        print(f"    final_showcase_d{{0-9}}.png")

    return test_loss


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a ByteTabNet MD4 checkpoint on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str,
        default="checkpoints_mnist_mdm/best_model.pt",
        help="Path to the .pt checkpoint file.",
    )
    p.add_argument(
        "--out-dir", type=str, default="eval_output",
        help="Directory to save evaluation outputs.",
    )
    p.add_argument(
        "--n-compare", type=int, default=8,
        help="Number of image pairs in the comparison plot.",
    )
    p.add_argument(
        "--n-showcase", type=int, default=100,
        help="Number of images in the final showcase grid.",
    )
    p.add_argument(
        "--eval-batches", type=int, default=50,
        help="Number of test batches to average loss over.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path = args.checkpoint,
        out_dir         = args.out_dir,
        n_compare       = args.n_compare,
        n_showcase      = args.n_showcase,
        eval_batches    = args.eval_batches,
    )
