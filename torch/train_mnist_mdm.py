"""
MNIST Generation with ByteTabNet MD4 — PyTorch

Implements the MD4 (Masked Diffusion) algorithm from
  github.com/darioShar/pytorch-md4

Key algorithmic choices (matching MD4):
  - Noise schedule: alpha_t = 1 - cos(π/2·(1-t))
      alpha_t = fraction of UNMASKED tokens; alpha_0=1 (clean), alpha_1=0 (fully masked)
  - Loss weight:   w(t) = (π/2)·tan(π/2·(1-t))
      Derived from the ELBO of the masked diffusion process.
  - Timestep sampling: quasi-stratified across the batch
      t_i = (rand + i/B) mod 1  for each batch element i
  - Sampling: per-position probabilistic unmasking (not top-k)
      p(unmask | masked at t, going to s) = (alpha_s - alpha_t) / (1 - alpha_t)

Encoding:
  - Pixel p ∈ [0, 255]  →  token p + BYTE_OFFSET (64)
  - Unconditional: [BOS, p_0, ..., p_783, EOS]          (length 786)
  - Conditional:   [BOS, class_tok, p_0, ..., p_783, EOS]  (length 787)
    class_tok = digit + 4   (reserved token slots 4-13)

Usage:
    python train_mnist_mdm.py
    python train_mnist_mdm.py --num-epochs 10 --batch-size 32 --hidden-dim 64
    python train_mnist_mdm.py --conditional
    python train_mnist_mdm.py --train-fraction 0.1   # quick smoke-test
    python train_mnist_mdm.py --amp                  # mixed-precision (CUDA)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from train_mdm import (
    ByteTabNetMDM,
    CheckpointManager,
    mdm_loss,
    get_device,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
)
from byte_tabnet import BYTE_OFFSET, VOCAB_SIZE, tabnet_sparsity_loss


# ============================================================================
# Constants
# ============================================================================

IMG_SIZE = 28 * 28          # 784 pixels
NUM_CLASSES = 10
DIGIT_TOKEN_OFFSET = 4      # digit d → token d+4  (reserved slots 4-13)


# ============================================================================
# MD4 Noise Schedule & Loss Weight
# ============================================================================

def md4_alpha(t: torch.Tensor) -> torch.Tensor:
    """
    Fraction of UNMASKED tokens at noise level t  (alpha_t in MD4).

    alpha_t = 1 - cos(π/2 · (1-t))

    At t=0 (clean):  alpha = 1  →  no tokens masked
    At t=1 (noisy):  alpha = 0  →  all tokens masked
    """
    return 1.0 - torch.cos((1.0 - t) * (math.pi / 2))


def md4_mask_rate(t: torch.Tensor) -> torch.Tensor:
    """Fraction of MASKED tokens at noise level t. = 1 - alpha_t."""
    return 1.0 - md4_alpha(t)


def md4_loss_weight(t: torch.Tensor) -> torch.Tensor:
    """
    ELBO-derived loss weight for the MD4 objective.

    w(t) = (π/2) · tan(π/2 · (1-t))

    This weight is derived as  w(t) = alpha_t' / (1 - alpha_t)
    where alpha_t' = d(alpha_t)/dt.

    Note: w→∞ as t→0 (weight blows up at the clean limit), so
    timesteps are sampled in (T_MIN, T_MAX) to avoid instability.
    """
    return (math.pi / 2) * torch.tan((1.0 - t) * (math.pi / 2))


# Safe bounds for t to keep the loss weight finite
T_MIN = 0.01
T_MAX = 0.99


# ============================================================================
# Encoding
#
# Each pixel p ∈ [0, 255] maps directly to token p + BYTE_OFFSET.
# Sequence layout:
#   Unconditional: [BOS, p_0, …, p_783, EOS]           length = IMG_SIZE + 2
#   Conditional:   [BOS, class_tok, p_0, …, p_783, EOS] length = IMG_SIZE + 3
# ============================================================================

SEQ_LEN_UNCOND = IMG_SIZE + 2   # BOS + 784 pixels + EOS
SEQ_LEN_COND   = IMG_SIZE + 3   # BOS + class_tok + 784 pixels + EOS


def encode_image(
    pixels: np.ndarray,
    label: Optional[int] = None,
    conditional: bool = False,
) -> np.ndarray:
    """
    Encode a flat (784,) uint8 pixel array into token IDs.

    Each pixel p is mapped to p + BYTE_OFFSET.

    Sequence layout:
        Unconditional: [BOS, p_0, …, p_783, EOS]
        Conditional:   [BOS, class_tok, p_0, …, p_783, EOS]
    """
    toks = (pixels.astype(np.int64) + BYTE_OFFSET).tolist()
    if conditional and label is not None:
        return np.array(
            [BOS_TOKEN_ID, DIGIT_TOKEN_OFFSET + label] + toks + [EOS_TOKEN_ID],
            dtype=np.int64,
        )
    return np.array([BOS_TOKEN_ID] + toks + [EOS_TOKEN_ID], dtype=np.int64)


# ============================================================================
# MNIST Loading
# ============================================================================

def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (train_images, train_labels, test_images, test_labels) as uint8 (N,784)."""

    # torchvision
    try:
        import torchvision
        print("Loading MNIST via torchvision...")
        tr = torchvision.datasets.MNIST(root="/tmp/mnist", train=True,  download=True)
        te = torchvision.datasets.MNIST(root="/tmp/mnist", train=False, download=True)
        return (
            tr.data.numpy().reshape(-1, IMG_SIZE),
            tr.targets.numpy(),
            te.data.numpy().reshape(-1, IMG_SIZE),
            te.targets.numpy(),
        )
    except Exception as e:
        print(f"[warn] torchvision failed ({type(e).__name__}: {e})")

    # HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading MNIST via HuggingFace datasets...")
        ds = load_dataset("mnist")
        def _np(split):
            imgs   = np.array([np.array(x["image"]).flatten() for x in ds[split]], dtype=np.uint8)
            labels = np.array([x["label"] for x in ds[split]], dtype=np.int32)
            return imgs, labels
        return *_np("train"), *_np("test")
    except Exception as e:
        print(f"[warn] HuggingFace datasets failed ({type(e).__name__}: {e})")

    # Manual download (Google Storage mirror)
    import gzip, struct, urllib.request
    print("Downloading MNIST from Google Storage...")
    base  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    cache = Path("/tmp/mnist_raw")
    cache.mkdir(exist_ok=True)

    def _get(fname):
        p = cache / fname
        if p.exists():
            with open(p, "rb") as f:
                if f.read(2) != b"\x1f\x8b":
                    p.unlink()  # corrupt cache
        if not p.exists():
            urllib.request.urlretrieve(base + fname, p)
        return p

    def _imgs(fname):
        with gzip.open(_get(fname), "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)

    def _lbls(fname):
        with gzip.open(_get(fname), "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)

    return (
        _imgs("train-images-idx3-ubyte.gz"), _lbls("train-labels-idx1-ubyte.gz"),
        _imgs("t10k-images-idx3-ubyte.gz"),  _lbls("t10k-labels-idx1-ubyte.gz"),
    )


# ============================================================================
# Dataset
# ============================================================================

class MNISTMDMDataset:
    """Wraps MNIST arrays into MD4 token sequences; all sequences have equal length."""

    def __init__(self, images, labels, conditional=False, device=torch.device("cpu")):
        self.images      = images
        self.labels      = labels
        self.conditional = conditional
        self.device      = device
        self.seq_len     = SEQ_LEN_COND if conditional else SEQ_LEN_UNCOND

    def __len__(self):
        return len(self.images)

    def make_batches(self, batch_size: int, shuffle: bool = True
                     ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield (token_seqs [long], attention_masks [float32]) batches."""
        idx = np.arange(len(self))
        if shuffle:
            np.random.shuffle(idx)
        for s in range(0, len(idx) - batch_size + 1, batch_size):
            bi = idx[s : s + batch_size]
            tokens = np.stack([
                encode_image(self.images[i], int(self.labels[i]), self.conditional)
                for i in bi
            ])
            masks = np.ones((batch_size, self.seq_len), dtype=np.float32)
            yield (
                torch.from_numpy(tokens).to(self.device),
                torch.from_numpy(masks).to(self.device),
            )


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MNISTMDMConfig:
    # Model
    embed_dim:          int   = 64
    hidden_dim:         int   = 64
    n_blocks:           int   = 6
    n_heads:            int   = 4
    n_steps:            int   = 3
    n_d:                int   = 64
    n_a:                int   = 64
    gamma:              float = 1.5
    n_shared:           int   = 2
    n_step:             int   = 2
    virtual_batch_size: int   = 32

    # Training
    num_epochs:     int   = 10
    batch_size:     int   = 32
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.01
    gradient_clip:  float = 1.0
    warmup_steps:   int   = 100
    noise_schedule: str   = "md4_cosine"

    # Regularisation
    sparsity_weight: float = 1e-3
    label_smoothing: float = 0.05

    # Logging / checkpointing
    log_every:      int = 10
    sample_every:   int = 50
    save_every:     int = 5
    checkpoint_dir: str = "./checkpoints_mnist_mdm"
    sample_dir:     str = "./samples_mnist_mdm"

    # Inference
    num_denoise_steps:  int   = 25
    sample_temperature: float = 1.0
    num_samples:        int   = 16

    # Data
    train_fraction: float = 0.5
    val_fraction:   float = 0.5
    conditional:    bool  = False

    # Hardware
    amp: bool = False

    # Misc
    seed: int = 42


# ============================================================================
# MD4 Loss
# ============================================================================

def md4_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    should_mask: torch.Tensor,
    tabnet_masks: torch.Tensor,
    t: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sparsity_weight: float = 1e-3,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    MD4 ELBO loss: w(t) · CE(model(x_t), x_0) at masked positions + TabNet sparsity.

    The weight w(t) = (π/2)·tan(π/2·(1-t)) is the key difference from a plain
    masked language model — it up-weights lightly-noised examples (small t)
    where the model must be more precise.

    Args:
        logits:          (batch, seq_len, vocab_size)
        targets:         (batch, seq_len) — original clean tokens
        should_mask:     (batch, seq_len) — True where tokens were masked
        tabnet_masks:    TabNet attention masks for sparsity regularisation
        t:               scalar timestep in (T_MIN, T_MAX)
        attention_mask:  (batch, seq_len) — 1 = real token, 0 = pad
        sparsity_weight: weight on TabNet entropy regularisation
        label_smoothing: label smoothing factor
    """
    ce       = mdm_loss(logits, targets, should_mask, attention_mask, label_smoothing)
    w        = md4_loss_weight(t)
    sparsity = tabnet_sparsity_loss(tabnet_masks)
    return w * ce + sparsity_weight * sparsity


# ============================================================================
# MD4 Sampling
# ============================================================================

@torch.no_grad()
def md4_sample(
    model: ByteTabNetMDM,
    seq_len: int,
    num_steps: int,
    temperature: float,
    device: torch.device,
    pixel_start: int = 1,
) -> torch.Tensor:
    """
    Generate a sequence via MD4 iterative probabilistic unmasking.

    At each reverse step from t=1 (fully masked) to t≈0 (clean), each
    currently-masked pixel position is independently unmasked with probability:

        p_unmask = (alpha_s - alpha_t) / (1 - alpha_t)

    where alpha_t = fraction of unmasked tokens at time t, and s < t.

    Args:
        model:       Trained ByteTabNetMDM (in eval mode).
        seq_len:     Total sequence length (including BOS / class token).
        num_steps:   Number of reverse steps.
        temperature: Softmax temperature for token sampling.
        device:      Torch device.
        pixel_start: Index from which pixel tokens start (1 uncond, 2 cond).

    Returns:
        Token IDs of shape (1, seq_len).
    """
    model.eval()

    # Start fully masked; index 0 is always BOS
    ids = torch.full((1, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device)
    ids[0, 0] = BOS_TOKEN_ID
    attn = torch.ones((1, seq_len), dtype=torch.float32, device=device)

    pixel_region = torch.zeros(seq_len, dtype=torch.bool, device=device)
    pixel_region[pixel_start:] = True

    # Reverse time: t goes from 1.0 down to 0.0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur  = timesteps[i]       # higher t = more masked
        t_next = timesteps[i + 1]   # lower  t = cleaner

        alpha_t = md4_alpha(t_cur)
        alpha_s = md4_alpha(t_next)

        # Probability that a currently masked token gets revealed this step
        denom   = (1.0 - alpha_t).clamp(min=1e-8)
        p_unmask = ((alpha_s - alpha_t) / denom).clamp(0.0, 1.0)

        logits, _ = model(ids, t_cur, attn)
        logits = logits / temperature

        probs   = F.softmax(logits[0], dim=-1)   # (seq_len, vocab)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        is_masked     = (ids[0] == MASK_TOKEN_ID) & pixel_region
        should_unmask = (torch.rand(seq_len, device=device) < p_unmask) & is_masked

        ids = torch.where(should_unmask, sampled, ids[0]).unsqueeze(0)

    # Any remaining MASK tokens → sample from near-clean model
    still_masked = (ids[0] == MASK_TOKEN_ID) & pixel_region
    if still_masked.any():
        logits, _ = model(ids, torch.tensor(T_MIN, device=device), attn)
        probs   = F.softmax(logits[0] / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        ids = torch.where(still_masked, sampled, ids[0]).unsqueeze(0)

    return ids


@torch.no_grad()
def md4_sample_conditional(
    model: ByteTabNetMDM,
    digit: int,
    seq_len: int,
    num_steps: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    MD4 generation conditioned on a digit class.

    Prepends [BOS, class_token] and unmasks only pixel positions (index 2+).
    """
    model.eval()

    init = ([BOS_TOKEN_ID, DIGIT_TOKEN_OFFSET + digit]
            + [MASK_TOKEN_ID] * (seq_len - 2))
    ids  = torch.tensor([init], dtype=torch.long, device=device)
    # class token at index 1 is fixed; BOS at 0 is fixed
    # pixel_start = 2 to leave them untouched
    return md4_sample(model, seq_len, num_steps, temperature, device, pixel_start=2)


# ============================================================================
# Visualisation
# ============================================================================

def save_image_grid(images: np.ndarray, path: str, nrow: int = 4, border: int = 2):
    """Save a (N, 28, 28) uint8 array as a grid PNG."""
    try:
        from PIL import Image
    except ImportError:
        print("[warn] PIL unavailable; skip grid. pip install Pillow")
        return

    n    = len(images)
    ncol = (n + nrow - 1) // nrow
    h    = ncol * (28 + border) + border
    w    = nrow * (28 + border) + border
    grid = np.full((h, w), 200, dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // nrow; c = idx % nrow
        grid[border + r*(28+border) : border + r*(28+border)+28,
             border + c*(28+border) : border + c*(28+border)+28] = img

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="L").save(path)
    print(f"  Saved grid → {path}")


# ============================================================================
# Helpers
# ============================================================================



def generate_samples(model, cfg: MNISTMDMConfig, device, n=None) -> np.ndarray:
    """Generate n samples; returns raw token ID arrays of shape (n, seq_len)."""
    n       = n or cfg.num_samples
    seq_len = SEQ_LEN_COND if cfg.conditional else SEQ_LEN_UNCOND
    seqs    = []
    model.eval()

    for i in range(n):
        if cfg.conditional:
            ids = md4_sample_conditional(
                model, i % NUM_CLASSES, seq_len,
                cfg.num_denoise_steps, cfg.sample_temperature, device,
            )
        else:
            ids = md4_sample(
                model, seq_len,
                cfg.num_denoise_steps, cfg.sample_temperature, device,
            )
        seqs.append(ids[0].cpu().numpy())
    return np.stack(seqs)




def _evaluate(model, dataset, cfg, device, num_batches=10) -> float:
    """Average MD4 loss over a dataset subset, evaluated at t = 0.25/0.5/0.75."""
    model.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for tokens, masks in dataset.make_batches(cfg.batch_size, shuffle=False):
            if count >= num_batches:
                break
            for t_val in (0.25, 0.5, 0.75):
                t          = torch.tensor(t_val, device=device)
                mask_rate  = md4_mask_rate(t)
                is_special = (tokens == PAD_TOKEN_ID) | (tokens == BOS_TOKEN_ID) | (tokens == EOS_TOKEN_ID)
                should_mask = (torch.rand_like(tokens, dtype=torch.float32) < mask_rate.item()) & ~is_special
                masked_ids  = torch.where(should_mask, MASK_TOKEN_ID, tokens)

                logits, tabnet_masks = model(masked_ids, t, masks)
                loss = md4_loss(logits, tokens, should_mask, tabnet_masks, t,
                                masks, cfg.sparsity_weight, cfg.label_smoothing)
                total += loss.item()
            count += 1

    model.train()
    return total / max(count * 3, 1)


# ============================================================================
# Final showcase  (~100 images)
# ============================================================================

def generate_final_showcase(model, cfg: MNISTMDMConfig, device, n_samples=100) -> np.ndarray:
    """
    Generate ~100 token sequences and return them as a (total, seq_len) array.

    Unconditional: 100 sequences
    Conditional:   10 sequences per digit class (100 total)
    """
    n_samples = ((n_samples + 9) // 10) * 10
    seq_len   = SEQ_LEN_COND if cfg.conditional else SEQ_LEN_UNCOND

    per_class = n_samples // NUM_CLASSES if cfg.conditional else None
    total     = per_class * NUM_CLASSES if cfg.conditional else n_samples
    mode      = f"conditional, {per_class}/class" if cfg.conditional else "unconditional"

    print(f"\nGenerating {total} final showcase sequences ({mode}) …")

    seqs: List[np.ndarray] = []
    model.eval()
    pbar = tqdm(total=total, desc="Generating", unit="seq")

    if cfg.conditional:
        for digit in range(NUM_CLASSES):
            for _ in range(per_class):
                ids = md4_sample_conditional(
                    model, digit, seq_len,
                    cfg.num_denoise_steps, cfg.sample_temperature, device,
                )
                seqs.append(ids[0].cpu().numpy())
                pbar.update(1)
    else:
        for _ in range(total):
            ids = md4_sample(
                model, seq_len,
                cfg.num_denoise_steps, cfg.sample_temperature, device,
            )
            seqs.append(ids[0].cpu().numpy())
            pbar.update(1)

    pbar.close()
    return np.stack(seqs)


# ============================================================================
# Training
# ============================================================================

def train(cfg: MNISTMDMConfig) -> ByteTabNetMDM:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    print(f"Device: {device}")
    if cfg.amp and device.type != "cuda":
        print("[warn] --amp requires CUDA; disabling.")
        cfg.amp = False

    # --- Data ---
    tr_imgs, tr_lbl, te_imgs, te_lbl = load_mnist()
    print(f"MNIST — train: {len(tr_imgs):,}  test: {len(te_imgs):,}")

    if cfg.train_fraction < 1.0:
        n = int(len(tr_imgs) * cfg.train_fraction)
        p = np.random.permutation(len(tr_imgs))[:n]
        tr_imgs, tr_lbl = tr_imgs[p], tr_lbl[p]

    n_val        = int(len(tr_imgs) * cfg.val_fraction)
    va_imgs, va_lbl = tr_imgs[:n_val], tr_lbl[:n_val]
    tr_imgs, tr_lbl = tr_imgs[n_val:], tr_lbl[n_val:]
    print(f"Splits — train: {len(tr_imgs):,}  val: {n_val:,}  test: {len(te_imgs):,}")

    seq_len  = SEQ_LEN_COND if cfg.conditional else SEQ_LEN_UNCOND
    train_ds = MNISTMDMDataset(tr_imgs, tr_lbl, cfg.conditional, device)
    val_ds   = MNISTMDMDataset(va_imgs, va_lbl, cfg.conditional, device)
    test_ds  = MNISTMDMDataset(te_imgs, te_lbl, cfg.conditional, device)

    # --- Model ---
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Sequence length: {seq_len}  conditional: {cfg.conditional}  AMP: {cfg.amp}\n")

    # --- Optimizer / scheduler ---
    steps_per_epoch = max(len(train_ds) // cfg.batch_size, 1)
    total_steps     = steps_per_epoch * cfg.num_epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.9,
        max_momentum=0.99,
        div_factor=0.68,
        final_div_factor=10000.0,
        three_phase=True,
        last_epoch=-1,
    )
    scaler = torch.amp.GradScaler(enabled=cfg.amp)

    ckpt_manager = CheckpointManager(cfg.checkpoint_dir, keep_best_k=3)
    Path(cfg.sample_dir).mkdir(parents=True, exist_ok=True)

    print(f"Training: {cfg.num_epochs} epochs  "
          f"{steps_per_epoch} steps/epoch  {total_steps} total\n")

    global_step   = 0
    best_val_loss = float("inf")
    print(model)
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        pbar = tqdm(
            train_ds.make_batches(cfg.batch_size, shuffle=True),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
        )
        #convert to string -> use the the tokenizer;
        for tokens, attn_mask in pbar:
            B = tokens.shape[0]

            # --- MD4 quasi-stratified timestep sampling ---
            # Each batch element gets t_i = (rand + i/B) mod 1, then clamped to (T_MIN, T_MAX)
            rand = torch.rand(1, device=device)
            t_batch = torch.fmod(
                rand + (1 + torch.arange(B, device=device)).float() / B, 1.0
            ).clamp(T_MIN, T_MAX)   # (B,)

            # For the model forward (scalar t), use the batch mean
            t_scalar = t_batch.mean()

            # --- Apply per-element masking using each element's own t ---
            is_special = (
                (tokens == PAD_TOKEN_ID) |
                (tokens == BOS_TOKEN_ID) |
                (tokens == EOS_TOKEN_ID)
            )
            # t_batch[:, None] broadcasts to (B, seq_len)
            mask_rates  = md4_mask_rate(t_batch)[:, None]
            should_mask = (torch.rand_like(tokens, dtype=torch.float32) < mask_rates) & ~is_special
            masked_ids  = torch.where(should_mask, MASK_TOKEN_ID, tokens)

            optimizer.zero_grad()

            #with torch.amp.autocast(device_type=device.type, enabled=cfg.amp):
            logits, tabnet_masks = model(masked_ids, t_scalar, attn_mask)
            loss = md4_loss(
                logits, tokens, should_mask, tabnet_masks, t_scalar,
                attn_mask, cfg.sparsity_weight, cfg.label_smoothing,
            )
            #import pdb; pdb.set_trace()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % cfg.log_every == 0:
                print(f"  step {global_step:6d}  loss={loss.item():.4f}"
                      f"  lr={scheduler.get_last_lr()[0]:.2e}")

            if global_step % cfg.sample_every == 0:
                generate_samples(model, cfg, device)
                model.train()

        avg_train = epoch_loss / max(num_batches, 1)
        val_loss  = _evaluate(model, val_ds, cfg, device, num_batches=10)
        print(f"Epoch {epoch+1:3d}/{cfg.num_epochs}  "
              f"train={avg_train:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_manager.save_checkpoint(
                model, epoch, val_loss,
                metadata={"cfg": cfg.__dict__, "global_step": global_step},
                is_best=True,
            )
            print(f"  → new best  val={val_loss:.4f}")

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_manager.save_checkpoint(
                model, epoch, val_loss,
                metadata={"cfg": cfg.__dict__, "global_step": global_step},
            )

    # --- Final evaluation & showcase ---
    print("\nEvaluating on test set...")
    test_loss = _evaluate(model, test_ds, cfg, device, num_batches=50)
    print(f"Test loss: {test_loss:.4f}")

    generate_final_showcase(model, cfg, device, n_samples=100)

    print(f"\nDone!  best_val_loss={best_val_loss:.4f}")
    print(f"  Checkpoints : {cfg.checkpoint_dir}")
    print(f"  Sample grids: {cfg.sample_dir}")
    return model


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train ByteTabNet MD4 on MNIST (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embed-dim",  type=int,   default=64)
    p.add_argument("--hidden-dim", type=int,   default=64)
    p.add_argument("--n-blocks",   type=int,   default=6)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--num-epochs",    type=int,   default=2)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps",  type=int,   default=100)
    p.add_argument("--amp", action="store_true", help="Mixed-precision (CUDA only)")
    p.add_argument("--num-denoise-steps",  type=int,   default=25)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--log-every",      type=int, default=10)
    p.add_argument("--sample-every",   type=int, default=50)
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints_mnist_mdm")
    p.add_argument("--sample-dir",     type=str, default="./samples_mnist_mdm")
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = MNISTMDMConfig(
        embed_dim          = args.embed_dim,
        hidden_dim         = args.hidden_dim,
        n_blocks           = args.n_blocks,
        n_heads            = args.n_heads,
        num_epochs         = args.num_epochs,
        batch_size         = args.batch_size,
        learning_rate      = args.learning_rate,
        warmup_steps       = args.warmup_steps,
        num_denoise_steps  = args.num_denoise_steps,
        sample_temperature = args.sample_temperature,
        conditional        = args.conditional,
        log_every          = args.log_every,
        sample_every       = args.sample_every,
        checkpoint_dir     = args.checkpoint_dir,
        sample_dir         = args.sample_dir,
        train_fraction     = args.train_fraction,
        amp                = args.amp,
        seed               = args.seed,
    )
    train(cfg)
