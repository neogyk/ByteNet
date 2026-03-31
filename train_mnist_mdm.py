"""
MNIST Generation with ByteTabNet Masked Diffusion Model (MDM)

Trains ByteTabNetMDM to generate MNIST handwritten digit images using
masked diffusion (MaskGIT-style) training.

Encoding:
  - Each 28×28 grayscale image → flat sequence of 784 pixel byte tokens
  - Pixel value p (0-255) → token ID p + 64  (byte vocabulary offset)
  - Unconditional:  [BOS, p_0, ..., p_783, EOS]        (length 786)
  - Conditional:    [BOS, class_tok, p_0, ..., p_783, EOS]  (length 787)
    where class_tok = digit_label + 4  (reserved token slots 4-13)

Training objective:
  - Uniformly sample a noise level t ~ U(0.01, 1.0)
  - Mask pixel tokens at cosine-scheduled rate  mask_rate = 1 - cos(π/2 · t)
  - Predict original pixel values at masked positions (cross-entropy)

Generation:
  - Start from fully-masked sequence (+ BOS, optional class token)
  - Iteratively unmask the highest-confidence positions over num_denoise_steps
  - Save generated image grids as PNG

Usage:
    python train_mnist_mdm.py
    python train_mnist_mdm.py --num-epochs 50 --batch-size 64 --hidden-dim 256
    python train_mnist_mdm.py --conditional   # class-conditional generation
    python train_mnist_mdm.py --train-fraction 0.1  # quick smoke-test
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import numpy as np
from tqdm import tqdm

from train_mdm import (
    ByteTabNetMDM,
    CheckpointManager,
    cosine_noise_schedule,
    iterative_demask,
    linear_noise_schedule,
    mdm_loss_with_sparsity,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
)


# ============================================================================
# Constants
# ============================================================================

IMG_SIZE = 28 * 28          # 784 pixels per MNIST image
BYTE_OFFSET = 64            # bytes are at token IDs 64..319 in the vocabulary
NUM_CLASSES = 10

# Special class tokens: use reserved slots 4-13  (digit d → token d + 4)
DIGIT_TOKEN_OFFSET = 4

# Unconditional sequence: [BOS, p_0, ..., p_783, EOS]
SEQ_LEN_UNCOND = IMG_SIZE + 2  # 786

# Conditional sequence: [BOS, class_tok, p_0, ..., p_783, EOS]
SEQ_LEN_COND = IMG_SIZE + 3    # 787


# ============================================================================
# Pixel ↔ Token Encoding
# ============================================================================

def pixels_to_tokens(pixels: np.ndarray) -> np.ndarray:
    """
    Map uint8 pixel values to byte token IDs.

    pixel p (0..255)  →  token p + BYTE_OFFSET
    """
    return pixels.astype(np.int32) + BYTE_OFFSET


def tokens_to_pixels(tokens: np.ndarray) -> np.ndarray:
    """
    Map byte token IDs back to uint8 pixel values.

    Clips to valid [0, 255] range before casting.
    """
    return np.clip(tokens.astype(np.int32) - BYTE_OFFSET, 0, 255).astype(np.uint8)


def encode_image(
    pixels: np.ndarray,
    label: Optional[int] = None,
    conditional: bool = False,
) -> np.ndarray:
    """
    Encode a flattened 28×28 image into a token sequence.

    Unconditional: [BOS, p_0, ..., p_783, EOS]
    Conditional:   [BOS, class_token, p_0, ..., p_783, EOS]

    Args:
        pixels: uint8 array of shape (784,).
        label: Digit label 0-9 (required when conditional=True).
        conditional: Whether to prepend a class token.

    Returns:
        int32 token array of shape (SEQ_LEN,).
    """
    pixel_tokens = pixels_to_tokens(pixels).tolist()
    if conditional and label is not None:
        seq = [BOS_TOKEN_ID, DIGIT_TOKEN_OFFSET + label] + pixel_tokens + [EOS_TOKEN_ID]
    else:
        seq = [BOS_TOKEN_ID] + pixel_tokens + [EOS_TOKEN_ID]
    return np.array(seq, dtype=np.int32)


def decode_image(token_seq: np.ndarray, conditional: bool = False) -> np.ndarray:
    """
    Decode a generated token sequence back to a 28×28 uint8 image.

    Skips BOS, optional class token, and stops at EOS / PAD / MASK.

    Args:
        token_seq: 1-D int array (may contain special tokens).
        conditional: Whether to skip an extra class token after BOS.

    Returns:
        uint8 array of shape (28, 28).
    """
    start = 2 if conditional else 1   # skip BOS (+ class token if conditional)

    pixel_tokens: List[int] = []
    for t in token_seq[start:]:
        t = int(t)
        if t in (EOS_TOKEN_ID, PAD_TOKEN_ID, MASK_TOKEN_ID):
            break
        if t >= BYTE_OFFSET:
            pixel_tokens.append(t)
        if len(pixel_tokens) == IMG_SIZE:
            break

    # Pad with "pixel 0" if the sequence is shorter than expected
    pixel_tokens.extend([BYTE_OFFSET] * (IMG_SIZE - len(pixel_tokens)))
    pixel_tokens = pixel_tokens[:IMG_SIZE]

    return tokens_to_pixels(np.array(pixel_tokens)).reshape(28, 28)


# ============================================================================
# MNIST Dataset Loading
# ============================================================================

def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset via torchvision, HuggingFace datasets, or manual download.

    Returns:
        (train_images, train_labels, test_images, test_labels)
        - images: uint8 arrays of shape (N, 784), values in [0, 255]
        - labels: int arrays of shape (N,)
    """
    # --- torchvision ---
    try:
        import torchvision
        print("Loading MNIST via torchvision...")
        train_ds = torchvision.datasets.MNIST(root="/tmp/mnist", train=True,  download=True)
        test_ds  = torchvision.datasets.MNIST(root="/tmp/mnist", train=False, download=True)
        train_imgs   = train_ds.data.numpy().reshape(-1, IMG_SIZE)
        train_labels = train_ds.targets.numpy()
        test_imgs    = test_ds.data.numpy().reshape(-1, IMG_SIZE)
        test_labels  = test_ds.targets.numpy()
        return train_imgs, train_labels, test_imgs, test_labels
    except Exception as e:
        print(f"[warn] torchvision load failed ({type(e).__name__}: {e}); trying next loader...")

    # --- HuggingFace datasets ---
    try:
        from datasets import load_dataset
        print("Loading MNIST via HuggingFace datasets...")
        ds = load_dataset("mnist")

        def _to_np(split: str):
            imgs   = np.array([
                np.array(item["image"]).flatten() for item in ds[split]
            ], dtype=np.uint8)
            labels = np.array([item["label"] for item in ds[split]], dtype=np.int32)
            return imgs, labels

        return *_to_np("train"), *_to_np("test")
    except Exception as e:
        print(f"[warn] HuggingFace datasets load failed ({type(e).__name__}: {e}); trying manual download...")

    # --- Manual binary download ---
    import gzip, struct, urllib.request
    print("Downloading MNIST manually (IDX binary format)...")
    # LeCun's original URL is unreliable; use Google Storage mirror instead.
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    cache = Path("/tmp/mnist_raw")
    cache.mkdir(exist_ok=True)

    def _download(fname: str) -> Path:
        p = cache / fname
        if p.exists():
            # Validate: a real gzip file starts with the magic bytes 1f 8b
            with open(p, "rb") as fh:
                magic = fh.read(2)
            if magic != b"\x1f\x8b":
                print(f"  Cached {fname} appears corrupt; re-downloading...")
                p.unlink()
        if not p.exists():
            print(f"  Downloading {fname} ...")
            urllib.request.urlretrieve(base + fname, p)
        return p

    def _fetch_images(fname: str) -> np.ndarray:
        p = _download(fname)
        with gzip.open(p, "rb") as f:
            _, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols)

    def _fetch_labels(fname: str) -> np.ndarray:
        p = _download(fname)
        with gzip.open(p, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int32)

    train_imgs   = _fetch_images("train-images-idx3-ubyte.gz")
    train_labels = _fetch_labels("train-labels-idx1-ubyte.gz")
    test_imgs    = _fetch_images("t10k-images-idx3-ubyte.gz")
    test_labels  = _fetch_labels("t10k-labels-idx1-ubyte.gz")
    return train_imgs, train_labels, test_imgs, test_labels


# ============================================================================
# Dataset Iterator
# ============================================================================

class MNISTMDMDataset:
    """
    Wraps MNIST image arrays into MDM token sequences.

    Each image is encoded as [BOS, pixel_0, ..., pixel_783, EOS] (unconditional)
    or [BOS, class_token, pixel_0, ..., pixel_783, EOS] (conditional).

    Attention masks are all 1s (no padding—every sequence has the same length).
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        conditional: bool = False,
    ):
        self.images      = images       # (N, 784) uint8
        self.labels      = labels       # (N,) int
        self.conditional = conditional
        self.seq_len     = SEQ_LEN_COND if conditional else SEQ_LEN_UNCOND

    def __len__(self) -> int:
        return len(self.images)

    def make_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[Tuple[jax.Array, jax.Array]]:
        """
        Yield (token_seqs, attention_masks) JAX arrays of shape:
            token_seqs:      (batch_size, seq_len)  int32
            attention_masks: (batch_size, seq_len)  float32  (all 1s)

        Incomplete final batches are dropped for consistent shapes under jit.
        """
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices) - batch_size + 1, batch_size):
            idx = indices[start : start + batch_size]
            tokens = np.stack([
                encode_image(self.images[i], int(self.labels[i]), self.conditional)
                for i in idx
            ])  # (batch, seq_len)
            masks = np.ones((batch_size, self.seq_len), dtype=np.float32)
            yield jnp.array(tokens), jnp.array(masks)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MNISTMDMConfig:
    """All hyperparameters for MNIST MDM training in one place."""

    # Model architecture
    embed_dim:           int   = 64
    hidden_dim:          int   = 256
    n_blocks:            int   = 6
    n_heads:             int   = 4
    n_steps:             int   = 5      # TabNet decision steps
    n_d:                 int   = 64
    n_a:                 int   = 64
    gamma:               float = 1.5
    n_shared:            int   = 2
    n_step:              int   = 2
    virtual_batch_size:  int   = 128

    # Training
    num_epochs:     int   = 100
    batch_size:     int   = 64
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.01
    gradient_clip:  float = 1.0
    warmup_steps:   int   = 1000
    noise_schedule: str   = "cosine"   # "cosine" | "linear"

    # Regularisation
    sparsity_weight: float = 1e-3
    label_smoothing: float = 0.05

    # Logging / checkpointing
    log_every:      int = 100
    sample_every:   int = 500     # steps between generating sample grids
    save_every:     int = 5       # epochs between periodic checkpoints
    checkpoint_dir: str = "./checkpoints_mnist_mdm"
    sample_dir:     str = "./samples_mnist_mdm"

    # Inference
    num_denoise_steps:  int   = 25
    sample_temperature: float = 1.0
    num_samples:        int   = 16    # images per generated grid (should be ≥ nrow²)

    # Data
    train_fraction: float = 1.0   # fraction of MNIST training set to use
    val_fraction:   float = 0.1   # fraction of (possibly sub-sampled) train set
    conditional:    bool  = False  # class-conditional generation

    # Misc
    seed: int = 42


# ============================================================================
# Visualisation
# ============================================================================

def save_image_grid(
    images: np.ndarray,
    path: str,
    nrow: int = 4,
    border: int = 2,
) -> None:
    """
    Save a grid of 28×28 grayscale images to a PNG file.

    Args:
        images: uint8 array of shape (N, 28, 28).
        path: Output file path.
        nrow: Number of columns in the grid.
        border: Pixel gap between images.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[warn] PIL not available; skipping image grid. pip install Pillow")
        return

    n    = len(images)
    ncol = (n + nrow - 1) // nrow
    h    = ncol * (28 + border) + border
    w    = nrow * (28 + border) + border
    grid = np.full((h, w), 200, dtype=np.uint8)  # light-grey background

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        y = border + row * (28 + border)
        x = border + col * (28 + border)
        grid[y : y + 28, x : x + 28] = img

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="L").save(path)
    print(f"  Saved image grid → {path}")


# ============================================================================
# Training Step (JIT-compiled via closure)
# ============================================================================

def make_train_step(optimizer, noise_fn, sparsity_weight: float, label_smoothing: float):
    """
    Return a JIT-compiled training step function that closes over the optimizer
    and hyperparameters so they are treated as compile-time constants.
    """

    @eqx.filter_jit
    def train_step(
        model: ByteTabNetMDM,
        opt_state,
        token_seqs: jax.Array,
        attention_mask: jax.Array,
        t: jax.Array,
        mask_key: jax.Array,
    ):
        print(model)
        def loss_fn(model):
            mask_rate = noise_fn(t)
            rand = jrandom.uniform(mask_key, token_seqs.shape)
            should_mask = rand < mask_rate

            # Never mask special tokens (BOS, EOS, PAD, digit class token)
            is_special = (
                (token_seqs == PAD_TOKEN_ID) |
                (token_seqs == BOS_TOKEN_ID) |
                (token_seqs == EOS_TOKEN_ID)
            )
            should_mask = should_mask & ~is_special

            masked_ids = jnp.where(should_mask, MASK_TOKEN_ID, token_seqs)

            logits, tabnet_masks, _ = model(
                masked_ids, t, attention_mask, inference=False,
            )
            loss = mdm_loss_with_sparsity(
                logits, token_seqs, should_mask, tabnet_masks,
                attention_mask, sparsity_weight, label_smoothing,
            )
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    return train_step


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model: ByteTabNetMDM,
    dataset: MNISTMDMDataset,
    batch_size: int,
    sparsity_weight: float,
    noise_fn,
    num_batches: int = 10,
) -> float:
    """
    Average MDM loss over a subset of the dataset.

    Evaluates at t = 0.25, 0.5, 0.75 and averages for a stable estimate.
    """
    key = jrandom.PRNGKey(0)
    total, count = 0.0, 0

    for tokens, masks in dataset.make_batches(batch_size, shuffle=False):
        if count >= num_batches:
            break
        for t_val in (0.25, 0.5, 0.75):
            key, mk = jrandom.split(key)
            t         = jnp.array(t_val)
            mask_rate = noise_fn(t)
            rand      = jrandom.uniform(mk, tokens.shape)
            is_special = (
                (tokens == PAD_TOKEN_ID) |
                (tokens == BOS_TOKEN_ID) |
                (tokens == EOS_TOKEN_ID)
            )
            should_mask = (rand < mask_rate) & ~is_special
            masked_ids  = jnp.where(should_mask, MASK_TOKEN_ID, tokens)

            logits, tabnet_masks, _ = model(masked_ids, t, masks, inference=True)
            loss = mdm_loss_with_sparsity(
                logits, tokens, should_mask, tabnet_masks,
                masks, sparsity_weight,
            )
            total += float(loss)
        count += 1

    return total / max(count * 3, 1)


# ============================================================================
# Generation
# ============================================================================

def _iterative_demask_conditional(
    model: ByteTabNetMDM,
    digit: int,
    seq_len: int,
    num_steps: int,
    temperature: float,
    key: jax.Array,
) -> jax.Array:
    """
    Iterative demasking for class-conditional generation.

    Keeps BOS + class token fixed; only unmasks pixel positions.

    Returns:
        Token IDs of shape (1, seq_len).
    """
    # Initial sequence: [BOS, class_token, MASK, ..., MASK]
    init = [BOS_TOKEN_ID, DIGIT_TOKEN_OFFSET + digit] + [MASK_TOKEN_ID] * (seq_len - 2)
    ids  = jnp.array([init], dtype=jnp.int32)               # (1, seq_len)
    mask = jnp.ones((1, seq_len), dtype=jnp.float32)

    # Pixel positions begin at index 2
    pixel_region = jnp.zeros(seq_len, dtype=jnp.bool_)
    pixel_region = pixel_region.at[2:].set(True)

    n_pixels = seq_len - 2  # positions eligible for unmasking

    for step in range(num_steps):
        key, subkey = jrandom.split(key)

        t_cur  = 1.0 - step / num_steps
        t_next = max(1.0 - (step + 1) / num_steps, 0.0)

        logits, _, _ = model(ids, jnp.array(t_cur), mask, inference=True)
        logits = logits / temperature

        probs      = jax.nn.softmax(logits[0], axis=-1)          # (seq_len, vocab)
        sampled    = jrandom.categorical(subkey, logits[0], axis=-1)  # (seq_len,)
        confidence = jnp.max(probs, axis=-1)                      # (seq_len,)

        is_masked  = (ids[0] == MASK_TOKEN_ID) & pixel_region
        confidence = jnp.where(is_masked, confidence, -1.0)

        cur_rate  = cosine_noise_schedule(jnp.array(t_cur))
        next_rate = cosine_noise_schedule(jnp.array(t_next))
        n_unmask  = jnp.maximum(
            jnp.round((cur_rate - next_rate) * n_pixels).astype(jnp.int32), 1
        )
        n_unmask = jnp.minimum(n_unmask, jnp.sum(is_masked))

        sorted_idx  = jnp.argsort(-confidence)
        ranks       = jnp.zeros(seq_len, dtype=jnp.int32)
        ranks       = ranks.at[sorted_idx].set(jnp.arange(seq_len))
        should_unmask = (ranks < n_unmask) & is_masked

        ids = jnp.where(should_unmask, sampled, ids[0])[None, :]

    return ids


def generate_samples(
    model: ByteTabNetMDM,
    cfg: MNISTMDMConfig,
    key: jax.Array,
) -> np.ndarray:
    """
    Generate ``cfg.num_samples`` MNIST images via iterative demasking.

    For conditional mode the samples cycle through digit classes 0-9.

    Returns:
        uint8 array of shape (num_samples, 28, 28).
    """
    seq_len = SEQ_LEN_COND if cfg.conditional else SEQ_LEN_UNCOND
    images  = []

    for i in range(cfg.num_samples):
        key, subkey = jrandom.split(key)

        if cfg.conditional:
            digit     = i % NUM_CLASSES
            token_ids = _iterative_demask_conditional(
                model, digit, seq_len,
                cfg.num_denoise_steps, cfg.sample_temperature, subkey,
            )
        else:
            token_ids = iterative_demask(
                model, seq_len,
                cfg.num_denoise_steps, cfg.sample_temperature,
                key=subkey,
            )

        img = decode_image(np.array(token_ids[0]), conditional=cfg.conditional)
        images.append(img)

    return np.stack(images)


# ============================================================================
# Final showcase  (≈100 images)
# ============================================================================

def generate_final_showcase(
    model: ByteTabNetMDM,
    cfg: MNISTMDMConfig,
    key: jax.Array,
    n_samples: int = 100,
) -> np.ndarray:
    """
    Generate ~100 images after training and save organised grids to disk.

    Unconditional:
        Generates ``n_samples`` images and saves a single 10-column grid
        → <sample_dir>/final_showcase.png

    Conditional:
        Rounds ``n_samples`` up to the nearest multiple of 10 and generates
        an equal number of samples per digit class (one class per grid row),
        so the grid reads left-to-right as samples of digits 0-9.
        → <sample_dir>/final_showcase.png          (full 10-column grid)
        → <sample_dir>/final_showcase_d{0-9}.png   (one 1-row strip per class)

    A progress bar is shown during generation.

    Args:
        model:     Trained ByteTabNetMDM (in inference mode—no grad tracking).
        cfg:       Training config (conditional flag, denoise steps, temperature, …).
        key:       PRNG key.
        n_samples: Target number of images (adjusted to a multiple of 10).

    Returns:
        uint8 array of shape (n_generated, 28, 28).
    """
    seq_len = SEQ_LEN_COND if cfg.conditional else SEQ_LEN_UNCOND

    # Round up to a multiple of 10 for clean grids
    n_samples = ((n_samples + 9) // 10) * 10

    if cfg.conditional:
        per_class = n_samples // NUM_CLASSES   # samples per digit
        total     = per_class * NUM_CLASSES
    else:
        total     = n_samples
        per_class = None

    print(f"\nGenerating {total} final showcase images "
          f"({'conditional, ' + str(per_class) + '/class' if cfg.conditional else 'unconditional'}) …")

    images: List[np.ndarray] = []
    labels: List[int]        = []   # only used for conditional

    pbar = tqdm(total=total, desc="Generating", unit="img")

    if cfg.conditional:
        # Generate per_class images for each digit 0-9
        for digit in range(NUM_CLASSES):
            for _ in range(per_class):
                key, subkey = jrandom.split(key)
                token_ids = _iterative_demask_conditional(
                    model, digit, seq_len,
                    cfg.num_denoise_steps, cfg.sample_temperature, subkey,
                )
                images.append(decode_image(np.array(token_ids[0]), conditional=True))
                labels.append(digit)
                pbar.update(1)
    else:
        for _ in range(total):
            key, subkey = jrandom.split(key)
            token_ids = iterative_demask(
                model, seq_len,
                cfg.num_denoise_steps, cfg.sample_temperature,
                key=subkey,
            )
            images.append(decode_image(np.array(token_ids[0]), conditional=False))
            pbar.update(1)

    pbar.close()

    images_arr = np.stack(images)  # (total, 28, 28)

    # --- Main grid (10 columns) ---
    grid_path = os.path.join(cfg.sample_dir, "final_showcase.png")
    save_image_grid(images_arr, grid_path, nrow=10)

    # --- Per-class strips for conditional ---
    if cfg.conditional:
        for digit in range(NUM_CLASSES):
            digit_imgs = images_arr[digit * per_class : (digit + 1) * per_class]
            strip_path = os.path.join(cfg.sample_dir, f"final_showcase_d{digit}.png")
            save_image_grid(digit_imgs, strip_path, nrow=per_class)

    return images_arr


# ============================================================================
# Main Training Function
# ============================================================================

def train(cfg: MNISTMDMConfig) -> ByteTabNetMDM:
    """Full training run.  Returns the final model."""
    print(f"JAX devices: {jax.devices()}")
    key = jrandom.PRNGKey(cfg.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_imgs, train_labels, test_imgs, test_labels = load_mnist()
    print(f"MNIST loaded — train: {len(train_imgs):,}  test: {len(test_imgs):,}")

    # Optional sub-sampling
    if cfg.train_fraction < 1.0:
        n_keep  = int(len(train_imgs) * cfg.train_fraction)
        rng     = np.random.RandomState(cfg.seed)
        perm    = rng.permutation(len(train_imgs))[:n_keep]
        train_imgs   = train_imgs[perm]
        train_labels = train_labels[perm]

    # Validation split (carved from training set)
    n_val        = int(len(train_imgs) * cfg.val_fraction)
    val_imgs,   val_labels   = train_imgs[:n_val],  train_labels[:n_val]
    train_imgs, train_labels = train_imgs[n_val:],  train_labels[n_val:]
    print(f"Splits — train: {len(train_imgs):,}  val: {n_val:,}  test: {len(test_imgs):,}")

    seq_len   = SEQ_LEN_COND if cfg.conditional else SEQ_LEN_UNCOND
    train_ds  = MNISTMDMDataset(train_imgs,  train_labels,  cfg.conditional)
    val_ds    = MNISTMDMDataset(val_imgs,    val_labels,    cfg.conditional)
    test_ds   = MNISTMDMDataset(test_imgs,   test_labels,   cfg.conditional)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    key, model_key = jrandom.split(key)
    model = ByteTabNetMDM(
        max_seq_length   = seq_len,
        embed_dim        = cfg.embed_dim,
        hidden_dim       = cfg.hidden_dim,
        n_blocks         = cfg.n_blocks,
        n_heads          = cfg.n_heads,
        n_steps          = cfg.n_steps,
        n_d              = cfg.n_d,
        n_a              = cfg.n_a,
        gamma            = cfg.gamma,
        n_shared         = cfg.n_shared,
        n_step           = cfg.n_step,
        virtual_batch_size = cfg.virtual_batch_size,
        use_positional   = True,
        vocab_size       = 320,
        key              = model_key,
    )

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"Model parameters: {n_params:,}")
    prefix = "BOS + class_token + " if cfg.conditional else "BOS + "
    print(f"Sequence layout: [{prefix}{IMG_SIZE} pixels + EOS]  (len={seq_len})")
    print(f"Conditional generation: {cfg.conditional}\n")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_ds) // cfg.batch_size
    total_steps     = steps_per_epoch * cfg.num_epochs

    schedule = optax.warmup_cosine_decay_schedule(
        init_value   = 0.0,
        peak_value   = cfg.learning_rate,
        warmup_steps = 10,#cfg.warmup_steps,
        decay_steps  = 100,#total_steps,
        end_value    = cfg.learning_rate * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.gradient_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    noise_fn   = cosine_noise_schedule if cfg.noise_schedule == "cosine" else linear_noise_schedule
    train_step = make_train_step(optimizer, noise_fn, cfg.sparsity_weight, cfg.label_smoothing)

    # ------------------------------------------------------------------
    # Checkpointing & output directories
    # ------------------------------------------------------------------
    ckpt_manager = CheckpointManager(cfg.checkpoint_dir, keep_best_k=3)
    Path(cfg.sample_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"Training for {cfg.num_epochs} epochs  ({steps_per_epoch} steps/epoch, {total_steps} total)\n")

    global_step  = 0
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        epoch_loss  = 0.0
        num_batches = 0

        pbar = tqdm(
            train_ds.make_batches(cfg.batch_size, shuffle=True),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
         )
        print("=== Model ===")
        print(model)
        for tokens, attn_mask in pbar:
            key, t_key, mask_key = jrandom.split(key, 3)
            t = jrandom.uniform(t_key, (), minval=0.01, maxval=1.0)

            model, opt_state, loss = train_step(
                model, opt_state, tokens, attn_mask, t, mask_key,
            )

            epoch_loss  += float(loss)
            num_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{float(loss):.4f}"})

            if global_step % cfg.log_every == 0:
                print(f"  step {global_step:6d}  loss={float(loss):.4f}")

            # Periodically generate and save a sample image grid
            if global_step % cfg.sample_every == 0:
                key, gen_key = jrandom.split(key)
                samples = generate_samples(model, cfg, gen_key)
                save_image_grid(
                    samples,
                    os.path.join(cfg.sample_dir, f"step_{global_step:07d}.png"),
                    nrow=4,
                )

        avg_train = epoch_loss / max(num_batches, 1)
        val_loss  = evaluate(model, val_ds, cfg.batch_size, cfg.sparsity_weight, noise_fn)
        print(
            f"Epoch {epoch + 1:3d}/{cfg.num_epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}"
        )

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_manager.save_checkpoint(
                model, epoch, val_loss,
                metadata={"cfg": cfg.__dict__, "global_step": global_step},
                is_best=True,
            )
            print(f"  → new best  val_loss={val_loss:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_manager.save_checkpoint(
                model, epoch, val_loss,
                metadata={"cfg": cfg.__dict__, "global_step": global_step},
            )

    # ------------------------------------------------------------------
    # Final evaluation & sample generation
    # ------------------------------------------------------------------
    print("\nEvaluating on test set...")
    test_loss = evaluate(
        model, test_ds, cfg.batch_size, cfg.sparsity_weight, noise_fn, num_batches=50,
    )
    print(f"Test loss: {test_loss:.4f}")

    key, gen_key = jrandom.split(key)
    generate_final_showcase(model, cfg, gen_key, n_samples=100)

    print(f"\nDone!  best_val_loss={best_val_loss:.4f}")
    print(f"  Checkpoints : {cfg.checkpoint_dir}")
    print(f"  Sample grids: {cfg.sample_dir}")
    return model


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ByteTabNet MDM on MNIST digit generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--embed-dim",   type=int,   default=64)
    p.add_argument("--hidden-dim",  type=int,   default=256,
                   help="Transformer hidden dimension (larger = better but slower)")
    p.add_argument("--n-blocks",    type=int,   default=6,
                   help="Number of bidirectional transformer blocks")
    p.add_argument("--n-heads",     type=int,   default=4)

    # Training
    p.add_argument("--num-epochs",     type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--learning-rate",  type=float, default=3e-4)
    p.add_argument("--warmup-steps",   type=int,   default=1000)
    p.add_argument("--noise-schedule", choices=["cosine", "linear"], default="cosine")

    # Generation
    p.add_argument("--num-denoise-steps", type=int,   default=25)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--conditional", action="store_true",
                   help="Class-conditional generation (prepends digit class token)")

    # Logging
    p.add_argument("--log-every",         type=int, default=100)
    p.add_argument("--sample-every",      type=int, default=500)
    p.add_argument("--checkpoint-dir",    type=str, default="./checkpoints_mnist_mdm")
    p.add_argument("--sample-dir",        type=str, default="./samples_mnist_mdm")

    # Data
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="Fraction of MNIST training set to use (0.1 for quick smoke-tests)")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = MNISTMDMConfig(
        embed_dim           = args.embed_dim,
        hidden_dim          = args.hidden_dim,
        n_blocks            = args.n_blocks,
        n_heads             = args.n_heads,
        num_epochs          = args.num_epochs,
        batch_size          = args.batch_size,
        learning_rate       = args.learning_rate,
        warmup_steps        = args.warmup_steps,
        noise_schedule      = args.noise_schedule,
        num_denoise_steps   = args.num_denoise_steps,
        sample_temperature  = args.sample_temperature,
        conditional         = args.conditional,
        log_every           = args.log_every,
        sample_every        = args.sample_every,
        checkpoint_dir      = args.checkpoint_dir,
        sample_dir          = args.sample_dir,
        train_fraction      = args.train_fraction,
        seed                = args.seed,
    )
    train(cfg)
