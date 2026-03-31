"""
Code Generation with ByteTabNet MD4 — PyTorch

Trains ByteTabNet MDM on XythicK/code-generation-dataset using the MD4
(Masked Diffusion) algorithm from github.com/darioShar/pytorch-md4.

Key algorithmic choices (matching MD4):
  - Noise schedule: alpha_t = 1 - cos(π/2·(1-t))
      alpha_t = fraction of UNMASKED tokens; alpha_0=1 (clean), alpha_1=0 (fully masked)
  - Loss weight:   w(t) = (π/2)·tan(π/2·(1-t))
  - Timestep sampling: quasi-stratified across the batch
  - Sampling: per-position probabilistic unmasking

Encoding:
  - Each sample: [BOS, prompt_bytes..., code_bytes..., EOS, PAD...]
    truncated and right-padded to max_seq_length
  - Byte b ∈ [0, 255] → token b + BYTE_OFFSET (64)
  - Newline / punctuation are included naturally in the byte encoding

Conditional generation (inference):
  - Prompt bytes are frozen; code positions start fully masked
  - MD4 iterative unmasking progressively reveals code tokens

Usage:
    python train_code_mdm.py
    python train_code_mdm.py --num-epochs 5 --batch-size 16 --max-seq-length 512
    python train_code_mdm.py --train-fraction 0.05   # quick smoke-test
    python train_code_mdm.py --amp                   # mixed-precision (CUDA)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
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
# MD4 Noise Schedule & Loss Weight
# ============================================================================

# Safe bounds for t to keep the loss weight finite
T_MIN = 0.01
T_MAX = 0.99


def md4_alpha(t: torch.Tensor) -> torch.Tensor:
    """
    Fraction of UNMASKED tokens at noise level t (alpha_t in MD4).

    alpha_t = 1 - cos(π/2 · (1-t))

    At t=0 (clean):  alpha = 1  →  no tokens masked
    At t=1 (noisy):  alpha = 0  →  all tokens masked
    """
    return 1.0 - torch.cos((1.0 - t) * (math.pi / 2))


def md4_mask_rate(t: torch.Tensor) -> torch.Tensor:
    """Fraction of MASKED tokens at noise level t = 1 - alpha_t."""
    return 1.0 - md4_alpha(t)


def md4_loss_weight(t: torch.Tensor) -> torch.Tensor:
    """
    ELBO-derived loss weight for the MD4 objective.

    w(t) = (π/2) · tan(π/2 · (1-t))

    Derived as w(t) = alpha_t' / (1 - alpha_t).
    Blows up as t→0, so timesteps are clamped to (T_MIN, T_MAX).
    """
    return (math.pi / 2) * torch.tan((1.0 - t) * (math.pi / 2))


# ============================================================================
# Text ↔ Token Encoding
# ============================================================================

def text_to_tokens(text: str) -> List[int]:
    """Encode UTF-8 text bytes to token IDs (shifted by BYTE_OFFSET)."""
    return [b + BYTE_OFFSET for b in text.encode("utf-8")]


def tokens_to_text(tokens) -> str:
    """Decode a sequence of token IDs back to a UTF-8 string."""
    byte_vals = []
    for t in tokens:
        t = int(t)
        if BYTE_OFFSET <= t < VOCAB_SIZE:
            byte_vals.append(t - BYTE_OFFSET)
    return bytes(byte_vals).decode("utf-8", errors="replace")


# ============================================================================
# Dataset Loading
# ============================================================================

def load_code_dataset(
    dataset_name: str = "XythicK/code-generation-dataset",
    train_fraction: float = 1.0,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Load XythicK/code-generation-dataset and return train/val/test splits.

    Dataset schema:
        documentation  (str) — natural-language docstring / prompt
        code_snippet   (str) — the corresponding code
        language       (str) — programming language tag

    The dataset has only a "train" split (5 000 examples in parquet format),
    so we create val/test splits ourselves with an 80/val_fraction/10 ratio.

    Returns:
        (train_prompts, train_codes, val_prompts, val_codes, test_prompts, test_codes)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install HuggingFace datasets: pip install datasets"
        )

    print(f"Loading {dataset_name} ...")

    # Try plain load first; fall back to explicit parquet path if the
    # datasets library cannot auto-detect the file format.
    try:
        ds = load_dataset(dataset_name)
    except Exception as e1:
        print(f"[warn] load_dataset failed ({e1}); trying explicit parquet path ...")
        try:
            ds = load_dataset(
                "parquet",
                data_files={"train": f"hf://datasets/{dataset_name}/data/train-*"},
            )
        except Exception as e2:
            raise RuntimeError(
                f"Could not load {dataset_name}.\n"
                f"  Attempt 1: {e1}\n"
                f"  Attempt 2: {e2}\n"
                "Make sure you are logged in (`huggingface-cli login`) and "
                "the datasets library is up to date (`pip install -U datasets`)."
            ) from e2

    # Collect all examples from all available splits
    all_prompts: List[str] = []
    all_codes: List[str] = []

    # Column name priority: dataset-specific first, then generic fallbacks
    PROMPT_COLS = ("documentation", "prompt", "instruction", "input", "question")
    CODE_COLS   = ("code_snippet",  "code",   "output",      "solution", "answer")

    for split_name in ds:
        rows = ds[split_name]
        # Detect column names from the first row
        sample = rows[0] if len(rows) > 0 else {}
        prompt_col = next((c for c in PROMPT_COLS if c in sample), None)
        code_col   = next((c for c in CODE_COLS   if c in sample), None)
        if prompt_col is None or code_col is None:
            raise ValueError(
                f"Cannot find prompt/code columns in split '{split_name}'. "
                f"Available columns: {list(sample.keys())}"
            )
        print(f"  split={split_name!r}  prompt_col={prompt_col!r}  code_col={code_col!r}")
        for row in rows:
            p = str(row[prompt_col] or "").strip()
            c = str(row[code_col]   or "").strip()
            if p and c:
                all_prompts.append(p)
                all_codes.append(c)

    print(f"Total examples: {len(all_prompts):,}")

    # Shuffle with fixed seed
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_prompts))
    if train_fraction < 1.0:
        idx = idx[: int(len(idx) * train_fraction)]

    n = len(idx)
    n_test = max(int(n * 0.1), 1)
    n_val  = max(int(n * val_fraction), 1)
    n_train = max(n - n_val - n_test, 1)

    def _sel(ix):
        return (
            [all_prompts[i] for i in ix],
            [all_codes[i]   for i in ix],
        )

    tr_p, tr_c = _sel(idx[:n_train])
    va_p, va_c = _sel(idx[n_train : n_train + n_val])
    te_p, te_c = _sel(idx[n_train + n_val :])

    print(f"Splits — train: {len(tr_p):,}  val: {len(va_p):,}  test: {len(te_p):,}")
    return tr_p, tr_c, va_p, va_c, te_p, te_c


# ============================================================================
# Dataset
# ============================================================================

class CodeGenMDMDataset:
    """
    MDM dataset for (prompt, code) pairs.

    Each sample is encoded as:
        [BOS, prompt_bytes..., code_bytes..., EOS, PAD...]

    All sequences are right-padded to max_seq_length.
    prompt_lengths[i] records the number of prompt byte tokens (not counting
    BOS) so that conditional generation can freeze the prompt region.
    """

    def __init__(
        self,
        prompts: List[str],
        codes: List[str],
        max_seq_length: int = 512,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_seq_length = max_seq_length
        self.device = device
        self.sequences: List[np.ndarray] = []
        self.attn_masks: List[np.ndarray] = []
        self.prompt_lengths: List[int] = []

        for prompt, code in zip(prompts, codes):
            seq, mask, plen = self._encode(prompt, code)
            self.sequences.append(seq)
            self.attn_masks.append(mask)
            self.prompt_lengths.append(plen)

    def _encode(
        self, prompt: str, code: str
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        prompt_toks = text_to_tokens(prompt)
        code_toks   = text_to_tokens(code)

        # budget = max_seq_length - 2  (reserve 1 for BOS, 1 for EOS)
        budget     = self.max_seq_length - 2
        max_prompt = min(len(prompt_toks), budget // 3)
        prompt_toks = prompt_toks[:max_prompt]
        code_toks   = code_toks[: budget - len(prompt_toks)]

        content    = [BOS_TOKEN_ID] + prompt_toks + code_toks + [EOS_TOKEN_ID]
        prompt_len = len(prompt_toks)

        pad_len = self.max_seq_length - len(content)
        seq  = np.array(content + [PAD_TOKEN_ID] * pad_len, dtype=np.int64)
        mask = np.array([1.0] * len(content) + [0.0] * pad_len, dtype=np.float32)

        return seq, mask, prompt_len

    def __len__(self) -> int:
        return len(self.sequences)

    def make_batches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """Yield (token_seqs [long], attn_masks [float32], prompt_lengths) batches."""
        idx = np.arange(len(self))
        if shuffle:
            np.random.shuffle(idx)
        for s in range(0, len(idx) - batch_size + 1, batch_size):
            bi    = idx[s : s + batch_size]
            tokens = np.stack([self.sequences[i]   for i in bi])
            masks  = np.stack([self.attn_masks[i]  for i in bi])
            plens  = [self.prompt_lengths[i]        for i in bi]
            yield (
                torch.from_numpy(tokens).to(self.device),
                torch.from_numpy(masks).to(self.device),
                plens,
            )


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CodeGenMDMConfig:
    # Model
    max_seq_length:     int   = 512
    embed_dim:          int   = 64
    hidden_dim:         int   = 128
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
    batch_size:     int   = 16
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.01
    gradient_clip:  float = 1.0
    warmup_steps:   int   = 200
    noise_schedule: str   = "md4_cosine"

    # Regularisation
    sparsity_weight: float = 1e-3
    label_smoothing: float = 0.05

    # Logging / checkpointing
    log_every:      int = 20
    sample_every:   int = 200
    save_every:     int = 1
    checkpoint_dir: str = "./checkpoints_code_mdm"
    sample_dir:     str = "./samples_code_mdm"

    # Inference
    num_denoise_steps:  int   = 32
    sample_temperature: float = 1.0
    num_samples:        int   = 4
    max_code_len:       int   = 256

    # Data
    dataset_name:   str   = "XythicK/code-generation-dataset"
    train_fraction: float = 1.0
    val_fraction:   float = 0.1

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
# Conditional MD4 Sampling (generate code given a fixed prompt)
# ============================================================================

@torch.no_grad()
def md4_generate_code(
    model: ByteTabNetMDM,
    prompt: str,
    max_code_len: int,
    num_steps: int,
    temperature: float,
    device: torch.device,
    max_seq_length: int = 512,
) -> str:
    """
    Generate code conditioned on a prompt using MD4 iterative unmasking.

    The prompt bytes are fixed throughout; only code positions (the region
    after the prompt) are unmasked step-by-step using the MD4 reverse process:

        p(unmask | masked at t, going to s) = (alpha_s - alpha_t) / (1 - alpha_t)

    Initial sequence: [BOS, prompt_bytes..., MASK * max_code_len, EOS]

    Args:
        model:          Trained ByteTabNetMDM (in eval mode).
        prompt:         Plain-text prompt string.
        max_code_len:   Maximum number of code byte tokens to generate.
        num_steps:      Number of MD4 reverse steps.
        temperature:    Softmax temperature for token sampling.
        device:         Torch device.
        max_seq_length: Model's maximum sequence length (must match training).

    Returns:
        Generated code as a UTF-8 string.
    """
    model.eval()

    prompt_toks = text_to_tokens(prompt)
    budget      = max_seq_length - 2  # BOS + EOS
    max_prompt  = min(len(prompt_toks), budget // 3)
    prompt_toks = prompt_toks[:max_prompt]
    max_code_len = min(max_code_len, budget - len(prompt_toks))

    # Build initial sequence: [BOS, prompt..., MASK * max_code_len, EOS]
    seq_len = 1 + len(prompt_toks) + max_code_len + 1
    ids = torch.full((1, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device)
    ids[0, 0] = BOS_TOKEN_ID
    for i, tok in enumerate(prompt_toks):
        ids[0, 1 + i] = tok
    ids[0, -1] = EOS_TOKEN_ID

    attn = torch.ones((1, seq_len), dtype=torch.float32, device=device)

    code_start = 1 + len(prompt_toks)
    code_end   = seq_len - 1  # exclusive; EOS is fixed

    code_region = torch.zeros(seq_len, dtype=torch.bool, device=device)
    code_region[code_start:code_end] = True

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur  = timesteps[i]
        t_next = timesteps[i + 1]

        alpha_t = md4_alpha(t_cur)
        alpha_s = md4_alpha(t_next)

        denom    = (1.0 - alpha_t).clamp(min=1e-8)
        p_unmask = ((alpha_s - alpha_t) / denom).clamp(0.0, 1.0)

        logits, _ = model(ids, t_cur, attn)
        logits = logits / temperature

        probs   = F.softmax(logits[0], dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        is_masked     = (ids[0] == MASK_TOKEN_ID) & code_region
        should_unmask = (torch.rand(seq_len, device=device) < p_unmask) & is_masked
        ids = torch.where(should_unmask, sampled, ids[0]).unsqueeze(0)

    # Flush any remaining MASK tokens with a near-clean model pass
    still_masked = (ids[0] == MASK_TOKEN_ID) & code_region
    if still_masked.any():
        logits, _ = model(ids, torch.tensor(T_MIN, device=device), attn)
        probs   = F.softmax(logits[0] / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        ids = torch.where(still_masked, sampled, ids[0]).unsqueeze(0)

    code_toks = ids[0, code_start:code_end].cpu().tolist()
    return tokens_to_text(code_toks)


# ============================================================================
# Helpers
# ============================================================================

def build_warmup_cosine_scheduler(
    optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01
):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_text_samples(
    model: ByteTabNetMDM,
    prompts: List[str],
    cfg: CodeGenMDMConfig,
    device: torch.device,
    path: str,
    n: Optional[int] = None,
) -> None:
    """Generate code for the first n prompts and write to a text file."""
    n = n or cfg.num_samples
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    with open(path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts[:n]):
            f.write(f"{'=' * 60}\n")
            f.write(f"Sample {i + 1}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"GENERATED CODE:\n")
            try:
                code = md4_generate_code(
                    model, prompt,
                    max_code_len=cfg.max_code_len,
                    num_steps=cfg.num_denoise_steps,
                    temperature=cfg.sample_temperature,
                    device=device,
                    max_seq_length=cfg.max_seq_length,
                )
                f.write(code)
            except Exception as e:
                f.write(f"[generation error: {e}]")
            f.write("\n\n")

    print(f"  Saved {min(n, len(prompts))} samples → {path}")
    model.train()


def _evaluate(
    model: ByteTabNetMDM,
    dataset: CodeGenMDMDataset,
    cfg: CodeGenMDMConfig,
    device: torch.device,
    num_batches: int = 10,
) -> float:
    """Average MD4 loss over a dataset subset, evaluated at t = 0.25 / 0.5 / 0.75."""
    model.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for tokens, masks, _ in dataset.make_batches(cfg.batch_size, shuffle=False):
            if count >= num_batches:
                break
            for t_val in (0.25, 0.5, 0.75):
                t         = torch.tensor(t_val, device=device)
                mask_rate = md4_mask_rate(t)
                is_special = (
                    (tokens == PAD_TOKEN_ID) |
                    (tokens == BOS_TOKEN_ID) |
                    (tokens == EOS_TOKEN_ID)
                )
                should_mask = (
                    torch.rand_like(tokens, dtype=torch.float32) < mask_rate.item()
                ) & ~is_special
                masked_ids = torch.where(should_mask, MASK_TOKEN_ID, tokens)

                logits, tabnet_masks = model(masked_ids, t, masks)
                loss = md4_loss(
                    logits, tokens, should_mask, tabnet_masks, t,
                    masks, cfg.sparsity_weight, cfg.label_smoothing,
                )
                total += loss.item()
            count += 1

    model.train()
    return total / max(count * 3, 1)


# ============================================================================
# Training
# ============================================================================

def train(cfg: CodeGenMDMConfig) -> ByteTabNetMDM:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    print(f"Device: {device}")
    if cfg.amp and device.type != "cuda":
        print("[warn] --amp requires CUDA; disabling.")
        cfg.amp = False

    # --- Data ---
    tr_p, tr_c, va_p, va_c, te_p, te_c = load_code_dataset(
        cfg.dataset_name, cfg.train_fraction, cfg.val_fraction, cfg.seed
    )

    train_ds = CodeGenMDMDataset(tr_p, tr_c, cfg.max_seq_length, device)
    val_ds   = CodeGenMDMDataset(va_p, va_c, cfg.max_seq_length, device)
    test_ds  = CodeGenMDMDataset(te_p, te_c, cfg.max_seq_length, device)

    # --- Model ---
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Max seq length: {cfg.max_seq_length}  AMP: {cfg.amp}\n")

    # --- Optimizer / scheduler ---
    steps_per_epoch = max(len(train_ds) // cfg.batch_size, 1)
    total_steps     = steps_per_epoch * cfg.num_epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(optimizer, cfg.warmup_steps, total_steps)
    scaler    = torch.amp.GradScaler(enabled=cfg.amp)

    ckpt_manager = CheckpointManager(cfg.checkpoint_dir, keep_best_k=3)
    Path(cfg.sample_dir).mkdir(parents=True, exist_ok=True)

    print(f"Training: {cfg.num_epochs} epochs  "
          f"{steps_per_epoch} steps/epoch  {total_steps} total\n")

    global_step   = 0
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        pbar = tqdm(
            train_ds.make_batches(cfg.batch_size, shuffle=True),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
        )

        for tokens, attn_mask, _ in pbar:
            B = tokens.shape[0]

            # MD4 quasi-stratified timestep sampling:
            # t_i = (rand + i/B) mod 1, clamped to (T_MIN, T_MAX)
            rand    = torch.rand(1, device=device)
            t_batch = torch.fmod(
                rand + (1 + torch.arange(B, device=device)).float() / B, 1.0
            ).clamp(T_MIN, T_MAX)
            t_scalar = t_batch.mean()

            # Per-element masking using each element's own t
            is_special = (
                (tokens == PAD_TOKEN_ID) |
                (tokens == BOS_TOKEN_ID) |
                (tokens == EOS_TOKEN_ID)
            )
            mask_rates  = md4_mask_rate(t_batch)[:, None]   # (B, 1) → broadcasts
            should_mask = (
                torch.rand_like(tokens, dtype=torch.float32) < mask_rates
            ) & ~is_special
            masked_ids = torch.where(should_mask, MASK_TOKEN_ID, tokens)

            optimizer.zero_grad()

            logits, tabnet_masks = model(masked_ids, t_scalar, attn_mask)
            loss = md4_loss(
                logits, tokens, should_mask, tabnet_masks, t_scalar,
                attn_mask, cfg.sparsity_weight, cfg.label_smoothing,
            )

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
                save_text_samples(
                    model, te_p, cfg, device,
                    os.path.join(cfg.sample_dir, f"step_{global_step:07d}.txt"),
                )
                model.train()

        avg_train = epoch_loss / max(num_batches, 1)
        val_loss  = _evaluate(model, val_ds, cfg, device, num_batches=20)
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

    save_text_samples(
        model, te_p, cfg, device,
        os.path.join(cfg.sample_dir, "final_samples.txt"),
        n=min(20, len(te_p)),
    )

    print(f"\nDone!  best_val_loss={best_val_loss:.4f}")
    print(f"  Checkpoints : {cfg.checkpoint_dir}")
    print(f"  Samples     : {cfg.sample_dir}")
    return model


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train ByteTabNet MD4 on XythicK/code-generation-dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-seq-length",     type=int,   default=512)
    p.add_argument("--embed-dim",          type=int,   default=64)
    p.add_argument("--hidden-dim",         type=int,   default=128)
    p.add_argument("--n-blocks",           type=int,   default=6)
    p.add_argument("--n-heads",            type=int,   default=4)
    p.add_argument("--num-epochs",         type=int,   default=10)
    p.add_argument("--batch-size",         type=int,   default=16)
    p.add_argument("--learning-rate",      type=float, default=3e-4)
    p.add_argument("--warmup-steps",       type=int,   default=200)
    p.add_argument("--amp",                action="store_true",
                   help="Mixed-precision training (CUDA only)")
    p.add_argument("--num-denoise-steps",  type=int,   default=32)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--max-code-len",       type=int,   default=256)
    p.add_argument("--log-every",          type=int,   default=20)
    p.add_argument("--sample-every",       type=int,   default=200)
    p.add_argument("--checkpoint-dir",     type=str,   default="./checkpoints_code_mdm")
    p.add_argument("--sample-dir",         type=str,   default="./samples_code_mdm")
    p.add_argument("--train-fraction",     type=float, default=1.0,
                   help="Fraction of dataset to use (e.g. 0.05 for a quick smoke-test)")
    p.add_argument("--seed",               type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = CodeGenMDMConfig(
        max_seq_length     = args.max_seq_length,
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
        max_code_len       = args.max_code_len,
        log_every          = args.log_every,
        sample_every       = args.sample_every,
        checkpoint_dir     = args.checkpoint_dir,
        sample_dir         = args.sample_dir,
        train_fraction     = args.train_fraction,
        amp                = args.amp,
        seed               = args.seed,
    )
    train(cfg)
