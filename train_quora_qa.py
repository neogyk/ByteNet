"""
Train ByteTabNetSeq2Seq on the Quora Question-Answer dataset.

Dataset: toughdata/quora-question-answer-dataset (HuggingFace)
  columns: question → answer

Usage:
    python train_quora_qa.py                         # default settings
    python train_quora_qa.py --batch-size 64         # override batch size
    python train_quora_qa.py --epochs 5 --quick      # fast smoke-test
    python train_quora_qa.py --resume ./checkpoints/quora_qa/best_model.eqx
"""

import argparse
from dataclasses import asdict

import jax.random as jrandom
import equinox as eqx
import numpy as np

from trainer import ModelConfig, TrainingConfig, Trainer
from dataset import DataConfig


# ---------------------------------------------------------------------------
# Defaults tuned for Quora QA
# ---------------------------------------------------------------------------

DEFAULT_MODEL = ModelConfig(
    max_seq_length=256,        # questions rarely exceed 256 bytes
    embed_dim=64,
    hidden_dim=256,
    n_steps=5,
    n_d=64,
    n_a=64,
    gamma=1.5,
    n_shared=2,
    n_step=3,
    virtual_batch_size=64,
    use_positional=True,
    vocab_size=320,
    n_decoder_layers=4,
    max_target_length=512,     # answers up to 512 tokens
)

DEFAULT_TRAINING = TrainingConfig(
    task_type="generation",
    learning_rate=5e-4,
    weight_decay=0.01,
    batch_size=32,
    num_epochs=20,
    optimizer="adamw",
    sparsity_weight=1e-3,
    label_smoothing=0.1,
    kl_weight=0.0,
    gradient_clip=1.0,
    lr_schedule="warmup_cosine",
    warmup_steps=300,
    min_lr=1e-6,
    patience=5,
    min_delta=1e-4,
    save_every=5,
    checkpoint_dir="./checkpoints/quora_qa",
    keep_best_k=3,
    log_every=50,
    eval_every=200,
    train_split=0.9,
    val_split=0.05,
    test_split=0.05,
    shuffle_seed=42,
)

DEFAULT_DATA = DataConfig(
    data_format="huggingface",
    hf_dataset_name="toughdata/quora-question-answer-dataset",
    hf_split_train="train",
    hf_split_test="test",
    text_column="question",
    target_column="answer",
    max_target_length=512,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ByteTabNet on Quora QA")
    p.add_argument("--batch-size",   type=int,   help="Training batch size")
    p.add_argument("--epochs",       type=int,   help="Number of training epochs")
    p.add_argument("--lr",           type=float, help="Peak learning rate")
    p.add_argument("--hidden-dim",   type=int,   help="Model hidden dimension")
    p.add_argument("--n-steps",      type=int,   help="TabNet decision steps")
    p.add_argument("--checkpoint-dir", type=str, help="Directory for checkpoints")
    p.add_argument("--resume",       type=str,   help="Path to checkpoint .eqx to resume from")
    p.add_argument("--quick",        action="store_true",
                   help="Smoke-test: 2 epochs, batch=8, max 500 samples")
    p.add_argument("--no-demo",      action="store_true",
                   help="Skip post-training inference demo")
    return p.parse_args()


def apply_overrides(model_cfg, train_cfg, args):
    """Apply CLI overrides to config dataclasses."""
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.epochs:
        train_cfg.num_epochs = args.epochs
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.hidden_dim:
        model_cfg.hidden_dim = args.hidden_dim
    if args.n_steps:
        model_cfg.n_steps = args.n_steps
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir

    if args.quick:
        train_cfg.num_epochs = 2
        train_cfg.batch_size = 8
        train_cfg.log_every = 5
        train_cfg.eval_every = 20
        train_cfg.warmup_steps = 10

    return model_cfg, train_cfg


# ---------------------------------------------------------------------------
# Post-training inference demo
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    "What is the best way to learn programming?",
    "How do I improve my memory?",
    "What are some tips for better sleep?",
]


def run_inference_demo(model, model_cfg, data_cfg):
    """Generate answers for a few sample questions."""
    from byte_tabnet import ByteEmbedding

    print("\n" + "=" * 60)
    print("Inference demo")
    print("=" * 60)

    key = jrandom.PRNGKey(0)
    for question in DEMO_QUESTIONS:
        input_ids, attention_mask = ByteEmbedding.encode_batch(
            [question],
            max_length=model_cfg.max_seq_length,
        )
        key, subkey = jrandom.split(key)
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=200,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            key=subkey,
        )
        # Decode: tokens 64-319 → bytes 0-255
        raw = [int(t) - 64 for t in output_ids[0] if 64 <= int(t) < 320]
        answer = bytes(raw).decode("utf-8", errors="replace")
        print(f"\nQ: {question}")
        print(f"A: {answer}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_cfg = DEFAULT_MODEL
    train_cfg = DEFAULT_TRAINING
    data_cfg  = DEFAULT_DATA

    model_cfg, train_cfg = apply_overrides(model_cfg, train_cfg, args)

    print("=" * 60)
    print("ByteTabNet — Quora QA Training")
    print("=" * 60)
    print(f"Dataset : {data_cfg.hf_dataset_name}")
    print(f"Task    : seq2seq generation (question → answer)")
    print(f"Epochs  : {train_cfg.num_epochs}  |  Batch : {train_cfg.batch_size}")
    print(f"LR      : {train_cfg.learning_rate}  |  Hidden : {model_cfg.hidden_dim}")
    print(f"Ckpts   : {train_cfg.checkpoint_dir}")
    print("=" * 60 + "\n")

    trainer = Trainer(model_cfg, train_cfg, data_cfg)
    trainer.prepare_data()

    # Optionally cap samples for quick smoke-test
    if args.quick:
        trainer.train_data = trainer.train_data[:500]
        trainer.val_data   = trainer.val_data[:100]
        trainer.test_data  = trainer.test_data[:50]
        print(f"[quick mode] using {len(trainer.train_data)} train / "
              f"{len(trainer.val_data)} val samples\n")

    # Resume from checkpoint
    if args.resume:
        key = jrandom.PRNGKey(train_cfg.shuffle_seed)
        from trainer import GenerationTask
        template = GenerationTask().create_model(model_cfg, key)
        trainer.model, meta = trainer.checkpoint_manager.load_checkpoint(
            template, args.resume
        )
        start_epoch = meta.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})\n")

    trainer.train()

    if not args.no_demo and trainer.model is not None:
        run_inference_demo(trainer.model, model_cfg, data_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
