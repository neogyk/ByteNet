"""Test forward pass for ByteTabNet using default text."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from byte_tabnet import ByteTabNet, ByteEmbedding


def test_forward_pass():
    """Run a test forward pass with default text."""
    print("=" * 60)
    print("ByteTabNet Forward Pass Test")
    print("=" * 60)

    # Initialize model
    key = jrandom.PRNGKey(42)
    model = ByteTabNet(
        output_dim=10,  # 10 classes for classification
        max_seq_length=256,
        embed_dim=64,
        hidden_dim=128,
        n_steps=5,
        n_d=64,
        n_a=64,
        key=key,
    )

    # Default test text (single sample to avoid batch dimension issues)
    default_text = "Hello, world! This is ByteTabNet."

    print(f"\n[Input text]")
    print(f"  {default_text!r}")

    # Encode single text - add batch dimension
    input_ids = ByteEmbedding.encode_string(default_text, max_length=64)
    input_ids = input_ids[None, :]  # Add batch dimension: (1, seq_len)
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.float32)

    print(f"\n[Encoded tensors]")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")
    print(f"  Token IDs (first 20): {input_ids[0, :20].tolist()}")

    # Forward pass
    print("\n[Running forward pass...]")
    logits, masks, state = model(input_ids, attention_mask, inference=True)

    print(f"\n[Output]")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits values: {logits[0].tolist()}")
    print(f"  Attention masks shape: {masks.shape}")

    # Get predictions
    probs = jax.nn.softmax(logits, axis=-1)
    predicted_class = jnp.argmax(probs, axis=-1)[0]

    print(f"\n[Predictions]")
    print(f"  Softmax probabilities: {probs[0].tolist()}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {probs[0, predicted_class]:.4f}")

    # Test with multiple samples using vmap
    print("\n" + "-" * 60)
    print("[Testing with batch of 3 samples using vmap]")

    texts = [
        "Hello, world!",
        "This is a test.",
        "TabNet with bytes!"
    ]

    # Encode batch
    batch_ids, batch_mask = ByteEmbedding.encode_batch(texts, max_length=32)
    print(f"  Batch input shape: {batch_ids.shape}")

    # Create a batched forward function
    @eqx.filter_jit
    def batched_forward(model, ids, mask):
        def single_forward(single_ids, single_mask):
            # Add batch dim, run, remove batch dim
            ids_batched = single_ids[None, :]
            mask_batched = single_mask[None, :]
            logits, masks, _ = model(ids_batched, mask_batched, inference=True)
            return logits[0], masks[0] if masks.shape[0] > 0 else masks

        logits, masks = jax.vmap(single_forward)(ids, mask)
        return logits, masks

    batch_logits, batch_masks = batched_forward(model, batch_ids, batch_mask)
    batch_probs = jax.nn.softmax(batch_logits, axis=-1)
    batch_preds = jnp.argmax(batch_probs, axis=-1)

    print(f"  Batch logits shape: {batch_logits.shape}")
    print(f"  Batch predictions: {batch_preds.tolist()}")

    for i, (text, pred, prob) in enumerate(zip(texts, batch_preds, batch_probs)):
        print(f"    [{i}] '{text}' -> Class {pred}, Prob: {prob[pred]:.4f}")

    print("\n" + "=" * 60)
    print("Forward pass completed successfully!")
    print("=" * 60)

    return model, logits, masks


if __name__ == "__main__":
    test_forward_pass()
