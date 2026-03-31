"""
Example usage of ByteTabNet with byte-level string inputs.

This demonstrates the same input encoding as:
    input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]]).to("cuda")
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx

from byte_tabnet import ByteTabNet, ByteEmbedding, tabnet_loss


def main():
    # Check available devices
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # Initialize random key
    key = jrandom.PRNGKey(42)

    # =========================================================================
    # Method 1: Manual byte encoding (similar to PyTorch example)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Method 1: Manual byte encoding")
    print("=" * 60)

    prompt = "Hello, TabNet!"

    # Encode exactly like the PyTorch example:
    # input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]])
    input_ids_manual = jnp.array([[1] + [b + 64 for b in prompt.encode("utf-8")]])

    print(f"Input string: '{prompt}'")
    print(f"Encoded bytes: {list(prompt.encode('utf-8'))}")
    print(f"Token IDs (with offset): {input_ids_manual[0].tolist()}")

    # =========================================================================
    # Method 2: Using ByteEmbedding utility functions
    # =========================================================================
    print("\n" + "=" * 60)
    print("Method 2: Using ByteEmbedding utilities")
    print("=" * 60)

    # Single string encoding
    input_ids_single = ByteEmbedding.encode_string(prompt)
    print(f"Single encoding: {input_ids_single.tolist()}")

    # Batch encoding with padding
    texts = [
        "Hello, TabNet!",
        "This is a longer example text for testing.",
        "Short",
    ]

    input_ids, attention_mask = ByteEmbedding.encode_batch(texts, max_length=64)
    print(f"\nBatch input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"First sequence (truncated): {input_ids[0, :20].tolist()}")
    print(f"Attention mask (first 20): {attention_mask[0, :20].tolist()}")

    # =========================================================================
    # Create and run the model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Model Forward Pass")
    print("=" * 60)

    model = ByteTabNet(
        output_dim=5,           # 5 output classes
        max_seq_length=128,     # Max sequence length
        embed_dim=64,           # Byte embedding dimension
        hidden_dim=128,         # Hidden dimension for TabNet
        n_steps=3,              # Number of decision steps
        n_d=32,                 # Decision embedding dim
        n_a=32,                 # Attention embedding dim
        gamma=1.5,              # Relaxation factor
        pooling="attention",    # Sequence pooling method
        key=key,
    )

    # Forward pass
    logits, masks, state = model(input_ids, attention_mask, inference=True)

    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits:\n{logits}")

    print(f"\nAttention masks shape: {masks.shape}")
    print(f"(batch_size, n_steps-1, hidden_dim)")

    # Get predictions
    predictions = jax.nn.softmax(logits, axis=-1)
    predicted_classes = jnp.argmax(predictions, axis=-1)

    print(f"\nPredicted probabilities:\n{predictions}")
    print(f"Predicted classes: {predicted_classes.tolist()}")

    # =========================================================================
    # Training Example
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Example")
    print("=" * 60)

    # Create dummy labels
    labels = jnp.array([0, 2, 1])

    # Setup optimizer
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Define loss function
    @eqx.filter_value_and_grad
    def compute_loss(model, input_ids, attention_mask, labels):
        logits, masks, _ = model(input_ids, attention_mask, inference=False)
        return tabnet_loss(logits, labels, masks, sparsity_weight=1e-3)

    # Training step
    @eqx.filter_jit
    def train_step(model, opt_state, input_ids, attention_mask, labels):
        loss, grads = compute_loss(model, input_ids, attention_mask, labels)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Run a few training steps
    print("Running training steps...")
    for step in range(5):
        model, opt_state, loss = train_step(model, opt_state, input_ids, attention_mask, labels)
        print(f"Step {step + 1}: Loss = {loss:.4f}")

    # =========================================================================
    # Feature Importance
    # =========================================================================
    print("\n" + "=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)

    # Get feature importance from attention masks
    importance = model.get_feature_importance(input_ids, attention_mask)
    print(f"Feature importance shape: {importance.shape}")
    print(f"Mean importance per sample: {jnp.mean(importance, axis=-1).tolist()}")

    # Top-k important features for first sample
    top_k = 10
    top_indices = jnp.argsort(importance[0])[-top_k:][::-1]
    top_values = importance[0][top_indices]
    print(f"\nTop {top_k} important feature indices: {top_indices.tolist()}")
    print(f"Top {top_k} importance values: {top_values.tolist()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
