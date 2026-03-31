Files Created
byte_tabnet.py - Main module with all components:

sparsemax - Sparse attention activation (outputs exact zeros)
GhostBatchNorm - Virtual batch normalization for better generalization
GLUBlock - Gated Linear Unit block (FC → BN → GLU)
FeatureTransformer - Shared + step-specific GLU layers with skip connections
AttentiveTransformer - Computes sparse attention masks for feature selection
ByteEmbedding - Byte-level encoder matching your PyTorch example
TabNetEncoder - Core TabNet architecture with decision steps
TabNetDecoder - For pretraining/reconstruction
ByteTabNet - Main model combining byte embedding + TabNet
ByteTabNetHax - Haliax version with named tensors
requirements.txt - Dependencies

example.py - Usage demonstration

Byte-Level Encoding
The encoding matches your PyTorch example exactly:


# Your PyTorch example:
input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]]).to("cuda")

# Equivalent JAX:
input_ids = jnp.array([[1] + [b + 64 for b in prompt.encode("utf-8")]])

# Or using the utility:
input_ids = ByteEmbedding.encode_string("Hello, world!")
input_ids, mask = ByteEmbedding.encode_batch(["text1", "text2"], max_length=128)
Quick Usage

from byte_tabnet import ByteTabNet, ByteEmbedding
import jax.random as jrandom

# Create model
model = ByteTabNet(
    output_dim=10,
    n_steps=5,
    n_d=64,
    n_a=64,
    key=jrandom.PRNGKey(42)
)

# Encode and run
input_ids, mask = ByteEmbedding.encode_batch(["Hello!", "World!"], max_length=64)
logits, attention_masks, state = model(input_ids, mask)
To run the example: python example.py