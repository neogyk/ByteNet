"""
Quantization-Aware Training (QAT) utilities for ByteTabNet.

Provides helpers to prepare a ByteTabNetSeq2Seq model for QAT,
fine-tune with fake quantizers, and convert to int8 for inference.

Approach:
- Eager-mode QAT via ``torch.ao.quantization``
- Only ``nn.Linear`` layers are quantized (dominant compute)
- Embeddings, GhostBatchNorm, sparsemax, and attention stay fp32
- GhostBatchNorm fusion is intentionally skipped (virtual-batch
  splitting makes standard Linear+BN fusion impractical)
"""

from __future__ import annotations

import io
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.ao.quantization as aoq

from byte_tabnet import (
    ByteEmbedding,
    GhostBatchNorm,
    ByteTabNet,
    ByteTabNetSeq2Seq,
    ByteTabNetSeq2SeqDecoder,
)


# ============================================================================
# QAT-wrapped Seq2Seq model
# ============================================================================

class ByteTabNetSeq2SeqQAT(nn.Module):
    """
    Thin wrapper around ByteTabNetSeq2Seq that inserts QuantStub /
    DeQuantStub at the encoder-decoder boundary.

    The inner ``model`` attribute is the original ByteTabNetSeq2Seq.
    """

    def __init__(self, model: ByteTabNetSeq2Seq):
        super().__init__()
        self.model = model
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        # Expose attributes used by the trainer / generation
        self.bos_token_id = model.bos_token_id
        self.eos_token_id = model.eos_token_id
        self.pad_token_id = model.pad_token_id
        self.vocab_size = model.vocab_size
        self.n_d = model.n_d

    # -- forwarding helpers ---------------------------------------------------

    def encode(self, input_ids, attention_mask=None):
        return self.model.encode(input_ids, attention_mask)

    def decode(self, decoder_input_ids, encoder_context,
               encoder_mask=None, decoder_mask=None):
        return self.model.decode(
            decoder_input_ids, encoder_context, encoder_mask, decoder_mask,
        )

    def forward(self, input_ids, decoder_input_ids,
                attention_mask=None, decoder_attention_mask=None):
        return self.model(
            input_ids, decoder_input_ids, attention_mask, decoder_attention_mask,
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(input_ids, attention_mask, **kwargs)


# ============================================================================
# Prepare / Convert helpers
# ============================================================================

def _set_qconfig_recursive(module: nn.Module, qconfig) -> None:
    """Set *qconfig* on every sub-module, then override to ``None``
    for modules that should stay in fp32."""
    module.qconfig = qconfig
    for child in module.children():
        _set_qconfig_recursive(child, qconfig)

    # Keep these in fp32
    if isinstance(module, (ByteEmbedding, GhostBatchNorm, nn.Embedding)):
        module.qconfig = None


def prepare_model_for_qat(
    model: ByteTabNetSeq2Seq,
    backend: str = "fbgemm",
) -> ByteTabNetSeq2SeqQAT:
    """
    Prepare a pre-trained ByteTabNetSeq2Seq for QAT.

    1. Wraps in ``ByteTabNetSeq2SeqQAT`` (adds QuantStub/DeQuantStub)
    2. Sets per-module qconfig (skipping embeddings / GhostBatchNorm)
    3. Calls ``torch.ao.quantization.prepare_qat`` to insert fake quantizers

    Args:
        model: Pre-trained fp32 model.
        backend: ``"fbgemm"`` (x86) or ``"qnnpack"`` (ARM/mobile).

    Returns:
        Model ready for QAT fine-tuning.
    """
    torch.backends.quantized.engine = backend

    wrapped = ByteTabNetSeq2SeqQAT(copy.deepcopy(model))

    qconfig = aoq.get_default_qat_qconfig(backend)
    _set_qconfig_recursive(wrapped, qconfig)

    # QuantStub / DeQuantStub keep the global qconfig
    wrapped.quant.qconfig = qconfig
    wrapped.dequant.qconfig = qconfig

    aoq.prepare_qat(wrapped, inplace=True)

    return wrapped


def convert_qat_to_quantized(model: ByteTabNetSeq2SeqQAT) -> nn.Module:
    """
    Convert a QAT-trained model to a fully quantized int8 model.

    Args:
        model: Model after QAT fine-tuning.

    Returns:
        Quantized int8 model (runs on CPU).
    """
    model.eval()
    quantized = aoq.convert(model.cpu(), inplace=False)
    return quantized


# ============================================================================
# Utilities
# ============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """Return the serialised size of *model* in megabytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)
