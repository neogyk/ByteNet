"""Tests for byte-level tokenization and detokenization."""

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import (
    BYTE_OFFSET,
    SPECIAL_TOKENS,
    encode_text_to_tokens,
    decode_token_ids,
    map_token_to_char,
    _strip_special,
)


# ---------------------------------------------------------------------------
# encode_text_to_tokens
# ---------------------------------------------------------------------------

class TestEncodeTextToTokens:
    def test_simple_ascii(self):
        ids, mask = encode_text_to_tokens(["hi"], max_length=10)
        expected = [1, ord("h") + BYTE_OFFSET, ord("i") + BYTE_OFFSET]
        assert ids.shape == (1, 10)
        assert list(ids[0, :3]) == expected
        # rest should be padding (0)
        assert jnp.all(ids[0, 3:] == 0)

    def test_attention_mask(self):
        ids, mask = encode_text_to_tokens(["abc"], max_length=8)
        # BOS + 3 chars = 4 real tokens
        assert list(mask[0]) == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    def test_truncation(self):
        long_text = "a" * 100
        ids, mask = encode_text_to_tokens([long_text], max_length=10)
        assert ids.shape == (1, 10)
        # First token should be BOS
        assert int(ids[0, 0]) == 1

    def test_batch(self):
        ids, mask = encode_text_to_tokens(["ab", "xyz"], max_length=8)
        assert ids.shape == (2, 8)
        # "ab" -> BOS + 2 bytes = 3 tokens, pad 5
        assert float(mask[0, 2]) == 1.0
        assert float(mask[0, 3]) == 0.0
        # "xyz" -> BOS + 3 bytes = 4 tokens, pad 4
        assert float(mask[1, 3]) == 1.0
        assert float(mask[1, 4]) == 0.0

    def test_empty_string(self):
        ids, mask = encode_text_to_tokens([""], max_length=4)
        # Only BOS token
        assert int(ids[0, 0]) == 1
        assert jnp.all(ids[0, 1:] == 0)
        assert list(mask[0]) == [1.0, 0.0, 0.0, 0.0]

    def test_multibyte_utf8(self):
        text = "\u00e9"  # e-acute, 2 bytes in UTF-8: 0xC3 0xA9
        ids, mask = encode_text_to_tokens([text], max_length=8)
        utf8_bytes = text.encode("utf-8")
        assert len(utf8_bytes) == 2
        # BOS + 2 byte tokens
        assert int(ids[0, 0]) == 1
        assert int(ids[0, 1]) == utf8_bytes[0] + BYTE_OFFSET
        assert int(ids[0, 2]) == utf8_bytes[1] + BYTE_OFFSET

    def test_emoji_utf8(self):
        text = "\U0001f600"  # grinning face, 4 bytes in UTF-8
        ids, mask = encode_text_to_tokens([text], max_length=10)
        utf8_bytes = text.encode("utf-8")
        assert len(utf8_bytes) == 4
        for i, b in enumerate(utf8_bytes):
            assert int(ids[0, i + 1]) == b + BYTE_OFFSET


# ---------------------------------------------------------------------------
# map_token_to_char
# ---------------------------------------------------------------------------

class TestMapTokenToChar:
    def test_printable_ascii(self):
        # 'A' = 65, token_id = 65 + 64 = 129
        assert map_token_to_char(ord("A") + BYTE_OFFSET) == "A"
        assert map_token_to_char(ord("z") + BYTE_OFFSET) == "z"
        assert map_token_to_char(ord(" ") + BYTE_OFFSET) == " "

    def test_special_tokens(self):
        assert map_token_to_char(0) == "<PAD>"
        assert map_token_to_char(1) == "<BOS>"
        assert map_token_to_char(2) == "<EOS>"

    def test_non_printable_byte(self):
        # byte 0 -> token 64, non-printable
        result = map_token_to_char(64)
        assert result == "\\x00"

    def test_high_byte(self):
        # byte 200 -> token 264
        result = map_token_to_char(264)
        assert result == "\\xc8"

    def test_out_of_range(self):
        result = map_token_to_char(999)
        assert result == "<UNK_999>"


# ---------------------------------------------------------------------------
# _strip_special
# ---------------------------------------------------------------------------

class TestStripSpecial:
    def test_strip_bos(self):
        assert _strip_special([1, 100, 110]) == [100, 110]

    def test_strip_eos(self):
        assert _strip_special([1, 100, 2, 999]) == [100]

    def test_no_special(self):
        assert _strip_special([100, 110, 120]) == [100, 110, 120]

    def test_only_bos_eos(self):
        assert _strip_special([1, 2]) == []

    def test_empty(self):
        assert _strip_special([]) == []

    def test_pad_tokens_kept(self):
        # PAD (0) inside the sequence is kept if before EOS
        assert _strip_special([1, 100, 0, 110, 2]) == [100, 0, 110]


# ---------------------------------------------------------------------------
# decode_token_ids
# ---------------------------------------------------------------------------

class TestDecodeTokenIds:
    def test_simple_ascii(self):
        tokens = [1, ord("h") + BYTE_OFFSET, ord("i") + BYTE_OFFSET, 2]
        assert decode_token_ids(tokens) == "hi"

    def test_with_padding(self):
        tokens = [1, ord("o") + BYTE_OFFSET, ord("k") + BYTE_OFFSET, 2, 0, 0]
        assert decode_token_ids(tokens) == "ok"

    def test_multibyte_utf8(self):
        text = "\u00e9"  # e-acute
        utf8_bytes = text.encode("utf-8")
        tokens = [1] + [b + BYTE_OFFSET for b in utf8_bytes] + [2]
        assert decode_token_ids(tokens) == text

    def test_emoji(self):
        text = "\U0001f600"
        utf8_bytes = text.encode("utf-8")
        tokens = [1] + [b + BYTE_OFFSET for b in utf8_bytes] + [2]
        assert decode_token_ids(tokens) == text

    def test_empty_after_strip(self):
        assert decode_token_ids([1, 2]) == ""

    def test_no_eos(self):
        tokens = [1, ord("a") + BYTE_OFFSET, ord("b") + BYTE_OFFSET]
        assert decode_token_ids(tokens) == "ab"

    def test_invalid_utf8_replaced(self):
        # Single continuation byte (0x80) is invalid start -> replacement char
        tokens = [1, 0x80 + BYTE_OFFSET, 2]
        result = decode_token_ids(tokens)
        assert "\ufffd" in result  # replacement character


# ---------------------------------------------------------------------------
# Round-trip: encode -> decode
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("text", [
        "hello world",
        "",
        "Hello, World! 123",
        "\u00e9\u00e8\u00ea",      # French accented chars
        "\U0001f600\U0001f4a9",    # Emoji
        "\u4f60\u597d",            # Chinese: "nihao"
        "line1\nline2\ttab",       # Whitespace
        'key="value"',             # Quotes
        "a" * 500,                 # Long string
    ])
    def test_roundtrip(self, text):
        max_len = len(text.encode("utf-8")) + 2  # BOS + bytes + slack
        ids, mask = encode_text_to_tokens([text], max_length=max_len)
        token_list = list(int(x) for x in ids[0])
        decoded = decode_token_ids(token_list)
        assert decoded == text
