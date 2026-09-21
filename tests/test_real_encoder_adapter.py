"""The real-encoder adapter and its on-disk vector cache.

``make_hf_encoder`` was shipped explicitly UNTESTED ("transformers is not installed in the
environment this was written in"). When a real checkpoint finally arrived it failed twice,
both times because transformers 5.x removed a tokenizer method the adapter called:

    AttributeError: DebertaV2Tokenizer has no attribute prepare_for_model
    AttributeError: DebertaV2Tokenizer has no attribute build_inputs_with_special_tokens

It now derives the special-token wrapper by comparing an encoding with and without special
tokens, which depends on no removable helper. These tests pin that behaviour with a fake
tokenizer/model, so the suite does not need a 700 MB download to protect the contract.

The disk cache is what makes a real encoder affordable at all: the suite is ~9 separate
scripts and the in-memory memo dies with each process, so without persistence every script
re-embeds all 273 statements (~6,000 words each, minutes apiece).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity import FrozenEncoder, make_hf_encoder  # noqa: E402

torch = pytest.importorskip("torch")
# The adapter imports transformers lazily, inside make_hf_encoder. Skip the whole module
# rather than error when it is absent: the rest of the suite runs fine without it, and the
# project's own interpreter does not carry transformers.
transformers = pytest.importorskip("transformers")

CLS, SEP, PAD = 1, 2, 0
DIM = 4


class _FakeTok:
    """Wraps content ids in [CLS] ... [SEP], like a BERT/DeBERTa tokenizer.

    Deliberately exposes NEITHER ``prepare_for_model`` NOR
    ``build_inputs_with_special_tokens``: that is exactly the transformers 5.x surface the
    adapter broke on, and __getattr__ raises the same AttributeError the real class does.
    """

    pad_token_id = PAD

    def __init__(self):
        self.calls: list[int] = []          # sequence length of every forward-bound row

    def __call__(self, text, add_special_tokens=True, truncation=False):
        ids = [ord(c) % 97 + 3 for c in text.replace(" ", "")]
        return {"input_ids": ([CLS] + ids + [SEP]) if add_special_tokens else ids}

    def __getattr__(self, name):
        raise AttributeError(f"_FakeTok has no attribute {name}")


class _FakeCfg:
    hidden_size = DIM


class _FakeModel:
    """Returns one deterministic vector per position; records the batches it was given."""

    config = _FakeCfg()

    def __init__(self):
        self.batches: list[tuple[int, int]] = []   # (rows, width) per forward
        self.seen_ids: list[list[int]] = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        self.batches.append(tuple(input_ids.shape))
        self.seen_ids.extend(input_ids.tolist())
        b, n = input_ids.shape
        vals = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, DIM)
        vals = vals * torch.arange(1, DIM + 1, dtype=torch.float32)
        return type("Out", (), {"last_hidden_state": vals})()


@pytest.fixture
def patched(monkeypatch):
    tok, model = _FakeTok(), _FakeModel()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: tok))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        classmethod(lambda cls, *a, **k: model))
    return tok, model


# ---- the two real bugs -------------------------------------------------------------------
def test_adapter_never_calls_the_removed_tokenizer_helpers(patched):
    """Regression: both helpers were removed in transformers 5.x; _FakeTok raises on any."""
    enc = make_hf_encoder("fake", max_tokens=8)
    v = enc.embed("hello world this is a statement")   # must not raise
    assert v.shape == (DIM,)


def test_special_tokens_wrap_every_chunk(patched):
    tok, model = patched
    enc = make_hf_encoder("fake", max_tokens=8)
    enc.embed("abcdefghijklmnop")                       # 16 chars -> several windows
    assert model.seen_ids, "model was never called"
    for row in model.seen_ids:
        real = [t for t in row if t != PAD]
        assert real[0] == CLS and real[-1] == SEP, f"chunk not wrapped: {real[:3]}..{real[-3:]}"
        assert len(real) <= 8, f"window exceeds max_tokens: {len(real)}"


# ---- correctness of the pooling ----------------------------------------------------------
def test_output_is_unit_norm_and_deterministic(patched):
    enc = make_hf_encoder("fake", max_tokens=8)
    a = enc.embed("some ecb statement text")
    b = enc.embed("some ecb statement text")
    assert np.isclose(np.linalg.norm(a), 1.0), "embeddings must be L2-normalised"
    assert np.array_equal(a, b)


def test_padding_does_not_leak_into_the_mean(patched):
    """Padded positions must be masked out; otherwise a short final chunk drags the mean."""
    tok, model = patched
    enc = make_hf_encoder("fake", max_tokens=8, batch_chunks=8)
    long = enc.embed("abcdefghijklmnopqrstuvwxyz")      # last window is short -> padding used
    widths = {w for _, w in model.batches}
    assert any(w for w in widths), "no forward recorded"
    # Same content, but forced one-chunk-per-forward so no padding can occur at all.
    model.batches.clear()
    enc2 = make_hf_encoder("fake", max_tokens=8, batch_chunks=1)
    unpadded = enc2.embed("abcdefghijklmnopqrstuvwxyz")
    assert np.allclose(long, unpadded, atol=1e-6), \
        "batched (padded) pooling disagrees with unpadded pooling"


def test_chunks_are_batched_not_looped_one_at_a_time(patched):
    tok, model = patched
    enc = make_hf_encoder("fake", max_tokens=8, batch_chunks=8)
    enc.embed("abcdefghijklmnopqrstuvwxyz")
    assert len(model.batches) == 1, f"expected one batched forward, got {model.batches}"
    assert model.batches[0][0] > 1, "batched forward should carry several chunks"


# ---- the on-disk cache -------------------------------------------------------------------
def test_cache_survives_a_new_encoder_instance(patched, tmp_path):
    tok, model = patched
    path = str(tmp_path / "emb.npz")
    enc = make_hf_encoder("fake", max_tokens=8, cache_path=path)
    v1 = enc.embed("an ecb press conference statement")
    enc.flush()
    assert os.path.exists(path)

    calls_before = len(model.batches)
    enc2 = make_hf_encoder("fake", max_tokens=8, cache_path=path)
    v2 = enc2.embed("an ecb press conference statement")
    assert np.array_equal(v1, v2)
    assert len(model.batches) == calls_before, "cache hit still ran the model"


def test_cache_miss_still_computes(patched, tmp_path):
    tok, model = patched
    path = str(tmp_path / "emb.npz")
    enc = make_hf_encoder("fake", max_tokens=8, cache_path=path)
    enc.embed("first text")
    enc.flush()
    enc2 = make_hf_encoder("fake", max_tokens=8, cache_path=path)
    before = len(model.batches)
    enc2.embed("a different text")
    assert len(model.batches) > before


def test_flush_is_atomic_and_leaves_no_temp_file(patched, tmp_path):
    """np.savez appends '.npz'; the temp name must too, or the replace cannot find it."""
    path = str(tmp_path / "emb.npz")
    enc = make_hf_encoder("fake", max_tokens=8, cache_path=path)
    enc.embed("text")
    enc.flush()                                        # must not raise FileNotFoundError
    leftovers = [f for f in os.listdir(tmp_path) if ".tmp" in f]
    assert not leftovers, f"temp file left behind: {leftovers}"


def test_flush_without_a_cache_path_is_a_no_op(patched):
    enc = make_hf_encoder("fake", max_tokens=8)
    enc.embed("text")
    enc.flush()                                        # no cache configured -> nothing to do


def test_encoder_without_cache_is_unchanged(patched):
    """The cache is opt-in: the no-cache path must behave exactly as before."""
    enc = FrozenEncoder(embed_fn=lambda t: np.ones(3) / np.sqrt(3), dim=3)
    v = enc.embed("x")
    assert v.shape == (3,) and not v.flags.writeable
    assert enc.embed("x") is v, "in-memory memo should return the same object"
