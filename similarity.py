"""
Encoder interface + hybrid similarity, matching the proposal's Sec 4 formula:

    sim(q, d_i) = lambda * cos(e_q, e_i) + (1 - lambda) * exp(-||m_q - m_i||^2 / tau)

OPEN QUESTION TO RAISE WITH YOUR TEAMMATE BEFORE THIS GOES IN THE PAPER:
The proposal states (Sec 4) that "at genuine prediction time, m_q is by
definition unobserved" -- the query event's own market outcome hasn't
happened yet. But the hybrid formula as written needs m_q to compute the
market-similarity term for retrieval. Those two statements are in tension:
either (a) the hybrid formula is only ever used at TRAIN time, where m_q is
known for historical query events, and at genuine out-of-sample prediction
time retrieval silently falls back to lambda=1 (text-only), or (b) there's a
different mechanism intended that isn't fully specified yet. This is not a
detail to guess at silently -- it changes what "hybrid retrieval" even means
at deployment/test time and affects how the ablation over lambda should be
read. `hybrid_similarity` below implements (a) explicitly: if m_q is None,
it forces lambda=1.0 and returns which mode was used, so the caller (and the
paper) can be honest about which regime produced which result.
"""
from __future__ import annotations

import dataclasses
import zlib
from typing import Callable, Optional

import numpy as np


class FrozenEncoder:
    """Interface for the frozen encoder from the concurrent submission.

    TO WIRE UP THE REAL ENCODER: replace `embed` with a call into the actual
    trained model (e.g. load checkpoint, run forward pass, return the pooled
    embedding). Everything downstream only depends on this interface -- a
    callable text -> np.ndarray of fixed dimension.
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray], dim: int):
        self._embed_fn = embed_fn
        self.dim = dim
        # A frozen encoder is a pure function of the text, so memoizing is value-preserving.
        # Without this the baselines re-embedded every candidate document on every comparison
        # (O(events^2) embeddings of ~6,000-word statements), which made run_real.py take hours.
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            v = self._embed_fn(text)
            assert v.shape == (self.dim,), f"expected shape ({self.dim},), got {v.shape}"
            v.setflags(write=False)  # shared across callers: make accidental mutation an error
            self._cache[text] = v
        return v


def make_stub_encoder(dim: int = 32, seed: int = 0) -> FrozenEncoder:
    """STUB ONLY. Produces a deterministic hash-based embedding so the
    pipeline is runnable before the real frozen encoder is wired in. This
    embedding carries no real semantic content beyond simple bag-of-words
    overlap -- do not use it to support any claim about retrieval quality.
    """
    rng_master = np.random.default_rng(seed)
    word_vectors: dict[str, np.ndarray] = {}

    def embed(text: str) -> np.ndarray:
        vecs = []
        for w in text.split():
            if w not in word_vectors:
                # Deterministic per-word vector, seeded off the word itself
                # so the same word always maps to the same vector.
                # zlib.crc32, NOT builtin hash(): str hash() is salted per
                # process (PYTHONHASHSEED), which made this "deterministic"
                # stub return different embeddings on every run.
                h = zlib.crc32(w.encode("utf-8"))
                word_vectors[w] = np.random.default_rng(h).normal(0, 1, size=dim)
            vecs.append(word_vectors[w])
        if not vecs:
            return np.zeros(dim)
        v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    return FrozenEncoder(embed_fn=embed, dim=dim)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclasses.dataclass
class SimilarityResult:
    score: float
    mode: str  # "hybrid" or "text_only_fallback"


def hybrid_similarity(
    e_q: np.ndarray,
    e_i: np.ndarray,
    m_i: np.ndarray,
    m_q: Optional[np.ndarray],
    lam: float,
    tau: float,
    squared: bool = True,
) -> SimilarityResult:
    """sim(q, d_i) per Sec 4. If m_q is None (genuine out-of-sample
    prediction time), falls back to text-only similarity and reports that
    explicitly via `mode` rather than silently mixing in a market term that
    doesn't exist yet.

    UNRESOLVED DISCREPANCY (flagged 2026-09-20): the proposal PDF (Sec 4,
    rendered page 4) shows the market term as exp(-||m_q - m_i||_2 / tau) --
    the L2 norm, NOT squared -- whereas this project's code and the formula
    quoted in the owner's original brief use the SQUARED distance,
    exp(-||m_q - m_i||^2 / tau). Default (squared=True) keeps what was
    built and run so far; squared=False gives the PDF's reading. The two
    are different kernels (for tau=1 they differ a lot), so confirm which
    one Sec 4 intends before any number is reported. See ablations.py for
    the sensitivity of the results to this choice.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    text_term = cosine_sim(e_q, e_i)
    if m_q is None:
        return SimilarityResult(score=text_term, mode="text_only_fallback")
    sq = float(np.sum((m_q - m_i) ** 2))
    market_term = float(np.exp(-(sq if squared else np.sqrt(sq)) / tau))
    score = lam * text_term + (1 - lam) * market_term
    return SimilarityResult(score=score, mode="hybrid")


# ---------------------------------------------------------------------------------------------
# Real-encoder switch. The scripts call make_default_encoder(); set the environment variable
#     PRECEDENT_ENCODER=<path-or-name of a Hugging Face-style checkpoint>
# to use a real model everywhere at once, then RE-RUN every experiment. Unset -> the stub.
# ---------------------------------------------------------------------------------------------
def make_hf_encoder(path: str, max_tokens: int = 512, pooling: str = "mean") -> FrozenEncoder:
    """UNTESTED against a real checkpoint (transformers is not installed in the environment this was
    written in, and the trained encoder from the companion work was never available). It follows the
    standard recipe -- chunk each long document into <= max_tokens windows, take the mean-pooled (or
    [CLS]) last hidden state of each chunk, average the chunks, L2-normalise -- but if the companion
    encoder uses a different interface (custom head, sentence-transformers, different pooling),
    replace this with a FrozenEncoder(embed_fn, dim) around that interface; nothing else changes.
    Statements are ~6,000 words, far beyond one window, hence the chunking.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval()
    dim = int(model.config.hidden_size)
    step = max_tokens - 2  # room for special tokens

    @torch.no_grad()
    def embed(text: str) -> np.ndarray:
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        chunks = [ids[i:i + step] for i in range(0, max(len(ids), 1), step)] or [[]]
        vecs = []
        for c in chunks:
            enc = tok.prepare_for_model(c, max_length=max_tokens, truncation=True, return_tensors="pt")
            out = model(**{k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in enc.items()}).last_hidden_state[0]
            vecs.append(out[0] if pooling == "cls" else out.mean(0))
        v = torch.stack(vecs).mean(0).numpy().astype(np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    return FrozenEncoder(embed_fn=embed, dim=dim)


def make_default_encoder() -> FrozenEncoder:
    """The stub unless PRECEDENT_ENCODER is set (see above)."""
    import os

    path = os.environ.get("PRECEDENT_ENCODER")
    return make_hf_encoder(path) if path else make_stub_encoder()
