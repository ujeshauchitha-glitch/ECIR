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

    def embed(self, text: str) -> np.ndarray:
        v = self._embed_fn(text)
        assert v.shape == (self.dim,), f"expected shape ({self.dim},), got {v.shape}"
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
) -> SimilarityResult:
    """sim(q, d_i) per Sec 4. If m_q is None (genuine out-of-sample
    prediction time), falls back to text-only similarity and reports that
    explicitly via `mode` rather than silently mixing in a market term that
    doesn't exist yet.
    """
    text_term = cosine_sim(e_q, e_i)
    if m_q is None:
        return SimilarityResult(score=text_term, mode="text_only_fallback")
    market_term = float(np.exp(-np.sum((m_q - m_i) ** 2) / tau))
    score = lam * text_term + (1 - lam) * market_term
    return SimilarityResult(score=score, mode="hybrid")
