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

import atexit
import dataclasses
import hashlib
import os
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

    def __init__(self, embed_fn: Callable[[str], np.ndarray], dim: int,
                 cache_path: Optional[str] = None):
        self._embed_fn = embed_fn
        self.dim = dim
        # A frozen encoder is a pure function of the text, so memoizing is value-preserving.
        # Without this the baselines re-embedded every candidate document on every comparison
        # (O(events^2) embeddings of ~6,000-word statements), which made run_real.py take hours.
        self._cache: dict[str, np.ndarray] = {}
        # Optional ON-DISK cache, keyed by sha256(text). The in-memory dict above dies with
        # the process, and the experiment suite is ~9 separate scripts: with a real
        # transformer that meant re-embedding all 273 statements (~6,000 words each) once per
        # script. Persisting makes the encoder a one-off cost for the whole suite.
        self._cache_path = cache_path
        self._dirty = False
        if cache_path and os.path.exists(cache_path):
            with np.load(cache_path) as z:
                self._disk = {k: z[k] for k in z.files}
        else:
            self._disk = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is not None:
            return v
        if self._cache_path:
            cached = self._disk.get(self._key(text))
            if cached is not None and cached.shape == (self.dim,):
                cached = np.asarray(cached, dtype=np.float64)
                cached.setflags(write=False)
                self._cache[text] = cached
                return cached
        v = self._embed_fn(text)
        assert v.shape == (self.dim,), f"expected shape ({self.dim},), got {v.shape}"
        if self._cache_path:
            self._disk[self._key(text)] = v
            self._dirty = True
        v.setflags(write=False)  # shared across callers: make accidental mutation an error
        self._cache[text] = v
        return v

    def flush(self) -> None:
        """Persist newly computed vectors. Written to a temp file then replaced, so an
        interrupted run cannot leave a half-written cache behind."""
        if not (self._cache_path and self._dirty):
            return
        d = os.path.dirname(os.path.abspath(self._cache_path))
        os.makedirs(d, exist_ok=True)
        # np.savez APPENDS ".npz" unless the name already ends with it, so the temp name
        # must too -- otherwise savez writes "<tmp>.npz" and the replace below cannot find
        # the file it just wrote.
        tmp = self._cache_path + ".tmp.npz"
        np.savez(tmp, **self._disk)
        os.replace(tmp, self._cache_path)
        self._dirty = False


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
def make_hf_encoder(path: str, max_tokens: int = 512, pooling: str = "mean",
                    device: str = "cpu", batch_chunks: int = 8,
                    cache_path: Optional[str] = None) -> FrozenEncoder:
    """Wrap a Hugging Face-style checkpoint as a FrozenEncoder.

    Tested against microsoft/deberta-v3-base and against the companion work's DAPT /
    fine-tuned DeBERTa checkpoints (transformers 5.x, torch 2.14).

    Recipe: chunk each long document into <= max_tokens windows, take the mean-pooled (or
    [CLS]) last hidden state of each chunk, average the chunks, L2-normalise. ECB statements
    run to several thousand tokens, far beyond one window, hence the chunking.

    If the companion encoder ever exposes a different interface (custom head,
    sentence-transformers, different pooling), replace this with a
    FrozenEncoder(embed_fn, dim) around that interface; nothing else changes.

    ``device`` stays "cpu" by default: the corpus is only a few hundred documents, so the
    GPU buys little and a CPU run cannot disturb a training job sharing the card.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(device)
    dim = int(model.config.hidden_size)

    # Discover this tokenizer's special-token wrapper empirically rather than calling a
    # helper: transformers 5.x removed both ``prepare_for_model`` and
    # ``build_inputs_with_special_tokens`` from the tokenizer API, and hard-coding
    # [CLS]/[SEP] would silently mis-wrap a model that uses a different convention.
    _bare = tok("probe", add_special_tokens=False)["input_ids"]
    _wrapped = tok("probe", add_special_tokens=True)["input_ids"]
    _at = next((i for i in range(len(_wrapped) - len(_bare) + 1)
                if _wrapped[i:i + len(_bare)] == _bare), 0)
    prefix, suffix = _wrapped[:_at], _wrapped[_at + len(_bare):]
    step = max_tokens - len(prefix) - len(suffix)
    if step < 1:
        raise ValueError(f"max_tokens={max_tokens} too small for this tokenizer's special tokens")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    @torch.no_grad()
    def embed(text: str) -> np.ndarray:
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        chunks = [ids[i:i + step] for i in range(0, max(len(ids), 1), step)] or [[]]
        rows = [list(prefix) + list(c) + list(suffix) for c in chunks]
        vecs = []
        # One padded forward pass per group of chunks instead of one per chunk: a 6,000-word
        # statement is ~17 windows, and looping them individually left the hardware idle
        # between tiny calls (~107 s/document on CPU).
        for i in range(0, len(rows), batch_chunks):
            group = rows[i:i + batch_chunks]
            w = max(len(r) for r in group)
            input_ids = torch.tensor([r + [pad_id] * (w - len(r)) for r in group], device=device)
            attention = torch.tensor([[1] * len(r) + [0] * (w - len(r)) for r in group], device=device)
            out = model(input_ids=input_ids, attention_mask=attention).last_hidden_state
            if pooling == "cls":
                vecs.append(out[:, 0])
            else:
                # Mean over REAL tokens only; padding must not drag the mean toward zero.
                m = attention.unsqueeze(-1).to(out.dtype)
                vecs.append((out * m).sum(1) / m.sum(1).clamp(min=1e-9))
        v = torch.cat(vecs, 0).mean(0).float().cpu().numpy().astype(np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    enc = FrozenEncoder(embed_fn=embed, dim=dim, cache_path=cache_path)
    atexit.register(enc.flush)
    return enc


def make_default_encoder() -> FrozenEncoder:
    """The stub unless PRECEDENT_ENCODER is set (see above).

    Two optional companions, both only meaningful with a real checkpoint:

    * ``PRECEDENT_ENCODER_DEVICE`` -- "cuda" to embed on the GPU ("cpu" default, so a
      training job sharing the card is never disturbed);
    * ``PRECEDENT_ENCODER_CACHE`` -- path to an .npz vector cache shared by every script
      in the suite. Defaults to ``cache/emb_<checkpoint>.npz`` so the ~273 statements are
      embedded once in total rather than once per script. The filename carries a hash of
      the checkpoint identifier, so switching checkpoints can never silently reuse the
      previous model's vectors.
    """
    import os

    path = os.environ.get("PRECEDENT_ENCODER")
    if not path:
        return make_stub_encoder()
    device = os.environ.get("PRECEDENT_ENCODER_DEVICE", "cpu")
    tag = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    cache = os.environ.get("PRECEDENT_ENCODER_CACHE",
                           os.path.join("cache", f"emb_{tag}.npz"))
    return make_hf_encoder(path, device=device, cache_path=cache)
