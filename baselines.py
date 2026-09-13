"""
Baselines, per proposal Sec 5:
  - Lexical (BM25)
  - Text-only dense retrieval (lambda = 1)
  - Market-signal-only retrieval (lambda = 0)
  - Random / nearest-date floor

BM25 is implemented from scratch below (no `rank_bm25` package available in
this environment) -- standard Robertson/Sparck-Jones BM25 with k1=1.5, b=0.75.
"""
from __future__ import annotations

import collections
import math
from typing import Optional

import numpy as np

from data import Event
from similarity import FrozenEncoder, cosine_sim, hybrid_similarity


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids = list(docs.keys())
        self.tokenized = {did: docs[did].lower().split() for did in self.doc_ids}
        self.doc_len = {did: len(toks) for did, toks in self.tokenized.items()}
        self.avgdl = float(np.mean(list(self.doc_len.values()))) if self.doc_ids else 0.0

        df: dict[str, int] = collections.defaultdict(int)
        self.tf: dict[str, dict[str, int]] = {}
        for did, toks in self.tokenized.items():
            counts = collections.Counter(toks)
            self.tf[did] = counts
            for term in counts:
                df[term] += 1
        n = len(self.doc_ids)
        self.idf = {
            term: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5)) for term, dfi in df.items()
        }

    def score(self, query: str, doc_id: str) -> float:
        q_terms = query.lower().split()
        tf = self.tf[doc_id]
        dl = self.doc_len[doc_id]
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (f * (self.k1 + 1)) / denom if denom > 0 else 0.0
        return s


def rank_candidates(
    query_event: Event,
    candidate_events: list[Event],
    mode: str,
    *,
    encoder: Optional[FrozenEncoder] = None,
    doc_text_by_event: Optional[dict[str, str]] = None,
    bm25: Optional[BM25] = None,
    lam: float = 0.5,
    tau: float = 1.0,
    seed: int = 0,
) -> list[str]:
    """Returns candidate event_ids ranked best-first. mode in:
    {"bm25", "dense_text_only", "market_only", "hybrid", "random"}.
    """
    cand_ids = [e.event_id for e in candidate_events]

    if mode == "random":
        rng = np.random.default_rng(seed)
        shuffled = cand_ids.copy()
        rng.shuffle(shuffled)
        return shuffled

    if mode == "bm25":
        assert bm25 is not None and doc_text_by_event is not None
        q_text = doc_text_by_event[query_event.event_id]
        # bm25.score(query_text, doc_id) -- BM25 must be constructed with
        # doc_id == event_id keys so `cid` indexes it correctly.
        scored = [(bm25.score(q_text, cid), cid) for cid in cand_ids]
        scored.sort(reverse=True)
        return [cid for _, cid in scored]

    if mode == "dense_text_only":
        assert encoder is not None and doc_text_by_event is not None
        e_q = encoder.embed(doc_text_by_event[query_event.event_id])
        scored = []
        for c in candidate_events:
            e_c = encoder.embed(doc_text_by_event[c.event_id])
            scored.append((cosine_sim(e_q, e_c), c.event_id))
        scored.sort(reverse=True)
        return [cid for _, cid in scored]

    if mode == "market_only":
        scored = []
        for c in candidate_events:
            d = float(np.linalg.norm(query_event.market_vector - c.market_vector))
            scored.append((-d, c.event_id))  # smaller distance -> higher score
        scored.sort(reverse=True)
        return [cid for _, cid in scored]

    if mode == "hybrid":
        assert encoder is not None and doc_text_by_event is not None
        e_q = encoder.embed(doc_text_by_event[query_event.event_id])
        scored = []
        for c in candidate_events:
            e_c = encoder.embed(doc_text_by_event[c.event_id])
            res = hybrid_similarity(
                e_q, e_c, c.market_vector, query_event.market_vector, lam=lam, tau=tau
            )
            scored.append((res.score, c.event_id))
        scored.sort(reverse=True)
        return [cid for _, cid in scored]

    raise ValueError(f"unknown mode: {mode}")
