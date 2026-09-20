import datetime as dt

import numpy as np
import pytest

from baselines import BM25, rank_candidates
from benchmark import build_ecb_precedent, pairwise_distances, relevance_lookup
from data import Event


def mk_events(vectors):
    return [Event(event_id=f"e{i}", doc_id=f"d{i}", date=dt.date(2000 + i, 1, 1), market_vector=np.array(v, float))
            for i, v in enumerate(vectors)]


# ---------------- benchmark ----------------
def test_pairwise_distances_hand_computed():
    ev = mk_events([[0, 0], [3, 4]])
    d = pairwise_distances(ev)
    assert d[0, 1] == pytest.approx(5.0) and d[0, 0] == 0.0 and d[1, 0] == d[0, 1]


def test_relevance_symmetric_no_self_pairs_and_bounded():
    rng = np.random.default_rng(0)
    ev = mk_events(rng.normal(size=(40, 3)))
    pairs, thr = build_ecb_precedent(ev, percentile=10.0)
    rel = relevance_lookup(pairs)
    assert all(a != b for a, b in rel)                              # never pairs an event with itself
    assert all(0.0 < r <= 1.0 for r in rel.values())
    for (a, b), r in rel.items():
        assert rel[(b, a)] == pytest.approx(r)                      # symmetric
    assert thr > 0


def test_threshold_is_bottom_decile_of_pairwise_distances():
    rng = np.random.default_rng(1)
    ev = mk_events(rng.normal(size=(60, 4)))
    pairs, thr = build_ecb_precedent(ev, percentile=10.0)
    n = len(ev)
    off = pairwise_distances(ev)[~np.eye(n, dtype=bool)]
    frac_below = np.mean(off < thr)
    assert 0.08 <= frac_below <= 0.11                               # ~10% of ordered pairs are 'relevant'
    assert len(pairs) == int(np.sum(off < thr))


def test_relevance_decreases_with_distance_and_closest_pair_is_best():
    ev = mk_events([[0.0], [0.1], [0.5], [5.0], [9.0], [20.0], [40.0], [80.0], [160.0], [320.0]])
    pairs, _ = build_ecb_precedent(ev, percentile=30.0)
    by_d = sorted(pairs, key=lambda p: p.distance)
    rels = [p.relevance for p in by_d]
    assert rels == sorted(rels, reverse=True)
    assert by_d[0].distance == pytest.approx(0.1)


def test_identical_vectors_do_not_crash_and_give_no_pairs():
    ev = mk_events([[1, 1]] * 6)
    pairs, thr = build_ecb_precedent(ev)
    assert thr == 0.0 and pairs == []                               # d<0 is never true: degenerate but safe


# ---------------- BM25 ----------------
def test_bm25_prefers_docs_containing_query_terms():
    docs = {"a": "inflation tighten inflation", "b": "accommodative support easing", "c": "inflation"}
    bm = BM25(docs)
    assert bm.score("inflation", "a") > bm.score("inflation", "b") == 0.0
    assert bm.score("inflation", "c") > 0


def test_bm25_empty_query_and_unseen_terms_score_zero():
    bm = BM25({"a": "x y z", "b": "y z"})
    assert bm.score("", "a") == 0.0 and bm.score("nonexistent", "a") == 0.0


def test_bm25_idf_rarer_term_scores_higher():
    docs = {"a": "common rare", "b": "common", "c": "common", "d": "common"}
    bm = BM25(docs)
    assert bm.score("rare", "a") > bm.score("common", "a")


def test_bm25_is_case_insensitive():
    bm = BM25({"a": "Inflation Tighten"})
    assert bm.score("INFLATION", "a") == bm.score("inflation", "a") > 0


# ---------------- rank_candidates ----------------
def test_market_only_ranks_nearest_outcome_first():
    ev = mk_events([[0.0], [10.0], [1.0], [5.0]])
    q, cands = ev[0], ev[1:]
    assert rank_candidates(q, cands, "market_only") == ["e2", "e3", "e1"]


def test_random_is_reproducible_by_seed_and_a_permutation():
    ev = mk_events(np.arange(10.0).reshape(-1, 1))
    q, cands = ev[0], ev[1:]
    a = rank_candidates(q, cands, "random", seed=3)
    assert a == rank_candidates(q, cands, "random", seed=3)
    assert sorted(a) == sorted(e.event_id for e in cands)


def test_unknown_mode_raises():
    ev = mk_events([[0.0], [1.0]])
    with pytest.raises(ValueError):
        rank_candidates(ev[0], ev[1:], "nope")


def test_rankings_are_permutations_of_candidates_for_every_mode():
    from similarity import make_stub_encoder
    ev = mk_events(np.random.default_rng(2).normal(size=(8, 2)))
    text = {e.event_id: f"word{i} inflation" for i, e in enumerate(ev)}
    enc, bm = make_stub_encoder(), BM25(text)
    for mode in ("bm25", "dense_text_only", "market_only", "hybrid", "random"):
        r = rank_candidates(ev[0], ev[1:], mode, encoder=enc, doc_text_by_event=text, bm25=bm)
        assert sorted(r) == sorted(e.event_id for e in ev[1:]), mode
