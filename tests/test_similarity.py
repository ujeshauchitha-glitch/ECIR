import os
import subprocess
import sys

import numpy as np
import pytest

from similarity import cosine_sim, hybrid_similarity, make_stub_encoder


def test_cosine_basics():
    a = np.array([1.0, 0.0])
    assert cosine_sim(a, a) == pytest.approx(1.0)
    assert cosine_sim(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_sim(a, -a) == pytest.approx(-1.0)
    assert cosine_sim(a, np.zeros(2)) == 0.0        # zero vector must not NaN


def test_hybrid_matches_formula_by_hand():
    e_q, e_i = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    m_q, m_i = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    lam, tau = 0.3, 2.0
    expected = lam * (1 / np.sqrt(2)) + (1 - lam) * np.exp(-1.0 / tau)   # ||m_q-m_i||^2 = 1
    r = hybrid_similarity(e_q, e_i, m_i, m_q, lam=lam, tau=tau)
    assert r.score == pytest.approx(expected) and r.mode == "hybrid"


def test_lambda_extremes():
    e_q, e_i = np.array([1.0, 2.0]), np.array([2.0, 1.0])
    m_q, m_i = np.array([0.0]), np.array([3.0])
    assert hybrid_similarity(e_q, e_i, m_i, m_q, lam=1.0, tau=1.0).score == pytest.approx(cosine_sim(e_q, e_i))
    assert hybrid_similarity(e_q, e_i, m_i, m_q, lam=0.0, tau=1.0).score == pytest.approx(np.exp(-9.0))


def test_market_term_is_one_for_identical_outcomes():
    m = np.array([1.0, -2.0, 3.0])
    r = hybrid_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]), m, m, lam=0.0, tau=1.0)
    assert r.score == pytest.approx(1.0)


def test_missing_query_market_vector_falls_back_to_text_only_regardless_of_lambda():
    e_q, e_i, m_i = np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([5.0])
    for lam in (0.0, 0.5, 1.0):
        r = hybrid_similarity(e_q, e_i, m_i, None, lam=lam, tau=1.0)
        assert r.mode == "text_only_fallback" and r.score == pytest.approx(cosine_sim(e_q, e_i))


def test_stub_encoder_shape_norm_and_empty_text():
    enc = make_stub_encoder(dim=16)
    v = enc.embed("inflation tighten firm")
    assert v.shape == (16,) and np.linalg.norm(v) == pytest.approx(1.0)
    assert np.all(enc.embed("") == 0)
    assert np.allclose(enc.embed("a b c"), enc.embed("a b c"))


def test_stub_encoder_identical_across_processes_regression():
    """Regression: the stub used salted builtin hash(), so every process produced different embeddings."""
    code = ("from similarity import make_stub_encoder; "
            "print(list(make_stub_encoder().embed('inflation tighten accommodative')[:4]))")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = []
    for seed in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                   env=env, cwd=root).stdout)
    assert outs[0] == outs[1] == outs[2] != ""


def test_invalid_hyperparameters_are_rejected_not_silently_used():
    e, m = np.array([1.0, 0.0]), np.array([0.0])
    for lam in (-0.1, 1.5):
        with pytest.raises(ValueError):
            hybrid_similarity(e, e, m, m, lam=lam, tau=1.0)
    for tau in (0.0, -1.0):
        with pytest.raises(ValueError):
            hybrid_similarity(e, e, m, m, lam=0.5, tau=tau)


def test_squared_vs_l2_kernel_both_hand_computed():
    e = np.array([1.0, 0.0])
    m_q, m_i = np.array([0.0, 0.0]), np.array([3.0, 4.0])          # ||d|| = 5, ||d||^2 = 25
    sq = hybrid_similarity(e, e, m_i, m_q, lam=0.0, tau=10.0, squared=True).score
    l2 = hybrid_similarity(e, e, m_i, m_q, lam=0.0, tau=10.0, squared=False).score
    assert sq == pytest.approx(np.exp(-25 / 10)) and l2 == pytest.approx(np.exp(-5 / 10))
    assert hybrid_similarity(e, e, m_i, m_q, lam=0.0, tau=10.0).score == pytest.approx(sq)   # default unchanged


def test_default_encoder_is_the_stub_unless_env_var_set(monkeypatch):
    from similarity import make_default_encoder
    monkeypatch.delenv("PRECEDENT_ENCODER", raising=False)
    a, b = make_default_encoder(), make_stub_encoder()
    assert a.dim == b.dim and np.array_equal(a.embed("inflation firm"), b.embed("inflation firm"))


def test_encoder_cache_returns_read_only_shared_vector():
    enc = make_stub_encoder()
    v = enc.embed("inflation")
    assert enc.embed("inflation") is v                       # memoized
    with pytest.raises(ValueError):
        v[0] = 1.0                                           # shared vector cannot be mutated by accident
