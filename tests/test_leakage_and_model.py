"""The tests that matter most: they pin the leakage bugs found on 2026-09-20 and the
structural promises the proposal makes about the model (Sec 4-5)."""
import datetime as dt

import numpy as np
import pytest
import torch

from data import Event, make_synthetic_corpus, stance_label, MARKET_VECTOR_COLUMNS
from fusion_head import FusionHead, NonAugmentedHead
from similarity import make_stub_encoder
import train


def synthetic_events(n=60, seed=0, market_dim=len(MARKET_VECTOR_COLUMNS)):
    """Synthetic events with 9-dim vectors so stance_label (which indexes OIS_1Y) works."""
    c = make_synthetic_corpus(n_docs=n, n_events=n, market_dim=market_dim, seed=seed)
    ev = sorted(c.events, key=lambda e: e.date)
    text = {e.event_id: c.doc_by_id(e.doc_id).text for e in ev}
    enc = make_stub_encoder()
    emb = {eid: enc.embed(t) for eid, t in text.items()}
    return ev, enc, emb


# ---------------- leakage boundary ----------------
def test_retrieved_precedents_are_strictly_earlier_than_query():
    ev, enc, emb = synthetic_events()
    date_of = {e.event_id: e.date for e in ev}
    for ex in train.build_examples(ev, enc, emb):
        assert all(date_of[r] < ex["event"].date for r in ex["retrieved_ids"])


def test_first_event_has_no_history_and_is_skipped_not_invented():
    ev, enc, emb = synthetic_events()
    exs = train.build_examples(ev, enc, emb)
    assert len(exs) == len(ev) - 1 and ev[0].event_id not in {e["event"].event_id for e in exs}


def test_prediction_time_retrieval_ignores_the_querys_own_market_vector():
    """THE bug: retrieval used m_q, which contains the stance label. Changing the query's market
    vector must NOT change what is retrieved (or the similarity scores) in the default setup."""
    ev, enc, emb = synthetic_events()
    base = train.build_examples(ev, enc, emb)
    ev2 = [Event(e.event_id, e.doc_id, e.date, e.market_vector + 1000.0) if i == 40 else e for i, e in enumerate(ev)]
    changed = train.build_examples(ev2, enc, emb)
    a = {x["event"].event_id: x for x in base}[ev[40].event_id]
    b = {x["event"].event_id: x for x in changed}[ev[40].event_id]
    assert a["retrieved_ids"] == b["retrieved_ids"]
    assert np.allclose(a["sim_scores"], b["sim_scores"])


def test_leaky_mode_does_depend_on_query_market_vector_proving_the_test_can_fail():
    ev, enc, emb = synthetic_events()
    ev2 = [Event(e.event_id, e.doc_id, e.date, e.market_vector + 1000.0) if i == 40 else e for i, e in enumerate(ev)]
    a = {x["event"].event_id: x for x in train.build_examples(ev, enc, emb, leaky=True)}[ev[40].event_id]
    b = {x["event"].event_id: x for x in train.build_examples(ev2, enc, emb, leaky=True)}[ev[40].event_id]
    assert not np.allclose(a["sim_scores"], b["sim_scores"])


def test_model_input_contains_no_trace_of_the_query_label():
    """Features = query TEXT embedding + PAST events' outcomes. Changing only the query's own
    market vector (hence its label) must leave every model input unchanged."""
    ev, enc, emb = synthetic_events()
    ev2 = [Event(e.event_id, e.doc_id, e.date, e.market_vector * 0 + 7.0) if i == 40 else e for i, e in enumerate(ev)]
    a = {x["event"].event_id: x for x in train.build_examples(ev, enc, emb)}[ev[40].event_id]
    b = {x["event"].event_id: x for x in train.build_examples(ev2, enc, emb)}[ev[40].event_id]
    assert a["label"] != b["label"]                                   # label DID change
    assert np.array_equal(a["e_q"], b["e_q"]) and np.array_equal(a["retrieved_m"], b["retrieved_m"])


def test_faithfulness_ranking_ignores_query_market_vector():
    from faithfulness import rank_history
    ev, enc, emb = synthetic_events()
    q, hist = ev[40], ev[:40]
    q2 = Event(q.event_id, q.doc_id, q.date, q.market_vector + 500.0)
    r1 = [c.event_id for _, c in rank_history(q, hist, emb)]
    r2 = [c.event_id for _, c in rank_history(q2, hist, emb)]
    assert r1 == r2


def test_stance_label_reads_the_documented_column():
    v = np.arange(len(MARKET_VECTOR_COLUMNS), dtype=float)
    assert stance_label(v) == float(MARKET_VECTOR_COLUMNS.index("OIS_1Y"))


# ---------------- model structure ----------------
def make_inputs(b=3, k=5, h=16, m=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, h, generator=g), torch.randn(b, k, m, generator=g), torch.rand(b, k, generator=g))


def test_fusion_output_shapes_positive_sigma_and_attention_sums_to_one():
    e, mv, s = make_inputs()
    out = FusionHead(16, 9)(e, mv, s)
    assert out.s_hat.shape == (3,) and out.sigma_hat.shape == (3,) and out.attn_weights.shape == (3, 5)
    assert (out.sigma_hat > 0).all()
    assert torch.allclose(out.attn_weights.sum(-1), torch.ones(3), atol=1e-5)


def test_fusion_head_is_not_a_knn_every_parameter_receives_gradient():
    """Proposal Sec 4: 'the model is not reducible to a non-parametric k-NN estimator'."""
    torch.manual_seed(0)
    m = FusionHead(16, 9)
    e, mv, s = make_inputs()
    out = m(e, mv, s)
    (out.s_hat.pow(2).sum() + out.sigma_hat.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_fusion_is_invariant_to_order_of_retrieved_precedents():
    torch.manual_seed(0)
    m = FusionHead(16, 9).eval()
    e, mv, s = make_inputs(b=1)
    perm = torch.randperm(5)
    a = m(e, mv, s).s_hat
    b = m(e, mv[:, perm], s[:, perm]).s_hat
    assert torch.allclose(a, b, atol=1e-5)


def test_fusion_responds_to_retrieved_outcomes():
    torch.manual_seed(0)
    m = FusionHead(16, 9).eval()
    e, mv, s = make_inputs(b=1)
    assert not torch.allclose(m(e, mv, s).s_hat, m(e, mv + 5.0, s).s_hat)


def test_padded_slots_are_ignored():
    """to_batch pads short retrieved sets with sim=-1e9; padded slots must not influence the output."""
    torch.manual_seed(0)
    m = FusionHead(16, 9).eval()
    e, mv, s = make_inputs(b=1, k=5)
    s = s.clone(); s[:, 3:] = -1e9
    mv2 = mv.clone(); mv2[:, 3:] = 12345.0
    assert torch.allclose(m(e, mv, s).s_hat, m(e, mv2, s).s_hat, atol=1e-4)


def test_non_augmented_head_cannot_see_retrieved_evidence():
    torch.manual_seed(0)
    b = NonAugmentedHead(16).eval()
    e, mv, s = make_inputs(b=2)
    out = b(e)
    assert out.s_hat.shape == (2,) and (out.sigma_hat > 0).all()
    import inspect
    assert list(inspect.signature(b.forward).parameters) == ["e_q"]


def test_gaussian_nll_hand_computed_and_minimised_at_truth():
    s_hat, sig, y = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])
    expected = 0.5 * (((3 - 1) / 2) ** 2 + 2 * np.log(2.0))
    assert float(train.gaussian_nll(s_hat, sig, y)) == pytest.approx(expected)
    assert float(train.gaussian_nll(y, sig, y)) < float(train.gaussian_nll(s_hat, sig, y))


def test_training_reduces_loss_on_a_learnable_toy_problem():
    """Sanity check the training loop can learn something: label = a function of the retrieved outcomes."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    exs = []
    for _ in range(80):
        rm = rng.normal(size=(5, 9)).astype(np.float32)
        exs.append({"e_q": rng.normal(size=16).astype(np.float32), "retrieved_m": rm,
                    "sim_scores": np.ones(5, np.float32), "label": float(rm[:, 3].mean())})
    model = FusionHead(16, 9)
    pre = np.mean([abs(p - e["label"]) for p, e in zip(train.predict(model, exs, 16, 9, 5, True, 0.0, 1.0), exs)])
    train.train_model(model, exs, 16, 9, 5, True, 0.0, 1.0, epochs=60)
    post = np.mean([abs(p - e["label"]) for p, e in zip(train.predict(model, exs, 16, 9, 5, True, 0.0, 1.0), exs)])
    assert post < 0.7 * pre


# ---------------- regimes ----------------
def test_regime_boundaries():
    from regime_robustness import regime
    assert regime(dt.date(2014, 5, 31)) == "pre_zlb" and regime(dt.date(2014, 6, 1)) == "zlb"
    assert regime(dt.date(2022, 6, 30)) == "zlb" and regime(dt.date(2022, 7, 1)) == "post_hiking"


# ---------------- input standardization ----------------
def test_fusion_head_is_identity_standardized_until_stats_are_set():
    m = FusionHead(16, 9)
    assert torch.equal(m.m_mean, torch.zeros(9)) and torch.equal(m.m_std, torch.ones(9))
    m.set_input_stats(np.full(9, 2.0), np.full(9, 4.0))
    e, mv, s = make_inputs(b=1)
    m.eval()
    a = m(e, mv, s).s_hat
    m2 = FusionHead(16, 9)
    m2.load_state_dict(m.state_dict())
    m2.set_input_stats(np.zeros(9), np.ones(9))
    assert not torch.allclose(a, m2(e, mv, s).s_hat)                     # stats actually change the output
    # ...and standardizing inside the model equals feeding pre-standardized inputs to an identity model:
    assert torch.allclose(a, m2.eval()(e, (mv - 2.0) / 4.0, s).s_hat, atol=1e-5)


def test_input_stats_come_from_training_examples_only():
    rng = np.random.default_rng(0)
    def ex(loc):
        return {"e_q": rng.normal(size=16).astype(np.float32), "retrieved_m": (loc + rng.normal(size=(5, 9))).astype(np.float32),
                "sim_scores": np.ones(5, np.float32), "label": float(rng.normal())}
    train_ex = [ex(100.0) for _ in range(20)]
    model = FusionHead(16, 9)
    train.train_model(model, train_ex, 16, 9, 5, True, 0.0, 1.0, epochs=1, verbose=False)
    assert np.allclose(model.m_mean.numpy(), 100.0, atol=1.0)            # fitted on training rows
    raw = FusionHead(16, 9)
    train.train_model(raw, train_ex, 16, 9, 5, True, 0.0, 1.0, epochs=1, verbose=False, standardize_inputs=False)
    assert torch.equal(raw.m_mean, torch.zeros(9))                        # switch off -> untouched


# ---------------- early stopping ----------------
def _toy_examples(n, flip_last_fraction=0.0, seed=0):
    rng = np.random.default_rng(seed)
    exs = []
    for i in range(n):
        rm = rng.normal(size=(5, 9)).astype(np.float32)
        y = float(rm[:, 3].mean())
        if i >= n * (1 - flip_last_fraction):
            y = -y                                        # validation slice contradicts the training signal
        exs.append({"e_q": rng.normal(size=16).astype(np.float32), "retrieved_m": rm,
                    "sim_scores": np.ones(5, np.float32), "label": y})
    return exs


def test_early_stopping_uses_the_latest_training_slice_and_restores_the_best_epoch():
    exs = _toy_examples(100, flip_last_fraction=0.2)      # last 20 (the validation slice) have flipped labels
    torch.manual_seed(0)
    m = FusionHead(16, 9)
    train.train_model(m, exs, 16, 9, 5, True, 0.0, 1.0, epochs=40, verbose=False)
    assert 0 <= m.best_epoch < 39                          # validation got worse as training proceeded -> stopped early
    val = exs[-20:]
    torch.manual_seed(0)
    m_full = FusionHead(16, 9)
    train.train_model(m_full, exs[:-20], 16, 9, 5, True, 0.0, 1.0, epochs=40, verbose=False, early_stopping_frac=0)  # same 80 fit examples
    assert m_full.best_epoch == -1
    v_es = train._val_mse(m, val, 16, 9, 5, True, 0.0, 1.0)
    v_full = train._val_mse(m_full, val, 16, 9, 5, True, 0.0, 1.0)
    assert v_es < v_full                                   # restoring the best epoch beat training to the end


def test_early_stopping_skipped_for_tiny_training_sets():
    exs = _toy_examples(20)
    m = FusionHead(16, 9)
    train.train_model(m, exs, 16, 9, 5, True, 0.0, 1.0, epochs=2, verbose=False)
    assert m.best_epoch == -1


def test_early_stopping_never_looks_at_examples_outside_the_list_it_is_given():
    exs = _toy_examples(60)
    heldout = _toy_examples(30, seed=9)
    torch.manual_seed(0)
    a = FusionHead(16, 9); train.train_model(a, exs, 16, 9, 5, True, 0.0, 1.0, epochs=5, verbose=False)
    for e in heldout:                                     # scramble held-out data completely
        e["label"] = 1e6; e["retrieved_m"] = e["retrieved_m"] * 1e3
    torch.manual_seed(0)
    b = FusionHead(16, 9); train.train_model(b, exs, 16, 9, 5, True, 0.0, 1.0, epochs=5, verbose=False)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_patience_stops_training_once_validation_stops_improving():
    exs = _toy_examples(100, flip_last_fraction=0.2)     # validation contradicts training -> best epoch is early
    calls = {"n": 0}
    orig = train._val_mse

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    train._val_mse = counting
    try:
        torch.manual_seed(0)
        m = FusionHead(16, 9)
        train.train_model(m, exs, 16, 9, 5, True, 0.0, 1.0, epochs=100, verbose=False, patience=5)
    finally:
        train._val_mse = orig
    assert calls["n"] < 100 and calls["n"] <= m.best_epoch + 1 + 5      # stopped 5 epochs after the best one
