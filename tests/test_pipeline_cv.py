"""Smoke + property tests for the experiment plumbing (pipeline.py, cv.py, ablations.py)."""
import numpy as np
import pytest

import pipeline
from data import MARKET_VECTOR_COLUMNS, make_synthetic_corpus
from similarity import make_stub_encoder
import train


def tiny_setup(n=45, seed=3):
    c = make_synthetic_corpus(n_docs=n, n_events=n, market_dim=len(MARKET_VECTOR_COLUMNS), seed=seed)
    ev = sorted(c.events, key=lambda e: e.date)
    enc = make_stub_encoder()
    emb = {e.event_id: enc.embed(c.doc_by_id(e.doc_id).text) for e in ev}
    return pipeline.Setup(ev, enc, emb, c)


def test_fit_predict_shapes_and_finiteness():
    s = tiny_setup()
    cfg = pipeline.Cfg(epochs=2)
    ex = pipeline.make_examples(s, cfg)
    tr, te = ex[:30], ex[30:]
    r = pipeline.fit_predict(tr, te, cfg, s.encoder.dim, len(s.events[0].market_vector))
    assert len(r["y_true"]) == len(r["fusion"]) == len(r["baseline"]) == len(r["train_mean"]) == len(te)
    assert np.isfinite(r["fusion"]).all() and np.isfinite(r["baseline"]).all()


def test_fit_predict_is_reproducible_given_seed():
    s = tiny_setup()
    cfg = pipeline.Cfg(epochs=2, seed=5)
    ex = pipeline.make_examples(s, cfg)
    a = pipeline.fit_predict(ex[:30], ex[30:], cfg, s.encoder.dim, 9)
    b = pipeline.fit_predict(ex[:30], ex[30:], cfg, s.encoder.dim, 9)
    assert np.allclose(a["fusion"], b["fusion"]) and np.allclose(a["baseline"], b["baseline"])


def test_shuffled_label_control_changes_training_labels_but_not_test_labels():
    s = tiny_setup()
    ex = pipeline.make_examples(s, pipeline.Cfg())
    tr, te = ex[:30], ex[30:]
    a = pipeline.fit_predict(tr, te, pipeline.Cfg(epochs=1), s.encoder.dim, 9)
    b = pipeline.fit_predict(tr, te, pipeline.Cfg(epochs=1, shuffle_labels=True), s.encoder.dim, 9)
    assert a["y_true"] == b["y_true"]                                   # evaluation labels untouched
    assert not np.allclose(a["fusion"], b["fusion"])                    # training was affected


def test_alternative_stance_labels_differ_and_are_computed_from_the_documented_columns():
    v = np.arange(len(MARKET_VECTOR_COLUMNS), dtype=float)
    c = MARKET_VECTOR_COLUMNS.index
    assert pipeline.LABELS["ois_1y"](v) == v[c("OIS_1Y")]
    assert pipeline.LABELS["de2y"](v) == v[c("DE2Y")]
    assert pipeline.LABELS["ois_short_mean"](v) == pytest.approx(np.mean([v[c(x)] for x in ("OIS_1M", "OIS_3M", "OIS_6M", "OIS_1Y")]))


def test_uncertainty_off_and_frozen_attention_train_without_error_and_freeze_attention():
    s = tiny_setup()
    ex = pipeline.make_examples(s, pipeline.Cfg())
    import torch
    from fusion_head import FusionHead
    m = FusionHead(s.encoder.dim, 9)
    before = m.key_proj.weight.detach().clone(), m.head[0].weight.detach().clone()
    train.train_model(m, ex[:20], s.encoder.dim, 9, 5, True, 0.0, 1.0, epochs=2, freeze_attention=True,
                      use_uncertainty=False, verbose=False)
    assert torch.equal(before[0], m.key_proj.weight.detach())          # attention frozen: unchanged
    assert not torch.equal(before[1], m.head[0].weight.detach())       # head trained: changed


def test_cv_pooled_report_handles_small_groups_and_constant_predictors():
    import cv
    y = list(np.random.default_rng(0).normal(size=40))
    regs = ["zlb"] * 20 + ["pre_zlb"] * 15 + ["post_hiking"] * 5
    rep = cv.pooled_report(y, {"m": list(np.random.default_rng(1).normal(size=40)), "const": [0.3] * 40}, regs)
    assert rep["m"]["all"]["n"] == 40 and "dir_acc" in rep["m"]["zlb"]
    assert rep["m"]["post_hiking"]["note"].startswith("too few")          # 5 < 8: not scored, not faked
    assert rep["const"]["all"]["spearman"] is None                        # constant predictor: undefined, not NaN


def test_rolling_origin_never_trains_on_the_future(monkeypatch):
    import cv
    s = tiny_setup(n=120)
    seen = []

    def spy(tr, te, cfg, ed, md):
        assert max(e["event"].date for e in tr) < min(e["event"].date for e in te)      # strict time ordering
        seen.append(len(te))
        return {"y_true": [e["label"] for e in te], "fusion": [0.0] * len(te), "baseline": [0.0] * len(te),
                "train_mean": [0.0] * len(te)}

    monkeypatch.setattr(cv, "fit_predict", spy)
    monkeypatch.setattr(cv, "FIRST_TEST_YEAR", 2005)
    monkeypatch.setattr(cv, "MIN_TRAIN", 10)
    out = cv.rolling_origin(s, seeds=(0,), epochs=1)
    assert seen and out["n_pooled"] == sum(seen)
