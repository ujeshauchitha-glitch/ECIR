"""
Shared experiment plumbing for ablations.py and cv.py, so every experiment trains
and scores models through exactly the same code path as train.py (leak-free
retrieval, strictly-earlier precedents, same model, same loss).

Nothing here is new modelling: it is train.py's pieces behind a small config object.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

import train
from data import MARKET_VECTOR_COLUMNS, load_real_corpus, stance_label
from fusion_head import FusionHead, NonAugmentedHead
from metrics import directional_accuracy, r_squared, spearman_rho
from similarity import make_stub_encoder

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def _col(name):
    return MARKET_VECTOR_COLUMNS.index(name)


# Alternative stance definitions for the sensitivity analysis. "ois_1y" is the
# project's stand-in (data.stance_label); the others are equally defensible
# single/composite short-rate readings. None of these is the proposal's definition
# (it has none) -- the point of ablations.py is to show how much conclusions depend on the choice.
LABELS = {
    "ois_1y": stance_label,
    "ois_short_mean": lambda v: float(np.mean([v[_col(c)] for c in ("OIS_1M", "OIS_3M", "OIS_6M", "OIS_1Y")])),
    "de2y": lambda v: float(v[_col("DE2Y")]),
}


@dataclasses.dataclass
class Cfg:
    k: int = 5
    epochs: int = 150
    lr: float = 1e-3
    seed: int = 0
    uncertainty: bool = True
    freeze_attention: bool = False
    label: str = "ois_1y"
    shuffle_labels: bool = False   # CONTROL: permute training labels; held-out skill must vanish


@dataclasses.dataclass
class Setup:
    events: list
    encoder: object
    embeddings: dict
    corpus: object


def load_setup(use_bis: bool = True) -> Setup:
    corpus = load_real_corpus(
        DATA_DIR / "all_ECB_speeches (1).csv",
        DATA_DIR / "speeches_bis.csv" if use_bis else None,
        DATA_DIR / "Dataset_EA-MPD.xlsx",
    )
    events = sorted(corpus.events, key=lambda e: e.date)
    encoder = make_stub_encoder()
    embeddings = {e.event_id: encoder.embed(corpus.doc_by_id(e.doc_id).text) for e in events}
    return Setup(events, encoder, embeddings, corpus)


def make_examples(setup: Setup, cfg: Cfg):
    return train.build_examples(setup.events, setup.encoder, setup.embeddings, k=cfg.k, label_fn=LABELS[cfg.label])


def fit_predict(train_ex, test_ex, cfg: Cfg, embed_dim: int, market_dim: int):
    """Trains FusionHead and NonAugmentedHead on train_ex; returns their predictions on test_ex
    (in raw label units) plus the true labels."""
    if cfg.shuffle_labels:
        perm = np.random.default_rng(1000 + cfg.seed).permutation(len(train_ex))
        train_ex = [dict(e, label=train_ex[j]["label"]) for e, j in zip(train_ex, perm)]
    labels = np.array([e["label"] for e in train_ex])
    mu, sd = float(labels.mean()), float(labels.std() + 1e-8)
    fusion = FusionHead(embed_dim, market_dim)
    train.train_model(fusion, train_ex, embed_dim, market_dim, cfg.k, True, mu, sd, epochs=cfg.epochs, lr=cfg.lr,
                      seed=cfg.seed, use_uncertainty=cfg.uncertainty, freeze_attention=cfg.freeze_attention, verbose=False)
    base = NonAugmentedHead(embed_dim)
    train.train_model(base, train_ex, embed_dim, market_dim, cfg.k, False, mu, sd, epochs=cfg.epochs, lr=cfg.lr,
                      seed=cfg.seed, use_uncertainty=cfg.uncertainty, verbose=False)
    return {
        "y_true": [e["label"] for e in test_ex],
        "fusion": train.predict(fusion, test_ex, embed_dim, market_dim, cfg.k, True, mu, sd),
        "baseline": train.predict(base, test_ex, embed_dim, market_dim, cfg.k, False, mu, sd),
        # constant "predict the training mean" reference: R^2 of a model that learned nothing
        "train_mean": [mu] * len(test_ex),
    }


def score(y_true, y_pred) -> dict:
    return {
        "dir_acc": directional_accuracy(y_true, y_pred),
        "spearman": spearman_rho(y_true, y_pred) if np.std(y_pred) > 0 else float("nan"),
        "r2": r_squared(y_true, y_pred),
    }


def save_json(name: str, obj) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / name
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return p


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return f"{time.time() - self.t0:6.0f}s"
