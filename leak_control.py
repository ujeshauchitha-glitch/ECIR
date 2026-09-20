"""
Quantifies the retrieval leak found on 2026-09-20 (see HANDOFF.md, train.build_examples).

LEAKY setup (INVALID, for measurement only): retrieval uses the query's own market vector m_q, which
contains the stance label.  FIXED setup: retrieval is text-only at prediction time.
Same data, same model, same seeds, same held-out set (2020 onward); only the retrieval differs.

Writes results/leak_control.json. Never quote the leaky numbers as results.
Usage: python leak_control.py
"""
from __future__ import annotations

import numpy as np

import train
from metrics import bootstrap_metric_ci, directional_accuracy
from pipeline import Cfg, fit_predict, load_setup, LABELS, save_json


def main():
    setup = load_setup()
    embed_dim, market_dim = setup.encoder.dim, len(setup.events[0].market_vector)
    out = {"protocol": "train < 2020, held-out >= 2020, FusionHead directional accuracy, 3 seeds", "runs": {}}
    for leaky in (True, False):
        preds, truth, per_seed = [], None, []
        for seed in (0, 1, 2):
            cfg = Cfg(seed=seed)
            ex = train.build_examples(setup.events, setup.encoder, setup.embeddings, k=cfg.k, leaky=leaky, label_fn=LABELS[cfg.label])
            tr = [e for e in ex if e["event"].date.year < 2020]
            ho = [e for e in ex if e["event"].date.year >= 2020]
            r = fit_predict(tr, ho, cfg, embed_dim, market_dim)
            truth = r["y_true"]
            per_seed.append(directional_accuracy(r["y_true"], r["fusion"]))
            preds.append(r["fusion"])
        # pool the three seeds' predictions for one CI over the held-out set
        pooled_true = truth * 3
        pooled_pred = [p for ps in preds for p in ps]
        pt, lo, hi = bootstrap_metric_ci(pooled_true, pooled_pred, directional_accuracy)
        out["runs"]["leaky" if leaky else "fixed"] = {
            "n_heldout": len(truth), "dir_acc_per_seed": per_seed, "dir_acc_pooled": [pt, lo, hi],
            "dir_acc_seed_mean": float(np.mean(per_seed)),
        }
        print(("LEAKY (invalid)" if leaky else "FIXED         "), f"per-seed {['%.3f' % x for x in per_seed]}  pooled {pt:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    save_json("leak_control.json", out)


if __name__ == "__main__":
    main()
