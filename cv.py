"""
Cross-validation (proposal Sec 6, "Reused evaluation machinery": rolling-origin and
leave-one-meeting-out).

ROLLING-ORIGIN (default): expanding window. For each test year Y from FIRST_TEST_YEAR on, train on
every event dated before 1 Jan Y and predict the events in year Y; pool all out-of-fold predictions
and score them together (much larger n than the single 2020+ split in train.py, so the CIs are
tighter -- and the pooled set spans all three regimes, so regime-stratified scores are possible
with real sample sizes). Nothing from the future is ever used to predict the past.

LEAVE-ONE-MEETING-OUT (--lomo): each held-out meeting is predicted by a model trained on ALL
other meetings (including later ones -- that is what LOMO means, and it is NOT a forecasting
setup, so treat it as a secondary, optimistic-in-principle check, not a substitute for rolling
origin). Retrieval for the held-out meeting still uses strictly-earlier precedents only. Because
each fold retrains two networks, --stride N evaluates every N-th meeting only (default 4,
epochs=60) to stay within a reasonable runtime; --stride 1 is the full protocol.

Both use train.py's leak-free retrieval (text-only at prediction time). Same stub-encoder /
pragmatic-stance-label caveats as everywhere else. Writes results/cv_rolling.json / cv_lomo.json.

Usage: python cv.py                # rolling origin
       python cv.py --lomo [--stride 4]
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np

from metrics import bootstrap_metric_ci, directional_accuracy, r_squared, spearman_rho
from pipeline import Cfg, Timer, fit_predict, load_setup, make_examples, save_json
from regime_robustness import regime

FIRST_TEST_YEAR = 2008
MIN_TRAIN = 60


def pooled_report(y, preds: dict, regimes: list[str]) -> dict:
    out = {"n": len(y)}
    groups = {"all": np.ones(len(y), bool)}
    for r in ("pre_zlb", "zlb", "post_hiking"):
        groups[r] = np.array([g == r for g in regimes])
    for who, p in preds.items():
        p = np.asarray(p)
        out[who] = {}
        for gname, mask in groups.items():
            if mask.sum() < 8:
                out[who][gname] = {"n": int(mask.sum()), "note": "too few events to score"}
                continue
            yt, yp = np.asarray(y)[mask], p[mask]
            rec = {"n": int(mask.sum())}
            for mname, fn in (("dir_acc", directional_accuracy), ("spearman", spearman_rho), ("r2", r_squared)):
                if mname == "spearman" and np.std(yp) == 0:
                    rec[mname] = None
                    continue
                pt, lo, hi = bootstrap_metric_ci(yt, yp, fn)
                rec[mname] = {"point": pt, "ci95": [lo, hi]}
            out[who][gname] = rec
    return out


def rolling_origin(setup, seeds=(0, 1), epochs=150):
    embed_dim, market_dim = setup.encoder.dim, len(setup.events[0].market_vector)
    cfg0 = Cfg(epochs=epochs)
    ex = make_examples(setup, cfg0)
    years = sorted({e["event"].date.year for e in ex})
    t = Timer()
    per_seed = {}
    meta = None
    for sd in seeds:
        y, fus, base, mean_, regs, dates = [], [], [], [], [], []
        for Y in [yr for yr in years if yr >= FIRST_TEST_YEAR]:
            tr = [e for e in ex if e["event"].date < dt.date(Y, 1, 1)]
            te = [e for e in ex if e["event"].date.year == Y]
            if len(tr) < MIN_TRAIN or not te:
                continue
            r = fit_predict(tr, te, Cfg(epochs=epochs, seed=sd), embed_dim, market_dim)
            y += r["y_true"]; fus += r["fusion"]; base += r["baseline"]; mean_ += r["train_mean"]
            regs += [regime(e["event"].date) for e in te]
            dates += [e["event"].date.isoformat() for e in te]
            print(f"[{t()}] seed={sd} test year {Y}: train={len(tr)} test={len(te)}", flush=True)
        per_seed[str(sd)] = pooled_report(y, {"fusion": fus, "baseline": base, "train_mean": mean_}, regs)
        meta = {"n_pooled": len(y), "first_test_year": FIRST_TEST_YEAR, "min_train": MIN_TRAIN}
    return {"protocol": "rolling-origin, expanding window, yearly test blocks", **meta, "per_seed": per_seed}


def lomo(setup, stride=4, epochs=60, seed=0):
    embed_dim, market_dim = setup.encoder.dim, len(setup.events[0].market_vector)
    ex = make_examples(setup, Cfg(epochs=epochs))
    held = list(range(0, len(ex), stride))
    t = Timer()
    y, fus, base, mean_, regs = [], [], [], [], []
    for n, i in enumerate(held):
        te = [ex[i]]
        tr = [e for j, e in enumerate(ex) if j != i]
        r = fit_predict(tr, te, Cfg(epochs=epochs, seed=seed), embed_dim, market_dim)
        y += r["y_true"]; fus += r["fusion"]; base += r["baseline"]; mean_ += r["train_mean"]
        regs.append(regime(ex[i]["event"].date))
        if n % 10 == 0:
            print(f"[{t()}] LOMO fold {n + 1}/{len(held)}", flush=True)
    return {"protocol": f"leave-one-meeting-out, every {stride}th meeting, epochs={epochs}",
            "report": pooled_report(y, {"fusion": fus, "baseline": base, "train_mean": mean_}, regs)}


def main():
    setup = load_setup()
    print(f"{len(setup.events)} events with statements.", flush=True)
    if "--lomo" in sys.argv:
        stride = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 4
        save_json("cv_lomo.json", lomo(setup, stride=stride))
    else:
        save_json("cv_rolling.json", rolling_origin(setup))
    print("done.")


if __name__ == "__main__":
    main()
