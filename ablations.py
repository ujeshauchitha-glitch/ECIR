"""
Ablations (proposal Sec 6): mixing weight lambda, neighbour count k, presence/absence of
the uncertainty output, frozen-vs-trained fusion attention -- plus three things the proposal
does not list but this project needs:

  * tau / kernel sensitivity. The market term is exp(-d/tau) with d in basis points; with the
    default tau=1 almost every pair scores ~0 on that term, so "hybrid" barely uses it. This
    sweeps tau as a multiple of the median pairwise distance, for BOTH the squared kernel (what
    the code has always used) and the plain-L2 kernel (what the proposal PDF appears to show --
    see similarity.hybrid_similarity's UNRESOLVED DISCREPANCY note).
  * stance-definition sensitivity (three defensible short-rate stances; none is "the" definition).
  * CONTROLS: a shuffled-training-labels run and a predict-the-training-mean reference. If the
    shuffled control shows skill, the pipeline leaks; if the real model can't beat the mean
    reference, there is nothing to explain yet.

Part 1 (retrieval quality: lambda x tau x kernel) is symmetric pairwise evaluation and
legitimately uses m_q (train-time regime). Part 2 (downstream) is leak-free: at prediction time
retrieval is text-only, so lambda and tau CANNOT affect downstream results -- that is by design
(no market term without m_q), and is why lambda is swept only in Part 1.

Stub encoder + pragmatic stance label caveats from train.py apply to every number here.
Writes results/ablations_retrieval.json and results/ablations_downstream.json.

Usage: python ablations.py [--quick]      (--quick: 2 seeds, 60 epochs, for smoke-testing)
"""
from __future__ import annotations

import sys

import numpy as np

from benchmark import build_ecb_precedent
from metrics import bootstrap_ci, ndcg_at_k, average_precision
from pipeline import Cfg, Timer, fit_predict, load_setup, make_examples, save_json, score

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
TAU_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]


def retrieval_sweep(setup) -> dict:
    ev = setup.events
    n = len(ev)
    E = np.stack([setup.embeddings[e.event_id] for e in ev])
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    cos = En @ En.T
    M = np.stack([e.market_vector for e in ev])
    D2 = ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
    D = np.sqrt(D2)

    pairs, thr = build_ecb_precedent(ev)
    idx = {e.event_id: i for i, e in enumerate(ev)}
    R = np.zeros((n, n))
    for p in pairs:
        R[idx[p.query_event_id], idx[p.candidate_event_id]] = p.relevance
    off = ~np.eye(n, dtype=bool)

    def evaluate(S):
        nd, ap = [], []
        for i in range(n):
            cand = np.where(off[i])[0]
            order = cand[np.argsort(-S[i, cand], kind="stable")]
            rels = list(R[i, order])
            nd.append(ndcg_at_k(rels, 10))
            ap.append(average_precision(rels))
        return nd, ap

    out = {"n_events": n, "benchmark_threshold": float(thr), "median_sq_dist": float(np.median(D2[off])),
           "median_l2_dist": float(np.median(D[off])), "grid": [], "refs": {}}
    for name, S in {"market_only": -D, "text_only": cos}.items():
        nd, ap = evaluate(S)
        out["refs"][name] = {"ndcg@10": bootstrap_ci(nd), "map": bootstrap_ci(ap)}

    for kernel, Dk in (("squared", D2), ("l2", D)):
        med = float(np.median(Dk[off]))
        # tau as multiples of the median distance, PLUS the code's historical default tau=1 for reference
        for tau_label, tau in [(f"{m}x median", m * med) for m in TAU_MULTS] + [("tau=1 (old default)", 1.0)]:
            for lam in LAMS:
                S = lam * cos + (1 - lam) * np.exp(-Dk / tau)
                nd, ap = evaluate(S)
                out["grid"].append({"kernel": kernel, "tau": tau_label, "tau_value": tau, "lam": lam,
                                    "ndcg@10": bootstrap_ci(nd), "map": bootstrap_ci(ap)})
    return out


def downstream(setup, quick: bool) -> dict:
    seeds_main = [0, 1] if quick else [0, 1, 2, 3, 4]
    seeds_abl = [0] if quick else [0, 1, 2]
    epochs = 60 if quick else 150
    embed_dim, market_dim = setup.encoder.dim, len(setup.events[0].market_vector)
    t = Timer()

    configs = {
        "main (k=5, uncertainty on, ois_1y)": (Cfg(k=5), seeds_main),
        "k=1": (Cfg(k=1), seeds_abl), "k=3": (Cfg(k=3), seeds_abl),
        "k=10": (Cfg(k=10), seeds_abl), "k=20": (Cfg(k=20), seeds_abl),
        "uncertainty OFF (MSE loss)": (Cfg(uncertainty=False), seeds_abl),
        "attention FROZEN at init": (Cfg(freeze_attention=True), seeds_abl),
        "stance = ois_short_mean": (Cfg(label="ois_short_mean"), seeds_abl),
        "stance = de2y": (Cfg(label="de2y"), seeds_abl),
        "CONTROL: shuffled training labels": (Cfg(shuffle_labels=True), seeds_abl),
    }
    results = {}
    for name, (cfg0, seeds) in configs.items():
        per_seed = []
        for sd in seeds:
            cfg = Cfg(**{**cfg0.__dict__, "seed": sd, "epochs": epochs})
            ex = make_examples(setup, cfg)
            tr = [e for e in ex if e["event"].date.year < 2020]
            ho = [e for e in ex if e["event"].date.year >= 2020]
            r = fit_predict(tr, ho, cfg, embed_dim, market_dim)
            per_seed.append({"seed": sd, "n_train": len(tr), "n_heldout": len(ho),
                             "fusion": score(r["y_true"], r["fusion"]),
                             "baseline": score(r["y_true"], r["baseline"]),
                             "train_mean": score(r["y_true"], r["train_mean"])})
            print(f"[{t()}] {name} seed={sd} done", flush=True)
        agg = {}
        for who in ("fusion", "baseline", "train_mean"):
            agg[who] = {m: (float(np.nanmean([s[who][m] for s in per_seed])), float(np.nanstd([s[who][m] for s in per_seed])))
                        for m in ("dir_acc", "spearman", "r2")}
        agg["fusion_beats_baseline_dir_acc_in"] = f"{sum(s['fusion']['dir_acc'] > s['baseline']['dir_acc'] for s in per_seed)}/{len(per_seed)} seeds"
        results[name] = {"per_seed": per_seed, "summary(mean,sd)": agg}
        save_json("ablations_downstream.json", results)   # incremental, so partial progress survives
    return results


def main():
    quick = "--quick" in sys.argv
    setup = load_setup()
    print(f"{len(setup.events)} events with statements.", flush=True)
    print("Part 1: retrieval-quality sweep (lambda x tau x kernel) ...", flush=True)
    r1 = retrieval_sweep(setup)
    save_json("ablations_retrieval.json", r1)
    print("Part 2: downstream ablations (multi-seed) ...", flush=True)
    downstream(setup, quick)
    print("done.")


if __name__ == "__main__":
    main()
