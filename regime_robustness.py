"""
Regime-stratified robustness analysis (proposal Sec 6 "Regime robustness"):

    All preceding analyses stratified by zero-lower-bound versus
    post-hiking test events, surfacing whether retrieval augmentation's
    benefit varies with the density of available historical precedent --
    treated as a substantive empirical question rather than a robustness
    formality.

REGIME BOUNDARIES USED, and why -- real ECB policy history, not invented:
  pre_zlb:     before 2014-06-01
  zlb:         2014-06-01 to 2022-06-30 (ECB deposit rate first went
               negative, -0.10%, in June 2014; stayed <= 0% until the
               July 2022 hike)
  post_hiking: 2022-07-01 onward (ECB's first hike since 2011 -- +50bp,
               decided 21 July 2022 -- began the cycle that took the
               deposit rate to ~4% by late 2023)
These are well-known, citable facts about actual ECB policy rate history,
not a choice made for this project the way the stance label was.

CORRECTION TO SOMETHING SAID EARLIER IN THIS PROJECT: train.py's val split
(2020-2022 by calendar year) was described in conversation as "the ZLB
era" -- that's only mostly true. Checked here: 15 of those 18 events are
genuinely zlb-regime, but 3 (Sep/Oct/Dec 2022) are already post_hiking by
the real date boundary above. This script uses the correct date-based
regime label throughout, not the calendar-year split.

WHAT THIS RUNS, reusing the exact same trained model / retrieval / metrics
code as train.py and faithfulness.py (no separate retraining logic):
  1. Retrieval quality (nDCG@5/10, MAP), leave-one-out over ALL 228 real
     events, grouped by which regime the QUERY falls in. This does not
     depend on the train/val/test split.
  2. Downstream prediction quality (directional accuracy, Spearman rho,
     R^2), using the SAME model trained in train.py (train < 2020, so no
     chronological leakage), evaluated separately on the zlb-regime and
     post_hiking-regime portions of the held-out (post-2020) events.
  3. Faithfulness (the Sec 5 perturbation protocol), same split, same two
     groups.
  4. A direct precedent-density diagnostic: for post_hiking queries
     specifically (a brand-new regime with almost no same-regime history
     to draw on), what fraction of their retrieved top-k precedents are
     ALSO post_hiking vs. from an earlier regime -- the concrete version
     of "density of available historical precedent" the proposal names.

Same caveats as train.py/faithfulness.py apply throughout: stub encoder,
pragmatic stance label, and here ADDITIONALLY small per-regime n
(post_hiking has only 17 events total, ever, in this dataset -- it is a
regime that only started in mid-2022). Read every CI, not just the point
estimates.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path

import numpy as np

from baselines import BM25, rank_candidates
from benchmark import build_ecb_precedent, relevance_lookup
from data import load_real_corpus
from faithfulness import fusion_predict, rank_history, run_perturbations
from fusion_head import FusionHead, NonAugmentedHead
from metrics import (
    bootstrap_ci, bootstrap_metric_ci, directional_accuracy, per_query_scores,
    r_squared, spearman_rho,
)
from similarity import make_stub_encoder
from train import K, build_examples, predict, train_model

DATA_DIR = Path(__file__).parent / "data"
ZLB_START = dt.date(2014, 6, 1)
HIKE_START = dt.date(2022, 7, 1)


def regime(date: dt.date) -> str:
    if date < ZLB_START:
        return "pre_zlb"
    if date < HIKE_START:
        return "zlb"
    return "post_hiking"


def main():
    print("=" * 72)
    print("Regime-stratified robustness analysis -- REAL data. Same caveats")
    print("as train.py/faithfulness.py, PLUS small per-regime n (post_hiking")
    print("has only 17 real events total -- it's a regime that only started")
    print("mid-2022). Read every CI.")
    print("=" * 72)

    corpus = load_real_corpus(
        DATA_DIR / "all_ECB_speeches (1).csv",
        DATA_DIR / "speeches_bis.csv",
        DATA_DIR / "Dataset_EA-MPD.xlsx",
    )
    events = sorted(corpus.events, key=lambda e: e.date)
    doc_text = {e.event_id: corpus.doc_by_id(e.doc_id).text for e in events}
    encoder = make_stub_encoder()
    embeddings = {eid: encoder.embed(text) for eid, text in doc_text.items()}

    counts = Counter(regime(e.date) for e in events)
    print(f"\nRegime counts across all 228 real events: {dict(counts)}")

    # ------------------------------------------------------------------
    # 1. Retrieval quality, stratified by query regime
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("1. RETRIEVAL QUALITY BY REGIME (leave-one-out over all 228 events)")
    print("-" * 72)
    pairs, threshold = build_ecb_precedent(events)
    rel_lookup = relevance_lookup(pairs)
    bm25 = BM25({e.event_id: doc_text[e.event_id] for e in events})

    # Only hybrid + market_only + bm25 -- skipping dense_text_only/random
    # here (already characterized in run_real.py) to keep this runtime
    # reasonable; they'd cost the same 228x228 comparisons again for no
    # new information about regime robustness specifically.
    modes = ["hybrid", "market_only", "bm25"]
    rels_by_mode_regime: dict[str, dict[str, list[list[float]]]] = {
        m: {r: [] for r in ("pre_zlb", "zlb", "post_hiking")} for m in modes
    }
    for q in events:
        candidates = [e for e in events if e.event_id != q.event_id]
        q_regime = regime(q.date)
        for mode in modes:
            ranked_ids = rank_candidates(
                q, candidates, mode, encoder=encoder, doc_text_by_event=doc_text,
                bm25=bm25, lam=0.5, tau=1.0,
            )
            rels = [rel_lookup.get((q.event_id, cid), 0.0) for cid in ranked_ids]
            rels_by_mode_regime[mode][q_regime].append(rels)

    header = f"{'mode':<14}{'regime':<14}{'n':>5}{'ndcg@5':>22}{'ndcg@10':>22}{'map':>22}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        for reg in ("pre_zlb", "zlb", "post_hiking"):
            rels_list = rels_by_mode_regime[mode][reg]
            if not rels_list:
                continue
            pq = per_query_scores(rels_list)
            row = f"{mode:<14}{reg:<14}{len(rels_list):>5}"
            for m in ("ndcg@5", "ndcg@10", "map"):
                mean, lo, hi = bootstrap_ci(pq[m])
                row += f"{mean:>7.3f} [{lo:.3f},{hi:.3f}]"
            print(row)

    # ------------------------------------------------------------------
    # 2 & 3. Downstream prediction + faithfulness, stratified by regime,
    # using the SAME model trained the SAME way as train.py (chronological,
    # no leakage). Only the held-out (post-train) events can be evaluated.
    # ------------------------------------------------------------------
    examples = build_examples(events, encoder, embeddings)
    train_ex = [e for e in examples if e["event"].date.year < 2020]
    held_out = [e for e in examples if e["event"].date.year >= 2020]

    held_out_by_regime = {"zlb": [], "post_hiking": []}
    for e in held_out:
        r = regime(e["event"].date)
        if r in held_out_by_regime:
            held_out_by_regime[r].append(e)
    # pre_zlb has no held-out events (training already covers all of it up
    # to 2019) -- nothing to evaluate there, correctly omitted rather than
    # padded with an empty/fake row.
    print(f"\nHeld-out (2020 onward) events by regime: "
          f"zlb={len(held_out_by_regime['zlb'])}, "
          f"post_hiking={len(held_out_by_regime['post_hiking'])}")

    train_labels = np.array([e["label"] for e in train_ex])
    label_mean, label_std = float(train_labels.mean()), float(train_labels.std() + 1e-8)
    embed_dim, market_dim = encoder.dim, len(events[0].market_vector)

    print("\nTraining FusionHead and NonAugmentedHead (identical setup to train.py) ...")
    fusion = FusionHead(embed_dim=embed_dim, market_dim=market_dim)
    train_model(fusion, train_ex, embed_dim, market_dim, K, True, label_mean, label_std)
    baseline = NonAugmentedHead(embed_dim=embed_dim)
    train_model(baseline, train_ex, embed_dim, market_dim, K, False, label_mean, label_std)

    print("\n" + "-" * 72)
    print("2. DOWNSTREAM PREDICTION QUALITY BY REGIME")
    print("-" * 72)
    for reg, exs in held_out_by_regime.items():
        if not exs:
            continue
        y_true = [e["label"] for e in exs]
        print(f"\n  regime={reg} (n={len(exs)})")
        for name, model, is_fusion in [("FusionHead", fusion, True), ("NonAugmented", baseline, False)]:
            y_pred = predict(model, exs, embed_dim, market_dim, K, is_fusion, label_mean, label_std)
            print(f"    {name}:")
            for metric_name, fn in [("dir. accuracy", directional_accuracy), ("Spearman rho", spearman_rho), ("R^2", r_squared)]:
                point, lo, hi = bootstrap_metric_ci(y_true, y_pred, fn)
                print(f"      {metric_name:<15} {point:>7.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\n" + "-" * 72)
    print("3. FAITHFULNESS BY REGIME (perturbation-magnitude vs shift correlation)")
    print("-" * 72)
    for reg, exs in held_out_by_regime.items():
        mags, shifts = [], []
        skipped = 0
        for ex in exs:
            q = ex["event"]
            history = [e for e in events if e.date < q.date]
            if len(history) <= K + 1:
                skipped += 1
                continue
            ranked = rank_history(q, history, embeddings)
            res = run_perturbations(fusion, ex["e_q"], ranked, K, __import__("random").Random(0), fusion_predict)
            orig = res["original"][0] * label_std + label_mean
            for cond in ("replace_with_next", "replace_with_random"):
                pred, mag = res[cond]
                mags.append(mag)
                shifts.append(abs(pred * label_std + label_mean - orig))
        if len(mags) < 4:
            print(f"\n  regime={reg}: only {len(mags)} perturbations available "
                  f"({skipped} events skipped, insufficient history) -- too few to report a correlation.")
            continue
        rho, lo, hi = bootstrap_metric_ci(mags, shifts, lambda a, b: spearman_rho(a, b))
        print(f"\n  regime={reg} (n={len(mags)} perturbations): "
              f"Spearman rho(perturbation magnitude, |shift|) = {rho:.3f}  [{lo:.3f}, {hi:.3f}]")

    # ------------------------------------------------------------------
    # 4. Precedent-density diagnostic for post_hiking queries specifically
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("4. PRECEDENT DENSITY: what regime do post_hiking queries actually")
    print("   retrieve their evidence from? (the proposal's literal question)")
    print("-" * 72)
    same_regime_fracs = []
    for q in [e for e in events if regime(e.date) == "post_hiking"]:
        candidates = [e for e in events if e.event_id != q.event_id]
        ranked_ids = rank_candidates(
            q, candidates, "hybrid", encoder=encoder, doc_text_by_event=doc_text, lam=0.5, tau=1.0,
        )[:K]
        cand_by_id = {e.event_id: e for e in candidates}
        regimes_retrieved = [regime(cand_by_id[cid].date) for cid in ranked_ids]
        frac_same = sum(1 for r in regimes_retrieved if r == "post_hiking") / len(regimes_retrieved)
        same_regime_fracs.append(frac_same)
    mean_frac = float(np.mean(same_regime_fracs))
    print(f"  Across {len(same_regime_fracs)} post_hiking queries, on average "
          f"{mean_frac:.0%} of each query's top-{K} retrieved precedents are "
          f"ALSO post_hiking-regime events; the rest are drawn from earlier "
          f"regimes (pre_zlb/zlb) because post_hiking is a brand-new regime "
          f"with little same-regime history yet.")

    print("\n" + "=" * 72)
    print("REMINDERS: stub encoder + pragmatic stance label still apply.")
    print("post_hiking has only 17 real events, ever, in this dataset --")
    print("every post_hiking number above is a small-n result. Report it as")
    print("such, and revisit once more post-2022 ECB events are available.")
    print("=" * 72)


if __name__ == "__main__":
    main()
