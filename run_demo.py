"""
End-to-end pipeline run on SYNTHETIC data (data.py::make_synthetic_corpus).

Purpose: prove the plumbing works (benchmark construction -> baselines ->
metrics -> significance testing all execute and produce well-formed output).

NO NUMBER THIS SCRIPT PRINTS BELONGS IN THE PAPER. The corpus, market
vectors, and encoder are all synthetic/stubbed (see README.md). Re-run this
same script against the real corpus + trained encoder once both are wired in
(data.py / similarity.py) and *those* numbers are the real ones.

Retrieval task, leave-one-out over events: for each event e_q in E, rank the
remaining events in E as candidates using each retrieval mode, score against
the ECB-Precedent graded relevance labels (benchmark.py), and evaluate with
nDCG@5, nDCG@10, and MAP (metrics.py), each with a 95% paired-bootstrap CI
over query events. Also reports paired-bootstrap significance of every
baseline vs. hybrid on nDCG@10, Holm-corrected across the comparison family.
"""
from __future__ import annotations

import sys
import time

from baselines import BM25, rank_candidates
from benchmark import build_ecb_precedent, relevance_lookup
from data import make_synthetic_corpus
from metrics import bootstrap_ci, holm_correction, paired_bootstrap_pvalue, per_query_scores
from similarity import make_default_encoder

MODES = ["random", "bm25", "dense_text_only", "market_only", "hybrid"]


def main() -> None:
    t0 = time.time()

    print("=" * 72)
    print("Precedent-RAG retrieval pipeline -- SYNTHETIC DATA DEMO")
    print("No number below is real. This only validates the pipeline runs.")
    print("=" * 72)

    corpus = make_synthetic_corpus()
    events = corpus.events
    n_events = len(events)
    print(f"\nSynthetic corpus: {len(corpus.documents)} documents, "
          f"{n_events} linked events.")

    pairs, threshold = build_ecb_precedent(events)
    rel_lookup = relevance_lookup(pairs)
    print(f"ECB-Precedent (bottom-decile): {len(pairs)} relevant pairs, "
          f"distance threshold={threshold:.4f}")

    doc_text_by_event = {
        e.event_id: corpus.doc_by_id(e.doc_id).text for e in events
    }
    encoder = make_default_encoder()
    bm25 = BM25({eid: text for eid, text in doc_text_by_event.items()})

    # ranked_relevances_by_query[mode] = list over query events of
    # [relevance of each ranked candidate, best-first]
    ranked_relevances_by_query: dict[str, list[list[float]]] = {m: [] for m in MODES}

    print(f"\nRunning leave-one-out retrieval over {n_events} query events "
          f"x {len(MODES)} modes ...")
    for qi, query_event in enumerate(events):
        candidates = [e for e in events if e.event_id != query_event.event_id]
        for mode in MODES:
            ranked_ids = rank_candidates(
                query_event, candidates, mode,
                encoder=encoder, doc_text_by_event=doc_text_by_event,
                bm25=bm25, lam=0.5, tau=1.0, seed=qi,
            )
            rels = [
                rel_lookup.get((query_event.event_id, cid), 0.0)
                for cid in ranked_ids
            ]
            ranked_relevances_by_query[mode].append(rels)

    print(f"Retrieval done in {time.time() - t0:.1f}s.\n")

    # --- Metrics + CIs ---------------------------------------------------
    per_query = {m: per_query_scores(ranked_relevances_by_query[m]) for m in MODES}
    metric_names = ["ndcg@5", "ndcg@10", "map"]

    header = f"{'mode':<18}" + "".join(f"{m:>22}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for mode in MODES:
        row = f"{mode:<18}"
        for m in metric_names:
            mean, lo, hi = bootstrap_ci(per_query[mode][m])
            row += f"{mean:>7.3f} [{lo:.3f},{hi:.3f}]"
        print(row)

    # --- Significance: every baseline vs. hybrid on nDCG@10, Holm-corrected
    print("\nPaired bootstrap significance vs. hybrid (nDCG@10), Holm-corrected:")
    pvalues = {}
    for mode in MODES:
        if mode == "hybrid":
            continue
        p = paired_bootstrap_pvalue(
            per_query["hybrid"]["ndcg@10"], per_query[mode]["ndcg@10"]
        )
        pvalues[f"hybrid_vs_{mode}"] = p
    holm = holm_correction(pvalues)
    for name, p in sorted(pvalues.items(), key=lambda kv: kv[1]):
        threshold, significant = holm[name]
        print(f"  {name:<28} p={p:.4f}  Holm-threshold={threshold:.4f}  "
              f"significant={significant}")

    # --- Standing flags, restated so they surface on every run -----------
    print("\n" + "=" * 72)
    print("FLAGS (see README.md for full discussion):")
    print("1. market_only is expected to be near-ceiling here -- relevance")
    print("   labels are DEFINED by market-outcome distance (benchmark.py),")
    print("   so a retriever using market distance directly is close to an")
    print("   oracle on this synthetic benchmark. Not a bug; frame it as")
    print("   such in the eval write-up. Confirm this pattern re-examined")
    print("   once real data is wired in -- it may not hold the same way.")
    print("2. m_q at genuine prediction time is unresolved (similarity.py).")
    print("   This demo runs hybrid in the TRAIN-time regime (m_q known,")
    print("   full hybrid formula). Needs a decision from whoever owns")
    print("   Sec 4 before deployment-time behavior is defined.")
    print("=" * 72)

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
