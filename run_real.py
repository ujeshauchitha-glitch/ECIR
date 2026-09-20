"""
End-to-end pipeline run on REAL data -- real ECB text, real press-conference
statements, real market reactions (EA-MPD). Companion to run_demo.py, which
runs the identical pipeline on synthetic data to validate the plumbing.

HOW REAL THIS IS, PRECISELY:
  - Corpus text (D): REAL. 3,052 real ECB speeches plus one press-conference
    statement per linked event (the ECB's own published text, fetched by
    fetch_ecb_pressconf.py; BIS-archive fallback for a few early dates).
  - Market-reaction vectors (m_i): REAL. From the ECB's own published
    EA-MPD dataset.
  - Event set (E): every EA-MPD event (of the proposal's stated 315) that has
    a press-conference statement. Events with none -- mostly 1999-2001
    meetings that produced only a press release -- are left out, never linked
    to a nearby unrelated document. The count actually used is printed at run
    time and MUST be the number reported, not 315.
  - Encoder (f_theta): STILL THE STUB. make_stub_encoder() is a hash-based
    placeholder with no real semantic understanding of central-bank
    language -- the real trained encoder from the concurrent submission is
    not wired in yet. This means BM25 (real word overlap) is more
    trustworthy right now than dense_text_only/hybrid (stub embeddings).
    market_only does not depend on the encoder at all and is fully real.

NOTE ON "hybrid" HERE: uses lam=0.5, tau=1.0, squared-distance kernel -- the
historical defaults. With distances in basis points, tau=1 makes the market
term ~0 for almost all pairs, so this "hybrid" is close to text-only. See
ablations.py for the lambda/tau/kernel sweep and a tuned setting.

BOTTOM LINE: market_only and BM25 numbers below are real and can be reasoned
about. dense_text_only and hybrid numbers reflect the stub encoder's
behavior, not the real model's -- re-run this exact script once the real
encoder is wired into similarity.py and treat today's dense/hybrid numbers
as pipeline-validation only, not results.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from baselines import BM25, rank_candidates
from benchmark import build_ecb_precedent, relevance_lookup
from data import load_real_corpus
from metrics import bootstrap_ci, holm_correction, paired_bootstrap_pvalue, per_query_scores
from similarity import make_default_encoder

MODES = ["random", "bm25", "dense_text_only", "market_only", "hybrid"]
DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    t0 = time.time()

    print("=" * 72)
    print("Precedent-RAG retrieval pipeline -- REAL DATA RUN")
    print("Real ECB text + real EA-MPD market data. Encoder is STILL A STUB")
    print("(see module docstring) -- only market_only and bm25 are fully real.")
    print("=" * 72)

    corpus = load_real_corpus(
        DATA_DIR / "all_ECB_speeches (1).csv",
        DATA_DIR / "speeches_bis.csv",
        DATA_DIR / "Dataset_EA-MPD.xlsx",
    )
    events = corpus.events
    n_events = len(events)
    print(f"\nReal corpus: {len(corpus.documents)} documents, "
          f"{n_events} linked events (of the proposal's stated 315; events with no "
          f"press-conference statement available are left out -- see data.py).")

    pairs, threshold = build_ecb_precedent(events)
    rel_lookup = relevance_lookup(pairs)
    print(f"ECB-Precedent (bottom-decile) on these {n_events} events: "
          f"{len(pairs)} relevant pairs, distance threshold={threshold:.4f}")

    doc_text_by_event = {
        e.event_id: corpus.doc_by_id(e.doc_id).text for e in events
    }
    encoder = make_default_encoder()
    bm25 = BM25({eid: text for eid, text in doc_text_by_event.items()})

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

    per_query = {m: per_query_scores(ranked_relevances_by_query[m]) for m in MODES}
    metric_names = ["ndcg@5", "ndcg@10", "map"]

    header = f"{'mode':<18}" + "".join(f"{m:>22}" for m in metric_names)
    print(header)
    print("-" * len(header))
    results = {"n_events": n_events, "n_documents": len(corpus.documents), "modes": {}, "holm": {}}
    for mode in MODES:
        row = f"{mode:<18}"
        results["modes"][mode] = {}
        for m in metric_names:
            mean, lo, hi = bootstrap_ci(per_query[mode][m])
            row += f"{mean:>7.3f} [{lo:.3f},{hi:.3f}]"
            results["modes"][mode][m] = [mean, lo, hi]
        print(row)

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
        thr, significant = holm[name]
        print(f"  {name:<28} p={p:.4f}  Holm-threshold={thr:.4f}  "
              f"significant={significant}")
        results["holm"][name] = {"p": p, "threshold": thr, "significant": bool(significant)}
    from pipeline import save_json
    save_json("retrieval_quality.json", results)

    print("\n" + "=" * 72)
    print("REMINDERS:")
    print(f"1. n={n_events} events, not 315. Say so wherever these numbers are used.")
    print("2. dense_text_only/hybrid use the STUB encoder -- re-run once the")
    print("   real trained encoder is wired into similarity.py.")
    print("3. market_only is expected to be near-ceiling by construction")
    print("   (relevance labels ARE market-outcome distance). Not a bug.")
    print("=" * 72)

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
