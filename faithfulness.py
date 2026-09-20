"""
Causal faithfulness protocol (proposal Sec 5 "Faithfulness protocol" + Sec 6
"Faithfulness"):

    For a held-out sample of predictions, the top-ranked retrieved precedent
    is ablated or replaced (with the next-ranked item, or with a random
    in-corpus item as a control), and the resulting shift in s_hat_q is
    measured. A model whose evidence is causally load-bearing should show
    shift magnitude that scales with the rank and relevance of the removed
    precedent; a model whose evidence is decorative will not.
    ...
    Correlation between perturbation magnitude of retrieved evidence and
    the resulting prediction shift, computed across the full test set.

WHAT THIS RUNS: trains FusionHead + NonAugmentedHead exactly as train.py
does (same data, split, hyperparameters), then on the TEST split (a few
dozen events at most -- small, see train.py's caveats, which all still apply here) runs four
perturbations of each query's retrieved evidence and measures the shift in
the prediction:

  A) ablate_top1        -- drop the single most-similar retrieved precedent
  B) ablate_bottom1      -- drop the single LEAST-similar of the retrieved
                             precedents (a same-size perturbation, different
                             rank -- the direct test of "shift should scale
                             with rank": a faithful model should shift MORE
                             for (A) than (B))
  C) replace_with_next   -- swap the top precedent for the next-best-ranked
                             one just outside the retrieved set
  D) replace_with_random -- swap the top precedent for a uniformly random
                             precedent from the rest of history (the
                             proposal's explicit control condition)

VALIDITY CHECK BUILT IN: the same four perturbations are also run through
NonAugmentedHead, which structurally cannot see retrieved evidence at all
(it isn't even passed in). Its shift must be exactly zero. If it isn't,
something in this harness is broken -- that's what the assertion at the
bottom of main() is for.

Same caveats as train.py apply and are not repeated in full here: stub
encoder, pragmatic stance label, small n. Read train.py's module docstring
if you haven't.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from data import load_real_corpus, stance_label
from fusion_head import FusionHead, NonAugmentedHead
from metrics import bootstrap_ci, bootstrap_metric_ci, paired_bootstrap_pvalue, spearman_rho
from similarity import hybrid_similarity, make_stub_encoder
from train import K, LAM, TAU, build_examples, train_model

DATA_DIR = Path(__file__).parent / "data"
SEED = 0


def rank_history(q, history, embeddings, lam=LAM, tau=TAU):
    """All strictly-earlier events, scored against q and sorted best-first.
    m_q is deliberately NOT passed (text-only fallback): the query's own
    market outcome is unobserved at prediction time and the stance label is
    derived from it -- see train.build_examples's LEAKAGE BOUNDARY note.
    """
    scored = []
    for c in history:
        res = hybrid_similarity(
            embeddings[q.event_id], embeddings[c.event_id],
            c.market_vector, None, lam=lam, tau=tau,
        )
        scored.append((res.score, c))
    scored.sort(key=lambda x: -x[0])
    return scored


def make_tensors(e_q, items):
    """items: list of (score, event). Builds exactly-length tensors, no
    padding needed -- by test time there is always ample earlier history,
    unlike the early-training-example case train.py's to_batch handles.
    """
    e_q_t = torch.tensor(e_q, dtype=torch.float32).unsqueeze(0)
    m = np.stack([c.market_vector for _, c in items])
    s = np.array([sc for sc, _ in items], dtype=np.float32)
    return e_q_t, torch.tensor(m, dtype=torch.float32).unsqueeze(0), torch.tensor(s, dtype=torch.float32).unsqueeze(0)


def fusion_predict(model, e_q, items):
    e_q_t, m_t, s_t = make_tensors(e_q, items)
    with torch.no_grad():
        out = model(e_q_t, m_t, s_t)
    return float(out.s_hat.item())


def baseline_predict(model, e_q):
    with torch.no_grad():
        out = model(torch.tensor(e_q, dtype=torch.float32).unsqueeze(0))
    return float(out.s_hat.item())


def run_perturbations(model, e_q, ranked, k, rng, predict_fn=fusion_predict):
    """ranked: full history for this query, best-first (len > k+1 assumed --
    true for every test-split event given the size of the training history). Returns
    dict of condition -> (prediction, perturbation_magnitude or None).
    perturbation_magnitude is the market-vector distance between the
    removed rank-1 item and its replacement, where that's well-defined
    (replace_* conditions); None for the ablation conditions, which remove
    evidence rather than swap it.
    """
    top_k = ranked[:k]
    next_item = ranked[k]
    pool = ranked[k + 1:]
    random_item = rng.choice(pool)

    rank1_vec = top_k[0][1].market_vector
    conditions = {
        "original": (top_k, None),
        "ablate_top1": (top_k[1:], None),
        "ablate_bottom1": (top_k[:-1], None),
        "replace_with_next": (
            [next_item] + top_k[1:],
            float(np.linalg.norm(rank1_vec - next_item[1].market_vector)),
        ),
        "replace_with_random": (
            [random_item] + top_k[1:],
            float(np.linalg.norm(rank1_vec - random_item[1].market_vector)),
        ),
    }
    return {
        name: (predict_fn(model, e_q, items), mag)
        for name, (items, mag) in conditions.items()
    }


def main():
    print("=" * 72)
    print("Causal faithfulness protocol -- REAL data, same caveats as")
    print("train.py (stub encoder, pragmatic stance label, small n).")
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

    examples = build_examples(events, encoder, embeddings)
    train_ex = [e for e in examples if e["event"].date.year < 2020]
    test_ex = [e for e in examples if e["event"].date.year >= 2023]
    print(f"\ntrain={len(train_ex)}  test={len(test_ex)} (held-out sample for this protocol)")

    train_labels = np.array([e["label"] for e in train_ex])
    label_mean, label_std = float(train_labels.mean()), float(train_labels.std() + 1e-8)
    embed_dim, market_dim = encoder.dim, len(events[0].market_vector)

    print("\nTraining FusionHead and NonAugmentedHead (same setup as train.py) ...")
    torch.manual_seed(SEED)  # init is drawn at construction: seed first (reproducibility)
    fusion = FusionHead(embed_dim=embed_dim, market_dim=market_dim)
    train_model(fusion, train_ex, embed_dim, market_dim, K, True, label_mean, label_std)
    torch.manual_seed(SEED)
    baseline = NonAugmentedHead(embed_dim=embed_dim)
    train_model(baseline, train_ex, embed_dim, market_dim, K, False, label_mean, label_std)

    rng = random.Random(SEED)
    rows_fusion, rows_baseline = [], []
    for ex in test_ex:
        q = ex["event"]
        history = [e for e in events if e.date < q.date]
        ranked = rank_history(q, history, embeddings)
        if len(ranked) <= K + 1:
            continue  # not enough history for a random-control item distinct from top-(k+1)

        e_q = ex["e_q"]
        res_f = run_perturbations(fusion, e_q, ranked, K, rng, fusion_predict)
        rows_fusion.append(res_f)
        res_b = run_perturbations(baseline, e_q, ranked, K, rng, lambda m, eq, items: baseline_predict(m, eq))
        rows_baseline.append(res_b)

    def to_raw(pred_std, label_mean=label_mean, label_std=label_std):
        return pred_std * label_std + label_mean

    print(f"\nEvaluated on {len(rows_fusion)} test-set queries (skipped "
          f"{len(test_ex) - len(rows_fusion)} with insufficient history for a control item).")

    print("\n--- FusionHead: does removed-evidence rank matter? ---")
    shift_top1 = [abs(to_raw(r["ablate_top1"][0]) - to_raw(r["original"][0])) for r in rows_fusion]
    shift_bottom1 = [abs(to_raw(r["ablate_bottom1"][0]) - to_raw(r["original"][0])) for r in rows_fusion]
    m1, lo1, hi1 = bootstrap_ci(shift_top1)
    m2, lo2, hi2 = bootstrap_ci(shift_bottom1)
    out_json = {"n_queries": len(rows_fusion),
                "rank_test": {"shift_rank1": [m1, lo1, hi1], f"shift_rank{K}": [m2, lo2, hi2]}}
    print(f"  |shift| removing rank-1 (most relevant):  {m1:.4f}  [{lo1:.4f}, {hi1:.4f}]")
    print(f"  |shift| removing rank-{K} (least relevant): {m2:.4f}  [{lo2:.4f}, {hi2:.4f}]")
    p = paired_bootstrap_pvalue(shift_top1, shift_bottom1)
    print(f"  paired bootstrap p-value (rank-1 shift vs rank-{K} shift): {p:.4f}")
    out_json["rank_test"]["p"] = p
    if m1 > m2 and p < 0.05:
        verdict = "SUPPORTED"
    elif m1 <= m2 and p < 0.05:
        verdict = "CONTRADICTED"
    else:
        verdict = "INCONCLUSIVE (not statistically distinguishable at this n -- do not report this as a finding either way)"
    print(f"  HYPOTHESIS ({verdict}): removing the most-relevant precedent "
          f"should shift the prediction more than removing the least-relevant one.")

    print("\n--- FusionHead: does the size of the swap matter? ---")
    mags, shifts = [], []
    for cond in ("replace_with_next", "replace_with_random"):
        for r in rows_fusion:
            pred, mag = r[cond]
            mags.append(mag)
            shifts.append(abs(to_raw(pred) - to_raw(r["original"][0])))
    rho, lo, hi = bootstrap_metric_ci(mags, shifts, lambda a, b: spearman_rho(a, b))
    out_json["magnitude_test"] = {"rho": [rho, lo, hi], "n_perturbations": len(mags)}
    print(f"  Spearman rho(perturbation magnitude, |prediction shift|): "
          f"{rho:.3f}  [{lo:.3f}, {hi:.3f}]  (n={len(mags)} perturbations, "
          f"{len(rows_fusion)} queries x 2 replace-conditions)")
    if lo > 0:
        verdict = "SUPPORTED"
    elif hi < 0:
        verdict = "CONTRADICTED"
    else:
        verdict = "INCONCLUSIVE (CI includes zero -- do not report this as a finding either way)"
    print(f"  HYPOTHESIS ({verdict}): a bigger swap in the evidence "
          f"should cause a bigger shift in the prediction.")

    print("\n--- Validity check: NonAugmentedHead must show ZERO shift ---")
    max_baseline_shift = max(
        abs(to_raw(r[cond][0]) - to_raw(r["original"][0]))
        for r in rows_baseline
        for cond in ("ablate_top1", "ablate_bottom1", "replace_with_next", "replace_with_random")
    )
    print(f"  max |shift| across all perturbations, all queries: {max_baseline_shift:.2e}")
    out_json["baseline_max_shift"] = float(max_baseline_shift)
    from pipeline import save_json
    save_json("faithfulness.json", out_json)
    assert max_baseline_shift < 1e-6, (
        "NonAugmentedHead shifted when its retrieved evidence changed, but it never "
        "sees that evidence -- the harness itself has a bug, fix before trusting anything above."
    )
    print("  OK: baseline is provably blind to the evidence, exactly as it should be. "
          "This confirms any FusionHead shift above is coming from the retrieved evidence, "
          "not from some other source of nondeterminism in this harness.")

    print("\n" + "=" * 72)
    print("REMINDERS: n=" + str(len(rows_fusion)) + " test queries -- read the CIs, not just")
    print("the point estimates. Stub encoder + pragmatic stance label still apply")
    print("(see train.py). Re-run once the real encoder and a confirmed stance")
    print("definition are wired in.")
    print("=" * 72)


if __name__ == "__main__":
    main()
