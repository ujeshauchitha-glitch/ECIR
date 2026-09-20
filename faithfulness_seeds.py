"""
Multi-seed faithfulness (proposal Sec 5-6 protocol), because a single trained model is not enough.

After the reproducibility fix (weights were previously unseeded), the single-model result in
faithfulness.py turned out to depend strongly on the initialisation: the same protocol gave
rho ~ 0.37 (CI excluding 0) for one initialisation and ~ 0.07 (CI including 0) for another.
So the honest statement of the faithfulness result is the distribution over initialisations,
which is what this script reports.

For each seed: train FusionHead exactly as train.py does (train < 2020), perturb every test-split
query's evidence (swap the top precedent for the next-ranked / a random one; ablate the most /
least similar) and record (a) Spearman rho between swap size and |prediction shift| and (b) the
rank test. Summary: per-seed values, their mean/sd, how many seeds have a CI excluding 0, and a
pooled estimate (pooling perturbations across seeds treats them as independent, which they are not
-- the per-seed numbers are the primary result, the pooled CI is an optimistic convenience).

Writes results/faithfulness_seeds.json.  Usage: python faithfulness_seeds.py [--seeds 5]
"""
from __future__ import annotations

import random
import sys

import numpy as np
import torch

import train
from faithfulness import fusion_predict, rank_history, run_perturbations
from fusion_head import FusionHead
from metrics import bootstrap_ci, bootstrap_metric_ci, paired_bootstrap_pvalue, spearman_rho
from pipeline import Cfg, load_setup, make_examples, save_json


def main():
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5
    setup = load_setup()
    events, embeddings = setup.events, setup.embeddings
    embed_dim, market_dim, K = setup.encoder.dim, len(events[0].market_vector), train.K
    ex = make_examples(setup, Cfg())
    tr = [e for e in ex if e["event"].date.year < 2020]
    te = [e for e in ex if e["event"].date.year >= 2023]
    labels = np.array([e["label"] for e in tr])
    mu, sd = float(labels.mean()), float(labels.std() + 1e-8)

    per_seed, all_mags, all_shifts = [], [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = FusionHead(embed_dim, market_dim)
        train.train_model(model, tr, embed_dim, market_dim, K, True, mu, sd, seed=seed, verbose=False)
        rng = random.Random(seed)
        mags, shifts, top1, bot1 = [], [], [], []
        for e in te:
            q = e["event"]
            ranked = rank_history(q, [x for x in events if x.date < q.date], embeddings)
            if len(ranked) <= K + 1:
                continue
            res = run_perturbations(model, e["e_q"], ranked, K, rng, fusion_predict)
            raw = lambda z: z * sd + mu
            orig = raw(res["original"][0])
            for cond in ("replace_with_next", "replace_with_random"):
                pred, mag = res[cond]
                mags.append(mag)
                shifts.append(abs(raw(pred) - orig))
            top1.append(abs(raw(res["ablate_top1"][0]) - orig))
            bot1.append(abs(raw(res["ablate_bottom1"][0]) - orig))
        rho, lo, hi = bootstrap_metric_ci(mags, shifts, lambda a, b: spearman_rho(a, b))
        rec = {"seed": seed, "n_queries": len(top1), "n_perturbations": len(mags), "rho": [rho, lo, hi],
               "rank_shift_top1": bootstrap_ci(top1), "rank_shift_bottom1": bootstrap_ci(bot1),
               "rank_p": paired_bootstrap_pvalue(top1, bot1)}
        per_seed.append(rec)
        all_mags += mags
        all_shifts += shifts
        print(f"seed {seed}: rho {rho:.3f} [{lo:.3f}, {hi:.3f}]  rank-test p={rec['rank_p']:.3f}", flush=True)

    rhos = [r["rho"][0] for r in per_seed]
    pooled = bootstrap_metric_ci(all_mags, all_shifts, lambda a, b: spearman_rho(a, b))
    out = {
        "n_seeds": n_seeds, "per_seed": per_seed,
        "rho_mean": float(np.mean(rhos)), "rho_sd": float(np.std(rhos)),
        "seeds_with_ci_above_zero": int(sum(r["rho"][1] > 0 for r in per_seed)),
        "seeds_with_rank_p_below_0.05": int(sum(r["rank_p"] < 0.05 for r in per_seed)),
        "pooled_rho_optimistic_ci": list(pooled),
    }
    save_json("faithfulness_seeds.json", out)
    print(f"mean rho {out['rho_mean']:.3f} +/- {out['rho_sd']:.3f}; CI>0 in {out['seeds_with_ci_above_zero']}/{n_seeds} seeds")


if __name__ == "__main__":
    main()
