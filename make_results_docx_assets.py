"""Step 1 of the Word results report: draws the charts (PNG) and writes payload.json, all from results/*.json.
Step 2 is make_results_docx.js (docx-js). Usage: python make_results_docx_assets.py <output_dir>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
R = lambda n: json.loads((ROOT / "results" / n).read_text(encoding="utf-8"))
rq, cv, fs, fsh = R("retrieval_quality.json"), R("cv_rolling.json"), R("faithfulness_seeds.json"), R("faithfulness_seeds_shuffled.json")
reg, lomo, ab, leak = R("regime_robustness.json"), R("cv_lomo.json"), R("ablations_downstream.json"), R("leak_control.json")

BLUE, GREY, GREEN = "#2f6fb0", "#9aa5b1", "#3a9d6b"


def hbar(path, title, rows, xmax, vline=None, vlabel="", xlabel=""):
    """rows: (label, value, color, lo, hi) top to bottom."""
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.4), dpi=200)
    ys = list(range(len(rows)))[::-1]
    for y, (lbl, v, c, lo, hi) in zip(ys, rows):
        ax.barh(y, max(v, 0), color=c, height=0.6)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color="#222", lw=1.6)
        ax.text(max(hi if hi is not None else v, v) + xmax * 0.015, y, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    if vline is not None:
        ax.axvline(vline, color="#c0392b", ls="--", lw=1.2)
        ax.text(vline + xmax * 0.012, len(rows) - 0.45, vlabel, color="#c0392b", fontsize=8.5, va="center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


m = rq["modes"]
nd = lambda k: (m[k]["ndcg@10"][0], m[k]["ndcg@10"][1], m[k]["ndcg@10"][2])
hbar(out / "c1.png", "Finding similar past events (0 = random, 1 = perfect)", [
    ("Random", *nd("random")[:1], GREY, *nd("random")[1:]),
    ("Keyword matching (BM25)", nd("bm25")[0], GREY, *nd("bm25")[1:]),
    ("Text-only (placeholder encoder)", nd("dense_text_only")[0], GREY, *nd("dense_text_only")[1:]),
    ("Text + market mix", nd("hybrid")[0], BLUE, *nd("hybrid")[1:]),
    ("Market outcome only", nd("market_only")[0], GREEN, *nd("market_only")[1:]),
], 1.0, xlabel="score (nDCG@10); whiskers = 95% uncertainty range")

a0, a1 = cv["per_seed"]["0"], cv["per_seed"]["1"]
P = lambda rep, who, k: rep[who]["all"][k]["point"]
C = lambda rep, who, k: tuple(rep[who]["all"][k]["ci95"])
hbar(out / "c2.png", "How often it gets the direction right (hawkish vs dovish)", [
    ("Retrieval model (run A)", P(a0, "fusion", "dir_acc"), BLUE, *C(a0, "fusion", "dir_acc")),
    ("Retrieval model (run B)", P(a1, "fusion", "dir_acc"), BLUE, *C(a1, "fusion", "dir_acc")),
    ("Plain model, no retrieval", P(a0, "baseline", "dir_acc"), GREY, *C(a0, "baseline", "dir_acc")),
    ("Always guess the same answer", P(a0, "train_mean", "dir_acc"), GREY, *C(a0, "train_mean", "dir_acc")),
], 1.0, vline=0.5, vlabel="coin flip", xlabel="share of correct directions; whiskers = 95% range")

hbar(out / "c4.png", "Does its answer move when the evidence is swapped?", [
    ("Real model (average of 5 runs)", fs["rho_mean"], BLUE, fs["rho_mean"] - fs["rho_sd"], fs["rho_mean"] + fs["rho_sd"]),
    ("Model trained on scrambled answers", fsh["rho_mean"], GREY, fsh["rho_mean"] - fsh["rho_sd"], fsh["rho_mean"] + fsh["rho_sd"]),
], 1.0, xlabel="correlation between size of swap and change in answer; whiskers = ± 1 sd across runs")

rg = {(x["mode"], x["regime"]): x["ndcg@10"] for x in reg["retrieval"]}
eras = [("pre_zlb", "Normal rates (to 2014)"), ("zlb", "Near-zero rates (2014–22)"), ("post_hiking", "Rate hikes (2022 on)")]
rows = []
for k, lbl in eras:
    n = reg["regime_counts"][k]
    rows.append((f"{lbl}, n={n}" + chr(10) + "market only", rg[("market_only", k)][0], GREEN, *rg[("market_only", k)][1:]))
    rows.append((f"{lbl}, n={n}" + chr(10) + "text + market", rg[("hybrid", k)][0], BLUE, *rg[("hybrid", k)][1:]))
hbar(out / "c5.png", "Finding similar past events, by economic era", rows, 1.0, xlabel="score (nDCG@10); whiskers = 95% uncertainty range")

lo = lomo["report"]
sh = lambda rep, who: rep[who]["all"]["r2"]["point"]
main = ab["main (k=5, uncertainty on, ois_1y)"]["summary(mean,sd)"]
payload = {
    "n_events": rq["n_events"], "n_docs": rq["n_documents"],
    "retrieval": {k: m[k]["ndcg@10"][0] for k in m},
    "cv_n": cv["n_pooled"],
    "dir": {"fusion": [P(a0, "fusion", "dir_acc"), P(a1, "fusion", "dir_acc")], "baseline": [P(a0, "baseline", "dir_acc"), P(a1, "baseline", "dir_acc")],
            "mean": [P(a0, "train_mean", "dir_acc"), P(a1, "train_mean", "dir_acc")]},
    "r2": {"fusion": [sh(a0, "fusion"), sh(a1, "fusion")], "baseline": [sh(a0, "baseline"), sh(a1, "baseline")], "mean": [sh(a0, "train_mean"), sh(a1, "train_mean")]},
    "lomo": {"n": lo["fusion"]["all"]["n"], "fusion_r2": lo["fusion"]["all"]["r2"]["point"], "base_r2": lo["baseline"]["all"]["r2"]["point"]},
    "faith": {"real": fs["rho_mean"], "real_sd": fs["rho_sd"], "real_pos": fs["seeds_with_ci_above_zero"], "seeds": fs["n_seeds"],
              "ctl": fsh["rho_mean"], "ctl_sd": fsh["rho_sd"], "ctl_pos": fsh["seeds_with_ci_above_zero"], "rank_sig": fs["seeds_with_rank_p_below_0.05"]},
    "regime_counts": reg["regime_counts"], "density": reg["precedent_density_post_hiking"]["mean_same_regime_fraction"],
    "leak": {"leaky": leak["runs"]["leaky"]["dir_acc_pooled"][0], "fixed": leak["runs"]["fixed"]["dir_acc_pooled"][0]},
    "ablation_main": {"fusion_dir": main["fusion"]["dir_acc"][0], "base_dir": main["baseline"]["dir_acc"][0], "fusion_r2": main["fusion"]["r2"][0]},
}
(out / "payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("assets written to", out)
