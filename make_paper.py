"""
Builds paper/paper.md (a DRAFT manuscript) from results/*.json.

Rule: no number in the manuscript is typed by hand. Every table cell and every count comes from a
results file produced by the experiment scripts; if a results file is missing the table says so
(TODO) instead of showing something plausible. Interpretive sentences that depend on numbers are
generated from the numbers by explicit, conservative rules (e.g. "CI includes 0.5").

Re-run after re-running the experiments (in particular after wiring in the real encoder):
    python make_paper.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

NL = chr(10)
ROOT = Path(__file__).parent
RES = ROOT / "results"
OUT = ROOT / "paper"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(v):
    """[point, lo, hi] or {'point', 'ci95'} -> '0.123 [0.045, 0.201]'."""
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        if "point" not in v:
            return "n/a"
        p, (lo, hi) = v["point"], v["ci95"]
    else:
        p, lo, hi = v
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def todo(script):
    return f"*TODO: results file missing -- run `python {script}`.*"


def includes(v, x):
    """True if the CI of v contains x."""
    if v is None:
        return None
    lo, hi = (v["ci95"] if isinstance(v, dict) else v[1:3])
    return lo <= x <= hi


# ---------------------------------------------------------------------------------- tables
def table_retrieval():
    r = load("retrieval_quality.json")
    if not r:
        return todo("run_real.py")
    names = {"random": "Random", "bm25": "BM25", "dense_text_only": "Text-only dense (lambda=1)",
             "market_only": "Market-only (lambda=0)", "hybrid": "Hybrid (lambda=0.5, tau=1, squared kernel)"}
    rows = [[names[m]] + [ci(r["modes"][m][k]) for k in ("ndcg@5", "ndcg@10", "map")] for m in names]
    return md_table(["Retriever", "nDCG@5", "nDCG@10", "MAP"], rows)


def table_grid():
    r = load("ablations_retrieval.json")
    if not r:
        return todo("ablations.py")
    parts = [f"Reference points (nDCG@10): market-only {ci(r['refs']['market_only']['ndcg@10'])}, "
             f"text-only {ci(r['refs']['text_only']['ndcg@10'])}. Median pairwise distance in the market space: "
             f"{r['median_l2_dist']:.2f} (L2), {r['median_sq_dist']:.1f} (squared).\n"]
    for kernel, label in (("squared", "squared distance, exp(-||m_q-m_i||^2 / tau)"),
                          ("l2", "plain L2 distance, exp(-||m_q-m_i||_2 / tau)")):
        taus = []
        for g in r["grid"]:
            if g["kernel"] == kernel and g["tau"] not in taus:
                taus.append(g["tau"])
        rows = []
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            row = [f"{lam:g}"]
            for t in taus:
                g = next(x for x in r["grid"] if x["kernel"] == kernel and x["tau"] == t and x["lam"] == lam)
                row.append(f3(g["ndcg@10"][0]))
            rows.append(row)
        parts.append(f"**Kernel: {label}** (mean nDCG@10; columns = tau)\n\n" + md_table(["lambda"] + taus, rows))
    return "\n\n".join(parts)


def _metric_cells(rec):
    if rec is None:
        return ["n/a"] * 3
    return [ci(rec.get("dir_acc")), ci(rec.get("spearman")), ci(rec.get("r2"))]


def table_downstream():
    r = load("downstream_train_py.json")
    if not r:
        return todo("train.py")
    rows = []
    for split in ("val", "test"):
        s = r["splits"].get(split)
        if not s:
            continue
        for key, name in (("fusion", "Retrieval-augmented (FusionHead)"), ("baseline", "Non-augmented head"),
                          ("train_mean", "Reference: predict training mean")):
            rows.append([f"{split} (n={s[key]['n']})", name] + _metric_cells(s[key]))
    return md_table(["Split", "Model", "Directional acc.", "Spearman rho", "R^2"], rows)


def table_cv_rolling():
    r = load("cv_rolling.json")
    if not r:
        return todo("cv.py")
    rows = []
    for seed, rep in r["per_seed"].items():
        for who, name in (("fusion", "Retrieval-augmented"), ("baseline", "Non-augmented"), ("train_mean", "Training-mean reference")):
            for grp in ("all", "pre_zlb", "zlb", "post_hiking"):
                g = rep[who][grp]
                if "note" in g:
                    rows.append([seed, name, f"{grp} (n={g['n']})", g["note"], "", ""])
                else:
                    rows.append([seed, name, f"{grp} (n={g['n']})", ci(g["dir_acc"]), ci(g["spearman"]), ci(g["r2"])])
    return md_table(["Seed", "Model", "Regime (pooled n)", "Directional acc.", "Spearman rho", "R^2"], rows)


def table_lomo():
    r = load("cv_lomo.json")
    if not r:
        return todo("cv.py --lomo --stride 3")
    rows = []
    for who, name in (("fusion", "Retrieval-augmented"), ("baseline", "Non-augmented"), ("train_mean", "Training-mean reference")):
        for grp in ("all", "pre_zlb", "zlb", "post_hiking"):
            g = r["report"][who][grp]
            if "note" in g:
                rows.append([name, f"{grp} (n={g['n']})", "too few events", "", ""])
                continue
            rho = ("n/a (leave-one-out mean is anti-correlated with the held-out label by construction)"
                   if who == "train_mean" else ci(g["spearman"]))
            rows.append([name, f"{grp} (n={g['n']})", ci(g["dir_acc"]), rho, ci(g["r2"])])
    proto = re.sub(r"every (\d+)th meeting", r"one in every \1 meetings", r["protocol"])
    return f"*Protocol: {proto}.*" + NL + NL + md_table(["Model", "Regime (n)", "Directional acc.", "Spearman rho", "R^2"], rows)


def table_faith():
    f, g = load("faithfulness.json"), load("regime_robustness.json")
    if not f:
        return todo("faithfulness.py")
    rt, mt = f["rank_test"], f["magnitude_test"]
    rk = [k for k in rt if k.startswith("shift_rank") and k != "shift_rank1"][0]
    rows = [
        [f"Mean |shift| when removing rank-1 (most similar) precedent (n={f['n_queries']} test queries)", ci(rt["shift_rank1"])],
        [f"Mean |shift| when removing {rk.replace('shift_rank', 'rank-')} (least similar retrieved) precedent", ci(rt[rk])],
        ["Paired bootstrap p-value, rank-1 vs last", f"{rt['p']:.3f}"],
        [f"Spearman rho(perturbation size, |prediction shift|), n={mt['n_perturbations']} swaps", ci(mt["rho"])],
        ["Validity check: max |shift| of the evidence-blind baseline", f"{f['baseline_max_shift']:.1e} (must be 0)"],
    ]
    if g:
        for reg, v in g["faithfulness"].items():
            rows.append([f"Spearman rho, {reg} events only (n={v['n_perturbations']} swaps)", ci(v["rho"])])
    t1 = md_table(["Quantity", "Estimate [95% CI]"], rows)
    fs = load("faithfulness_seeds.json")
    if not fs:
        return t1 + NL + NL + todo("faithfulness_seeds.py")
    rows = [[str(r["seed"]), ci(r["rho"]), f"{r['rank_p']:.3f}"] for r in fs["per_seed"]]
    t2 = md_table(["Seed (independent initialisation)", "rho(swap size, |shift|) [95% CI]", "Rank-test p"], rows)
    summ = (f"Across {fs['n_seeds']} initialisations: mean rho = {fs['rho_mean']:.3f} (sd {fs['rho_sd']:.3f}); the CI excludes 0 in "
            f"{fs['seeds_with_ci_above_zero']}/{fs['n_seeds']} seeds; the rank test is significant (p<0.05) in "
            f"{fs['seeds_with_rank_p_below_0.05']}/{fs['n_seeds']} seeds. Pooled over seeds (optimistic, perturbations are not independent): {ci(fs['pooled_rho_optimistic_ci'])}.")
    note = ("**Stability across initialisations** (the single-model result above depends on the initialisation, "
            "so this is the primary faithfulness result):")
    ctl = load("faithfulness_seeds_shuffled.json")
    if ctl:
        cs = (f"**Control -- models trained on shuffled labels (no possible skill):** mean rho = {ctl['rho_mean']:.3f} (sd {ctl['rho_sd']:.3f}); "
              f"CI excludes 0 in {ctl['seeds_with_ci_above_zero']}/{ctl['n_seeds']} seeds. "
              + ("Because the no-skill control shows a comparable swap-size/shift correlation, this metric measures the model's smooth "
                 "sensitivity to its inputs and **cannot by itself show that retrieved evidence is causally load-bearing**."
                 if ctl["rho_mean"] > 0.5 * fs["rho_mean"] else
                 "The control shows a clearly weaker correlation than the real models."))
    else:
        cs = todo("faithfulness_seeds.py --shuffle")
    return t1 + NL + NL + note + NL + NL + t2 + NL + NL + summ + NL + NL + cs


def table_regime():
    g = load("regime_robustness.json")
    if not g:
        return todo("regime_robustness.py")
    rows = [[x["mode"], f"{x['regime']} (n={x['n']})", ci(x["ndcg@5"]), ci(x["ndcg@10"]), ci(x["map"])] for x in g["retrieval"]]
    t1 = md_table(["Retriever", "Query regime (n)", "nDCG@5", "nDCG@10", "MAP"], rows)
    rows = []
    for reg, v in g["downstream"].items():
        for name in ("FusionHead", "NonAugmented"):
            m = v[name]
            rows.append([f"{reg} (n={v['n']})", name, ci(m["dir. accuracy"]), ci(m["Spearman rho"]), ci(m["R^2"])])
    t2 = md_table(["Held-out regime (n)", "Model", "Directional acc.", "Spearman rho", "R^2"], rows)
    d = g.get("precedent_density_post_hiking")
    t3 = (f"Precedent density: for the {d['n_queries']} post-hiking queries, on average {d['mean_same_regime_fraction']:.0%} of the "
          f"top-5 retrieved precedents are themselves post-hiking events; the rest come from earlier regimes.") if d else ""
    return "\n\n".join([t1, t2, t3])


def table_ablation():
    a = load("ablations_downstream.json")
    if not a:
        return todo("ablations.py")
    rows = []
    for name, v in a.items():
        s = v["summary(mean,sd)"]
        ms = lambda who, m: f"{s[who][m][0]:.3f} +/- {s[who][m][1]:.3f}"
        rows.append([name, str(len(v["per_seed"])), ms("fusion", "dir_acc"), ms("baseline", "dir_acc"),
                     ms("fusion", "r2"), ms("baseline", "r2"), ms("train_mean", "r2"), s["fusion_beats_baseline_dir_acc_in"]])
    return md_table(["Configuration", "Seeds", "Fusion dir. acc.", "Baseline dir. acc.", "Fusion R^2", "Baseline R^2",
                     "Train-mean R^2", "Fusion > baseline (dir. acc.)"], rows)


# ---------------------------------------------------------------------------------- findings
def auto_findings():
    """Conservative, rule-generated statements. Nothing here is written from memory."""
    out = []
    r = load("retrieval_quality.json")
    if r:
        m = r["modes"]
        out.append(f"Retrieval quality (n={r['n_events']} events): market-only nDCG@10 = {f3(m['market_only']['ndcg@10'][0])}, "
                   f"hybrid (lambda=0.5, tau=1) = {f3(m['hybrid']['ndcg@10'][0])}, BM25 = {f3(m['bm25']['ndcg@10'][0])}, "
                   f"text-only dense = {f3(m['dense_text_only']['ndcg@10'][0])}, random = {f3(m['random']['ndcg@10'][0])}. "
                   "Because relevance is defined by market-outcome distance, market-only is near-ceiling by construction.")
    d = load("downstream_train_py.json")
    if d:
        for split, s in d["splits"].items():
            f, b = s["fusion"], s["baseline"]
            out.append(f"{split} split (n={f['n']}): retrieval-augmented directional accuracy {ci(f['dir_acc'])} vs non-augmented "
                       f"{ci(b['dir_acc'])}; chance (0.5) is {'inside' if includes(f['dir_acc'], 0.5) else 'outside'} the retrieval-augmented CI; "
                       f"R^2 {f3(f['r2'][0])} vs {f3(b['r2'][0])} (a value below 0 means worse than predicting the training mean).")
    c = load("cv_rolling.json")
    if c:
        for seed, rep in c["per_seed"].items():
            f, b = rep["fusion"]["all"], rep["baseline"]["all"]
            out.append(f"Rolling-origin CV, seed {seed} (pooled n={f['n']}): retrieval-augmented directional accuracy {ci(f['dir_acc'])} "
                       f"(chance {'inside' if includes(f['dir_acc'], 0.5) else 'outside'} CI), baseline {ci(b['dir_acc'])}; "
                       f"R^2 {ci(f['r2'])} vs {ci(b['r2'])}.")
    a = load("ablations_downstream.json")
    if a:
        ctl = a.get("CONTROL: shuffled training labels")
        if ctl:
            s = ctl["summary(mean,sd)"]["fusion"]
            out.append(f"Control (training labels shuffled): fusion directional accuracy {s['dir_acc'][0]:.3f} +/- {s['dir_acc'][1]:.3f}, "
                       f"R^2 {s['r2'][0]:.3f} -- this is the level a model with NO real skill reaches through this pipeline.")
    fs = load("faithfulness_seeds.json")
    if fs:
        out.append(f"Faithfulness across {fs['n_seeds']} initialisations: rho(swap size, |shift|) mean {fs['rho_mean']:.3f} (sd {fs['rho_sd']:.3f}); "
                   f"CI excludes 0 in {fs['seeds_with_ci_above_zero']}/{fs['n_seeds']} seeds -- a result that varies this much with the initialisation is not a stable finding.")
    ctl = load("faithfulness_seeds_shuffled.json")
    if ctl:
        out.append(f"Faithfulness control (shuffled-label models): mean rho {ctl['rho_mean']:.3f} (sd {ctl['rho_sd']:.3f}), CI>0 in {ctl['seeds_with_ci_above_zero']}/{ctl['n_seeds']} seeds "
                   "-- compare with the real models above before reading anything into the faithfulness numbers.")
    f = load("faithfulness.json")
    if f:
        mt = f["magnitude_test"]
        out.append(f"Faithfulness: rho(perturbation size, |shift|) = {ci(mt['rho'])} over {mt['n_perturbations']} swaps -- "
                   f"{'the CI excludes 0' if not includes(mt['rho'], 0) else 'the CI includes 0 (inconclusive)'}; rank test p = {f['rank_test']['p']:.3f} "
                   f"({'inconclusive' if f['rank_test']['p'] >= 0.05 else 'significant'}).")
    return "\n".join(f"- {x}" for x in out) if out else todo("all experiment scripts")


# ---------------------------------------------------------------------------------- manuscript
def leak_sentence():
    l = load("leak_control.json")
    if not l:
        return "[TODO: run leak_control.py for the measured size of this effect.]"
    a, b = l["runs"]["leaky"], l["runs"]["fixed"]
    diff = a["dir_acc_pooled"][0] - b["dir_acc_pooled"][0]
    overlap = a["dir_acc_pooled"][1] <= b["dir_acc_pooled"][2] and b["dir_acc_pooled"][1] <= a["dir_acc_pooled"][2]
    verdict = ("the intervals overlap, so with the current placeholder encoder and sample size the inflation is not distinguishable from noise; "
               "the leaky setup is nevertheless invalid, and a stronger encoder could exploit the leak more" if overlap else
               "the intervals do not overlap, so the leak measurably inflates results")
    return (f"We measured this directly (same data, model, seeds and held-out set; only the retrieval differs): held-out directional accuracy is "
            f"{ci(a['dir_acc_pooled'])} with the leak versus {ci(b['dir_acc_pooled'])} without (difference {diff:+.3f}; n={a['n_heldout']} events x 3 seeds); {verdict}.")


def counts():
    r = load("retrieval_quality.json") or {}
    g = load("regime_robustness.json") or {}
    return r.get("n_events"), r.get("n_documents"), g.get("regime_counts", {})


def build() -> str:
    n_ev, n_doc, rc = counts()
    n_ev_s = str(n_ev) if n_ev else "[TODO: run run_real.py]"
    rc_s = ", ".join(f"{k}: {v}" for k, v in rc.items()) if rc else "[TODO]"
    generated = dt.date.today().isoformat()
    return f"""<!-- GENERATED by make_paper.py on {generated} from results/*.json. Do not hand-edit numbers; edit make_paper.py. -->

# Precedent-RAG: Retrieval-Augmented Evidence Grounding for Explainable Central Bank Communication Stance Prediction

*Anonymous authors (double-blind submission).*

> **DRAFT STATUS -- read before using any sentence below.** (1) The text encoder used in every experiment is a
> **placeholder** (hashed bag-of-words), not the intended trained encoder, so all text-similarity results are
> placeholder-quality and must be regenerated (`python make_paper.py` after re-running the experiments). (2) The
> stance target is a **stand-in** (1-year OIS change) -- see Sec. 3.4. (3) Bracketed `[TODO]` items are things this
> draft cannot fill honestly. (4) Related-work citations marked *[verify]* were written from memory and need checking.

## Abstract

Central bank communication moves markets, and a substantial body of work uses language models to classify or score
its hawkish-dovish stance, treating each communication as an independent prediction target. We propose
**Precedent-RAG**, a retrieval-augmented framework that grounds continuous stance prediction for European Central Bank
(ECB) communications in a retrieved set of historically analogous events, using a similarity that combines textual
representation with proximity in realized market-reaction space; the retrieved precedents both inform the prediction and
serve as a tabular, auditable explanation. We introduce **ECB-Precedent**, a benchmark in which retrieval relevance is
derived from realized market-outcome proximity across {n_ev_s} ECB monetary-policy events (1999--2025), so retrieval
quality can be evaluated independently of downstream accuracy. We test whether retrieved evidence is causally faithful
via controlled evidence perturbation, and stratify all analyses by policy regime (pre-zero-lower-bound, zero-lower-bound,
post-hiking). **[TODO: one sentence with the headline result -- to be written only after the trained encoder is wired in.
With the placeholder encoder used so far, no out-of-sample prediction skill is observed; see Sec. 5.]**

## 1 Introduction

Human analysts read a new central bank communication by comparing it with earlier, similar communications and
remembering what markets did afterwards. Existing computational approaches instead classify each communication in
isolation. We reformulate stance prediction as *evidence-grounded inference*: each prediction is conditioned on a retrieved
set of historical precedents and their realized market outcomes.

**Contributions.** (1) A retrieval-augmented prediction framework for continuous stance. (2) *ECB-Precedent*, a benchmark whose
relevance labels come from realized market outcomes instead of human judgement. (3) A hybrid retriever that mixes text and
market-outcome similarity under one tunable weight, evaluated against lexical, dense and market-only baselines. (4) A causal
faithfulness protocol that tests, by perturbing the evidence, whether retrieved precedents actually drive the prediction.
(5) A regime-stratified robustness analysis (zero lower bound vs. post-hiking) that treats non-stationarity as an experimental
variable. Beyond these, we report (6) an explicit *leakage boundary* for retrieval-augmented prediction with outcome-derived
labels, including a measured example of how much a violation inflates results (Sec. 3.3).

## 2 Related Work

**Central bank stance classification.** The dominant paradigm labels hawkish/dovish/neutral at sentence or document level;
manually labelled FOMC sentence corpora (Gorodnichenko et al., 2023 *[verify]*) underlie much of it, including domain-adapted
central-bank language models (Gambacorta et al., 2024 *[verify]*). Retrieval-based approaches over Federal Reserve reports exist but do
not fuse retrieval with market-outcome supervision, do not evaluate retrieval as an independent object, and do not address the
ECB. **Retrieval-augmented generation and explainability.** RAG (Lewis et al., 2020) and evidence-attribution work such as RAG-Ex
(Sudhi et al., 2024 *[verify]*) concentrate on open-domain generation; retrieval-augmented *regression* over an outcome-labelled
corpus, where correctness is externally verifiable, is comparatively unexplored. **Market-based measurement of monetary policy.**
High-frequency event-study data (Gürkaynak et al., 2005; Altavilla et al., 2019) provide the realized reactions we use as outcomes.
[TODO: extend with a proper literature search; this section was condensed from the project proposal and has not been re-verified.]

## 3 Method

### 3.1 Setting
Let D be the ECB communication corpus and E a subset of D of monetary-policy events, each with a realized market-reaction vector
m_i in R^k. A frozen encoder f maps a document to e_i = f(d_i).

### 3.2 Hybrid similarity
For a query q and candidate d_i in E:
`sim(q, d_i) = lambda * cos(e_q, e_i) + (1 - lambda) * exp(-dist(m_q, m_i) / tau)`. We evaluate both dist = ||.||_2^2 (squared, used in
our main runs) and dist = ||.||_2 (plain L2); the proposal text is ambiguous between them and Table 2 shows their behaviour.
**[TODO(owner): confirm which kernel is intended.]** The market term requires m_q, which is available for *historical* query events
(benchmark construction; retrieval-quality evaluation) but is unobserved at genuine prediction time.

### 3.3 Leakage boundary
At prediction time retrieval is therefore **text-only** (m_q not used) and restricted to **strictly earlier** events; only the
*candidates'* known outcomes enter the model, as evidence. This boundary matters: our stance target is computed from the query's own
market vector, so retrieving with m_q hands the model part of the answer. {leak_sentence()}
**[TODO(owner): confirm that "text-only retrieval at prediction time" is the intended deployment meaning of hybrid retrieval.]**

### 3.4 Prediction target
The proposal does not define a numeric stance target. As an explicit modelling choice, we define stance as the realized 1-year OIS rate
change (basis points) in the EA-MPD monetary-event window: positive = hawkish surprise, negative = dovish, in the spirit of single-rate
policy-shock measures (Gürkaynak et al., 2005). We test sensitivity to two alternatives (Table 8).
**[TODO(owner): replace/compare with the definition from the companion work once available.]**

### 3.5 Retrieval-augmented prediction head
For the k retrieved precedents with outcomes m_(1..k) and similarities s_(1..k), a cross-attention layer takes a query projection of
e_q and key/value projections of each m_(j); attention logits are the learned query-key score **plus** s_(j) (retrieval-biased
attention). The attended context is concatenated with e_q and passed through a two-layer MLP producing a stance estimate and a
positive uncertainty, trained with Gaussian negative log-likelihood (Adam, lr 1e-3, 150 epochs, batch size 1, k = 5). All layers are
trainable, so the model is not a k-NN average (every parameter receives gradient -- unit-tested). The **non-augmented** baseline
uses the same head on e_q alone. The explanation output is the retrieved precedents with similarity, attention weight and realized outcome.

## 4 ECB-Precedent benchmark
Two events are a relevant pair when the Euclidean distance of their market vectors falls below the bottom-decile threshold of all
pairwise distances; graded relevance is `max(0, 1 - d/threshold)` **(this grading function is an implementation choice, to be endorsed
or replaced)**. This yields a benchmark over {n_ev_s} events. **Circularity, stated plainly:** because relevance is a function of market
distance, a retriever that uses market distance (market-only) is close to an oracle here; that is expected, and the informative comparisons are
between *text-based* retrievers and against random -- and the extent to which text-derived similarity recovers outcome proximity.
Relevance is a proxy for, not a guarantee of, what a domain expert would judge relevant **[TODO: small expert spot-check, if time permits]**.

## 5 Experiments

### 5.1 Data and setup
Corpus: {n_doc if n_doc else '[TODO]'} documents = 3,052 ECB speeches (ECB "All speeches" export) plus one official Governing Council
press-conference statement per event, fetched from the ECB website and date-validated. Events: EA-MPD monetary-event windows
(Altavilla et al., 2019); of the 315 windows, **{n_ev_s}** have a press-conference statement on the ECB site and are used (the remainder,
mostly 1999--2001 meetings that produced only a press release, are excluded, not approximated). Regime counts: {rc_s}. Market vector
(9 dims): OIS 1M/3M/6M/1Y, German 2Y/5Y/10Y yields, EURO STOXX 50, EUR/USD -- columns without missing values across the used events; the
proposal's "volatility" component has no series in this dataset (EUR/USD stands in). Regimes by real policy dates: pre-ZLB (< June 2014), ZLB
(June 2014 -- June 2022), post-hiking (>= July 2022). Chronological split: train < 2020, validation 2020--2022, test >= 2023; rolling-origin and
leave-one-meeting-out cross-validation are also reported. Confidence intervals: bootstrap (1,000 resamples), 95%; paired bootstrap with
Holm correction for retrieval comparisons. **Encoder: placeholder (see status box).**

### 5.2 Retrieval quality (ECB-Precedent)
**Table 1.** Leave-one-out retrieval over all events.

{table_retrieval()}

**Table 2.** Hybrid retriever: sensitivity to lambda, tau and the distance kernel (nDCG@10). With distances in basis points, the default
tau = 1 makes the market term vanish for nearly all pairs; tau is therefore swept as a multiple of the median pairwise distance.

{table_grid()}

### 5.3 Downstream prediction
**Table 3.** Chronological split (train < 2020). "Reference" = always predict the training mean (R^2 <= 0 by construction on new data).

{table_downstream()}

**Table 4.** Rolling-origin cross-validation (expanding window, yearly test blocks from 2008; pooled out-of-fold predictions), per seed.

{table_cv_rolling()}

**Table 5.** Leave-one-meeting-out (secondary check; trains on all other meetings incl. later ones -- not a forecasting setup).

{table_lomo()}

### 5.4 Faithfulness
**Table 6.** Evidence-perturbation protocol on the test split (rank test: remove the most vs least similar retrieved precedent; magnitude test:
swap the top precedent for the next-ranked / a random one and correlate the size of the swap with the prediction shift).

{table_faith()}

### 5.5 Regime robustness
**Table 7.** Retrieval quality, downstream prediction and precedent density by regime.

{table_regime()}

### 5.6 Ablations and controls
**Table 8.** Multi-seed downstream ablations (mean +/- sd over seeds; held-out = 2020 onward). **The sd reflects random initialisation only -- it is NOT
the sampling uncertainty of the small held-out set, so "fusion > baseline in k/k seeds" is descriptive, not a significance test; compare with the
pooled rolling-origin evaluation in Table 4, which uses many more events.** Neighbour count k, presence/absence of the
uncertainty output, frozen cross-attention (one reading of "frozen vs fine-tuned fusion"; encoder fine-tuning is not possible with the
placeholder encoder), alternative stance definitions, and a **shuffled-label control**. lambda and tau cannot affect downstream results
because prediction-time retrieval is text-only; they are swept in Table 2.

{table_ablation()}

### 5.7 Summary of measured results (auto-generated)
{auto_findings()}

## 6 Limitations
- **Placeholder encoder.** All text-similarity results are placeholder-quality; the paper's central retrieval claims cannot be made until the trained encoder is used. **[TODO]**
- **Stance target is a stand-in** (Sec. 3.4); conclusions may depend on it (Table 8).
- **Small samples.** Held-out sets are a few dozen events; post-hiking has few events at all. CIs are wide and single point estimates are unstable.
- **Coverage.** {n_ev_s} of 315 events have a statement; the 1999--2001 period is underrepresented.
- **Circular benchmark.** Relevance is defined from market outcomes (Sec. 4); expert validation is not yet done.
- **Non-stationarity.** Retrieval from a differently-regimed past is a substantive limitation (Table 7).
- **Unresolved design questions** (kernel; meaning of hybrid retrieval at prediction time) are flagged in Sec. 3.

## 7 Conclusion
**[TODO: write after the trained-encoder run.]**

## Reproducibility
Code, the test suite, and the scripts that generate every table are provided; corpus and event data are referenced to their public sources
(ECB website; EA-MPD) rather than redistributed. Runs are seeded; results JSON files accompany the code.

## References
- Altavilla, C., Brugnolini, L., Gürkaynak, R. S., Motto, R., Ragusa, G. (2019). Measuring euro area monetary policy. *Journal of Monetary Economics* 108, 162--179.
- Gürkaynak, R. S., Sack, B., Swanson, E. (2005). Do actions speak louder than words? The response of asset prices to monetary policy actions and statements. *International Journal of Central Banking* 1(1), 55--93.
- Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 33*.
- Robertson, S., Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in IR* 3(4), 333--389.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65--70.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Kingma, D., Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS 30*.
- Gorodnichenko, Y., Pham, T., Talavera, O. (2023). The voice of monetary policy. *American Economic Review* 113(2). *[verify]*
- Gambacorta, L. et al. (2024). CB-LMs: language models for central banking. BIS Working Paper. *[verify]*
- Sudhi, V. et al. (2024). RAG-Ex: a generic framework for explaining retrieval-augmented generation. *SIGIR*. *[verify]*
- Istrefi, K., Odendahl, F., Sestieri, G. (2024). ECB communication and its impact on financial markets. CEPR Discussion Paper 19242. *(related event-study resource; not used in this draft)*
"""


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "paper.md").write_text(build(), encoding="utf-8")
    print("wrote", OUT / "paper.md")


if __name__ == "__main__":
    main()
