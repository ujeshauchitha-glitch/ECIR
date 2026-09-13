"""
Retrieval evaluation metrics, per proposal Sec 6:
  - nDCG@{5,10} and MAP on ECB-Precedent
  - Significance via paired bootstrap resampling over query events
    (1,000 resamples), 95% CIs, Holm-corrected pairwise significance
  - Downstream prediction quality: directional accuracy, Spearman's rho,
    R^2, each with bootstrap confidence intervals
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def dcg_at_k(relevances: list[float], k: int) -> float:
    r = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(r))


def ndcg_at_k(ranked_relevances: list[float], k: int) -> float:
    """ranked_relevances: relevance of each retrieved item, in ranked order."""
    ideal = sorted(ranked_relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / idcg


def average_precision(ranked_relevances: list[float], relevance_threshold: float = 0.0) -> float:
    """Binary AP: an item counts as relevant if its graded relevance exceeds
    the threshold. Standard mean-average-precision definition.
    """
    n_relevant = sum(1 for r in ranked_relevances if r > relevance_threshold)
    if n_relevant == 0:
        return 0.0
    hits = 0
    precisions = []
    for i, rel in enumerate(ranked_relevances):
        if rel > relevance_threshold:
            hits += 1
            precisions.append(hits / (i + 1))
    return sum(precisions) / n_relevant


def per_query_scores(
    ranked_relevances_by_query: list[list[float]],
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, list[float]]:
    """Returns per-query metric values (one entry per query event), needed
    for paired bootstrap resampling over queries.
    """
    out: dict[str, list[float]] = {f"ndcg@{k}": [] for k in k_values}
    out["map"] = []
    for rels in ranked_relevances_by_query:
        for k in k_values:
            out[f"ndcg@{k}"].append(ndcg_at_k(rels, k))
        out["map"].append(average_precision(rels))
    return out


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) via paired bootstrap over queries."""
    rng = np.random.default_rng(seed)
    values_arr = np.asarray(values)
    n = len(values_arr)
    means = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[b] = values_arr[idx].mean()
    alpha = 1 - ci
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values_arr.mean()), float(lower), float(upper)


def paired_bootstrap_pvalue(
    values_a: list[float], values_b: list[float], n_resamples: int = 1000, seed: int = 0
) -> float:
    """Two-sided paired bootstrap test: H0 is that system A and system B have
    equal mean per-query score. Returns an approximate p-value.
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    assert len(a) == len(b), "paired test requires equal-length, aligned per-query arrays"
    diff = a - b
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    n = len(diff)
    # Center the diffs under H0, then resample to build the null distribution.
    centered = diff - observed
    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = centered[idx].mean()
    p = float(np.mean(np.abs(resampled_means) >= np.abs(observed)))
    return p


def directional_accuracy(y_true: list[float], y_pred: list[float]) -> float:
    """Fraction of examples where the predicted and actual stance have the
    same sign (both hawkish, both dovish). A sign of exactly 0 counts as
    matching only if the other side is also exactly 0.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def r_squared(y_true: list[float], y_pred: list[float]) -> float:
    """Coefficient of determination, 1 - SS_res/SS_tot. Can be negative
    (worse than predicting the mean) -- do not clip that away.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def spearman_rho(y_true: list[float], y_pred: list[float]) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def bootstrap_metric_ci(
    y_true: list[float],
    y_pred: list[float],
    metric_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap over (y_true, y_pred) PAIRS (not queries --
    for metrics computed on one flat set of predictions rather than
    per-query lists). Returns (point_estimate, lower, upper).
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    point = metric_fn(yt, yp)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[b] = metric_fn(yt[idx], yp[idx])
    alpha = 1 - ci
    lower, upper = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lower), float(upper)


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni step-down correction. Returns {name: (adjusted_alpha_threshold, significant)}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    result: dict[str, tuple[float, bool]] = {}
    for rank, (name, p) in enumerate(items):
        threshold = alpha / (m - rank)
        significant = p < threshold
        result[name] = (threshold, significant)
        if not significant:
            # Holm's step-down: once one comparison fails, all subsequent
            # (larger p-value) comparisons are also declared non-significant.
            for name2, p2 in items[rank + 1:]:
                result[name2] = (alpha / (m - (rank + 1)), False)
            break
    return result
