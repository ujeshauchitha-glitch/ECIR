# Precedent-RAG retrieval pipeline — status

Runnable end-to-end (`python3 run_demo.py`) on **synthetic data**. Every module has a
clean interface so real data drops in without touching the logic.

## What's real vs. stubbed

| Piece | Status | File |
|---|---|---|
| Hybrid similarity formula (Sec 4) | Implemented as specified | `similarity.py` |
| ECB-Precedent benchmark construction (Sec 5) | Implemented as specified | `benchmark.py` |
| BM25 baseline | Implemented from scratch (no `rank_bm25` pkg available, no network here) | `baselines.py` |
| Text-only dense, market-only, random baselines | Implemented | `baselines.py` |
| nDCG@5/10, MAP, bootstrap CI, Holm correction (Sec 6) | Implemented as specified | `metrics.py` |
| ECB speech corpus + market-reaction vectors | **STUBBED — synthetic.** Teammate has the real data. | `data.py::make_synthetic_corpus` |
| Frozen encoder | **STUBBED — deterministic hash embedding, no real semantics.** Teammate's is trained and available. | `similarity.py::make_stub_encoder` |

## To go from synthetic to real (two swaps, nothing else changes)

1. **Corpus/market data**: write a loader that returns `data.Corpus` (list of `Document`,
   list of `Event`) from the real ECB corpus + market-reaction data. Replace
   `make_synthetic_corpus()` calls in `run_demo.py`.
2. **Encoder**: wrap the teammate's trained encoder to match the `FrozenEncoder` interface
   in `similarity.py` (a callable `text -> np.ndarray` of fixed dim). Replace
   `make_stub_encoder()`.

Everything else — benchmark construction, baselines, metrics — runs unchanged.

## Two things to flag with your teammate before this goes in the paper

**1. The `market_only` baseline is a near-ceiling baseline, not a weak one.**
The demo run shows `market_only` scoring almost as high as `hybrid` (0.867 vs 0.840
nDCG@5 on synthetic data). This isn't a bug — ECB-Precedent's relevance labels are
*defined* by market-outcome distance (Sec 5), so a retriever that directly uses market
distance is close to an oracle for that specific benchmark. This is the circularity
concern already named in your risk register (§10, mitigated via the leakage-boundary
claim in Sec 4). The demo makes it concrete: expect `market_only` to look strong, and
frame it that way in the eval section rather than being surprised by it, or a reviewer
will ask why your best baseline beats your proposed method.

**2. `m_q` at genuine prediction time.** The proposal states (Sec 4) that at prediction
time `m_q` is unobserved by construction, but the hybrid similarity formula as written
needs `m_q` to compute the market-similarity term. `hybrid_similarity()` in
`similarity.py` implements an explicit fallback: when `m_q` is `None`, it forces
`lambda=1` (text-only) and reports which mode produced the score, rather than silently
picking a resolution. **This needs a decision from whoever owns Sec 4** — is hybrid
retrieval only ever used at train time (with historical `m_q`), with test-time retrieval
falling back to text-only? Or is there a different mechanism intended? This changes what
"hybrid" means at deployment and how the lambda ablation (Sec 6) should be read.

## Files

- `data.py` — schemas (`Document`, `Event`, `Corpus`) + synthetic data generator
- `similarity.py` — frozen encoder interface, hybrid similarity (Sec 4)
- `benchmark.py` — ECB-Precedent construction (Sec 5)
- `baselines.py` — BM25, dense-text-only, market-only, random retrieval
- `metrics.py` — nDCG@k, MAP, bootstrap CI, Holm-corrected paired significance (Sec 6)
- `run_demo.py` — end-to-end run on synthetic data (validates the pipeline, produces
  no number usable in the paper)
