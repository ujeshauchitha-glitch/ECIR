# Precedent-RAG

Retrieval-augmented, evidence-grounded prediction of ECB monetary-policy communication stance
(ECIR 2027 submission; full paper due **5 Oct 2026**). Each prediction is conditioned on
historically analogous ECB events retrieved by a hybrid of text similarity and realized
market-outcome proximity, and the retrieved precedents double as an auditable explanation.

> **Read `HANDOFF.md` first** — it is the running log of what's done, what's blocked, what was
> found wrong and fixed, and what still needs a human decision.

## Status in one paragraph
All five contributions in the proposal have working, tested code running on **real** data:
the ECB-Precedent benchmark, the hybrid retriever + baselines, the retrieval-augmented prediction
model (cross-attention fusion head), the causal faithfulness protocol, and the regime-stratified
robustness analysis, plus ablations and rolling-origin / leave-one-meeting-out cross-validation.
**No number in this repo is a paper result yet**, for three reasons that only the owner/teammate can
remove: (1) the text encoder is still a **stub** (`similarity.make_stub_encoder`, a hash bag-of-words
with no semantics), so every text-similarity number is placeholder-quality; (2) the **stance
definition** is a stand-in (1-year OIS change), because the proposal never defines one and the
teammate's own definition doesn't exist yet; (3) `m_q`-at-prediction-time and the exact market
kernel (squared vs plain L2) are unresolved design questions in the proposal (see HANDOFF.md).

## Data (not in git — `data/` is ignored; ~450 MB)
| File in `data/` | What / where from |
|---|---|
| `all_ECB_speeches (1).csv` | ECB "All speeches" export (pipe-delimited, 3,052 rows). https://www.ecb.europa.eu/press/key/html/downloads.en.html |
| `Dataset_EA-MPD.xlsx` | Euro Area Monetary Policy Event-Study Database (Altavilla et al., JME 2019); the 315 event windows. https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx |
| `ecb_pressconf/*.txt` | The ECB's own press-conference statement for each event date — **produced by `python fetch_ecb_pressconf.py`** (273 of 315 dates have one; the rest have no statement on the ECB site). |
| `speeches_bis.csv` | BIS all-central-bank speech archive; only used as a strict-title fallback for statement text (currently contributes 0 events). Optional. |
| others (`EA-CED…xlsx`, `y1.csv`, `mps*.csv`, …) | **Not used.** The U.S./Fed files were added by mistake; `EA-CED` is real ECB data but its speech-ID link is undocumented and was not used. |

## Run it
```
pip install -r requirements.txt
python fetch_ecb_pressconf.py     # once; ~10 min, polite 1 req/s, validated + cached under data/ecb_pressconf
python -m pytest tests -q         # 64 tests
python run_demo.py                # synthetic-data plumbing check (numbers meaningless)
python run_real.py                # retrieval quality          -> results/retrieval_quality.json
python train.py                   # downstream prediction      -> results/downstream_train_py.json
python faithfulness.py            # perturbation protocol      -> results/faithfulness.json
python regime_robustness.py       # ZLB / post-hiking analysis -> results/regime_robustness.json
python ablations.py               # lambda/tau/kernel, k, uncertainty, frozen attention, stance defs, controls
python cv.py                      # rolling-origin CV          -> results/cv_rolling.json
python cv.py --lomo --stride 3    # leave-one-meeting-out (every 3rd meeting)
```
Set `OMP_NUM_THREADS=1` and run the long jobs in parallel — the networks are tiny (batch size 1).

## Files
| File | Role |
|---|---|
| `data.py` | Schemas; synthetic generator; real loaders (`load_real_corpus`, `load_market_vectors`), `stance_label` |
| `similarity.py` | Frozen-encoder interface, stub encoder, `hybrid_similarity` (with `squared=` kernel option) |
| `benchmark.py` | ECB-Precedent: graded relevance from market-outcome distance (bottom decile) |
| `baselines.py` | BM25 (from scratch), text-only dense, market-only, random |
| `metrics.py` | nDCG/MAP, directional accuracy, Spearman, R², bootstrap CIs, paired bootstrap, Holm |
| `fusion_head.py` | `FusionHead` (cross-attention over retrieved outcomes → stance + uncertainty) and `NonAugmentedHead` baseline |
| `train.py` | Leak-free training/eval, chronological split (train<2020, val 2020–22, test ≥2023) |
| `faithfulness.py` | Evidence-perturbation protocol + built-in validity check |
| `regime_robustness.py` | pre-ZLB / ZLB / post-hiking stratification |
| `pipeline.py`, `ablations.py`, `cv.py` | Shared experiment plumbing; ablations; cross-validation |
| `fetch_ecb_pressconf.py` | Fetches/validates the ECB's official press-conference statements |
| `tests/` | 64 tests: hand-computed metric answers, leakage-boundary tests, model-structure tests, data-link checks |
| `results/` | JSON outputs of every experiment (+ `logs/`) |
| `paper/` | Draft manuscript; tables are generated from `results/*.json` by `make_paper.py` |

## Two design rules worth knowing before touching anything
1. **Leakage boundary.** At prediction time the query's own market outcome doesn't exist, and the stance
   label is computed from it. So prediction-time retrieval is **text-only** (`m_q=None`) and only
   **strictly earlier** events are retrievable. An earlier version retrieved with `m_q` and looked
   ~25 points better in directional accuracy; that was the leak. `build_examples(leaky=True)` exists only
   to reproduce it. Tests in `tests/test_leakage_and_model.py` pin this.
2. **The retrieval-quality benchmark is circular by construction** — relevance is defined by market-outcome
   distance, so `market_only` is near-ceiling. That is expected, not a bug; the write-up must say so.
