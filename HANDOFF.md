# HANDOFF LOG — read this first if you're picking the project up in a new session

Project: **Precedent-RAG** (ECIR 2027 submission; full paper due **5 Oct 2026**).
Repo: https://github.com/ujeshauchitha-glitch/ECIR  (`data/` is gitignored — see "Data").
Owner's ground rules: don't invent numbers (mark TODO instead); run everything before claiming it works;
flag anything that looks wrong rather than working around it silently; explain things in plain language.
Last updated: 2026-09-20 (all phases below complete; state is consistent with `results/*.json`).

## Where things stand (one paragraph)
All five contributions in the proposal have working, tested code on real data, plus ablations, controls and two
cross-validation schemes; `python make_paper.py` builds a draft manuscript whose every number comes from
`results/*.json`. **The pipeline is trustworthy; the science is not finished**, because (1) the text encoder is a
placeholder, (2) the stance target is a stand-in, (3) two design questions in the proposal are unresolved. With the
placeholder encoder there is **no out-of-sample prediction skill** (see results). Nothing here is a paper result until
the real encoder is wired in and everything is re-run.

## Phase status
| Phase | What | Status |
|---|---|---|
| 0 | Fix the retrieval leak; make stub encoder deterministic | DONE |
| 1 | Data: 273/315 events now use the ECB's own press-conference statements (was 228 via BIS matching, 3 wrong links) | DONE |
| 2 | Tests: 86 passing (hand-computed metrics, leakage boundary, model structure, data links, results contract) | DONE |
| 3 | Ablations + controls (lambda/tau/kernel, k, uncertainty, frozen attention, stance defs, raw inputs, shuffled labels) | DONE -> results/ablations_*.json |
| 4 | Rolling-origin + leave-one-meeting-out CV | DONE -> results/cv_*.json |
| 5 | README rewritten; unrelated "Presample" files moved to `../ECIR_unrelated_presample/` | DONE |
| 6 | Paper draft `paper/paper.md` (generated) | DONE as a draft; owner TODOs remain (see below) |
| 7 | Zip package | DONE (see bottom) |

## Only the owner / teammate can unblock these
1. **Real trained encoder** (from the companion work). Every text-similarity number is placeholder-quality.
   Swap: `export PRECEDENT_ENCODER=<checkpoint>` (HF adapter in `similarity.make_hf_encoder`, **untested** — no checkpoint
   or `transformers` was available) or wrap it as `FrozenEncoder(embed_fn, dim)`. Then re-run in this order and rebuild:
   `run_real.py, train.py, faithfulness.py, faithfulness_seeds.py [--shuffle], regime_robustness.py, ablations.py,
   cv.py, cv.py --lomo --stride 3, leak_control.py`, then `python make_paper.py`. (Set `OMP_NUM_THREADS=1`; run in parallel.)
2. **Stance definition.** The proposal has none. Stand-in = 1-year OIS change in bp (`data.stance_label`). Teammate (14 Sep)
   hadn't built theirs; "we can compare later". Sensitivity to 2 alternatives is in ablations. Write it up as a modelling choice.
3. **What "hybrid" means at prediction time** (m_q unobserved). Implemented, leak-free: prediction-time retrieval is text-only.
   Confirm with whoever owns Sec 4.
4. **Kernel**: proposal PDF (p.4) shows exp(-||m_q-m_i||_2/tau) (plain L2); code default is the squared distance.
   `hybrid_similarity(..., squared=False)` gives the PDF reading. Table 2 in the paper shows both.
5. Paper: abstract headline, conclusion, related-work verification (*[verify]* items were written from memory), the
   ECB-Precedent grading function (`max(0, 1-d/threshold)` is an implementation choice), LNCS formatting/anonymisation.
   Dual-submission discipline: do NOT cross-reference the teammate's paper.

## Final measured results (placeholder encoder, 273 events) — NOT paper results
- **Retrieval (ECB-Precedent nDCG@10):** random 0.046, BM25 0.081, text-only dense (stub) 0.071, market-only 0.784, hybrid
  (lambda .5, tau 1, squared) 0.477. Market-only is near-ceiling **by construction** (relevance is defined by market distance);
  say so. With tau scaled to the data (0.25x median) and lambda .25 the hybrid reaches 0.782 ~ market-only: text never adds to
  market-only with the stub. tau=1 (old default) was badly mis-scaled (distances are in bp).
- **Downstream: no skill.** Rolling-origin CV (2008-2025, pooled n=171): R^2 fusion -0.04 / baseline -0.005 / training-mean
  -0.006; directional accuracy 0.45-0.49 with chance inside every CI. LOMO (n=91): fusion R^2 -0.09 vs baseline -0.03. The
  single 2020+ split shows small positive fusion R^2 (0.01-0.05) and fusion>baseline in dir. acc. in 5/5 seeds, but that does
  NOT survive the much larger rolling-origin evaluation and the seed sd is initialisation-only, not sampling uncertainty.
  Shuffled-label control behaves as it should (no skill). Raw (unstandardised) inputs are worse than standardised.
- **Faithfulness:** rho(swap size, |prediction shift|) averages 0.57 (sd 0.19) over 5 initialisations, CI>0 in 4/5; the
  no-skill shuffled-label control gives 0.13 (sd 0.10), CI>0 in 1/5 -> the metric does separate learned reliance from mere
  sensitivity. The rank test (remove most vs least similar precedent) is never significant (0/5). Caveat: models are
  barely trained (R^2~0), so this is sensitivity of an under-trained network, not evidence of useful reasoning.
- **Regimes:** retrieval is hardest in the post-hiking regime (market-only nDCG@10 0.593 vs 0.782 pre-ZLB / 0.866 ZLB; hybrid
  0.214 vs 0.444 / 0.674); 59% of a post-hiking query's top-5 precedents are themselves post-hiking (rest borrowed from
  earlier regimes). Only 27 post-hiking events exist.
- **Leak:** leaky 0.518 vs fixed 0.553 directional accuracy (3 seeds, held-out n=47) — no measurable inflation with this encoder;
  still an invalid setup. (An earlier single-seed figure of "0.688 vs 0.438" quoted in conversation was WRONG/overstated.)

## Things found wrong and fixed today (so you don't rediscover them)
- **Retrieval leak** (query's own market vector, which contains the label, used to pick precedents) — fixed; tests pin it.
- **Non-reproducible training**: weights were drawn before `torch.manual_seed`; now seeded first everywhere. Earlier
  "faithfulness supported" single-run results were partly initialisation luck.
- **No input standardisation / no early stopping**: raw bp vectors + 150 unregularised epochs on ~250 noisy examples gave
  held-out R^2 of -9 to -33. Now: standardise retrieved outcomes with training-only stats; early stopping on the latest 20%
  of the training window (patience 20). Held-out data is never touched (tests).
- **Data linking**: first ECB fetcher could link an event to the NEXT entry's page (or a podcast page); now per-entry parsing,
  statement-URL whitelist, printed-date/URL-date validation. Old BIS loose fallback had 3 wrong links; now title-strict.
  The 42 dates with no statement (mostly 1999-2001 second-monthly meetings; also 2002-08-01, 2003-07-31, 2004-08-05,
  2005-08-04, 2008-10-08) are left OUT (not verified beyond the ECB index; BIS rescued none).
- **Metrics**: bootstrap p-value could be exactly 0 (now (k+1)/(n+1)); Holm reported wrong thresholds after the first failure.
- **Efficiency**: baselines re-embedded every document per comparison + BM25 looped over ~6,000 query tokens -> run_real.py went
  from hours to ~1 minute (memoised encoder; unique-term BM25, proven equal by test).
- **Silent patch failures**: several scripted edits (JSON saving, LOMO table) silently did not apply. `tests/test_results_contract.py`
  now checks every results file the paper reads is written by a script. If you patch files via shell heredocs, verify they applied.
- Earlier claim that the val split "is the ZLB era" was only ~83% true; scripts use per-event date regimes.

## Data (not in git)
`data/`: `all_ECB_speeches (1).csv` (3,052 ECB speeches, `|`-delimited), `Dataset_EA-MPD.xlsx`
(https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx), `ecb_pressconf/*.txt` (273 files from
`python fetch_ecb_pressconf.py`, ~10 min), `speeches_bis.csv` (optional BIS archive, fallback only, contributes 0 events),
`bis_pressconf_cache.json`. `EA-CED Database_IOS.xlsx` is real ECB data (Istrefi/Odendahl/Sestieri 2024) but its speech-ID link is
undocumented (a guessed join matched ~30%) so it is NOT used. `y1.csv, mps*.csv, monetary-policy-surprises-data*.csv,
all_indices_data.csv` are U.S./Fed data added by mistake — unused, ignore.

## Repo map / how to run: see README.md
Key modules: `data.py similarity.py benchmark.py baselines.py metrics.py fusion_head.py train.py faithfulness.py
faithfulness_seeds.py regime_robustness.py pipeline.py ablations.py cv.py leak_control.py fetch_ecb_pressconf.py make_paper.py`.
`results/` (JSON + `logs/`) is what the paper is generated from; `results/archive_before_input_standardization/` holds superseded outputs.
