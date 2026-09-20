# HANDOFF LOG — read this first if you're picking the project up in a new session

Project: **Precedent-RAG** (ECIR 2027 submission; full paper due **5 Oct 2026**).
Repo: https://github.com/ujeshauchitha-glitch/ECIR  (data/ is gitignored — see "Data" below)
Ground rules the owner set: don't invent numbers (mark TODO instead); run everything before
claiming it works; flag anything that looks wrong rather than working around it silently;
explain things in plain, non-jargon language.

## Status by phase  (update this table as work lands)

| Phase | What | Status |
|---|---|---|
| 0 | Fix retrieval leak in the prediction pipeline; fix non-deterministic stub encoder | DONE (commit b7d3332) |
| 1 | Data: official ECB press-conference statements fetched + validated (273/315 events, was 228); title-strict BIS fallback | DONE (commit 76cbef4) |
| 2 | Tests (64, all passing) + bugs found by them fixed (p-value=0, Holm thresholds, hyperparameter validation) | DONE |
| 3 | Ablations (lambda/tau/kernel, k, uncertainty, frozen attention, stance defs, controls, multi-seed) | code DONE; full run -> results/ablations_*.json (see logs/) |
| 4 | Rolling-origin + leave-one-meeting-out CV | code DONE; full run -> results/cv_*.json |
| 5 | README rewrite; unrelated "Presample" files moved to ../ECIR_unrelated_presample/ | DONE |
| 6 | Paper draft (numbers auto-filled from results/*.json only) | see paper/ |
| 7 | Zip package | see bottom of this file |

## Things only the owner / teammate can unblock (cannot be done from code)
1. **Real trained encoder** (teammate's, from the concurrent submission). Everything that depends on
   text similarity currently uses `similarity.make_stub_encoder()` — a bag-of-words hash embedding with
   no real semantics. Swap: implement `FrozenEncoder(embed_fn, dim)` around the real checkpoint, replace
   `make_stub_encoder()` calls in run_real.py / train.py / faithfulness.py / regime_robustness.py.
   Then RE-RUN every script; every text-based number so far is placeholder-quality.
2. **Stance definition.** Proposal never defines it. Teammate confirmed (14 Sep 2026) they haven't built
   theirs yet; agreed to compare later. Current stand-in: 1-year OIS change in bp
   (`data.stance_label`, `data.STANCE_COLUMN`). Must be written up as an explicit modeling choice.
3. **m_q at prediction time** — proposal Sec 4 is internally in tension (see similarity.py docstring).
   Current resolution (implemented, leak-free): at prediction time retrieval is TEXT-ONLY (m_q=None);
   the retrieved candidates' known market outcomes still feed the fusion head. Someone who owns Sec 4
   must confirm this is the intended meaning of "hybrid" at deployment.
4. Paper framing decisions (dual-submission discipline: do NOT cross-reference the teammate's paper).

## What was found / fixed on 2026-09-20 (Phase 0) — IMPORTANT
- **Leak (fixed):** train.py / faithfulness.py originally retrieved precedents using the QUERY's own
  market vector, which contains the stance label (1y OIS). CORRECTION OF AN EARLIER CLAIM: on 2026-09-20 I first
  quoted "0.688 leaky vs 0.438 fixed (n=32)". That was ONE seed on the older 228-event data and overstated the
  effect. The proper measurement (leak_control.py, 3 seeds, 273 events, held-out n=47 x 3): leaky pooled directional
  accuracy 0.525 [0.440, 0.610] vs fixed 0.489 [0.411, 0.574] -- overlapping CIs, i.e. small/noisy with the stub
  encoder. The leak is still a methodological error (and could matter more with a real encoder); results from before
  the fix are invalid as claims, but the size of the inflation was small. `build_examples(..., leaky=True)` exists
  ONLY to reproduce the leaky number.
- **Stub encoder was non-deterministic** (used salted builtin `hash()`); now `zlib.crc32`. Earlier
  run-to-run instability in faithfulness results was partly this.
- **regime_robustness.py section 4** (precedent density) drew from all events incl. future ones; now
  restricted to strictly-earlier events (result: ~67% of a post-hiking query's top-5 are same-regime).
- Earlier statement that train.py's val split "is the ZLB era" was only ~83% true (3/18 are already
  post-hiking). Scripts now use per-event date regimes (data: pre_zlb <2014-06, zlb 2014-06..2022-06,
  post_hiking >=2022-07).

## Honest current results (leak-free, stub encoder, n small) — do NOT quote as paper results
- Retrieval augmentation shows NO reliable benefit over the no-retrieval head (CIs include chance).
- Faithfulness (perturbation size vs prediction shift): rho=0.29 [-0.13, 0.69] on test (inconclusive);
  by regime zlb 0.54 [0.19,0.79], post_hiking 0.37 [0.03,0.64].
- Retrieval-QUALITY benchmark (symmetric, uses m_q legitimately as train-time regime): market_only is
  near-ceiling by construction (labels are defined by market distance) — must be framed that way.

## Additional findings on 2026-09-20 (Phases 1-2)
- **Data bug caught before it bit:** first version of fetch_ecb_pressconf.py parsed the ECB index with one greedy
  regex; when an entry lacked a link it borrowed the NEXT entry's page (one date got the previous meeting's
  statement, another a podcast page). Rewritten to parse per-entry, accept only genuine statement URLs, and
  REJECT any page whose printed date (or URL date code) doesn't match the event. One ECB page (2006-06-08) prints
  the wrong date but its URL is right; accepted with a note.
- 273 of 315 EA-MPD dates have a statement on the ECB site. The other 42 (1999-2001 mostly, plus 2002-08-01,
  2003-07-31, 2004-08-05, 2005-08-04, 2008-10-08) have none listed -- consistent with meetings that produced only a
  press release (not independently verified beyond the ECB's index). They are LEFT OUT, never approximated.
  The BIS strict-title fallback rescued 0 of them.
- Old BIS loose fallback had linked 3 events to unrelated speeches (e.g. "The EURO. OUR money"); that path is gone.
- **Kernel discrepancy:** proposal PDF (Sec 4, p.4) shows exp(-||m_q-m_i||_2/tau) (plain L2); the code and the
  owner's brief use the squared distance. `hybrid_similarity(..., squared=True|False)`; default unchanged. UNRESOLVED.
- **tau scale problem:** distances are in basis points (median squared distance ~90), so the historical default
  tau=1 makes the market term ~0 for almost every pair; hybrid at tau=1 is badly handicapped. ablations.py sweeps tau
  as multiples of the median distance.
- Baseline code re-embedded every candidate on every comparison; with ~6,000-word statements run_real.py became
  hours-long. FrozenEncoder.embed now memoizes (value-preserving).
- `results/logs/` has raw logs; `results/*.json` has every number the paper tables are generated from.

## Data (not in git)
`data/` (local only): `all_ECB_speeches (1).csv` (3,052 ECB speeches, pipe-delimited),
`speeches_bis.csv` (BIS all-central-bank speeches, used only to find ECB press-conference texts),
`Dataset_EA-MPD.xlsx` (downloaded from https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx),
`EA-CED Database_IOS.xlsx` (Istrefi/Odendahl/Sestieri 2024; NOT wired in — its SpeechID link is
undocumented; a best-guess join matched only ~30%, so it was not used).
Files in data/ that are US Fed / S&P data (y1.csv, mps*.csv, monetary-policy-surprises-data*.csv,
all_indices_data.csv) were added by mistake and are NOT used anywhere. Ignore them.

## How to run
    pip install -r requirements.txt
    python run_demo.py            # synthetic-data plumbing check (numbers are meaningless)
    python run_real.py            # retrieval quality on real data
    python train.py               # downstream prediction (leak-free)
    python faithfulness.py        # perturbation protocol
    python regime_robustness.py   # regime-stratified analysis
