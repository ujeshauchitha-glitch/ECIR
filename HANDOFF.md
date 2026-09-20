# HANDOFF LOG — read this first if you're picking the project up in a new session

Project: **Precedent-RAG** (ECIR 2027 submission; full paper due **5 Oct 2026**).
Repo: https://github.com/ujeshauchitha-glitch/ECIR  (data/ is gitignored — see "Data" below)
Ground rules the owner set: don't invent numbers (mark TODO instead); run everything before
claiming it works; flag anything that looks wrong rather than working around it silently;
explain things in plain, non-jargon language.

## Status by phase  (update this table as work lands)

| Phase | What | Status |
|---|---|---|
| 0 | Fix retrieval leak in the prediction pipeline; fix non-deterministic stub encoder | DONE |
| 1 | Data: fetch missing press-conference texts from ECB site; audit existing links | TODO |
| 2 | Tests / stress tests (pytest) | TODO |
| 3 | Ablations (lambda, k, uncertainty on/off, stance-definition sensitivity, multi-seed) | TODO |
| 4 | Rolling-origin + leave-one-meeting-out CV | TODO |
| 5 | Cleanup: README rewrite, move unrelated "Presample" files out of the repo | TODO |
| 6 | Paper draft (numbers auto-filled from results/*.json only) | TODO |
| 7 | Zip package | TODO |

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
  market vector, which contains the stance label (1y OIS). Quantified: FusionHead held-out directional
  accuracy 0.688 [0.531,0.844] with the leak vs 0.438 [0.250,0.595] without (n=32). All earlier
  "retrieval helps" claims from before this fix are INVALID. `build_examples(..., leaky=True)` still
  exists ONLY to reproduce the leaky number for comparison.
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
