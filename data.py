"""
Data schemas for Precedent-RAG.

Two corpora, per the proposal's formal definition (Sec 4):
  - D: full ECB communication corpus, |D| = N documents
  - E subset of D: the subset linked to discrete monetary-policy events,
    each with a realized market-reaction vector m_i in R^h

REAL DATA NOT YET AVAILABLE: the teammate has the actual ECB speech corpus
and market-reaction vectors from the concurrent submission. Until that's
handed over, `make_synthetic_corpus()` below generates structurally-identical
fake data so the rest of the pipeline (retriever, benchmark, eval harness)
can be built and tested end-to-end now.

TO SWITCH TO REAL DATA: replace calls to `make_synthetic_corpus()` with a
loader that returns the same two dataclasses (Document, Event) populated
from the real corpus. Nothing downstream needs to change.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import sys
from pathlib import Path
from typing import Optional

import numpy as np


@dataclasses.dataclass
class Document:
    doc_id: str
    date: dt.date
    text: str
    # Set only for documents linked to a monetary-policy event (subset E).
    event_id: Optional[str] = None


@dataclasses.dataclass
class Event:
    event_id: str
    doc_id: str
    date: dt.date
    # Realized market-reaction vector m_i in R^h: yield changes, EA-MPD
    # surprise factors, volatility, equity reaction, short-rate expectations.
    market_vector: np.ndarray


@dataclasses.dataclass
class Corpus:
    documents: list[Document]
    events: list[Event]  # subset linked to E, |E| should match len(events)

    def doc_by_id(self, doc_id: str) -> Document:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        raise KeyError(doc_id)

    def event_by_doc_id(self, doc_id: str) -> Optional[Event]:
        for e in self.events:
            if e.doc_id == doc_id:
                return e
        return None


_HAWKISH_WORDS = [
    "inflation", "tighten", "restrictive", "vigilant", "upside risks",
    "elevated", "persistent", "firm", "withdraw accommodation",
]
_DOVISH_WORDS = [
    "accommodative", "support", "downside risks", "patient", "gradual",
    "easing", "subdued", "fragile recovery", "ample",
]


def make_synthetic_corpus(
    n_docs: int = 400,
    n_events: int = 315,
    market_dim: int = 5,
    seed: int = 0,
) -> Corpus:
    """Structurally matches the proposal's stated scale (N=3052 docs total in
    the real corpus, |E|=315 events) but is SYNTHETIC — every word and number
    here is generated, not observed. Use only to validate the pipeline runs
    correctly, never to support a claim in the paper.
    """
    rng = np.random.default_rng(seed)
    start = dt.date(1999, 1, 1)
    span_days = (dt.date(2025, 1, 1) - start).days

    documents: list[Document] = []
    events: list[Event] = []

    event_doc_indices = set(
        rng.choice(n_docs, size=min(n_events, n_docs), replace=False)
    )

    for i in range(n_docs):
        date = start + dt.timedelta(days=int(rng.integers(0, span_days)))
        # Latent "true stance" drives both the text's word mix and the
        # market vector, so retrieval on text has genuine (synthetic)
        # signal rather than being pure noise.
        stance = rng.normal(0, 1)
        n_words = rng.integers(40, 90)
        hawkish_p = 1 / (1 + np.exp(-stance))
        words = rng.choice(
            _HAWKISH_WORDS, size=int(n_words * hawkish_p)
        ).tolist() + rng.choice(
            _DOVISH_WORDS, size=int(n_words * (1 - hawkish_p))
        ).tolist()
        rng.shuffle(words)
        text = " ".join(words)

        doc_id = f"doc_{i:04d}"
        event_id = f"evt_{i:04d}" if i in event_doc_indices else None
        documents.append(Document(doc_id=doc_id, date=date, text=text, event_id=event_id))

        if event_id is not None:
            market_vector = stance * rng.normal(1.0, 0.3, size=market_dim) \
                + rng.normal(0, 0.4, size=market_dim)
            events.append(
                Event(
                    event_id=event_id,
                    doc_id=doc_id,
                    date=date,
                    market_vector=market_vector,
                )
            )

    return Corpus(documents=documents, events=events)


def load_ecb_corpus(csv_path: str | Path) -> Corpus:
    """Loads the REAL ECB speech corpus (D) from the "All ECB speeches" CSV
    (pipe-delimited: date|speakers|title|subtitle|contents -- this is the
    ECB's own published speeches export, confirmed by row count against the
    proposal's stated N=3052).

    STILL STUBBED: this returns `events=[]`. There is no market-reaction
    data wired in yet -- the E subset (Event objects, each needing a
    market_vector in R^h) cannot be built until that data is available and
    a corresponding loader is written. Everything downstream that needs
    Events (benchmark.py::build_ecb_precedent, hybrid_similarity's market
    term, the market_only/hybrid baselines) stays blocked on that, even
    with this function wired in. Text-only baselines (bm25, dense_text_only)
    CAN run on the Documents this returns.

    Data-quality note, not silently resolved: some rows have an empty
    `contents` field (slide-deck-only entries with no prose body -- 164 of
    3052 as of the file handed off on 2026-09-13). They are kept as
    Documents with empty text, rather than dropped, so `len(documents)`
    matches the corpus's real stated scale; callers that need non-empty
    text should filter explicitly and should decide that themselves rather
    than have it decided here.
    """
    csv.field_size_limit(sys.maxsize)
    documents: list[Document] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        missing = {"date", "speakers", "title", "subtitle", "contents"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(
                f"{csv_path}: expected columns date|speakers|title|subtitle|contents, "
                f"missing {missing}. Got {reader.fieldnames}."
            )
        for i, row in enumerate(reader):
            date = dt.date.fromisoformat(row["date"].strip())
            title = (row.get("title") or "").strip()
            subtitle = (row.get("subtitle") or "").strip()
            contents = (row.get("contents") or "").strip()
            # title/subtitle carry signal even when contents is empty
            # (slides-only entries); include them so no row is pure noise.
            text = " ".join(p for p in (title, subtitle, contents) if p)
            documents.append(
                Document(doc_id=f"ecb_{i:04d}", date=date, text=text, event_id=None)
            )
    return Corpus(documents=documents, events=[])


# Governing Council press-conference statements are how each of the 315
# EA-MPD monetary-policy events is actually communicated, but the "All ECB
# speeches" export (load_ecb_corpus above) is missing almost all of them --
# confirmed by date-matching: only 33/315 EA-MPD dates have a same-day row
# in that file. They exist instead in the BIS "Central bankers' speeches"
# archive (speeches_bis.csv), which covers many central banks, not just the
# ECB -- filtered below to rows authored by an ECB President on a date that
# is one of the 315 EA-MPD events.
_ECB_PRESIDENTS = ("duisenberg", "trichet", "draghi", "lagarde")

# Market-vector columns, chosen from Dataset_EA-MPD.xlsx's "Monetary Event
# Window" sheet. DESIGN CHOICE, not dictated by the proposal text: many
# columns (longer OIS/sovereign-yield maturities) are NaN for older events
# because those instruments didn't exist yet, so a column is only used here
# if it has zero missing values across the events we can actually match to
# a document (228 of 315 -- see load_real_corpus). This set covers the
# proposal's four named categories approximately, not exactly: there is no
# explicit "volatility" series in this file (e.g. no VSTOXX), so EURUSD
# (FX reaction) stands in for it here. Revisit with whoever owns Sec 4.
MARKET_VECTOR_COLUMNS = (
    "OIS_1M", "OIS_3M", "OIS_6M", "OIS_1Y",  # short-rate expectations
    "DE2Y", "DE5Y", "DE10Y",  # yield changes (German Bund curve)
    "STOXX50",  # equity reaction
    "EURUSD",  # FX reaction (stand-in; no volatility series available)
)

# ---------------------------------------------------------------------------
# STANCE LABEL -- NOT specified anywhere in the proposal. This is a pragmatic
# choice made on 2026-09-13 to unblock training fusion_head.py, after
# confirming with the user that neither this project nor the concurrent
# encoder-training submission has a stance definition yet ("I didnt / I am
# working on continuous stance predictions / that part is yet to implement
# / for me as well" -- teammate, same day). It is NOT an authoritative
# construct and must be written up in the paper as an explicit modeling
# choice ("we define stance as ..., because ..."), not presented as if it
# were always the plan.
#
# Definition: stance = the realized change in the 1-year OIS rate (basis
# points) in the event window. Positive = rates expected higher = hawkish
# surprise; negative = dovish surprise. This is a standard, citable move in
# the monetary-surprise literature (a single short/medium-maturity rate as
# the summary "policy shock", in the spirit of Gurkaynak-Sack-Swanson 2005's
# "target factor") -- chosen over averaging multiple OIS maturities for
# interpretability (one real, named instrument) and because OIS_1Y has zero
# missing values across all 228 matched events (see MARKET_VECTOR_COLUMNS
# comment above re: missingness in longer maturities).
# REVISIT if/when the teammate's own definition becomes available -- swap
# this out and re-run rather than maintaining two divergent definitions.
STANCE_COLUMN = "OIS_1Y"


def stance_label(market_vector: np.ndarray) -> float:
    """Extracts the stance label (see STANCE_COLUMN note above) from a
    market_vector built with MARKET_VECTOR_COLUMNS. Basis points, signed.
    """
    return float(market_vector[MARKET_VECTOR_COLUMNS.index(STANCE_COLUMN)])


def _parse_eamPd_date(value) -> dt.date:
    """Dataset_EA-MPD.xlsx mixes real Excel datetimes (older rows) with
    DD/MM/YYYY strings (rows from Dec 2023 onward) in the same date column
    -- a real inconsistency in the source file, not a parsing bug here.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.datetime.strptime(str(value).strip(), "%d/%m/%Y").date()


def load_market_vectors(xlsx_path: str | Path) -> dict[dt.date, np.ndarray]:
    """Loads realized market-reaction vectors from the real EA-MPD dataset
    (Euro Area Monetary Policy Event-Study Database, Altavilla, Brugnolini,
    Gurkaynak, Motto & Ragusa 2019 -- downloaded from the ECB's own site,
    https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx).

    Sheet "Monetary Event Window" has exactly 315 rows, matching the
    proposal's stated |E|=315 exactly. Returns {event_date: vector in R^9}
    using MARKET_VECTOR_COLUMNS. Requires openpyxl.
    """
    import openpyxl  # local import: only needed for this loader

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Monetary Event Window"]
    rows = list(ws.iter_rows(values_only=True))
    header, data_rows = rows[0], rows[1:]
    col_idx = {name: i for i, name in enumerate(header)}
    missing = set(MARKET_VECTOR_COLUMNS) - set(col_idx)
    if missing:
        raise ValueError(f"{xlsx_path}: missing expected columns {missing}")

    out: dict[dt.date, np.ndarray] = {}
    for row in data_rows:
        if row[col_idx["date"]] is None:
            continue
        date = _parse_eamPd_date(row[col_idx["date"]])
        vec = np.array([row[col_idx[c]] for c in MARKET_VECTOR_COLUMNS], dtype=float)
        if np.isnan(vec).any():
            # Shouldn't happen given MARKET_VECTOR_COLUMNS was chosen to be
            # complete, but never silently substitute a guess for real NaNs.
            continue
        out[date] = vec
    return out


def load_real_corpus(
    ecb_csv_path: str | Path,
    bis_csv_path: str | Path,
    market_xlsx_path: str | Path,
) -> Corpus:
    """Assembles the REAL corpus: real ECB speeches (D), real press-conference
    statements pulled from the BIS archive to fill the gap load_ecb_corpus
    can't (see _ECB_PRESIDENTS note above), and real market-reaction vectors
    from EA-MPD (see load_market_vectors).

    Only 228 of the 315 EA-MPD events have a press-conference statement
    findable in either provided file (checked +/- 3 days around each of the
    other 87 dates too -- genuinely not there, not a matching bug). Those
    228 are real, not approximated. The other 87 are simply absent from
    this Corpus's `events` list -- NOT filled with a nearby unrelated
    speech, which would silently corrupt the market-outcome labels.
    `len(corpus.events)` is the honest number to report, not 315.
    """
    csv.field_size_limit(sys.maxsize)

    corpus = load_ecb_corpus(ecb_csv_path)
    documents = list(corpus.documents)
    market_vectors = load_market_vectors(market_xlsx_path)
    event_dates = set(market_vectors)

    candidates_by_date: dict[dt.date, list[dict]] = {}
    with open(bis_csv_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_date = (row.get("date") or "").strip()
            if not raw_date:
                continue
            try:
                date = dt.datetime.fromisoformat(raw_date).date()
            except ValueError:
                continue
            if date not in event_dates:
                continue
            author = (row.get("author") or "").lower()
            if any(p in author for p in _ECB_PRESIDENTS):
                candidates_by_date.setdefault(date, []).append(row)

    events: list[Event] = []
    for i, date in enumerate(sorted(event_dates)):
        cands = candidates_by_date.get(date)
        if not cands:
            continue
        # Prefer the row actually titled as the press conference statement
        # over another speech the same president happened to give that day.
        preferred = [
            c for c in cands
            if "press conference" in (c.get("title") or "").lower()
            or "introductory statement" in (c.get("title") or "").lower()
        ]
        row = preferred[0] if preferred else cands[0]

        doc_id = f"bis_pc_{i:04d}"
        text = " ".join(
            p for p in (row.get("title", ""), row.get("description", ""), row.get("text", ""))
            if p
        ).strip()
        documents.append(Document(doc_id=doc_id, date=date, text=text, event_id=f"evt_{i:04d}"))
        events.append(
            Event(
                event_id=f"evt_{i:04d}",
                doc_id=doc_id,
                date=date,
                market_vector=market_vectors[date],
            )
        )

    return Corpus(documents=documents, events=events)
