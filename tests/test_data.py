import datetime as dt
from pathlib import Path

import numpy as np
import pytest

import data as D

DATA = Path(__file__).resolve().parent.parent / "data"
ECB = DATA / "all_ECB_speeches (1).csv"
XLSX = DATA / "Dataset_EA-MPD.xlsx"
PC = DATA / "ecb_pressconf"
needs_data = pytest.mark.skipif(not (ECB.exists() and XLSX.exists()), reason="real data files not present")


def test_synthetic_corpus_is_internally_consistent():
    c = D.make_synthetic_corpus(n_docs=50, n_events=30, market_dim=4, seed=1)
    assert len(c.documents) == 50 and len(c.events) == 30
    ids = {d.doc_id for d in c.documents}
    assert all(e.doc_id in ids for e in c.events) and len({e.event_id for e in c.events}) == 30
    assert all(e.market_vector.shape == (4,) for e in c.events)


def test_ea_mpd_date_parser_handles_both_formats_found_in_the_source_file():
    assert D._parse_eamPd_date(dt.datetime(2019, 3, 7)) == dt.date(2019, 3, 7)
    assert D._parse_eamPd_date("14/12/2023") == dt.date(2023, 12, 14)          # string rows from Dec 2023 on
    with pytest.raises(ValueError):
        D._parse_eamPd_date("2023-12-14")                                       # unexpected format must fail loudly


@needs_data
def test_market_vectors_are_complete_finite_and_315():
    mv = D.load_market_vectors(XLSX)
    assert len(mv) == 315                                                       # proposal's |E|
    assert all(v.shape == (len(D.MARKET_VECTOR_COLUMNS),) and np.isfinite(v).all() for v in mv.values())
    assert dt.date(2023, 12, 14) in mv and dt.date(1999, 1, 7) in mv            # both date formats survived


@needs_data
def test_stance_label_matches_raw_spreadsheet_value():
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, data_only=True)["Monetary Event Window"]
    rows = list(ws.iter_rows(values_only=True))
    hdr, body = rows[0], rows[1:]
    row = next(r for r in body if r[0] is not None and D._parse_eamPd_date(r[0]) == dt.date(2022, 7, 21))
    raw = float(row[hdr.index("OIS_1Y")])
    mv = D.load_market_vectors(XLSX)[dt.date(2022, 7, 21)]
    assert D.stance_label(mv) == pytest.approx(raw)


@needs_data
def test_ecb_speech_corpus_has_expected_size():
    c = D.load_ecb_corpus(ECB)
    assert len(c.documents) == 3052 and all(isinstance(d.date, dt.date) for d in c.documents)


@needs_data
@pytest.mark.skipif(not PC.exists(), reason="press-conference cache not fetched")
def test_real_corpus_links_are_valid_and_dates_agree():
    from fetch_ecb_pressconf import printed_date
    c = D.load_real_corpus(ECB, None, XLSX)          # ECB-website source only (no slow BIS scan)
    assert len(c.events) > 250
    assert len({e.event_id for e in c.events}) == len(c.events) == len({e.date for e in c.events})
    mv = D.load_market_vectors(XLSX)
    for e in c.events:
        doc = c.doc_by_id(e.doc_id)                                             # link resolves
        assert doc.date == e.date and e.date in mv and np.array_equal(e.market_vector, mv[e.date])
        assert len(doc.text.split()) > 300
        assert c.provenance[e.doc_id] == "ecb_website"
        # printed date OR the date code in the source URL must match (the ECB's own 2006-06-08 page
        # prints "6 June 2006" but its URL is is060608 -- a typo on their side, accepted by the fetcher).
        src = (PC / f"{e.date.isoformat()}.txt").read_text(encoding="utf-8").split("\n", 1)[0]
        url_code = src.split("is")[-1][:6] if "is" in src else ""
        assert printed_date(doc.text) == e.date or url_code == e.date.strftime("%y%m%d"), \
            f"{e.date}: neither the printed date nor the source URL matches"


@needs_data
def test_events_without_a_statement_are_left_out_not_faked():
    c = D.load_real_corpus(ECB, None, XLSX, pressconf_dir=DATA / "definitely_missing_dir")
    assert c.events == []                                                       # no source -> no events, no invention


def test_strict_bis_title_rule_rejects_unrelated_speeches():
    for bad in ("The EURO. OUR money", "speech at the 11th Frankfurt International Banking Evening"):
        assert not any(m in bad.lower() for m in D._PC_TITLE_MARKERS)
    for good in ("Mr Duisenberg reports on the outcome of the latest meeting of the ECB Governing Council",
                 "European Central Bank: Press briefing - transcript",
                 "European Central Bank: Press conference - introductory statement"):
        assert any(m in good.lower() for m in D._PC_TITLE_MARKERS)
