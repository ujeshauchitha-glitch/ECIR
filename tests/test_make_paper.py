import tempfile
from pathlib import Path

import make_paper as mp


def test_paper_builds_without_results_and_marks_todos_instead_of_inventing(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(mp, "RES", Path(d))                  # empty results dir
        text = mp.build()
    assert "TODO: results file missing" in text
    assert "0.688" not in text and "0.438" not in text           # no hand-typed numbers left behind
    for marker in ("{table_", "{leak_", "{auto_", "{rc_s}", "{n_ev_s}", "{n_doc"):
        assert marker not in text                                # no unformatted f-string placeholders


def test_ci_formatter_handles_both_result_shapes():
    assert mp.ci([0.5, 0.25, 0.75]) == "0.500 [0.250, 0.750]"
    assert mp.ci({"point": 0.5, "ci95": [0.25, 0.75]}) == "0.500 [0.250, 0.750]"
    assert mp.ci(None) == "n/a" and mp.ci({"n": 3, "note": "x"}) == "n/a"


def test_includes_helper():
    assert mp.includes([0.5, 0.4, 0.6], 0.5) and not mp.includes([0.5, 0.4, 0.6], 0.7)
    assert mp.includes({"point": 0.1, "ci95": [-0.2, 0.3]}, 0)
