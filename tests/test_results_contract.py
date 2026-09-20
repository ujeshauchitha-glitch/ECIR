"""Contract between the experiment scripts and the paper generator.

Regression test for a real failure: scripted edits that were meant to make train.py and
regime_robustness.py write their results silently did not apply, so the paper generator had nothing to
read for Table 3 and the per-regime faithfulness rows. Every results file the generator loads must be
written by some script."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["run_real.py", "train.py", "faithfulness.py", "regime_robustness.py", "ablations.py", "cv.py",
           "leak_control.py", "faithfulness_seeds.py"]


def json_names(text):
    return set(re.findall(r'["\']([a-z0-9_]+\.json)["\']', text))


def test_every_results_file_the_paper_reads_is_written_by_a_script():
    written = set()
    for s in SCRIPTS:
        written |= json_names((ROOT / s).read_text(encoding="utf-8"))
    read = json_names((ROOT / "make_paper.py").read_text(encoding="utf-8"))
    assert read, "make_paper.py loads no results files?"
    assert read <= written, f"paper reads results no script writes: {sorted(read - written)}"


def test_every_experiment_script_actually_calls_save_json():
    for s in SCRIPTS:
        text = (ROOT / s).read_text(encoding="utf-8")
        assert re.search(r"save_json\(", text), f"{s} never calls save_json"


def test_results_keys_used_by_the_paper_exist_in_the_latest_outputs():
    """If results files are present, the fields make_paper indexes must be there (skipped otherwise)."""
    import json
    res = ROOT / "results"
    checks = {
        "downstream_train_py.json": ["splits"],
        "regime_robustness.json": ["retrieval", "downstream", "faithfulness", "precedent_density_post_hiking"],
        "faithfulness.json": ["rank_test", "magnitude_test", "baseline_max_shift"],
        "retrieval_quality.json": ["modes", "n_events"],
    }
    for name, keys in checks.items():
        p = res / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for k in keys:
            assert k in d, f"{name} lacks '{k}'"
        if name == "regime_robustness.json":
            assert d["faithfulness"], "regime_robustness.json faithfulness section is empty"
