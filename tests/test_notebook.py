import json

from src.config import ROOT

NOTEBOOK_PATH = ROOT / "notebooks" / "01_exploration.ipynb"


def _load_code_cells():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_was_restart_and_run_all_not_edited_out_of_order():
    # a notebook with gaps or an out-of-order execution_count sequence means cells were run,
    # re-run, and reordered during development without a final clean pass — this is the check
    # kill-slop calls out as the biggest tell in data-science portfolios.
    cells = _load_code_cells()
    counts = [c["execution_count"] for c in cells]
    assert counts == list(range(1, len(cells) + 1)), f"execution counts not 1..n: {counts}"


def test_notebook_has_no_error_outputs():
    cells = _load_code_cells()
    errors = [
        (i, o.get("ename"))
        for i, c in enumerate(cells)
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"notebook has error outputs: {errors}"


def test_notebook_documents_a_real_failed_attempt():
    # kill-slop, notebooks: "keep one clearly-labelled section showing an approach that failed
    # and why" — this checks that section still exists and still says what it found, not that it
    # was quietly deleted in a later edit.
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    all_source = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "tried and rejected" in all_source.lower() or "tried and reject" in all_source.lower()
