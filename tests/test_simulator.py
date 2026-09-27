import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from app.simulator import (
    format_probability_difference,
    format_probability_level,
    plot_cate_gauge,
)


def test_plot_cate_gauge_returns_a_figure():
    eval_cate = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    fig = plot_cate_gauge(0.05, eval_cate)
    assert fig is not None
    assert len(fig.axes) == 1


def test_probability_level_and_probability_difference_have_distinct_units():
    assert format_probability_level(0.0601) == "6.01%"
    assert format_probability_difference(0.0601) == "+6.01 pp"


def test_streamlit_request_path_does_not_fit_causal_models():
    app_path = Path(__file__).resolve().parent.parent / "app" / "simulator.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "fit_causal_forest" not in called
    assert "fit_policy_forest" not in called


def test_policy_tab_exposes_manifest_backed_decision_controls():
    source = (Path(__file__).resolve().parent.parent / "app" / "simulator.py").read_text(
        encoding="utf-8"
    )

    assert '"email everyone (womens creative)"' in source or "recommendation_shares" in source
    assert "Assumed gross margin" in source
    assert "Cost per email" in source
    assert "may be top-coded" in source
    assert "not profit" in source


def test_policy_interval_caption_preserves_literal_currency(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import app.simulator as simulator
    from src.config import EVALUATION_ARTIFACT_PATH, RESULTS_PATH

    # Isolate caption rendering from artifact provenance; the full app test validates loading.
    artifact = joblib.load(EVALUATION_ARTIFACT_PATH)["payload"]
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(simulator, "get_public_artifacts", lambda: (artifact, results))
    monkeypatch.setattr(
        simulator, "get_eval_artifacts", lambda: (artifact["eval_df"], artifact["cate"])
    )
    at = AppTest.from_string("from app.simulator import render_policy_tab\nrender_policy_tab()")
    at.run(timeout=60)
    assert not at.exception
    captions = [c.value for c in at.caption if c.value.startswith("95% paired conditional")]
    assert len(captions) == 2
    assert all(caption.count(r"\$") == 2 for caption in captions)


@pytest.mark.slow
def test_app_runs_end_to_end_without_exceptions():
    # Exercises both tabs against the committed artifacts. This remains marked slow because
    # Streamlit starts a full script runtime and unpickles the 34MB production forest.
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "simulator.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=120)

    assert not at.exception

    metrics = {m.label: m.value for m in at.get("metric")}
    # The default top-30% slider uses a treated-minus-control mean difference within the selected
    # segment, the counterfactual all-email estimand. It must not revert to the old Qini-gain
    # denominator (+0.0499) that divided by all selected rows.
    assert metrics["Customers targeted"] == "5,760 / 19,200"
    assert metrics["Incremental visit-rate difference per customer emailed"] == "+6.03 pp"
    warnings = [item.value for item in at.warning]
    assert any("Normalized Qini" in text and "crosses zero" in text for text in warnings)
    assert any(item.label == "Assumed gross margin" for item in at.get("slider"))
    assert any(item.label == "Cost per email" for item in at.get("number_input"))
    assert any("may be top-coded" in text and "not profit" in text for text in warnings)
