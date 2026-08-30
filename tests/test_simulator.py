import ast
from pathlib import Path

import numpy as np
import pytest

from app.simulator import plot_cate_gauge


def test_plot_cate_gauge_returns_a_figure():
    eval_cate = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    fig = plot_cate_gauge(0.05, eval_cate)
    assert fig is not None
    assert len(fig.axes) == 1


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
    assert metrics["Incremental visits per customer emailed"] == "+0.0603"
    warnings = [item.value for item in at.warning]
    assert any("Normalized Qini" in text and "includes zero" in text for text in warnings)
