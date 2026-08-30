import numpy as np
import pytest

from src.interactions import interaction_test


@pytest.fixture(scope="module")
def result():
    from src.data_loader import load_hillstrom

    return interaction_test(load_hillstrom())


def test_interaction_test_reports_the_prespecified_family(result):
    table, joint_p = result

    assert table.index.tolist() == ["mens", "womens", "recency", "history"]
    assert table.index.name == "term"
    assert table.columns.tolist() == [
        "estimate",
        "std_error",
        "ci_low",
        "ci_high",
        "p_value",
        "p_holm",
    ]
    assert np.isfinite(table.to_numpy()).all()
    assert 0 <= joint_p <= 1


def test_holm_adjustment_never_reduces_a_p_value(result):
    table, _joint_p = result

    assert (table["p_holm"] >= table["p_value"]).all()
    assert (table["p_holm"] <= 1).all()


def test_purchase_history_interactions_are_reproduced_from_data(result):
    table, joint_p = result

    assert table.loc["mens", "p_value"] == pytest.approx(0.006146, abs=1e-5)
    assert table.loc["womens", "p_value"] < 1e-6
    assert table.loc["recency", "p_holm"] > 0.1
    assert table.loc["history", "p_holm"] > 0.1
    assert joint_p < 1e-6
