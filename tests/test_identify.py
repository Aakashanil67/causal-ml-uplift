import pytest

from src.identify import identify


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_hillstrom

    return load_hillstrom()


@pytest.mark.parametrize("outcome", ["visit", "conversion", "spend"])
def test_backdoor_set_is_empty_on_the_rct_graph(df, outcome):
    # the graph in src/dag.py has no edges into treatment, so the backdoor adjustment set should
    # be empty for every outcome — this is the formal identification claim the report leans on.
    _model, identified = identify(df, outcome)
    assert identified.get_backdoor_variables() == []


def test_estimand_type_is_nonparametric_ate(df):
    _model, identified = identify(df, "visit")
    assert "ATE" in str(identified.estimand_type)
