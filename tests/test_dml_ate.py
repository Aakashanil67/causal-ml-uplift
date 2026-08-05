import pytest

from src.dml_ate import dml_ate, per_arm_ate_table, pooled_ate_table


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_hillstrom

    return load_hillstrom()


def test_pooled_dml_ate_close_to_naive_diff_on_visit(df):
    # naive diff-in-means on visit is +0.0609 (reports/02_naive_estimate.md); DML uses flexible
    # nuisance models but on an RCT should land near it, not just share a sign.
    from src.config import TREATMENT_COL

    result = dml_ate(df, TREATMENT_COL, "visit")
    assert 0.04 < result["ate"] < 0.08
    assert result["ci_low"] < result["ate"] < result["ci_high"]


def test_pooled_ate_table_covers_all_outcomes(df):
    table = pooled_ate_table(df)
    assert set(table.index) == {"visit", "conversion", "spend"}
    assert (table["ate"] > 0).all()  # email lifts every outcome in this data


def test_per_arm_table_shows_mens_beats_womens_on_visit(df):
    # a real, checked fact about this data (reports/data_dictionary.md arm rates), not assumed:
    # mens email lifts visit more than womens email.
    table = per_arm_ate_table(df)
    mens_visit = table.loc[("Mens E-Mail", "visit"), "ate"]
    womens_visit = table.loc[("Womens E-Mail", "visit"), "ate"]
    assert mens_visit > womens_visit
    assert 0.05 < mens_visit < 0.10
    assert 0.02 < womens_visit < 0.07
