"""Pre-specified treatment-interaction tests used to check the forest's segmentation story."""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.config import TREATMENT_COL

INTERACTION_TERMS = ("mens", "womens", "recency", "history")


def interaction_test(
    df: pd.DataFrame,
    outcome: str = "visit",
    terms: tuple[str, ...] = INTERACTION_TERMS,
) -> tuple[pd.DataFrame, float]:
    """Fit one HC1-robust OLS and return individual and joint interaction tests."""
    rhs = "+".join(terms)
    model = smf.ols(f"{outcome} ~ {TREATMENT_COL}*({rhs})", data=df).fit(cov_type="HC1")
    names = [f"{TREATMENT_COL}:{term}" for term in terms]
    ci = model.conf_int().loc[names]
    raw_p = model.pvalues.loc[names].to_numpy()
    holm_p = multipletests(raw_p, method="holm")[1]

    table = pd.DataFrame(
        {
            "estimate": model.params.loc[names].to_numpy(),
            "std_error": model.bse.loc[names].to_numpy(),
            "ci_low": ci[0].to_numpy(),
            "ci_high": ci[1].to_numpy(),
            "p_value": raw_p,
            "p_holm": holm_p,
        },
        index=list(terms),
    )
    table.index.name = "term"
    constraints = ", ".join(f"{name} = 0" for name in names)
    joint_p = float(model.wald_test(constraints, scalar=True).pvalue)
    return table, joint_p


if __name__ == "__main__":
    from src.data_loader import load_hillstrom

    interaction_table, joint_pvalue = interaction_test(load_hillstrom())
    print(interaction_table)
    print(f"joint Wald p-value: {joint_pvalue:.6g}")
