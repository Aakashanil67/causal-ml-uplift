"""HC1 logit and OLS baselines with interpretable average effects.

Binary covariates are reported as average 0-to-1 changes in predicted probability; categorical
covariates are contrasted as valid profiles against their reference category. Continuous
covariates retain average derivatives.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import BINARY_COVARIATES, CATEGORICAL_LEVELS, NUMERIC_COVARIATES, TREATMENT_COL
from src.data_loader import build_covariate_matrix


def build_design_matrix(df):
    X = build_covariate_matrix(df)
    X[TREATMENT_COL] = df[TREATMENT_COL].astype(float)
    # `add_constant` otherwise mistakes an all-zero/all-one treatment counterfactual for an
    # existing intercept and silently drops `const` from one side of a contrast.
    return sm.add_constant(X, has_constant="add")


def fit_logit(df, outcome_col: str):
    X = build_design_matrix(df)
    y = df[outcome_col].astype(float)
    return sm.Logit(y, X).fit(disp=0, cov_type="HC1")


def fit_ols(df, outcome_col: str):
    X = build_design_matrix(df)
    y = df[outcome_col].astype(float)
    return sm.OLS(y, X).fit(cov_type="HC1")


def average_discrete_change(logit_result, design_zero: pd.DataFrame, design_one: pd.DataFrame):
    """Average predicted probability change between two valid design profiles.

    The standard error uses the fitted model's covariance matrix (HC1 in this module) and the
    delta method on the average counterfactual prediction difference.
    """
    if list(design_zero.columns) != list(design_one.columns):
        raise ValueError("Counterfactual design matrices must have identical columns and order.")
    if len(design_zero) != len(design_one):
        raise ValueError("Counterfactual design matrices must have the same number of rows.")
    columns = list(logit_result.params.index)
    x0 = design_zero.loc[:, columns].to_numpy(dtype=float)
    x1 = design_one.loc[:, columns].to_numpy(dtype=float)
    p0 = np.asarray(logit_result.predict(design_zero.loc[:, columns]), dtype=float)
    p1 = np.asarray(logit_result.predict(design_one.loc[:, columns]), dtype=float)
    gradient = np.mean(
        p1[:, None] * (1 - p1[:, None]) * x1 - p0[:, None] * (1 - p0[:, None]) * x0,
        axis=0,
    )
    variance = float(gradient @ np.asarray(logit_result.cov_params()) @ gradient)
    se = float(np.sqrt(max(variance, 0.0)))
    effect = float(np.mean(p1 - p0))
    return {"effect": effect, "se": se, "ci_low": effect - 1.96 * se, "ci_high": effect + 1.96 * se}


def average_marginal_effects(logit_result, df: pd.DataFrame) -> pd.DataFrame:
    """Return average effects, using coherent counterfactual profiles for discrete features."""
    rows = []
    derivatives = logit_result.get_margeff(at="overall").summary_frame()
    for column in NUMERIC_COVARIATES:
        result = derivatives.loc[column]
        rows.append(
            {
                "contrast": column,
                "effect": result["dy/dx"],
                "se": result["Std. Err."],
                "ci_low": result["Conf. Int. Low"],
                "ci_high": result["Cont. Int. Hi."],
                "kind": "average derivative",
            }
        )

    for column in (TREATMENT_COL, *BINARY_COVARIATES):
        zero = df.copy()
        one = df.copy()
        zero[column] = 0
        one[column] = 1
        result = average_discrete_change(
            logit_result, build_design_matrix(zero), build_design_matrix(one)
        )
        rows.append({"contrast": column, **result, "kind": "average 0-to-1 change"})

    for column, levels in CATEGORICAL_LEVELS.items():
        reference = levels[0]
        reference_profile = df.copy()
        reference_profile[column] = reference
        reference_design = build_design_matrix(reference_profile)
        for level in levels[1:]:
            comparison_profile = df.copy()
            comparison_profile[column] = level
            result = average_discrete_change(
                logit_result, reference_design, build_design_matrix(comparison_profile)
            )
            rows.append(
                {
                    "contrast": f"{column}={level} vs {reference}",
                    **result,
                    "kind": "category profile contrast",
                }
            )
    return pd.DataFrame(rows).set_index("contrast")


def write_regression_report(
    visit_ame, conversion_ame, spend_result, naive_diffs: dict, out_path
) -> None:
    lines = [
        "# Regression baseline: logit and OLS with robust standard errors",
        "",
        "Logit for `visit` and `conversion` (binary), OLS for `spend`, all with heteroscedasticity-",
        "robust (HC1) standard errors and the full covariate set as controls. Binary regressors use",
        "average 0-to-1 prediction changes; categorical regressors change as valid category",
        "profiles. Continuous covariates use average derivatives. The treatment contrast is the",
        "average predicted difference between assignment=1 and assignment=0, in percentage points.",
        "",
        "## Treatment effect, three ways to read it",
        "",
        "| outcome | adjusted estimate | 95% CI | naive diff-in-means |",
        "|---|---|---|---|",
    ]
    v_row = visit_ame.loc[TREATMENT_COL]
    c_row = conversion_ame.loc[TREATMENT_COL]
    s_row = spend_result.params[TREATMENT_COL]
    s_ci = spend_result.conf_int().loc[TREATMENT_COL]
    lines.append(
        f"| visit | {v_row['effect'] * 100:+.2f}pp | [{v_row['ci_low'] * 100:+.2f}pp, "
        f"{v_row['ci_high'] * 100:+.2f}pp] | {naive_diffs['visit'] * 100:+.2f}pp |"
    )
    lines.append(
        f"| conversion | {c_row['effect'] * 100:+.2f}pp | "
        f"[{c_row['ci_low'] * 100:+.2f}pp, {c_row['ci_high'] * 100:+.2f}pp] | "
        f"{naive_diffs['conversion'] * 100:+.2f}pp |"
    )
    lines.append(
        f"| spend | +${s_row:.4f} | [{s_ci[0]:.4f}, {s_ci[1]:.4f}] | +${naive_diffs['spend']:.4f} |"
    )
    lines += [
        "",
        "The regression estimate and the naive diff-in-means agree closely on all three outcomes,",
        "which is exactly what should happen on a randomised experiment: adding covariate controls",
        "should barely move the estimate, because treatment is uncorrelated with those covariates by",
        "design (`reports/02_naive_estimate.md`). Controls earn their place here for precision (a",
        "tighter CI on `spend`, where outcome variance is high), not for removing confounding that",
        "was never there.",
        "",
        "## Full average-effects table, `visit`",
        "",
        "| contrast | calculation | estimate | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for cov, row in visit_ame.iterrows():
        lines.append(
            f"| {cov} | {row['kind']} | {row['effect'] * 100:+.2f}pp | "
            f"[{row['ci_low'] * 100:+.2f}pp, {row['ci_high'] * 100:+.2f}pp] |"
        )
    newbie_change = visit_ame.loc["newbie", "effect"] * 100
    lines += [
        "",
        "Reading a control contrast: `newbie` has an average predicted change",
        f"of {newbie_change:+.2f}pp when the same profiles are changed from an existing account",
        "to a new account, with other covariates held fixed. That is a customer-type difference, not a treatment",
        "effect: `newbie` predicts engagement, it does not predict which arm anyone was assigned to",
        "(see the covariate balance table in `reports/02_naive_estimate.md`).",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.config import REPORTS_DIR
    from src.data_loader import load_hillstrom
    from src.naive import naive_estimates

    df = load_hillstrom()

    visit_model = fit_logit(df, "visit")
    conversion_model = fit_logit(df, "conversion")
    spend_model = fit_ols(df, "spend")

    visit_ame = average_marginal_effects(visit_model, df)
    conversion_ame = average_marginal_effects(conversion_model, df)

    naive = naive_estimates(df, TREATMENT_COL)
    naive_diffs = {o: naive.loc[o, "diff"] for o in ["visit", "conversion", "spend"]}

    print(visit_model.summary())
    print(visit_ame)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_regression_report(
        visit_ame,
        conversion_ame,
        spend_model,
        naive_diffs,
        REPORTS_DIR / "03_regression_baseline.md",
    )


if __name__ == "__main__":
    main()
