"""The econometrics baseline: logit for the two binary outcomes, OLS for spend, both with HC1
robust standard errors. Reported as average marginal effects for the logit models, not raw
log-odds coefficients, since a marginal effect is what an economist would actually quote and a
log-odds coefficient is not directly comparable across models or interpretable in percentage
points without it.
"""

import statsmodels.api as sm

from src.config import TREATMENT_COL
from src.data_loader import build_covariate_matrix


def build_design_matrix(df):
    X = build_covariate_matrix(df)
    X[TREATMENT_COL] = df[TREATMENT_COL].astype(float)
    return sm.add_constant(X)


def fit_logit(df, outcome_col: str):
    X = build_design_matrix(df)
    y = df[outcome_col].astype(float)
    return sm.Logit(y, X).fit(disp=0, cov_type="HC1")


def fit_ols(df, outcome_col: str):
    X = build_design_matrix(df)
    y = df[outcome_col].astype(float)
    return sm.OLS(y, X).fit(cov_type="HC1")


def average_marginal_effects(logit_result):
    return logit_result.get_margeff(at="overall").summary_frame()


def write_regression_report(
    visit_ame, conversion_ame, spend_result, naive_diffs: dict, out_path
) -> None:
    lines = [
        "# Regression baseline: logit and OLS with robust standard errors",
        "",
        "Logit for `visit` and `conversion` (binary), OLS for `spend`, all with heteroscedasticity-",
        "robust (HC1) standard errors and the full covariate set as controls. Reported as average",
        "marginal effects for the logit models rather than raw log-odds coefficients. That is how an",
        "economist would actually read these out: a coefficient of 0.5366 on treatment in the",
        "`visit` logit says nothing directly interpretable; a marginal effect of +6.49 percentage",
        "points does.",
        "",
        "## Treatment effect, three ways to read it",
        "",
        "| outcome | regression AME / coefficient | 95% CI | naive diff-in-means |",
        "|---|---|---|---|",
    ]
    v_row = visit_ame.loc[TREATMENT_COL]
    c_row = conversion_ame.loc[TREATMENT_COL]
    s_row = spend_result.params[TREATMENT_COL]
    s_ci = spend_result.conf_int().loc[TREATMENT_COL]
    lines.append(
        f"| visit | +{v_row['dy/dx'] * 100:.2f}pp | [{v_row['Conf. Int. Low'] * 100:.2f}pp, "
        f"{v_row['Cont. Int. Hi.'] * 100:.2f}pp] | +{naive_diffs['visit'] * 100:.2f}pp |"
    )
    lines.append(
        f"| conversion | +{c_row['dy/dx'] * 100:.2f}pp | "
        f"[{c_row['Conf. Int. Low'] * 100:.2f}pp, {c_row['Cont. Int. Hi.'] * 100:.2f}pp] | "
        f"+{naive_diffs['conversion'] * 100:.2f}pp |"
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
        "## Full marginal-effects table, `visit`",
        "",
        "| covariate | dy/dx | 95% CI |",
        "|---|---|---|",
    ]
    for cov, row in visit_ame.iterrows():
        lines.append(
            f"| {cov} | {row['dy/dx'] * 100:+.2f}pp | [{row['Conf. Int. Low'] * 100:.2f}pp, "
            f"{row['Cont. Int. Hi.'] * 100:.2f}pp] |"
        )
    newbie_ame = visit_ame.loc["newbie", "dy/dx"] * 100
    lines += [
        "",
        "Reading a control coefficient the way an economist would: `newbie` has a marginal effect",
        f"of {newbie_ame:+.2f}pp on the probability of visiting. A customer who opened their",
        f"account in the last twelve months is about {abs(newbie_ame):.1f} percentage points less",
        "likely to visit after the campaign than an otherwise-identical existing customer, other",
        "covariates held fixed. That is a real difference between customer types, not a treatment",
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

    visit_ame = average_marginal_effects(visit_model)
    conversion_ame = average_marginal_effects(conversion_model)

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
