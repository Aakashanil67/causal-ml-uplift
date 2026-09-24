"""Stress-tests DML under a constructed observational selection mechanism.

Hillstrom is randomised, so nothing in the ladder so far has had real confounding to correct —
every method from the naive diff-in-means onward agrees, because there was nothing to disagree
about. This module manufactures two observational datasets *from the same real (T, X, Y) rows* by
retaining customers with a probability that depends on both their arm and their covariates, then
compares selected-sample estimates with the full-RCT experimental reference.

Variant 1 confounds on `recency`/`history`, both included in the estimators' covariate matrix
(selection on observables). Variant 2 confounds on `newbie`, real in the data but withheld from
every estimator's covariate matrix in this variant (selection on an unobservable). No outcome
value is ever fabricated; only which real rows are retained changes.

Caveat: the full-RCT estimate is itself estimated, and selection changes the covariate distribution.
Under heterogeneous effects the selected-sample ATE may differ from the original-population ATE.
These examples are selection diagnostics against an experimental reference, not known-bias
benchmarks. Bias and coverage claims belong to the fully synthetic experiment in `src/simulation.py`,
where the generating probabilities and sample-average target are known.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import RANDOM_SEED, TREATMENT_COL
from src.dag import build_dag
from src.data_loader import build_covariate_matrix
from src.dml_ate import dml_ate


def make_confounded_sample(
    df: pd.DataFrame,
    confounder_cols: list[str],
    directions: dict[str, int],
    strength: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Retention probability rises with a z-scored, signed combination of `confounder_cols`,
    flipped in sign between treated and control rows — so in the retained sample, treated
    customers systematically differ from control customers on exactly those covariates. This is
    what "selection on observables" looks like when built from real experimental data rather than
    simulated from scratch: every row kept is a real customer with their real observed outcome."""
    rng = np.random.default_rng(seed)
    z = pd.DataFrame(
        {c: (df[c] - df[c].mean()) / df[c].std() * directions[c] for c in confounder_cols}
    )
    score = z.sum(axis=1)
    logit = strength * score * np.where(df[TREATMENT_COL] == 1, 1, -1)
    keep_prob = 1 / (1 + np.exp(-logit))
    keep = rng.uniform(size=len(df)) < keep_prob
    return df[keep].reset_index(drop=True)


def build_confounded_dag(confounder_cols: list[str]):
    """The graph in src/dag.py with edges added from each confounder into treatment — the
    structural change that makes the backdoor adjustment set non-empty (see identify_confounded
    below)."""
    g = build_dag()
    for c in confounder_cols:
        g.add_edge(c, TREATMENT_COL)
    return g


def identify_confounded(df: pd.DataFrame, outcome_col: str, confounder_cols: list[str]):
    from dowhy import CausalModel

    from src.dag import to_gml

    gml = to_gml(build_confounded_dag(confounder_cols))
    model = CausalModel(data=df, treatment=TREATMENT_COL, outcome=outcome_col, graph=gml)
    return model.identify_effect(proceed_when_unidentifiable=True)


def naive_diff(df: pd.DataFrame, outcome_col: str) -> float:
    treated = df.loc[df[TREATMENT_COL] == 1, outcome_col]
    control = df.loc[df[TREATMENT_COL] == 0, outcome_col]
    return float(treated.mean() - control.mean())


def ols_estimate(df: pd.DataFrame, outcome_col: str, drop_cols: list[str] | None = None) -> float:
    X = build_covariate_matrix(df)
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X[TREATMENT_COL] = df[TREATMENT_COL].astype(float)
    X = sm.add_constant(X)
    y = df[outcome_col].astype(float)
    return float(sm.OLS(y, X).fit(cov_type="HC1").params[TREATMENT_COL])


def dml_estimate(df: pd.DataFrame, outcome_col: str, drop_cols: list[str] | None = None) -> dict:
    if not drop_cols:
        return dml_ate(df, TREATMENT_COL, outcome_col)
    X = build_covariate_matrix(df).drop(columns=drop_cols).to_numpy()
    T = df[TREATMENT_COL].to_numpy()
    Y = df[outcome_col].to_numpy(dtype=float)
    from econml.dml import LinearDML
    from lightgbm import LGBMClassifier, LGBMRegressor

    est = LinearDML(
        model_y=LGBMRegressor(n_estimators=100, verbose=-1, random_state=RANDOM_SEED),
        model_t=LGBMClassifier(n_estimators=100, verbose=-1, random_state=RANDOM_SEED),
        discrete_treatment=True,
        cv=3,
        random_state=RANDOM_SEED,
    )
    est.fit(Y, T, X=X)
    ci_low, ci_high = est.ate_interval(X)
    return {"ate": float(est.ate(X)), "ci_low": float(ci_low), "ci_high": float(ci_high)}


def run_variant(
    df: pd.DataFrame,
    outcome_col: str,
    confounder_cols: list[str],
    directions: dict[str, int],
    strength: float,
    withhold: list[str] | None,
) -> dict:
    sub = make_confounded_sample(df, confounder_cols, directions, strength)
    return {
        "n": len(sub),
        "naive": naive_diff(sub, outcome_col),
        "ols": ols_estimate(sub, outcome_col, drop_cols=withhold),
        "dml": dml_estimate(sub, outcome_col, drop_cols=withhold),
        "sample": sub,
    }


def write_confounding_report(benchmark: float, v1: dict, v2: dict, out_path) -> None:
    def pp(value: float) -> str:
        return f"{value * 100:+.2f}pp"

    lines = [
        "# Constructed confounding stress test",
        "",
        "This section constructs selection using the same real Hillstrom rows. The full-RCT DML",
        "estimate on `visit`,",
        f"**{pp(benchmark)}** (`reports/05_dml_ate.md`), is an estimated experimental reference.",
        "It is not known truth for either selected sample: selection changes the covariate",
        "distribution, and heterogeneous effects can change the selected-population ATE.",
        "",
        "The differences below are selection diagnostics, not numerical bias estimates. A",
        "common-target comparison would be needed to make a real-data bias claim. Bias and",
        "coverage are evaluated in the fully synthetic experiment, where the generating",
        "probabilities and sample-average target are known. LinearDML also uses a linear final",
        "effect stage; flexible nuisance models do not guarantee recovery under an arbitrary",
        "response surface.",
        "",
        "## Variant 1: selection on observables (`recency`, `history`)",
        "",
        f"Retained {v1['n']:,} of 64,000 customers. Treated customers were kept preferentially",
        "when recent and high-spending, and control customers preferentially when lapsed and",
        "low-spending. Both variables remain available to each estimator.",
        "",
        "| estimator | selected-sample estimate | difference from full-RCT estimate |",
        "|---|---:|---:|",
        f"| naive diff-in-means | {pp(v1['naive'])} | {pp(v1['naive'] - benchmark)} |",
        f"| OLS with `recency` and `history` | {pp(v1['ols'])} | {pp(v1['ols'] - benchmark)} |",
        f"| LinearDML | {pp(v1['dml']['ate'])} [{pp(v1['dml']['ci_low'])}, "
        f"{pp(v1['dml']['ci_high'])}] | {pp(v1['dml']['ate'] - benchmark)} |",
        "",
        "These differences describe how the estimators move under this selection mechanism. They",
        "do not show how far an estimator is from a known causal target for the selected rows.",
        "",
        "## Variant 2: selection on `newbie`, withheld from the estimator",
        "",
        f"Retained {v2['n']:,} customers. Treated customers were kept preferentially when `newbie=0`,",
        "and control customers when `newbie=1`. The fitted estimators omit `newbie`.",
        f"Its regression profile contrast is {pp(v2['newbie_effect'])} when changing the same",
        "profiles from existing to new account (`reports/03_regression_baseline.md`).",
        "",
        "| estimator | selected-sample estimate | difference from full-RCT estimate |",
        "|---|---:|---:|",
        f"| naive diff-in-means | {pp(v2['naive'])} | {pp(v2['naive'] - benchmark)} |",
        f"| OLS with `newbie` withheld | {pp(v2['ols'])} | {pp(v2['ols'] - benchmark)} |",
        f"| LinearDML with `newbie` withheld | {pp(v2['dml']['ate'])} "
        f"[{pp(v2['dml']['ci_low'])}, {pp(v2['dml']['ci_high'])}] | "
        f"{pp(v2['dml']['ate'] - benchmark)} |",
        "",
        "The `newbie` variable is omitted from the estimator by construction. This illustrates",
        "the importance of measured adjustment inputs, but the difference from the original RCT",
        "estimate is not a bias estimate for this selected population. The synthetic experiment",
        "provides the known-target test for omitted-confounder bias.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.config import REPORTS_DIR
    from src.data_loader import load_hillstrom

    df = load_hillstrom()
    benchmark = dml_ate(df, TREATMENT_COL, "visit")["ate"]
    print(f"benchmark (full RCT DML ATE, visit): {benchmark:.4f}")

    v1 = run_variant(
        df,
        "visit",
        ["recency", "history"],
        {"recency": -1, "history": 1},
        strength=1.2,
        withhold=None,
    )
    print("variant 1:", {k: v for k, v in v1.items() if k != "sample"})

    v2 = run_variant(df, "visit", ["newbie"], {"newbie": -1}, strength=1.5, withhold=["newbie"])
    print("variant 2:", {k: v for k, v in v2.items() if k != "sample"})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_confounding_report(benchmark, v1, v2, REPORTS_DIR / "06_confounding_benchmark.md")


if __name__ == "__main__":
    main()
