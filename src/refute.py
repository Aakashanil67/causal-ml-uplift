"""Placebo treatment, random common cause, and data-subset refutation, run directly against
`src/dml_ate.py`'s own `dml_ate()` rather than through DoWhy's `refute_estimate()` wrapper.

That wrapper's econml integration passes effect-modifier columns straight to econml's numeric API
without dummy-encoding them (`dowhy/causal_estimators/econml.py:effect()`), so it fails on this
project's categorical covariates (`channel`, `zip_code`) with a `KeyError` before any refutation
logic runs. Implementing the three refuters by hand keeps the actual estimator this project reports
under test — the same `LinearDML` + `build_covariate_matrix()` pipeline every other report uses —
rather than switching to a different, less-integrated estimator just to satisfy the wrapper.

None of these three refuters tests whether the identification assumption holds. They record
whether the fitted estimate responds to label shuffling, irrelevant noise, and selected subsamples.
The constructed Hillstrom sample targets a different covariate distribution from the full RCT, so
its difference from the RCT estimate is not known bias. The fully synthetic simulation is where
bias and coverage are checked against known targets.
"""

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, TREATMENT_COL
from src.data_loader import build_covariate_matrix
from src.dml_ate import _make_dml


def placebo_treatment_refuter(
    df: pd.DataFrame, outcome_col: str, drop_cols: list[str] | None = None, seed: int = RANDOM_SEED
) -> dict:
    """Replaces the real treatment column with a random permutation of itself, breaking any link
    to potential outcomes while keeping the marginal treated/control split identical. A real effect
    should vanish; what's left over is whatever the estimator finds when there is nothing to find.

    `drop_cols` must match whatever was withheld from the estimate under test — otherwise this
    silently refutes a *different*, better-specified model than the one it's meant to be checking
    (caught during development: running this on `src/confounded.py`'s newbie-withheld estimate
    without also dropping `newbie` here gave a refuter result for the wrong, unbiased model)."""
    rng = np.random.default_rng(seed)
    placebo = df.copy()
    placebo[TREATMENT_COL] = rng.permutation(placebo[TREATMENT_COL].to_numpy())
    X = build_covariate_matrix(placebo)
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X = X.to_numpy()
    T = placebo[TREATMENT_COL].to_numpy()
    Y = placebo[outcome_col].to_numpy(dtype=float)
    est = _make_dml(seed)
    est.fit(Y, T, X=X)
    ate = float(est.ate(X))
    ci_low, ci_high = est.ate_interval(X)
    return {"ate": ate, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def random_common_cause_refuter(
    df: pd.DataFrame, outcome_col: str, drop_cols: list[str] | None = None, seed: int = RANDOM_SEED
) -> dict:
    """Adds a column of pure random noise, independent of treatment, outcome and every real
    covariate, to the covariate matrix and refits. A well-behaved estimator should barely move,
    since there is nothing in the new column for it to (mis)use. See `placebo_treatment_refuter`
    for why `drop_cols` has to match the estimate under test."""
    rng = np.random.default_rng(seed)
    X = build_covariate_matrix(df)
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X["_random_noise"] = rng.normal(size=len(df))
    T = df[TREATMENT_COL].to_numpy()
    Y = df[outcome_col].to_numpy(dtype=float)
    est = _make_dml(seed)
    est.fit(Y, T, X=X.to_numpy())
    ate = float(est.ate(X.to_numpy()))
    ci_low, ci_high = est.ate_interval(X.to_numpy())
    return {"ate": ate, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def data_subset_refuter(
    df: pd.DataFrame,
    outcome_col: str,
    drop_cols: list[str] | None = None,
    n_runs: int = 5,
    subset_frac: float = 0.8,
    seed: int = RANDOM_SEED,
) -> list[float]:
    """Refits on several independent 80% subsamples. A stable estimator gives similar answers each
    time; wide swings mean the headline number depends heavily on which rows happened to be
    sampled, not on a real, generalisable effect. See `placebo_treatment_refuter` for why
    `drop_cols` has to match the estimate under test."""
    rng = np.random.default_rng(seed)
    ates = []
    for i in range(n_runs):
        idx = rng.choice(len(df), size=int(len(df) * subset_frac), replace=False)
        sub = df.iloc[idx]
        X = build_covariate_matrix(sub)
        if drop_cols:
            X = X.drop(columns=drop_cols)
        X = X.to_numpy()
        T = sub[TREATMENT_COL].to_numpy()
        Y = sub[outcome_col].to_numpy(dtype=float)
        est = _make_dml(seed + i)
        est.fit(Y, T, X=X)
        ates.append(float(est.ate(X)))
    return ates


def _ci_contains_zero(r: dict) -> bool:
    return r["ci_low"] <= 0 <= r["ci_high"]


def _close_to(value: float, target: float, tol: float = 0.01) -> bool:
    return abs(value - target) < tol


def _refuter_table_rows(
    results: dict, reference_ate: float, pass_word: str, fail_word: str
) -> list[str]:
    placebo_pass = _ci_contains_zero(results["placebo"])
    cause_pass = _close_to(results["random_cause"]["ate"], reference_ate)
    subset_pass = np.std(results["subset"]) < 0.01
    return [
        "| refuter | result | read |",
        "|---|---|---|",
        f"| placebo treatment | {results['placebo']['ate']:+.4f} "
        f"[{results['placebo']['ci_low']:.4f}, {results['placebo']['ci_high']:.4f}] | "
        f"{pass_word if placebo_pass else fail_word}, CI "
        f"{'contains' if placebo_pass else 'excludes'} 0 |",
        f"| random common cause | {results['random_cause']['ate']:+.4f} "
        f"[{results['random_cause']['ci_low']:.4f}, {results['random_cause']['ci_high']:.4f}] | "
        f"{pass_word if cause_pass else fail_word}, "
        f"{'close to' if cause_pass else 'moved materially from'} the original |",
        f"| data subset (5 runs) | mean {np.mean(results['subset']):+.4f}, "
        f"std {np.std(results['subset']):.4f} | "
        f"{pass_word if subset_pass else fail_word}, "
        f"{'stable' if subset_pass else 'unstable'} across subsamples |",
    ], (placebo_pass, cause_pass, subset_pass)


def write_refutation_report(
    original_ate: float,
    rct_results: dict,
    selected_sample_ate: float,
    experimental_reference: float,
    selected_sample_results: dict,
    out_path,
) -> None:
    rct_rows, rct_passes = _refuter_table_rows(rct_results, original_ate, "PASS", "FAIL")
    selected_rows, selected_passes = _refuter_table_rows(
        selected_sample_results, selected_sample_ate, "PASS", "FAIL"
    )
    lines = [
        "# Refutation checks: what these results do and do not establish",
        "",
        "Placebo treatment, random common cause, and data-subset checks run against this",
        "project's `LinearDML` implementation (`src/dml_ate.py`). They record the outcomes of",
        "these particular checks; passing does not certify identification or general estimator",
        "behavior.",
        "",
        "## On the randomized Hillstrom sample",
        "",
        f"Pooled DML estimate on `visit`: **{original_ate:+.4f}**.",
        "",
        *rct_rows,
        "",
    ]
    if all(rct_passes):
        lines += [
            "All three checks met their stated criteria on this sample. The placebo interval",
            "contains zero, adding random noise leaves the estimate close to its original value,",
            "and the selected 80% refits vary little. These outcomes do not replace the randomized",
            "assignment design or test for an omitted common cause.",
        ]
    else:
        lines.append(
            "At least one check missed its stated criterion on this run. That result calls for",
            "inspection of that diagnostic; the thresholds are not a universal correctness test.",
        )
    lines += [
        "",
        "## On a constructed selected sample",
        "",
        f"The selected-sample estimate is **{selected_sample_ate:+.4f}**. The full-RCT estimate",
        f"of **{experimental_reference:+.4f}** is an estimated reference for a different population.",
        "Selection changes the covariate distribution, so the difference between these point",
        "estimates is descriptive and does not establish bias for the selected population.",
        "",
        *selected_rows,
        "",
    ]
    if all(selected_passes):
        lines += [
            "These checks met their stated criteria for this fitted selected-sample model. That",
            "does not establish that its estimate is correct for the selected population, and it",
            "does not show that the checks would detect other forms of confounding.",
        ]
    else:
        lines.append(
            "At least one check missed its stated criterion here. That outcome is specific to",
            "this fitted sample and does not determine the estimate's bias for its target population.",
        )
    lines += [
        "",
        "The fully synthetic experiment in `reports/results.json` is where the generating",
        "probabilities, sample-average target, bias, and coverage are known. Refutation checks are",
        "useful challenges to particular estimates, not universal certificates of correctness.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.config import REPORTS_DIR
    from src.confounded import run_variant
    from src.data_loader import load_hillstrom
    from src.dml_ate import dml_ate

    df = load_hillstrom()
    original = dml_ate(df, TREATMENT_COL, "visit")
    rct_results = {
        "placebo": placebo_treatment_refuter(df, "visit"),
        "random_cause": random_common_cause_refuter(df, "visit"),
        "subset": data_subset_refuter(df, "visit"),
    }

    selected = run_variant(df, "visit", ["newbie"], {"newbie": -1}, 1.5, withhold=["newbie"])
    selected_results = {
        "placebo": placebo_treatment_refuter(selected["sample"], "visit", drop_cols=["newbie"]),
        "random_cause": random_common_cause_refuter(
            selected["sample"], "visit", drop_cols=["newbie"]
        ),
        "subset": data_subset_refuter(selected["sample"], "visit", drop_cols=["newbie"]),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_refutation_report(
        original["ate"],
        rct_results,
        selected["dml"]["ate"],
        original["ate"],
        selected_results,
        REPORTS_DIR / "08_refutations.md",
    )


if __name__ == "__main__":
    main()
