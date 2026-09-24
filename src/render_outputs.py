"""Render public Markdown from the versioned results manifest."""

import hashlib
import re
from pathlib import Path

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ROOT
from src.policy import policy_comparison_conclusion
from src.results import load_results

README_PATH = ROOT / "README.md"
UPLIFT_REPORT_PATH = REPORTS_DIR / "07_uplift_policy.md"
CAUSAL_REPORT_PATH = REPORTS_DIR / "causal_report.md"
CAUSAL_REPORT_TEMPLATE_PATH = REPORTS_DIR / "templates" / "causal_report.md"
INTERVIEW_PATH = REPORTS_DIR / "interview_qa.md"
INTERVIEW_TEMPLATE_PATH = REPORTS_DIR / "templates" / "interview_qa.md"
MODEL_README_PATH = MODELS_DIR / "README.md"
MODEL_README_TEMPLATE_PATH = MODELS_DIR / "README.template.md"
NAIVE_REPORT_PATH = REPORTS_DIR / "02_naive_estimate.md"
REGRESSION_REPORT_PATH = REPORTS_DIR / "03_regression_baseline.md"
IDENTIFICATION_REPORT_PATH = REPORTS_DIR / "04_identification.md"
DML_REPORT_PATH = REPORTS_DIR / "05_dml_ate.md"
CONFOUNDING_REPORT_PATH = REPORTS_DIR / "06_confounding_benchmark.md"
REFUTATION_REPORT_PATH = REPORTS_DIR / "08_refutations.md"
MODEL_ARTIFACT_PATHS = (
    MODELS_DIR / "causal_forest.joblib",
    MODELS_DIR / "evaluation_artifacts.joblib",
)


def replace_markdown_section(text: str, heading: str, body: str) -> str:
    match = re.search(rf"(?m)^{re.escape(heading)}\s*$", text)
    if not match:
        raise ValueError(f"Heading not found: {heading}")
    level = len(heading) - len(heading.lstrip("#"))
    next_heading = re.search(rf"(?m)^#{{{level}}} (?!#)", text[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(text)
    prefix = text[: match.end()].rstrip()
    suffix = text[end:].lstrip("\n")
    rendered = f"{prefix}\n\n{body.strip()}\n"
    if suffix:
        rendered += f"\n{suffix}"
    return rendered


def _remove_markdown_section(text: str, heading: str) -> str:
    """Remove one peer section, including its body, from a Markdown document."""
    match = re.search(rf"(?m)^{re.escape(heading)}\s*$", text)
    if not match:
        return text
    level = len(heading) - len(heading.lstrip("#"))
    next_heading = re.search(rf"(?m)^#{{{level}}} (?!#)", text[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(text)
    prefix = text[: match.start()].rstrip()
    suffix = text[end:].lstrip("\n")
    return f"{prefix}\n\n{suffix}" if suffix else f"{prefix}\n"


def _clean_causal_report_template(text: str) -> str:
    """Drop obsolete duplicate sections left by earlier renderer headings."""
    obsolete = (
        "## 3. The methods ladder: naive, regression, and Double ML",
        "## 4. Constructed selection diagnostics",
        "## 5. Heterogeneity: what the model suggests",
        "## 6. Uplift ranking and reported-spend sensitivity",
        "## 7. Three actions, not two: held-out policy evaluation",
        "## 8. Refutation checks and their limits",
    )
    for heading in obsolete:
        text = _remove_markdown_section(text, heading)

    headings = re.findall(r"(?m)^## (?!#).+$", text)
    for heading in dict.fromkeys(headings):
        matches = list(re.finditer(rf"(?m)^{re.escape(heading)}\s*$", text))
        for duplicate in reversed(matches[1:]):
            next_heading = re.search(r"(?m)^## (?!#)", text[duplicate.end() :])
            end = duplicate.end() + next_heading.start() if next_heading else len(text)
            prefix = text[: duplicate.start()].rstrip()
            suffix = text[end:].lstrip("\n")
            text = f"{prefix}\n\n{suffix}" if suffix else f"{prefix}\n"
    return text


def _table_by(records: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in records}


def _interval_includes_zero(low: float, high: float) -> bool:
    return low <= 0 <= high


def _interval_conclusion(low: float, high: float, subject: str) -> str:
    if _interval_includes_zero(low, high):
        return f"The interval crosses zero, so this analysis does not establish {subject}."
    if low > 0:
        return f"The interval is entirely positive for {subject}."
    return f"The interval is entirely negative for {subject}."


def _policy_conclusion(comparison: dict) -> str:
    return policy_comparison_conclusion(comparison["ci_low"], comparison["ci_high"])


def _pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def _pp_interval(low: float, high: float) -> str:
    return f"[{_pp(low)}, {_pp(high)}]"


def _percent_interval(low: float, high: float) -> str:
    return f"[{low:.2%}, {high:.2%}]"


def _records_table(rows: list[dict], columns: list[tuple[str, str]], formats: dict) -> list[str]:
    """Render a compact Markdown table from manifest records."""
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "|" + "|".join("---:" if fmt else "---" for _, fmt in columns) + "|",
    ]
    for row in rows:
        values = []
        for _, field in columns:
            value = row.get(field)
            formatter = formats.get(field)
            values.append(
                "unavailable" if value is None else formatter(value) if formatter else str(value)
            )
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _effect_pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def _money(value: float) -> str:
    return f"${value:+.3f}"


def render_naive_report(results: dict) -> str:
    estimates = results["naive"]["estimates"]
    balance = results["naive"]["balance"]
    diffs = [row["standardised_diff"] for row in balance if row["standardised_diff"] is not None]
    largest = max((abs(value) for value in diffs), default=0.0)
    excessive = sum(abs(value) > 0.1 for value in diffs)
    lines = [
        "# Naive estimate: difference in means",
        "",
        "Treated customers received either email (Mens or Womens); controls received No E-Mail. "
        "Under the randomized assignment design, the difference in means estimates the average "
        "effect of assignment to the email mixture. The balance table is a descriptive check on "
        "this file, not proof that randomization worked.",
        "",
        "## Outcome differences",
        "",
        "| outcome | treated mean | control mean | difference | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in estimates:
        if row["outcome"] in ("visit", "conversion"):
            treated_mean = f"{row['treated_mean']:.2%}"
            control_mean = f"{row['control_mean']:.2%}"
            diff = _effect_pp(row["diff"])
            interval = _pp_interval(row["ci_low"], row["ci_high"])
        else:
            treated_mean = f"${row['treated_mean']:.3f}"
            control_mean = f"${row['control_mean']:.3f}"
            diff = f"${row['diff']:+.3f}"
            interval = f"[${row['ci_low']:.3f}, ${row['ci_high']:.3f}]"
        lines.append(
            f"| {row['outcome']} | {treated_mean} | {control_mean} | {diff} | {interval} |"
        )
    lines += [
        "",
        "## Covariate balance",
        "",
        "Standardised difference is the treated-minus-control mean divided by the pooled "
        "standard deviation. A value near zero is consistent with balance on that measured "
        "covariate; balance cannot test unmeasured causes.",
        "",
        "| covariate | treated mean | control mean | standardised difference |",
        "|---|---:|---:|---:|",
    ]
    for row in balance:
        value = row["standardised_diff"]
        diff = "undefined (zero within-group variance) ⚠" if value is None else f"{value:+.4f}"
        lines.append(
            f"| {row['covariate']} | {row['treated_mean']:.4f} | {row['control_mean']:.4f} | {diff} |"
        )
    lines += [
        "",
        f"Largest defined absolute standardised difference: {largest:.4f}; "
        f"{excessive} of {len(diffs)} defined covariates exceed 0.1. This is a descriptive "
        "randomization check. The constructed selection exercise in `reports/06_confounding_benchmark.md` "
        "shows how results move under specified selection mechanisms; differences from the "
        "full experiment are not known bias for a changed target population.",
    ]
    return "\n".join(lines) + "\n"


def render_regression_report(results: dict) -> str:
    estimates = _table_by(results["regression"]["estimates"], "outcome")
    naive = _table_by(results["naive"]["estimates"], "outcome")
    lines = [
        "# Regression baseline: logit and OLS with robust standard errors",
        "",
        "Logit models for `visit` and `conversion` use HC1-robust uncertainty and average "
        "counterfactual prediction differences for binary regressors, valid profile contrasts "
        "for categorical covariates, and average derivatives for continuous covariates. The "
        "treatment contrast sets assignment to 0 and 1 for every observed profile. OLS estimates "
        "the spend coefficient with HC1 uncertainty.",
        "",
        "## Treatment effects",
        "",
        "| outcome | adjusted contrast | 95% CI | naive difference |",
        "|---|---:|---:|---:|",
    ]
    for outcome in ("visit", "conversion", "spend"):
        row, base = estimates[outcome], naive[outcome]
        if outcome == "spend":
            effect = f"${row['effect']:+.4f}"
            interval = f"[${row['ci_low']:.4f}, ${row['ci_high']:.4f}]"
            naive_effect = f"${base['diff']:+.4f}"
        else:
            effect, interval, naive_effect = (
                _effect_pp(row["effect"]),
                _pp_interval(row["ci_low"], row["ci_high"]),
                _effect_pp(base["diff"]),
            )
        lines.append(f"| {outcome} | {effect} | {interval} | {naive_effect} |")
    for outcome in ("visit", "conversion"):
        rows = results["regression"]["logit_contrasts"][outcome]
        lines += [
            "",
            f"## Average logit contrasts: `{outcome}`",
            "",
            "Numeric covariates show average derivatives. Binary covariates use 0-to-1 "
            "counterfactual changes; categorical contrasts compare valid profiles to the "
            "reference level. These are associations/adjustment contrasts for covariates, "
            "not treatment effects.",
            "",
            "| contrast | calculation | estimate | 95% CI |",
            "|---|---|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['contrast']} | {row['kind']} | {_effect_pp(row['effect'])} | {_pp_interval(row['ci_low'], row['ci_high'])} |"
            )
    lines += [
        "",
        "The treatment contrast is an average discrete change in predicted probability. "
        "It is not the derivative of the logit probability with respect to treatment.",
    ]
    return "\n".join(lines) + "\n"


def render_identification_report(results: dict) -> str:
    outcomes = results["identification"]["outcomes"]
    lines = [
        "# Identification: what DoWhy recovers before any model is fit",
        "",
        "Identification asks whether the causal estimand follows from the assumed causal graph "
        "and observed distribution before choosing an estimator. DoWhy evaluates the graph in "
        "`src/dag.py`; it does not inspect the outcome values.",
        "",
        "## Result",
        "",
        "| outcome | estimand type | backdoor variables |",
        "|---|---|---|",
    ]
    for row in outcomes:
        backdoor = ", ".join(row["backdoor_variables"]) or "empty set"
        lines.append(f"| {row['outcome']} | {row['estimand_type']} | {backdoor} |")
    lines += [
        "",
        "Under the graph's randomized-assignment assumption, the backdoor set is empty "
        "and the estimand is the assignment-group difference. The graph encodes the "
        "experiment's design; the measured balance table is a descriptive check and cannot "
        "verify the absence of unmeasured causes. Identification and estimation remain "
        "separate: this step gives the target, while regression and DML compute estimates.",
    ]
    return "\n".join(lines) + "\n"


def render_dml_report(results: dict) -> str:
    pooled = _table_by(results["headline"]["pooled_ate"], "outcome")
    per_arm = results["headline"]["per_arm_ate"]
    regression = _table_by(results["regression"]["estimates"], "outcome")
    lines = [
        "# Average treatment effects via Double Machine Learning",
        "",
        "LinearDML uses LightGBM nuisance models and three-fold cross-fitting. Cross-fitting "
        "keeps each row's nuisance predictions out of that row's nuisance-model fit. On a "
        "randomized experiment, adjustment is not needed for identification, but flexible nuisance "
        "models can still overfit. LinearDML uses a linear final effect stage; flexible nuisance "
        "models alone do not guarantee recovery under arbitrary treatment-effect heterogeneity.",
        "",
        "## Pooled treatment: either email versus no email",
        "",
        "| outcome | DML effect | 95% CI | regression discrete change / coefficient |",
        "|---|---:|---:|---:|",
    ]
    for outcome in ("visit", "conversion", "spend"):
        row = pooled[outcome]
        reg = regression[outcome]
        if outcome == "spend":
            effect, interval, reg_effect = (
                f"${row['ate']:+.4f}",
                f"[${row['ci_low']:.4f}, ${row['ci_high']:.4f}]",
                f"${reg['effect']:+.4f}",
            )
        else:
            effect, interval, reg_effect = (
                _effect_pp(row["ate"]),
                _pp_interval(row["ci_low"], row["ci_high"]),
                _effect_pp(reg["effect"]),
            )
        lines.append(f"| {outcome} | {effect} | {interval} | {reg_effect} |")
    lines += [
        "",
        "## By creative versus no email",
        "",
        "| arm | outcome | DML effect | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in per_arm:
        if row["outcome"] == "spend":
            effect, interval = (
                f"${row['ate']:+.4f}",
                f"[${row['ci_low']:.4f}, ${row['ci_high']:.4f}]",
            )
        else:
            effect, interval = _effect_pp(row["ate"]), _pp_interval(row["ci_low"], row["ci_high"])
        lines.append(f"| {row['arm']} | {row['outcome']} | {effect} | {interval} |")
    lines += [
        "",
        "The pooled estimate averages the historical mixture of the two creatives. The "
        "separate arm estimates show why a decision analysis should preserve the three "
        "available actions. Differences in arm estimates are descriptive and do not by "
        "themselves establish customer-level personalization value.",
    ]
    return "\n".join(lines) + "\n"


def render_confounding_report(results: dict) -> str:
    confounding = results["confounding"]
    reference = confounding["experimental_reference"]
    variants = confounding["variants"]
    newbie = results["regression"]["logit_contrasts"]["visit"]
    newbie_effect = next(row["effect"] for row in newbie if row["contrast"] == "newbie")
    pp = _effect_pp
    lines = [
        "# Constructed selection diagnostics",
        "",
        f"The full randomized sample's DML estimate for `visit` is {pp(reference['ate'])} "
        f"[{pp(reference['ci_low'])}, {pp(reference['ci_high'])}]. This is an estimated "
        "experimental reference, not known truth for either selected sample. Selection changes "
        "the covariate distribution and can change the target-population effect. Differences "
        "below are descriptive selection diagnostics, not numerical bias estimates.",
        "",
        "Bias and coverage against known targets are evaluated in the fully synthetic Monte "
        "Carlo benchmark. LinearDML has flexible nuisance models but a linear final effect stage; "
        "neither property guarantees recovery under arbitrary effect heterogeneity.",
    ]
    for title, key, detail in (
        (
            "Selection on observed `recency` and `history`",
            "observable",
            "Both variables remain available to the estimators.",
        ),
        (
            "Selection on `newbie`, withheld from estimators",
            "unmeasured",
            "The fitted estimators omit `newbie` by construction.",
        ),
    ):
        variant = variants[key]
        lines += [
            "",
            f"## {title}",
            "",
            f"Retained {variant['n']:,} rows. {detail}",
            "",
            "| estimator | selected-sample estimate | difference from full-RCT estimate |",
            "|---|---:|---:|",
        ]
        for estimator, estimate in (
            ("Naive difference in means", variant["naive"]),
            ("OLS with selected covariates", variant["ols"]),
            ("LinearDML", variant["dml"]["ate"]),
        ):
            lines.append(f"| {estimator} | {pp(estimate)} | {pp(estimate - reference['ate'])} |")
        lines += [
            "",
            "These contrasts describe estimator movement under this constructed selection "
            "mechanism. They do not determine bias for the selected-population causal target.",
        ]
    lines += [
        "",
        f"In the visit logit, the `newbie` profile contrast is {pp(newbie_effect)}. "
        "This is an adjusted customer-profile contrast, not an effect of treatment.",
    ]
    return "\n".join(lines) + "\n"


def render_refutation_report(results: dict) -> str:
    reference = results["confounding"]["experimental_reference"]["ate"]
    lines = [
        "# Refutation checks: what these results do and do not establish",
        "",
        "Placebo treatment, random common cause, and data-subset checks challenge particular "
        "fitted estimates. Their outcomes do not certify identification or general estimator "
        "behavior.",
    ]
    for title, key in (
        ("Randomized Hillstrom sample", "rct"),
        ("Constructed selected sample", "selected_sample"),
    ):
        item = results["refutations"][key]
        lines += [
            "",
            f"## {title}",
            "",
            f"Reference estimate: {item['reference_estimate']:+.4f}.",
            "",
            "| check | result |",
            "|---|---:|",
        ]
        lines.append(
            f"| placebo treatment | {item['placebo']['ate']:+.4f} [{item['placebo']['ci_low']:+.4f}, {item['placebo']['ci_high']:+.4f}] |"
        )
        lines.append(
            f"| random common cause | {item['random_cause']['ate']:+.4f} [{item['random_cause']['ci_low']:+.4f}, {item['random_cause']['ci_high']:+.4f}] |"
        )
        subset = item["subset"]
        lines.append(
            f"| data subset ({len(subset)} refits) | mean {sum(subset) / len(subset):+.4f}, SD {float(__import__('numpy').std(subset)):.4f} |"
        )
        lines += [
            "",
            "These diagnostics show how this fitted estimate responds to the stated perturbations. "
            "For the selected sample, the full-RCT estimate (+"
            + f"{reference:.4f}) is a reference "
            "for a different population; the difference is descriptive, not a known-bias test.",
        ]
    lines += [
        "",
        "The fully synthetic simulation is where the generating probabilities, sample-average "
        "target, bias, and coverage are known. Refutation checks are diagnostics, not correctness certificates.",
    ]
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_model_readme() -> str:
    readme = MODEL_README_TEMPLATE_PATH.read_text(encoding="utf-8")
    marker = "Current artifact checksums:"
    prefix, separator, _old_hashes = readme.partition(marker)
    if not separator:
        raise ValueError(f"Heading not found: {marker}")
    hashes = "\n".join(f"- `{path.name}`: `{_sha256(path)}`" for path in MODEL_ARTIFACT_PATHS)
    return f"{prefix}{marker}\n\n{hashes}\n"


def render_readme_results(results: dict) -> str:
    pooled = _table_by(results["headline"]["pooled_ate"], "outcome")
    qini = results["ranking"]["normalized_qini"]
    repeats = results["ranking"]["repeated_summary"]
    policy = _table_by(results["policy"]["values"], "policy")
    comparison = _table_by(results["policy"]["comparisons"], "comparison")[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    lines = [
        "Pooled `LinearDML` effects for assignment to either email rather than no email:",
        "",
        "| outcome | DML ATE | 95% CI |",
        "|---|---:|---:|",
    ]
    for outcome in ("visit", "conversion", "spend"):
        row = pooled[outcome]
        if outcome == "spend":
            effect = f"+${row['ate']:.3f}"
            interval = f"[${row['ci_low']:.3f}, ${row['ci_high']:.3f}]"
        else:
            effect = _effect_pp(row["ate"])
            interval = _pp_interval(row["ci_low"], row["ci_high"])
        lines.append(f"| {outcome} | {effect} | {interval} |")
    lines += [
        "",
        "The pooled visit effect corresponds to about "
        f"{pooled['visit']['ate'] * 1000:.0f} additional visits per 1,000 customers assigned email. "
        "The forest produces varying conditional-effect predictions. "
        f"Normalized Qini is {qini['value']:.4f} "
        f"[{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]; across five honest splits it ranges "
        f"from {repeats['min']:.4f} to {repeats['max']:.4f}. "
        f"{_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')} "
        "The repository does not claim that individual uplift ordering is deployment-ready.",
        "",
        "Three-action policies are evaluated with cross-fitted doubly robust scores:",
        "",
        "| policy | expected visit rate | 95% CI |",
        "|---|---:|---:|",
    ]
    for name in (
        "learned (DRPolicyForest)",
        "email everyone (mens creative)",
        "email everyone (womens creative)",
        "purchase-history heuristic",
        "email nobody",
    ):
        row = policy[name]
        lines.append(
            f"| {name} | {row['value']:.2%} | {_percent_interval(row['ci_low'], row['ci_high'])} |"
        )
    lines += [
        "",
        f"Learned minus blanket mens is {_effect_pp(comparison['difference'])} "
        f"{_pp_interval(comparison['ci_low'], comparison['ci_high'])}. "
        f"{_policy_conclusion(comparison)}",
    ]
    return "\n".join(lines)


def render_readme_run() -> str:
    return """```powershell
py -3.12 -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m streamlit run app/simulator.py
```

On macOS or Linux, use `.venv/bin/python` in place of the Windows path. The committed manifest
and serving artifacts support the quick start. To rebuild all estimates, figures, reports, model
artifacts and PDF, run `.venv\\Scripts\\python.exe -m src.pipeline`. The optional 500-repetition
synthetic benchmark can be run separately with `-m src.simulation --repetitions 500`.
`scripts/verify.ps1` runs the pinned-environment quality gate."""


def render_readme_status(results: dict) -> str:
    qini = results["ranking"]["normalized_qini"]
    spend = results["ranking"]["gross_spend_top_30"]
    comparison = _table_by(results["policy"]["comparisons"], "comparison")[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    return f"""The average treatment effect is well identified by the randomised design. Targeting
evidence from this held-out evaluation is summarised below:

- The normalized Qini ranking score is {qini["value"]:.4f}
  [{qini["ci_low"]:.4f}, {qini["ci_high"]:.4f}]. {_interval_conclusion(qini["ci_low"], qini["ci_high"], "a positive ranking advantage")}
- Learned minus blanket mens emailing is {_effect_pp(comparison["difference"])}
  {_pp_interval(comparison["ci_low"], comparison["ci_high"])}. {_policy_conclusion(comparison)}

Reported spend among the top-30% diagnostic is ${spend["value"]:.4f}
[${spend["ci_low"]:.4f}, ${spend["ci_high"]:.4f}], but Hillstrom supplies neither gross margin nor
uncapped spend. It is therefore a gross-spend sensitivity, not a profit estimate. The experiment
covers one US retailer in March 2008; its effect sizes do not transfer to a 2026 South African bank
or telecoms campaign."""


def render_uplift_report(results: dict) -> str:
    interaction_rows = results["interactions"]["segment_effects"]
    contrast_rows = results["interactions"]["contrasts"]
    ranking = results["ranking"]
    qini = ranking["normalized_qini"]
    repeat = ranking["repeated_summary"]
    policy_values = results["policy"]["values"]
    comparisons = results["policy"]["comparisons"]
    lines = [
        "# Heterogeneous effects and targeting policy",
        "",
        "## Interaction check",
        "",
        "This post-hoc held-out interaction audit uses HC1-robust OLS with mens-only as the",
        "observed reference segment. Direct segment effects are evaluated at mean recency and",
        f"history; the joint Wald p-value is {results['interactions']['joint_p_value']:.3g}.",
        "",
        "| segment | effect on visit | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in interaction_rows:
        lines.append(
            f"| {row['segment']} | {_effect_pp(row['estimate'])} | "
            f"{_pp_interval(row['ci_low'], row['ci_high'])} |"
        )
    lines += [
        "",
        "The effect-modification contrasts are reported below with Holm-adjusted p-values across",
        "the four reported contrasts. This is a post-hoc exploratory audit defined after the primary analysis.",
        "",
        "| contrast | estimate | 95% CI | Holm-adjusted p |",
        "|---|---:|---:|---:|",
    ]
    for row in contrast_rows:
        lines.append(
            f"| {row['contrast']} | {_effect_pp(row['estimate'])} | "
            f"{_pp_interval(row['ci_low'], row['ci_high'])} | {row['p_holm']:.4g} |"
        )
    lines += [
        "",
        "",
        "## Ranking evidence",
        "",
        f"Raw Qini area is {ranking['raw_qini']:.2f}. The comparable normalized score is "
        f"{qini['value']:.4f} [{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. Across five honest "
        f"splits the score averages {repeat['mean']:.4f}, with a range from {repeat['min']:.4f} "
        f"to {repeat['max']:.4f}. {_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')} "
        "The CATE surface "
        "may contain real segment-level structure while still failing to rank individual customers "
        "reliably enough for deployment.",
        "",
        "| decile (0 = highest predicted uplift) | predicted CATE | observed uplift | n |",
        "|---|---:|---:|---:|",
    ]
    for row in ranking["deciles"]:
        lines.append(
            f"| {row['decile']} | {_effect_pp(row['pred_cate_mean'])} | "
            f"{_effect_pp(row['observed_uplift'])} | {row['n']:,} |"
        )
    lines += [
        "",
        "## Pooled-email targeting diagnostic",
        "",
        "The table estimates the treated-minus-control visit difference inside each selected",
        "segment. It evaluates assignment to the historical mixture of two creatives, not a",
        "creative-specific action.",
        "",
        "| top-k targeted | visit-rate difference | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in ranking["top_k"]:
        lines.append(
            f"| {row['k']:.0%} | {_effect_pp(row['value'])} | "
            f"{_pp_interval(row['ci_low'], row['ci_high'])} |"
        )
    spend = ranking["gross_spend_top_30"]
    lines += [
        "",
        f"At 30%, reported gross spend is ${spend['value']:.4f} "
        f"[${spend['ci_low']:.4f}, ${spend['ci_high']:.4f}] per targeted customer. Hillstrom "
        "does not provide margin, and twelve spend observations sit at the dataset maximum of "
        "$499. This cannot be read as profit or break-even evidence.",
        "",
        "## Three-action policy",
        "",
        "Values use cross-fitted outcome models, nominal one-third randomisation probabilities and",
        "doubly robust scores. Bootstrap samples preserve the arm counts; intervals are fixed-model",
        "conditional evaluation intervals.",
        "",
        "| policy | visit-rate value | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in policy_values:
        lines.append(
            f"| {row['policy']} | {row['value']:.2%} | "
            f"{_percent_interval(row['ci_low'], row['ci_high'])} |"
        )
    spend = results["reported_spend_sensitivity"]
    lines += [
        "",
        "The learned policy was trained to maximise visits, not spend or profit. Reported-spend",
        "sensitivity is shown separately because margin and email cost are not identified by this",
        "experiment.",
        "",
        "| gross margin assumption | incremental net value vs email nobody |",
        "|---:|---:|",
    ]
    for row in spend["margin_sensitivity"]:
        lines.append(f"| {row['gross_margin']:.0%} | ${row['incremental_net_value']:+.4f} |")
    lines += [
        "",
        "| paired comparison | difference | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {_effect_pp(row['difference'])} | "
            f"{_pp_interval(row['ci_low'], row['ci_high'])} |"
        )
    learned_mens = _table_by(comparisons, "comparison")[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    lines += ["", _policy_conclusion(learned_mens)]
    lines += [
        "",
        "## Fully synthetic estimator benchmark",
        "",
        "The original fast DML check uses ten repetitions per confounding level. Its target "
        "is the sample average of the generated probability differences, `mean(p1 - p0)`. "
        "The interval below reports Monte Carlo uncertainty in mean bias; coverage is the "
        "share of run-level 95% intervals covering that known target.",
        "",
        "| confounding strength | estimator | mean bias | 95% Monte Carlo CI for bias | coverage | Wilson 95% CI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in results["simulation"]["rows"]:
        bias_ci = (
            "unavailable"
            if row["bias_mc_ci_low"] is None
            else f"[{row['bias_mc_ci_low']:+.4f}, {row['bias_mc_ci_high']:+.4f}]"
        )
        coverage = "unavailable" if row["coverage"] is None else f"{row['coverage']:.1%}"
        coverage_ci = (
            "unavailable"
            if row["coverage_ci_low"] is None
            else f"[{row['coverage_ci_low']:.1%}, {row['coverage_ci_high']:.1%}]"
        )
        lines.append(
            f"| {row['confounding_strength']:.2f} | {row['estimator']} | {row['bias']:+.4f} | {bias_ci} | {coverage} | {coverage_ci} |"
        )
    quick = results["simulation"]["rows"]
    strongest_quick = max(row["confounding_strength"] for row in quick)
    quick_strong = {
        row["estimator"]: row for row in quick if row["confounding_strength"] == strongest_quick
    }
    naive_bias = quick_strong.get("naive", {}).get("bias")
    adjusted_bias = quick_strong.get("adjusted_dml", {}).get("bias")
    if naive_bias is not None and adjusted_bias is not None:
        if abs(adjusted_bias) < abs(naive_bias):
            lines += [
                "",
                f"At the strongest simulated confounding level ({strongest_quick:.2f}), the fast adjusted DML run has smaller absolute mean bias ({adjusted_bias:+.4f}) than the naive run ({naive_bias:+.4f}); its remaining error is still reported with Monte Carlo uncertainty. This is improvement in this finite simulation, not automatic recovery under arbitrary effect surfaces.",
            ]
        else:
            lines += [
                "",
                f"At the strongest simulated confounding level ({strongest_quick:.2f}), the fast adjusted DML run does not have smaller absolute mean bias ({adjusted_bias:+.4f}) than the naive run ({naive_bias:+.4f}). Results remain specific to this finite simulation and its estimator specification.",
            ]
    lines += [
        "",
        f"## Extended fully synthetic benchmark ({results['simulation']['extended_repetitions']} repetitions)",
        "",
        f"The following {results['simulation']['extended_repetitions']}-repetition Monte Carlo "
        "uses known potential-outcome probabilities. The target is each generated sample's "
        "average `p1 - p0`. Bias intervals show Monte Carlo uncertainty across repetitions; "
        "coverage intervals are Wilson intervals for the share of run-specific 95% intervals "
        "covering the target.",
        "",
        "| confounding strength | estimator | mean estimate | bias | 95% Monte Carlo CI for bias | coverage | Wilson 95% CI | RMSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["simulation"]["extended_rows"]:
        bias_ci = (
            "unavailable"
            if row["bias_mc_ci_low"] is None
            else f"[{row['bias_mc_ci_low']:+.4f}, {row['bias_mc_ci_high']:+.4f}]"
        )
        coverage = "unavailable" if row["coverage"] is None else f"{row['coverage']:.1%}"
        coverage_ci = (
            "unavailable"
            if row["coverage_ci_low"] is None
            else f"[{row['coverage_ci_low']:.1%}, {row['coverage_ci_high']:.1%}]"
        )
        lines.append(
            f"| {row['confounding_strength']:.2f} | {row['estimator']} | {row['mean_estimate']:.4f} | {row['bias']:+.4f} | {bias_ci} | {coverage} | {coverage_ci} | {row['rmse']:.4f} |"
        )
    lines += [
        "",
        "Naive is the unadjusted estimator; adjusted AIPW uses cross-fitted estimated "
        "propensity and outcome models; omitted-confounder AIPW intentionally excludes "
        "the confounder; oracle AIPW uses generating nuisance probabilities as a diagnostic. "
        "This is a synthetic benchmark, not evidence that a particular observational study is identified.",
    ]
    dedup = results["headline"]["deduplicated_visit_sensitivity"]
    lines += [
        "",
        "## Additional sensitivity checks",
        "",
        "| check | configuration | result |",
        "|---|---|---:|",
    ]
    for row in ranking["forest_sensitivity"]:
        depth = "uncapped" if row["max_depth"] is None else str(row["max_depth"])
        lines.append(
            f"| forest | {row['label']}: min leaf {row['min_samples_leaf']}, max depth {depth} | normalized Qini {row['normalized_qini']:.4f} |"
        )
    lines.append(
        f"| duplicate rows | {dedup['deduplicated_rows']:,} rows after exact-row deduplication (from {dedup['full_rows']:,}) | visit difference {_effect_pp(dedup['deduplicated_visit_difference'])}; full {_effect_pp(dedup['full_visit_difference'])} |"
    )
    lines += [
        "",
        "| policy split seed | learned value | blanket mens | learned minus blanket | action shares: no email / mens / womens |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in results["policy"]["split_sensitivity"]:
        shares = row["recommendation_shares"]
        lines.append(
            f"| {row['seed']} | {row['learned_value']:.2%} | {row['blanket_mens_value']:.2%} | {_effect_pp(row['learned_minus_blanket_mens'])} | {shares.get('No E-Mail', 0):.1%} / {shares.get('Mens E-Mail', 0):.1%} / {shares.get('Womens E-Mail', 0):.1%} |"
        )
    return "\n".join(lines) + "\n"


def render_interview_answers(results: dict) -> dict[int, str]:
    pooled = _table_by(results["headline"]["pooled_ate"], "outcome")["visit"]
    cate = results["ranking"]["cate_summary"]
    policy = _table_by(results["policy"]["values"], "policy")
    comparison = _table_by(results["policy"]["comparisons"], "comparison")[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    qini = results["ranking"]["normalized_qini"]
    repeated = results["ranking"]["repeated_summary"]
    top30 = results["ranking"]["top_k"][2]
    spend_comparisons = _table_by(
        results["reported_spend_sensitivity"]["comparisons"], "comparison"
    )
    learned_no_email = spend_comparisons["learned (DRPolicyForest) - email nobody"]
    learned_mens_spend = spend_comparisons[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    return {
        1: (
            "ATE averages the effect over the target population; ATT averages it over treated "
            "customers; CATE conditions on a covariate profile. Random assignment makes treated "
            "and control customers represent the same population in expectation, though finite-"
            "sample ATE and ATT need not be algebraically identical. The pooled DML ATE on visit "
            f"is {_effect_pp(pooled['ate'])} {_pp_interval(pooled['ci_low'], pooled['ci_high'])}. "
            f"The held-out forest's mean profile CATE is {_effect_pp(cate['mean'])}, ranging from "
            f"{_effect_pp(cate['min'])} to {_effect_pp(cate['max'])}. Those are model-based conditional averages, "
            "not individual treatment effects."
        ),
        7: (
            "Cross-fitted doubly robust evaluation gives the learned policy "
            f"{policy['learned (DRPolicyForest)']['value']:.2%} and blanket mens emailing "
            f"{policy['email everyone (mens creative)']['value']:.2%}. Their paired difference is "
            f"{_effect_pp(comparison['difference'])} {_pp_interval(comparison['ci_low'], comparison['ci_high'])}. "
            f"{_policy_conclusion(comparison)}"
        ),
        10: (
            f"The top-30% pooled-email diagnostic is {_effect_pp(top30['value'])} "
            f"{_pp_interval(top30['ci_low'], top30['ci_high'])} in visit-rate difference per emailed "
            "customer. Reported spend is a separate policy sensitivity: learned minus no email is "
            f"${learned_no_email['difference']:+.3f} [${learned_no_email['ci_low']:+.3f}, "
            f"${learned_no_email['ci_high']:+.3f}], and learned minus blanket mens is "
            f"${learned_mens_spend['difference']:+.3f} [${learned_mens_spend['ci_low']:+.3f}, "
            f"${learned_mens_spend['ci_high']:+.3f}]. Reported spend may be top-coded at $499; "
            "gross margin and uncapped spend are unavailable, so these are not profit estimates."
        ),
        11: (
            "Raw Qini is the area between cumulative incremental gain under the model ranking and "
            f"random targeting; here it is {results['ranking']['raw_qini']:.2f}, which depends on sample size. "
            f"The normalized score is {qini['value']:.4f} [{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. "
            f"Five honest sample splits range from {repeated['min']:.4f} to {repeated['max']:.4f}; "
            "these are sensitivity evidence, not independent replications. The interval is "
            "conditional on the fitted ranking. "
            f"{_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')} "
            "This conditional interval does not itself validate transport to a new campaign."
        ),
        15: (
            "I would run a new campaign designed around action-level trade-offs and validate the "
            "policy on another campaign. Positive average effects for both creatives would not rule "
            "out personalization: relative effects could cross across customers. This analysis "
            "shows exploratory segment differences. "
            f"{_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')} "
            f"{_policy_conclusion(comparison)} I would strengthen the existing fully synthetic known-truth "
            "benchmark across confounding strengths and test transport across campaigns before "
            "revisiting sparse-outcome spend CATEs."
        ),
    }


def render_causal_intro(results: dict) -> str:
    pooled = _table_by(results["headline"]["pooled_ate"], "outcome")["visit"]
    qini = results["ranking"]["normalized_qini"]
    comparison = _table_by(results["policy"]["comparisons"], "comparison")[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    return (
        "Hillstrom's 64,000-customer randomized experiment estimates an average visit-rate effect "
        f"of {_effect_pp(pooled['ate'])} {_pp_interval(pooled['ci_low'], pooled['ci_high'])}, "
        f"about {pooled['ate'] * 1000:.0f} additional visits per 1,000 customers assigned email. "
        f"The normalized Qini estimate is {qini['value']:.4f} [{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. "
        f"Learned minus blanket mens policy value is {_effect_pp(comparison['difference'])} "
        f"{_pp_interval(comparison['ci_low'], comparison['ci_high'])}. "
        "The evidence supports an average effect. "
        f"{_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')} "
        f"{_policy_conclusion(comparison)}"
    )


def _causal_sections(results: dict) -> dict[str, str]:
    segment_effects = _table_by(results["interactions"]["segment_effects"], "segment")
    contrasts = _table_by(results["interactions"]["contrasts"], "contrast")
    ranking = results["ranking"]
    qini = ranking["normalized_qini"]
    values = _table_by(results["policy"]["values"], "policy")
    comparisons = _table_by(results["policy"]["comparisons"], "comparison")
    learned_gap = comparisons["learned (DRPolicyForest) - email everyone (mens creative)"]
    naive = _table_by(results["naive"]["estimates"], "outcome")
    regression = _table_by(results["regression"]["estimates"], "outcome")
    dml = _table_by(results["headline"]["pooled_ate"], "outcome")
    conf = results["confounding"]
    variants = conf["variants"]
    refutations = results["refutations"]
    balance = results["naive"]["balance"]
    finite_balance = [
        row["standardised_diff"] for row in balance if row["standardised_diff"] is not None
    ]
    max_balance = max((abs(value) for value in finite_balance), default=0.0)
    spend_comparisons = _table_by(
        results["reported_spend_sensitivity"]["comparisons"], "comparison"
    )
    learned_no_email = spend_comparisons["learned (DRPolicyForest) - email nobody"]
    learned_mens_spend = spend_comparisons[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    ladder = [
        "The treatment effects below are expressed in percentage points for binary outcomes and dollars for spend. Agreement is expected under randomized assignment; it does not establish that DML is superior. The adjusted logit treatment effect is an average discrete prediction change. Continuous covariates use average derivatives, binary covariates use 0-to-1 changes, and categorical covariates use valid profile contrasts. LinearDML allows effect modification through a final effect model, which is linear here. Interactions can also estimate heterogeneity. Randomization does not prevent nuisance-model overfitting, so cross-fitting remains useful.",
        "",
        "| method | visit | conversion | spend |",
        "|---|---:|---:|---:|",
        f"| naive difference in means | {_effect_pp(naive['visit']['diff'])} | {_effect_pp(naive['conversion']['diff'])} | ${naive['spend']['diff']:+.3f} |",
        f"| adjusted logit/OLS | {_effect_pp(regression['visit']['effect'])} | {_effect_pp(regression['conversion']['effect'])} | ${regression['spend']['effect']:+.3f} |",
        f"| LinearDML | {_effect_pp(dml['visit']['ate'])} | {_effect_pp(dml['conversion']['ate'])} | ${dml['spend']['ate']:+.3f} |",
    ]
    constructed = [
        f"The full randomized sample's estimated DML visit effect is {_effect_pp(conf['experimental_reference']['ate'])} {_pp_interval(conf['experimental_reference']['ci_low'], conf['experimental_reference']['ci_high'])}. Selected-sample comparisons below are descriptive: selection changes the covariate distribution and can change the target-population effect. They are not known-bias estimates. The fully synthetic benchmark evaluates bias and coverage against known sample-average potential-outcome targets. LinearDML's flexible nuisance functions do not remove the restriction of its linear final effect stage.",
        "",
        "| selection variant | estimator | estimate | difference from full-RCT reference |",
        "|---|---|---:|---:|",
    ]
    for label, key in (
        ("recency/history observed", "observable"),
        ("newbie withheld", "unmeasured"),
    ):
        for estimator, estimate in (
            ("naive", variants[key]["naive"]),
            ("OLS", variants[key]["ols"]),
            ("LinearDML", variants[key]["dml"]["ate"]),
        ):
            constructed.append(
                f"| {label} | {estimator} | {_effect_pp(estimate)} | {_effect_pp(estimate - conf['experimental_reference']['ate'])} |"
            )
    heterogeneity = [
        "A held-out HC1-robust interaction audit finds exploratory segment-level differences. These are average effects within observed segments, not individual effects.",
        "",
        "| segment | visit effect | 95% CI |",
        "|---|---:|---:|",
    ]
    for name in ("mens only", "womens only", "both"):
        row = segment_effects[name]
        heterogeneity.append(
            f"| {name} | {_effect_pp(row['estimate'])} | {_pp_interval(row['ci_low'], row['ci_high'])} |"
        )
    heterogeneity += [
        "",
        f"The womens-only minus mens-only contrast has Holm-adjusted p={contrasts['womens only - mens only']['p_holm']:.3g}; the joint Wald p-value is {results['interactions']['joint_p_value']:.3g}. These post-hoc checks do not show that one customer-level ranking is reliable.",
    ]
    uplift = [
        f"Normalized Qini is {qini['value']:.4f} [{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. Across honest splits it ranges from {ranking['repeated_summary']['min']:.4f} to {ranking['repeated_summary']['max']:.4f}; repeated splits are sensitivity evidence, not independent replications. {_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')}",
        f"The top-30% pooled-email visit difference is {_effect_pp(ranking['top_k'][2]['value'])} {_pp_interval(ranking['top_k'][2]['ci_low'], ranking['top_k'][2]['ci_high'])} per emailed customer. This diagnostic concerns the historical creative mixture and does not select between creatives.",
        "",
        f"Reported-spend policy sensitivity is learned minus no email: ${learned_no_email['difference']:+.3f} [${learned_no_email['ci_low']:+.3f}, ${learned_no_email['ci_high']:+.3f}], and learned minus blanket mens: ${learned_mens_spend['difference']:+.3f} [${learned_mens_spend['ci_low']:+.3f}, ${learned_mens_spend['ci_high']:+.3f}]. Reported spend may be top-coded at $499; no gross margin or uncapped outcome is available, so these are not profit estimates.",
    ]
    policy_section = [
        "Cross-fitted doubly robust evaluation estimates expected visit rates under three-action policies. Intervals are paired fixed-model conditional evaluation intervals.",
        "",
        "| policy | expected visit rate | 95% CI |",
        "|---|---:|---:|",
    ]
    for name in (
        "learned (DRPolicyForest)",
        "email everyone (mens creative)",
        "email everyone (womens creative)",
        "purchase-history heuristic",
        "email nobody",
    ):
        row = values[name]
        policy_section.append(
            f"| {name} | {row['value']:.2%} | {_percent_interval(row['ci_low'], row['ci_high'])} |"
        )
    policy_section += [
        "",
        f"Learned minus blanket mens is {_effect_pp(learned_gap['difference'])} {_pp_interval(learned_gap['ci_low'], learned_gap['ci_high'])}. {_policy_conclusion(learned_gap)}",
    ]
    refutation = [
        "Placebo treatment, random common cause, and data-subset refits diagnose responses to specific perturbations. They do not test whether the causal identification assumptions hold and do not certify correctness.",
    ]
    for label, key in (("Randomized sample", "rct"), ("Selected sample", "selected_sample")):
        row = refutations[key]
        refutation += ["", f"### {label}", "", "| check | estimate / summary |", "|---|---:|"]
        refutation.append(
            f"| placebo treatment | {row['placebo']['ate']:+.4f} [{row['placebo']['ci_low']:+.4f}, {row['placebo']['ci_high']:+.4f}] |"
        )
        refutation.append(
            f"| random common cause | {row['random_cause']['ate']:+.4f} [{row['random_cause']['ci_low']:+.4f}, {row['random_cause']['ci_high']:+.4f}] |"
        )
        refutation.append(
            f"| data subset | mean {sum(row['subset']) / len(row['subset']):+.4f}, SD {float(__import__('numpy').std(row['subset'])):.4f} |"
        )
    refutation += [
        "",
        "The selected-sample estimate is compared with an estimated full-RCT reference for a different population. That difference is descriptive, not an omitted-confounder bias demonstration. Claims about bias and coverage against known targets are limited to the fully synthetic benchmark.",
    ]
    limitations = [
        "- The study is one US retailer campaign from March 2008. Its effect sizes do not establish what would happen in another country, retailer, or current campaign.",
        "- Spend is reported and may be top-coded at $499. The maximum alone does not establish the cap mechanism or motive. The uncapped-spend effect and profitability are unidentified here.",
        "- The observed order of visit, conversion, and spend is compatible with mediation, but sequence alone does not establish a causal graph. This analysis estimates total effects and does not decompose mediation.",
        "- The constructed real-data selection examples show estimates under specified mechanisms. They do not establish bias under omitted confounding in an observational target population.",
        f"- Segment contrasts are exploratory. Qini evidence: {_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')} Policy evidence: {_policy_conclusion(learned_gap)}",
    ]
    transfer = [
        "The estimation and validation workflow can inform a new randomized campaign, but the Hillstrom effect sizes and policy ranking do not transfer automatically. A local campaign should define its outcome, eligible actions, costs, target population, assignment probabilities, and follow-up window; estimate effects under its own design; and validate any learned policy on held-out or later campaign data. Observational use would additionally require estimated propensities, overlap checks, suitable adjusted estimators, and explicit assumptions. The current benchmark does not establish those assumptions for another dataset.",
    ]
    return {
        "## 1. The causal question and why identification is clean here": (
            "The assumed graph has no causes of treatment assignment other than the randomizer. "
            "Under the documented randomized design, the pooled email-versus-control comparison "
            "identifies an intention-to-treat effect. The graph states the design assumption; the "
            "measured balance table is a descriptive finite-sample check and cannot prove the "
            "absence of unmeasured causes. DoWhy derives an empty backdoor adjustment set under "
            "this graph."
        ),
        "## 2. The naive estimate, and the check that it is allowed to be naive": (
            "Under random assignment, the difference in means estimates the effect of assignment "
            "to the historical email mixture. The balance table is descriptive evidence about "
            "measured covariates, not proof of the assignment mechanism. The estimated effects are "
            f"visit {_effect_pp(naive['visit']['diff'])} {_pp_interval(naive['visit']['ci_low'], naive['visit']['ci_high'])}, "
            f"conversion {_effect_pp(naive['conversion']['diff'])} {_pp_interval(naive['conversion']['ci_low'], naive['conversion']['ci_high'])}, "
            f"and spend ${naive['spend']['diff']:+.3f} [${naive['spend']['ci_low']:.3f}, ${naive['spend']['ci_high']:.3f}]. "
            f"The largest defined absolute standardized difference is {max_balance:.4f}."
        ),
        "## 3. The methods ladder: naive, regression, and Double ML agree": "\n".join(ladder),
        "## 4. Constructed confounding stress test": "\n".join(constructed),
        "## 5. Heterogeneity: who the email actually helps": "\n".join(heterogeneity),
        "## 6. Uplift ranking and targeting economics": "\n".join(uplift),
        "## 7. Three actions, not two: an honest result": "\n".join(policy_section),
        "## 8. Refutation tests, and what they do not prove": "\n".join(refutation),
        "## 9. Limitations, stated rather than buried": "\n".join(limitations),
        "## 10. What would and would not transfer to a South African retention campaign": "\n".join(
            transfer
        ),
        "## Closing": (
            "The project estimates an average effect from a randomized experiment, examines "
            "exploratory treatment-effect variation, and evaluates a learned action policy. "
            "The average effect is supported by the design. "
            f"{_interval_conclusion(qini['ci_low'], qini['ci_high'], 'a positive ranking advantage')} "
            f"{_policy_conclusion(learned_gap)} The synthetic benchmark checks estimator "
            "behavior against known generated targets, while the selected-sample examples and "
            "refutation diagnostics remain limited to descriptive stress tests."
        ),
    }


def _references() -> str:
    return """## References

- Hillstrom, K. (2008), [MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/05/best-answer-e-mail-analytics-challenge.html).
- Radcliffe, N. J. and Surry, P. D. (2011), [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf).
- Sharma, A. and Kiciman, E. (2020), [DoWhy: An End-to-End Library for Causal Inference](https://arxiv.org/abs/2011.04216).
- Microsoft Research, [EconML 0.16 documentation](https://econml.azurewebsites.net/), including `CausalForestDML` and `DRPolicyForest`."""


def _rendered_outputs(results: dict) -> dict[Path, str]:
    readme = README_PATH.read_text(encoding="utf-8")
    readme = replace_markdown_section(readme, "## Results", render_readme_results(results))
    readme = replace_markdown_section(readme, "## How to run it", render_readme_run())
    readme = replace_markdown_section(readme, "## Status", render_readme_status(results))
    report = _clean_causal_report_template(CAUSAL_REPORT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    report = replace_markdown_section(
        report,
        "## Can causal ML find deployable treatment-effect heterogeneity, or only a reliable average effect?",
        render_causal_intro(results),
    )
    for heading, body in _causal_sections(results).items():
        report = replace_markdown_section(report, heading, body)
    if "## References" in report:
        report = re.sub(
            r"(?ms)^## References\n.*?(?=\n---\n)",
            _references().strip(),
            report,
        )
    else:
        report = report.replace("\n---\n", f"\n\n{_references()}\n\n---\n")
    old_inventory = "`reports/01` through `reports/" + "09`"
    report = report.replace(old_inventory, "`reports/01` through `reports/08`")

    interview = INTERVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")
    for number, answer in render_interview_answers(results).items():
        heading_match = re.search(rf"(?m)^## {number}\. .+$", interview)
        if not heading_match:
            raise ValueError(f"Interview question {number} not found.")
        interview = replace_markdown_section(interview, heading_match.group(), answer)
    return {
        README_PATH: readme,
        NAIVE_REPORT_PATH: render_naive_report(results),
        REGRESSION_REPORT_PATH: render_regression_report(results),
        IDENTIFICATION_REPORT_PATH: render_identification_report(results),
        DML_REPORT_PATH: render_dml_report(results),
        CONFOUNDING_REPORT_PATH: render_confounding_report(results),
        REFUTATION_REPORT_PATH: render_refutation_report(results),
        UPLIFT_REPORT_PATH: render_uplift_report(results),
        CAUSAL_REPORT_PATH: report,
        INTERVIEW_PATH: interview,
        MODEL_README_PATH: render_model_readme(),
    }


def outputs_are_current() -> bool:
    try:
        results = load_results()
        for path, expected in _rendered_outputs(results).items():
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                return False
        for figure in results["figures"]["files"]:
            path = FIGURES_DIR / figure["path"]
            if not path.exists() or _sha256(path) != figure["sha256"]:
                return False
        return True
    except (OSError, ValueError):
        return False


def render_all() -> None:
    for path, rendered in _rendered_outputs(load_results()).items():
        path.write_text(rendered, encoding="utf-8")


def main() -> None:
    render_all()
    print(f"rendered {README_PATH}, {UPLIFT_REPORT_PATH}, and {CAUSAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
