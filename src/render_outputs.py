"""Render public Markdown from the versioned results manifest."""

import re
from pathlib import Path

from src.config import REPORTS_DIR, ROOT
from src.results import load_results

README_PATH = ROOT / "README.md"
UPLIFT_REPORT_PATH = REPORTS_DIR / "07_uplift_policy.md"
CAUSAL_REPORT_PATH = REPORTS_DIR / "causal_report.md"
INTERVIEW_PATH = REPORTS_DIR / "interview_qa.md"


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


def _table_by(records: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in records}


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
            effect = f"{row['ate']:+.2%}"
            interval = f"[{row['ci_low']:.2%}, {row['ci_high']:.2%}]"
        lines.append(f"| {outcome} | {effect} | {interval} |")
    lines += [
        "",
        "The forest finds stable variation in predicted CATEs, but its ranking is weak. "
        f"Normalized Qini is {qini['value']:.4f} "
        f"[{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]; across five honest splits it ranges "
        f"from {repeats['min']:.4f} to {repeats['max']:.4f}. The interval includes zero, so the "
        "repo does not claim that individual uplift ordering is deployment-ready.",
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
            f"| {name} | {row['value']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines += [
        "",
        f"Learned minus blanket mens is {comparison['difference']:+.4f} "
        f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]. "
        "That is not a policy win. Blanket mens emailing is the simpler action supported by this "
        "experiment; personalised deployment needs new evidence.",
    ]
    return "\n".join(lines)


def render_readme_run() -> str:
    return """```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline
streamlit run app/simulator.py
```

On macOS or Linux, activate with `source .venv/bin/activate`. The pipeline regenerates the
manifest, serving artifacts, Markdown reports, figures, production model and PDF in dependency
order. Run `pytest -v` for tests and `ruff check . && ruff format --check .` for the code-quality
gate."""


def render_readme_status(results: dict) -> str:
    qini = results["ranking"]["normalized_qini"]
    spend = results["ranking"]["gross_spend_top_30"]
    return f"""The average treatment effect is well identified by the randomised design. Two narrower
claims are not established:

- The uplift ranking is weak: normalized Qini {qini["value"]:.4f}
  [{qini["ci_low"]:.4f}, {qini["ci_high"]:.4f}] includes zero.
- The learned policy does not beat blanket mens emailing on the held-out sample.

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
            f"| {row['segment']} | {row['estimate']:+.5f} | "
            f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] |"
        )
    lines += [
        "",
        "The effect-modification contrasts are reported below with Holm-adjusted p-values across",
        "the four reported contrasts. This is a post-hoc exploratory audit, not a pre-specified test.",
        "",
        "| contrast | estimate | 95% CI | Holm-adjusted p |",
        "|---|---:|---:|---:|",
    ]
    for row in contrast_rows:
        lines.append(
            f"| {row['contrast']} | {row['estimate']:+.5f} | "
            f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] | {row['p_holm']:.4g} |"
        )
    lines += [
        "",
        "",
        "## Ranking evidence",
        "",
        f"Raw Qini area is {ranking['raw_qini']:.2f}. The comparable normalized score is "
        f"{qini['value']:.4f} [{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. Across five honest "
        f"splits the score averages {repeat['mean']:.4f}, with a range from {repeat['min']:.4f} "
        f"to {repeat['max']:.4f}. The primary fixed-model conditional bootstrap interval includes "
        "zero. The CATE surface "
        "may contain real segment-level structure while still failing to rank individual customers "
        "reliably enough for deployment.",
        "",
        "| decile (0 = highest predicted uplift) | predicted CATE | observed uplift | n |",
        "|---|---:|---:|---:|",
    ]
    for row in ranking["deciles"]:
        lines.append(
            f"| {row['decile']} | {row['pred_cate_mean']:+.4f} | "
            f"{row['observed_uplift']:+.4f} | {row['n']:,} |"
        )
    lines += [
        "",
        "## Pooled-email targeting diagnostic",
        "",
        "The table estimates the treated-minus-control visit difference inside each selected",
        "segment. It evaluates assignment to the historical mixture of two creatives, not a",
        "creative-specific action.",
        "",
        "| top-k targeted | visits per customer emailed | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in ranking["top_k"]:
        lines.append(
            f"| {row['k']:.0%} | {row['value']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
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
            f"| {row['policy']} | {row['value']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
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
            f"| {row['comparison']} | {row['difference']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    lines += ["", results["policy"]["conclusion"]]
    return "\n".join(lines) + "\n"


def render_interview_answers(results: dict) -> dict[int, str]:
    pooled = _table_by(results["headline"]["pooled_ate"], "outcome")["visit"]
    cate = results["ranking"]["cate_summary"]
    policy = _table_by(results["policy"]["values"], "policy")
    comparison = _table_by(results["policy"]["comparisons"], "comparison")[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    spend = results["ranking"]["gross_spend_top_30"]
    qini = results["ranking"]["normalized_qini"]
    repeated = results["ranking"]["repeated_summary"]
    return {
        1: (
            "ATE averages the effect over the target population; ATT averages it over treated "
            "customers; CATE conditions on a covariate profile. Random assignment means treated "
            "and control customers represent the same population in expectation, not that finite-"
            "sample ATE and ATT are algebraically identical. This project reports a pooled DML ATE "
            f"of {pooled['ate']:+.4f} on `visit`. The held-out forest's mean profile CATE is "
            f"{cate['mean']:+.4f}, with estimates from {cate['min']:+.4f} to {cate['max']:+.4f}."
        ),
        7: (
            "It is the decision result. Cross-fitted doubly robust evaluation gives the learned "
            f"policy {policy['learned (DRPolicyForest)']['value']:.4f} and blanket mens emailing "
            f"{policy['email everyone (mens creative)']['value']:.4f}. Their paired difference is "
            f"{comparison['difference']:+.4f} [{comparison['ci_low']:+.4f}, "
            f"{comparison['ci_high']:+.4f}]. The interval includes zero. I would deploy the simpler "
            "blanket action on this evidence and treat policy learning as a design for the next "
            "experiment, not as a production win."
        ),
        10: (
            f"The corrected top-30% estimate is ${spend['value']:.4f} in reported gross spend per "
            f"targeted customer [{spend['ci_low']:.4f}, {spend['ci_high']:.4f}]. Its interval is "
            "positive, but profitability is still unidentified: the data contain no gross margin, "
            "and twelve observations sit at the $499 maximum. The report therefore presents a "
            "gross-spend sensitivity rather than comparing it directly with email cost."
        ),
        11: (
            "Raw Qini is the area between cumulative incremental gain under the model ranking and "
            "random targeting; here it is 17.84, which depends on sample size. The normalized score "
            f"is {qini['value']:.4f} [{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. Five honest "
            f"sample splits range from {repeated['min']:.4f} to {repeated['max']:.4f}; these are "
            "sensitivity evidence, not independent replications. The primary interval is "
            "conditional on the fitted ranking and includes zero, so a positive raw area is not "
            "enough to claim a deployment-quality ranking."
        ),
        15: (
            "I would run a new campaign designed around action-level trade-offs. Both creatives "
            "help almost every segment here, which leaves little policy headroom. I would also add "
            "a semi-synthetic Monte Carlo benchmark with known heterogeneous effects, bias and "
            "coverage across confounding strengths, then revisit spend CATEs only after collecting "
            "enough non-zero purchase outcomes to support honest leaves."
        ),
    }


def render_causal_intro(results: dict) -> str:
    pooled = _table_by(results["headline"]["pooled_ate"], "outcome")["visit"]
    qini = results["ranking"]["normalized_qini"]
    comparison = _table_by(results["policy"]["comparisons"], "comparison")[
        "learned (DRPolicyForest) - email everyone (mens creative)"
    ]
    return (
        "Hillstrom's 64,000-customer randomised experiment identifies a clear average result: "
        f"assignment to either email raises the two-week visit rate by {pooled['ate']:.2%} "
        f"[{pooled['ci_low']:.2%}, {pooled['ci_high']:.2%}]. The individual-targeting result is "
        f"weaker. Normalized Qini is {qini['value']:.4f} "
        f"[{qini['ci_low']:.4f}, {qini['ci_high']:.4f}], and the learned policy's advantage over "
        f"blanket mens emailing is {comparison['difference']:+.4f} "
        f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]. Neither interval supports "
        "personalised deployment. The decision supported by this campaign is blanket mens "
        "emailing, while the causal-ML work diagnoses what a better follow-up experiment must "
        "change."
    )


def _causal_sections(results: dict) -> dict[str, str]:
    segment_effects = _table_by(results["interactions"]["segment_effects"], "segment")
    contrasts = _table_by(results["interactions"]["contrasts"], "contrast")
    ranking = results["ranking"]
    qini = ranking["normalized_qini"]
    values = _table_by(results["policy"]["values"], "policy")
    comparisons = _table_by(results["policy"]["comparisons"], "comparison")
    learned_gap = comparisons["learned (DRPolicyForest) - email everyone (mens creative)"]
    return {
        "## 5. Heterogeneity: who the email actually helps": (
            "The causal forest estimates profile-level conditional average effects on `visit`. "
            "I did not treat its segments as evidence on their own. A post-hoc held-out HC1-robust "
            f"audit estimates {segment_effects['mens only']['estimate']:+.4f} for mens-only, "
            f"{segment_effects['womens only']['estimate']:+.4f} for womens-only and "
            f"{segment_effects['both']['estimate']:+.4f} for both-category customers. The "
            f"womens-only minus mens-only contrast has Holm p={contrasts['womens only - mens only']['p_holm']:.3g}; "
            f"the joint Wald p-value is {results['interactions']['joint_p_value']:.3g}. This is "
            "exploratory evidence of segment differences, not an individual treatment effect."
        ),
        "## 6. Uplift ranking and targeting economics": (
            f"The primary split's normalized Qini is {qini['value']:.4f} "
            f"[{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. Five repeated sample splits are all "
            "positive in this run, but range from "
            f"{ranking['repeated_summary']['min']:.4f} to "
            f"{ranking['repeated_summary']['max']:.4f}; the primary fixed-model conditional "
            "bootstrap interval still "
            "includes zero. That is weak ranking evidence. The top-30% pooled-mixture diagnostic "
            f"estimates {ranking['top_k'][2]['value']:+.4f} visits per emailed customer, but it "
            "does not tell a marketer which creative to send. The three-action analysis answers "
            "that separate question."
        ),
        "## 7. Three actions, not two: an honest result": (
            "Cross-fitted doubly robust evaluation gives the learned policy a visit-rate value of "
            f"{values['learned (DRPolicyForest)']['value']:.4f}, against "
            f"{values['email everyone (mens creative)']['value']:.4f} for blanket mens emailing. "
            f"The paired difference is {learned_gap['difference']:+.4f} "
            f"[{learned_gap['ci_low']:+.4f}, {learned_gap['ci_high']:+.4f}]. Personalisation does "
            "not earn its operational complexity here. The evidence-supported decision is the "
            "simpler one: use the stronger mens creative broadly, then test a new campaign designed "
            "to create genuine action-level trade-offs."
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
    report = CAUSAL_REPORT_PATH.read_text(encoding="utf-8")
    report = replace_markdown_section(
        report,
        "## Estimating who a marketing email persuades, on a randomised experiment, with constructed stress tests and refutation checks",
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
    report = report.replace(
        "`reports/01` through `reports/09`", "`reports/01` through `reports/08`"
    )

    interview = INTERVIEW_PATH.read_text(encoding="utf-8")
    for number, answer in render_interview_answers(results).items():
        heading_match = re.search(rf"(?m)^## {number}\. .+$", interview)
        if not heading_match:
            raise ValueError(f"Interview question {number} not found.")
        interview = replace_markdown_section(interview, heading_match.group(), answer)
    return {
        README_PATH: readme,
        UPLIFT_REPORT_PATH: render_uplift_report(results),
        CAUSAL_REPORT_PATH: report,
        INTERVIEW_PATH: interview,
    }


def outputs_are_current() -> bool:
    return all(
        path.read_text(encoding="utf-8") == expected
        for path, expected in _rendered_outputs(load_results()).items()
    )


def render_all() -> None:
    for path, rendered in _rendered_outputs(load_results()).items():
        path.write_text(rendered, encoding="utf-8")


def main() -> None:
    render_all()
    print(f"rendered {README_PATH}, {UPLIFT_REPORT_PATH}, and {CAUSAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
