from src.config import ROOT
from src.render_outputs import (
    CAUSAL_REPORT_PATH,
    CAUSAL_REPORT_TEMPLATE_PATH,
    INTERVIEW_TEMPLATE_PATH,
    _causal_sections,
    _rendered_outputs,
    outputs_are_current,
    render_causal_intro,
    render_interview_answers,
    render_model_readme,
    render_readme_results,
    render_readme_status,
    render_uplift_report,
    replace_markdown_section,
)
from src.results import load_results


def test_replace_markdown_section_stops_at_the_next_peer_heading():
    text = "# Title\n\n## Results\n\nold\n\n## Methods\n\nkeep\n"

    rendered = replace_markdown_section(text, "## Results", "new")

    assert rendered == "# Title\n\n## Results\n\nnew\n\n## Methods\n\nkeep\n"


def test_readme_results_are_rendered_from_the_manifest():
    rendered = render_readme_results(load_results())

    assert "18.23%" in rendered
    assert "18.10%" in rendered
    assert "0.0111" in rendered
    assert "email everyone (womens creative)" in rendered


def test_uplift_report_contains_executable_interaction_and_policy_evidence():
    rendered = render_uplift_report(load_results())

    assert "Holm-adjusted" in rendered
    assert "post-hoc held-out interaction audit" in rendered
    assert "fixed-model conditional evaluation intervals" in " ".join(rendered.split())
    assert "email everyone (womens creative)" in rendered
    assert "does not establish" in rendered
    assert "reported-spend" in rendered.lower()
    assert "conditional evaluation intervals" in rendered


def test_tracked_public_outputs_are_current():
    assert outputs_are_current()


def test_interview_answers_use_current_policy_and_ranking_evidence():
    answers = render_interview_answers(load_results())

    assert "blanket mens" in answers[7]
    assert "0.0111" in answers[11]
    assert "Reported spend" in answers[10]
    assert "top-coded" in answers[10]


def test_causal_report_intro_leads_with_the_supported_decision():
    intro = render_causal_intro(load_results())

    assert "0.0111" in intro
    assert "blanket mens" in intro
    assert "beats naive heuristics" not in intro


def test_causal_report_uses_the_recruiter_facing_subtitle():
    report = (ROOT / "reports" / "causal_report.md").read_text(encoding="utf-8")

    assert (
        "## Can causal ML find deployable treatment-effect heterogeneity, or only a reliable average effect?"
        in report
    )
    assert "Estimating who a marketing email persuades" not in report


def test_causal_report_rendering_does_not_duplicate_sections():
    rendered = _rendered_outputs(load_results())[CAUSAL_REPORT_PATH]
    headings = (
        "## 1. The causal question and why identification is clean here",
        "## 2. The naive estimate, and the check that it is allowed to be naive",
        "## 3. The methods ladder: naive, regression, and Double ML agree",
        "## 4. Constructed confounding stress test",
        "## 5. Heterogeneity: who the email actually helps",
        "## 6. Uplift ranking and targeting economics",
        "## 7. Three actions, not two: an honest result",
        "## 8. Refutation tests, and what they do not prove",
        "## 9. Limitations, stated rather than buried",
        "## 10. What would and would not transfer to a South African retention campaign",
    )

    assert all(rendered.count(heading) == 1 for heading in headings)


def test_generated_synthesis_and_interview_use_value_free_templates():
    causal_template = CAUSAL_REPORT_TEMPLATE_PATH.read_text(encoding="utf-8")
    interview_template = INTERVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "<!-- Rendered from the validated numerical manifest. -->" in causal_template
    assert "<!-- Rendered from the validated numerical manifest. -->" in interview_template
    assert "18.23%" not in causal_template + interview_template
    assert "0.0111" not in causal_template + interview_template


def test_model_readme_renders_when_generated_output_is_absent(tmp_path, monkeypatch):
    import src.render_outputs as render_outputs

    monkeypatch.setattr(render_outputs, "MODEL_README_PATH", tmp_path / "missing-README.md")

    rendered = render_model_readme()

    assert "Current artifact checksums:" in rendered
    assert "causal_forest.joblib" in rendered
    assert "evaluation_artifacts.joblib" in rendered


def test_retired_report_generators_do_not_embed_result_numbers():
    uplift_source = (ROOT / "src" / "uplift.py").read_text(encoding="utf-8")
    policy_source = (ROOT / "src" / "policy.py").read_text(encoding="utf-8")

    assert "p=0.006" not in uplift_source
    assert "close to twice as large" not in uplift_source
    assert "append_policy_section" not in policy_source


def test_synthesis_report_inventory_includes_the_final_gate():
    report = (ROOT / "reports" / "causal_report.md").read_text(encoding="utf-8")

    assert "`reports/01` through `reports/08`" in report


def test_public_claim_contract_uses_the_correct_causal_boundaries():
    report = (ROOT / "reports" / "causal_report.md").read_text(encoding="utf-8")
    questions = (ROOT / "reports" / "interview_qa.md").read_text(encoding="utf-8")
    causal_question = (ROOT / "reports" / "01_causal_question.md").read_text(encoding="utf-8")

    assert "post-hoc held-out interaction audit" in render_uplift_report(load_results())
    assert "pre-" + "specified" not in "\n".join((report, questions, causal_question))
    assert "individual treatment " + "effect" not in render_causal_intro(load_results()).lower()
    assert "conditional on the fitted ranking" in questions
    assert "intention-to-treat" in causal_question
    assert "no " + "colliders" not in questions.lower()


def test_changed_manifest_fixture_updates_public_values_and_conclusions():
    import copy

    results = copy.deepcopy(load_results())
    results["headline"]["pooled_ate"][0]["ate"] = 0.07
    visit_naive = next(row for row in results["naive"]["estimates"] if row["outcome"] == "visit")
    visit_naive.update(diff=0.07, ci_low=0.06, ci_high=0.08)
    results["ranking"]["normalized_qini"].update(value=0.03, ci_low=0.01, ci_high=0.05)
    comparison = next(
        row
        for row in results["policy"]["comparisons"]
        if row["comparison"] == "learned (DRPolicyForest) - email everyone (mens creative)"
    )
    comparison.update(difference=0.01, ci_low=0.002, ci_high=0.018)

    assert "+7.00pp" in render_causal_intro(results)
    assert "positive" in render_readme_status(results).lower()
    assert "positive" in render_uplift_report(results).lower()
    assert "higher expected visit value" in render_interview_answers(results)[7].lower()
    assert (
        "+7.00pp"
        in _causal_sections(results)[
            "## 2. The naive estimate, and the check that it is allowed to be naive"
        ]
    )


def test_positive_policy_interval_updates_every_published_conclusion():
    import copy

    results = copy.deepcopy(load_results())
    qini = results["ranking"]["normalized_qini"]
    qini.update(value=0.03, ci_low=0.01, ci_high=0.05)
    comparison = next(
        row
        for row in results["policy"]["comparisons"]
        if row["comparison"] == "learned (DRPolicyForest) - email everyone (mens creative)"
    )
    comparison.update(difference=0.01, ci_low=0.002, ci_high=0.018)

    outputs = [
        render_readme_results(results),
        render_readme_status(results),
        render_uplift_report(results),
        render_causal_intro(results),
        render_interview_answers(results)[7],
        render_interview_answers(results)[11],
        render_interview_answers(results)[15],
        *_causal_sections(results).values(),
    ]
    combined = "\n".join(outputs).lower()

    assert "higher expected visit value for the learned policy" in combined
    assert "entirely positive for a positive ranking advantage" in combined
    assert "two narrower claims are not established" not in combined
    assert "ranking is weak" not in combined
    assert "no established learned-policy win" not in combined
    assert "learned policy has no established advantage" not in combined
    assert "does not establish a personalized policy win" not in combined
