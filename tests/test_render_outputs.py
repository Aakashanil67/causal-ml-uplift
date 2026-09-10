from src.config import ROOT
from src.render_outputs import (
    outputs_are_current,
    render_causal_intro,
    render_interview_answers,
    render_readme_results,
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

    assert "0.1823" in rendered
    assert "0.1810" in rendered
    assert "0.0111" in rendered
    assert "email everyone (womens creative)" in rendered


def test_uplift_report_contains_executable_interaction_and_policy_evidence():
    rendered = render_uplift_report(load_results())

    assert "Holm-adjusted" in rendered
    assert "post-hoc held-out interaction audit" in rendered
    assert "fixed-model conditional" in rendered
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
    assert "reported gross spend" in answers[10]


def test_causal_report_intro_leads_with_the_supported_decision():
    intro = render_causal_intro(load_results())

    assert "0.0111" in intro
    assert "blanket mens" in intro
    assert "beats naive heuristics" not in intro


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
    assert "pre-specified" not in "\n".join((report, questions, causal_question))
    assert "individual treatment effect" not in render_causal_intro(load_results()).lower()
    assert "conditional on the fitted ranking" in questions
    assert "intention-to-treat" in causal_question
    assert "no colliders" not in questions.lower()
