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
    assert "0.1831" not in rendered


def test_uplift_report_contains_executable_interaction_and_policy_evidence():
    rendered = render_uplift_report(load_results())

    assert "Holm-adjusted" in rendered
    assert "0.0184" in rendered
    assert "[-0.0114, 0.0331]" in rendered
    assert "+0.0013" in rendered
    assert "does not establish" in rendered


def test_tracked_public_outputs_are_current():
    assert outputs_are_current()


def test_interview_answers_use_current_policy_and_ranking_evidence():
    answers = render_interview_answers(load_results())

    assert "0.1823" in answers[7]
    assert "0.1810" in answers[7]
    assert "0.0111" in answers[11]
    assert "0.1831" not in answers[7]
    assert "$0.1196" not in answers[10]


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

    assert "`reports/01` through `reports/09`" in report
