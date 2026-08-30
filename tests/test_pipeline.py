from src.pipeline import run_pipeline


def test_pipeline_runs_artifacts_then_markdown_then_pdf():
    calls = []

    run_pipeline(
        build=lambda: calls.append("artifacts"),
        render=lambda: calls.append("markdown"),
        pdf=lambda: calls.append("pdf"),
    )

    assert calls == ["artifacts", "markdown", "pdf"]
