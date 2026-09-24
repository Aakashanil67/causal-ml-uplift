"""One-command regeneration of numerical, serving and report artifacts."""

from collections.abc import Callable

from src.build_artifacts import build_all
from src.render_outputs import outputs_are_current, render_all
from src.render_report import pdf_is_current, render_pdf


def verify_generated_outputs() -> None:
    if not outputs_are_current():
        raise RuntimeError("Generated Markdown outputs do not match the results manifest.")
    if not pdf_is_current():
        raise RuntimeError("Generated PDF is missing or stale relative to causal_report.md.")


def run_pipeline(
    build: Callable[[], object] = build_all,
    render: Callable[[], object] = render_all,
    pdf: Callable[[], object] = render_pdf,
    verify: Callable[[], object] | None = verify_generated_outputs,
) -> None:
    build()
    render()
    pdf()
    if verify is not None:
        verify()


if __name__ == "__main__":
    run_pipeline()
