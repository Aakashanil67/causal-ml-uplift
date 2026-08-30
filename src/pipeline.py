"""One-command regeneration of numerical, serving and report artifacts."""

from collections.abc import Callable

from src.build_artifacts import build_all
from src.render_outputs import render_all
from src.render_report import render_pdf


def run_pipeline(
    build: Callable[[], object] = build_all,
    render: Callable[[], object] = render_all,
    pdf: Callable[[], object] = render_pdf,
) -> None:
    build()
    render()
    pdf()


if __name__ == "__main__":
    run_pipeline()
