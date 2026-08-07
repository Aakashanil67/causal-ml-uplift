import pytest

from src.render_report import find_chrome, render_pdf


def test_render_pdf_produces_a_real_pdf(tmp_path):
    # CHROME_CANDIDATES is a Windows-path list (this project's dev machine); on a CI runner
    # without one of those paths or a `chrome`/`google-chrome` on PATH, skip rather than fail —
    # the actual PDF-rendering logic still gets exercised locally and wherever Chrome exists.
    try:
        find_chrome()
    except FileNotFoundError:
        pytest.skip("no Chrome executable available in this environment")

    md_path = tmp_path / "tiny_report.md"
    md_path.write_text(
        "# Title\n\nSome **content** with a table.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )

    out_path = render_pdf(md_path, out_path=tmp_path / "tiny_report.pdf")

    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # a real rendered page, not an empty/error stub
    with open(out_path, "rb") as f:
        assert f.read(5) == b"%PDF-"
    # the intermediate HTML file should be cleaned up, not left behind
    assert not md_path.with_suffix(".html").exists()


def test_render_pdf_raises_clearly_if_chrome_is_missing(tmp_path, monkeypatch):
    import src.render_report as render_report

    monkeypatch.setattr(render_report, "CHROME_CANDIDATES", ["Z:\\nonexistent\\chrome.exe"])
    md_path = tmp_path / "report.md"
    md_path.write_text("# Title\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No Chrome executable found"):
        render_report.render_pdf(md_path, out_path=tmp_path / "report.pdf")
