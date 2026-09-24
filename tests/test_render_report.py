import hashlib
import json

import pytest

from src.render_report import find_chrome, pdf_freshness_path, pdf_is_current, render_pdf


def test_pdf_freshness_normalizes_markdown_newlines_and_hashes_pdf_as_binary(tmp_path):
    md_path = tmp_path / "report.md"
    pdf_path = tmp_path / "report.pdf"
    pdf_bytes = b"%PDF-1.7\nfixture bytes\n"
    md_path.write_bytes(b"# Report\r\nSame text\r\n")
    pdf_path.write_bytes(pdf_bytes)
    pdf_freshness_path(pdf_path).write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(b"# Report\nSame text\n").hexdigest(),
                "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert pdf_is_current(md_path, pdf_path)

    md_path.write_bytes(b"# Report\nSame text\n")
    assert pdf_is_current(md_path, pdf_path)

    pdf_path.write_bytes(b"%PDF-1.7\nchanged bytes\n")
    assert not pdf_is_current(md_path, pdf_path)


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
    assert pdf_is_current(md_path, out_path)
    assert pdf_freshness_path(out_path).exists()
    md_path.write_text("# Updated source\n", encoding="utf-8")
    assert not pdf_is_current(md_path, out_path)


def test_render_pdf_raises_clearly_if_chrome_is_missing(tmp_path, monkeypatch):
    import src.render_report as render_report

    # a real bug in the first version of this test: it only patched CHROME_CANDIDATES, but
    # find_chrome()'s PATH fallback (shutil.which) finds the real system Chrome on any machine
    # that has one on PATH — including GitHub's ubuntu-latest runners — so the "missing" case
    # was never actually exercised there. shutil.which has to be patched too.
    monkeypatch.setattr(render_report, "CHROME_CANDIDATES", ["Z:\\nonexistent\\chrome.exe"])
    monkeypatch.setattr(render_report.shutil, "which", lambda _name: None)
    md_path = tmp_path / "report.md"
    md_path.write_text("# Title\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No Chrome executable found"):
        render_report.render_pdf(md_path, out_path=tmp_path / "report.pdf")
