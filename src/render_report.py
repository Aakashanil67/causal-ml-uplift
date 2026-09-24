"""Renders `reports/causal_report.md` to PDF via headless Chrome.

`weasyprint` fails to import on this machine (`OSError: cannot load library 'libgobject-2.0-0'`,
needs a GTK/Pango runtime not installed here). Chrome's own
`--headless --print-to-pdf` needs no extra system install and was verified working before
committing to the approach, so the path here is markdown -> styled HTML -> Chrome print, not a
PDF library.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import markdown

from src.config import REPORTS_DIR

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 880px; margin: 40px auto;
       line-height: 1.5; color: #1a1a1a; font-size: 11pt; }
h1 { font-size: 20pt; margin-bottom: 4px; }
h2 { font-size: 13pt; margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 11.5pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 5px 8px; text-align: left; }
th { background: #eef2f7; }
img { max-width: 100%; display: block; margin: 12px auto; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
hr { border: none; border-top: 1px solid #ccc; margin: 24px 0; }
a { color: #2a5885; }
"""


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        return found
    raise FileNotFoundError("No Chrome executable found for PDF rendering.")


def render_pdf(
    md_path: Path = REPORTS_DIR / "causal_report.md", out_path: Path | None = None
) -> Path:
    out_path = out_path or md_path.with_suffix(".pdf")
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"]
    )
    html = (
        "<html><head><meta charset='utf-8'><title>Causal ML Uplift Analysis</title>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    chrome = find_chrome()
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={out_path}",
            "--no-pdf-header-footer",
            html_path.resolve().as_uri(),
        ],
        check=True,
        timeout=60,
    )
    html_path.unlink()
    sidecar = pdf_freshness_path(out_path)
    sidecar.write_text(
        json.dumps(
            {
                "source_sha256": _markdown_sha256(md_path),
                "pdf_sha256": _sha256(out_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_sha256(path: Path) -> str:
    text = path.read_bytes().decode("utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def pdf_freshness_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(pdf_path.suffix + ".json")


def pdf_is_current(
    md_path: Path = REPORTS_DIR / "causal_report.md", pdf_path: Path | None = None
) -> bool:
    pdf_path = pdf_path or md_path.with_suffix(".pdf")
    sidecar = pdf_freshness_path(pdf_path)
    if not md_path.exists() or not pdf_path.exists() or not sidecar.exists():
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == {
        "source_sha256": _markdown_sha256(md_path),
        "pdf_sha256": _sha256(pdf_path),
    }


def main() -> None:
    out_path = render_pdf()
    size_kb = out_path.stat().st_size / 1024
    print(f"rendered {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
