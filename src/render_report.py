"""Renders `reports/causal_report.md` to PDF via headless Chrome.

`weasyprint` fails to import on this machine (`OSError: cannot load library 'libgobject-2.0-0'`,
needs a GTK/Pango runtime not installed here — see `CLAUDE.md`). Chrome's own
`--headless --print-to-pdf` needs no extra system install and was verified working before
committing to the approach, so the path here is markdown -> styled HTML -> Chrome print, not a
PDF library.
"""

import shutil
import subprocess
from pathlib import Path

import markdown

from src.config import REPORTS_DIR

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
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
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
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
            "--print-to-pdf-no-header",
            html_path.resolve().as_uri(),
        ],
        check=True,
        timeout=60,
    )
    html_path.unlink()
    return out_path


def main() -> None:
    out_path = render_pdf()
    size_kb = out_path.stat().st_size / 1024
    print(f"rendered {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
