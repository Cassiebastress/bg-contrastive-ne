"""Assemble REPORT_scrna.md from scripts/REPORT_scrna.template.md and the generated files in results/scrna/. Each
{{name}} marker is replaced verbatim by a generated file, so no table is typed by hand.

Run with the cne-bg env (after scripts/scrna_report.py):
    python scripts/scrna_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scrna"


def body(path, demote=1):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    parts = {"tables": body(OUT / "tables.md"), "prechecks": body(OUT / "prechecks.md")}
    text = (ROOT / "scripts" / "REPORT_scrna.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_scrna.md").write_text(text)


if __name__ == "__main__":
    main()
