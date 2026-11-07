"""Assemble REPORT_single_stage.md from scripts/REPORT_single_stage.template.md and the generated files in
results/single_stage/. Each {{name}} marker is replaced verbatim by a generated file, so no table is typed by hand.

Run with the cne-bg env (after scripts/single_stage_report.py):
    python scripts/single_stage_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "single_stage"


def body(path, demote=1):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    parts = {"tables": body(OUT / "tables.md"), "criterion": body(OUT / "criterion.md")}
    text = (ROOT / "scripts" / "REPORT_single_stage.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_single_stage.md").write_text(text)


if __name__ == "__main__":
    main()
