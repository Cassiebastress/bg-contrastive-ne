"""Assemble REPORT_bsgraph.md from scripts/REPORT_bsgraph.template.md and the generated tables in results/bsgraph/.
Each {{name}} marker is replaced verbatim by a generated file, so no table is typed by hand.

Run with the cne-bg env (after scripts/bsgraph_phase1.py --merge and scripts/bsgraph_report.py):
    python scripts/bsgraph_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bsgraph"


def body(path, demote=2):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    parts = {"phase1": body(OUT / "phase1.md"), "phase2": body(OUT / "tables_phase2.md")}
    text = (ROOT / "scripts" / "REPORT_bsgraph.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_bsgraph.md").write_text(text)


if __name__ == "__main__":
    main()
