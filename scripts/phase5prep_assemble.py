"""Assemble REPORT_phase5prep.md from scripts/REPORT_phase5prep.template.md and the generated tables in
results/phase5prep/. Each {{name}} marker is replaced verbatim by a generated file, so no table is typed by hand.

Run with the cne-bg env (after scripts/phase5prep_summary.py):  python scripts/phase5prep_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "results" / "phase5prep"


def body(path, demote=2):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    parts = {"validation": body(P5 / "metric_validation.md", demote=1), "cpca": body(P5 / "cpca_summary.md"),
             "outside": (P5 / "signal_outside_background.md").read_text().strip()}
    text = (ROOT / "scripts" / "REPORT_phase5prep.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_phase5prep.md").write_text(text)


if __name__ == "__main__":
    main()
