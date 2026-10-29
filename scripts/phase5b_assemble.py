"""Assemble REPORT_phase5b.md from scripts/REPORT_phase5b.template.md and the generated tables in results/phase5b/.
Each {{name}} marker is replaced verbatim by a generated file, so no table is typed by hand.

Run with the cne-bg env (after scripts/phase5b_report.py):  python scripts/phase5b_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P5B = ROOT / "results" / "phase5b"


def body(path, demote=2):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    parts = {"predictions": (P5B / "predictions_status.md").read_text().strip(),
             "calibration": body(P5B / "calibration.md"), "tables_cyclic": body(P5B / "tables_cyclic.md"),
             "tables_between": body(P5B / "tables_between.md"),
             "tables_cyclic_local": body(P5B / "tables_cyclic_local_cpca.md"),
             "tables_between_local": body(P5B / "tables_between_local_cpca.md")}
    text = (ROOT / "scripts" / "REPORT_phase5b.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_phase5b.md").write_text(text)


if __name__ == "__main__":
    main()
