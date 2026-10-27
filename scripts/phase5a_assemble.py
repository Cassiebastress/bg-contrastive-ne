"""Assemble REPORT_phase5a.md from scripts/REPORT_phase5a.template.md and the generated tables in results/phase5a/.
Each {{name}} marker is replaced verbatim by a generated file, so no table is typed by hand.

Run with the cne-bg env (after scripts/phase5a_report.py):  python scripts/phase5a_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P5A = ROOT / "results" / "phase5a"


def body(path, demote=2):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    grid = (P5A / "grid_J.md").read_text().split("<!-- generated below -->", 1)[1]
    parts = {"grid_J": "\n".join("#" * 2 + ln if ln.startswith("#") else ln for ln in grid.splitlines()).strip(),
             "validity": body(P5A / "validity.md"), "diagnostics": body(P5A / "data_diagnostics.md"),
             "k_calibration": body(P5A / "k_calibration.md"), "tables_J": body(P5A / "tables_J.md"),
             "predictions": (P5A / "predictions_status.md").read_text().strip(),
             "extension_o": body(P5A / "extension_o.md")}
    text = (ROOT / "scripts" / "REPORT_phase5a.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_phase5a.md").write_text(text)


if __name__ == "__main__":
    main()
