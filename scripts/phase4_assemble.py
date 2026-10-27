"""Assemble REPORT.md from scripts/REPORT.template.md and the generated tables in results/phase4/.

Each {{name}} marker is replaced verbatim by a generated file, so no table in REPORT.md is typed by hand. The stress
section stays a placeholder until results/phase4/stress/tables_stress.md exists.

Run with the cne-bg env (after scripts/phase4_report.py):  python scripts/phase4_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "results" / "phase4"
PENDING = ("Pending: the stress test runs once, after the settings in `results/phase4/stress_settings.json` are "
           "committed.")


def body(path, drop_title=True):
    """A generated Markdown file, with its own top-level title removed and headings demoted one level."""
    lines = path.read_text().splitlines()
    if drop_title and lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    stress = P4 / "stress" / "tables_stress.md"
    parts = {
        "headline": (P4 / "headline.md").read_text().strip(),
        "posthoc": body(P4 / "posthoc.md"),
        "tables": body(P4 / "tables.md"),
        "excluded": body(P4 / "excluded_runs_note.md"),
        "stress": (body(stress) + "\n\nFigure: `results/phase4/stress/stress_embeddings.png` (seed 0).")
        if stress.exists() else PENDING,
    }
    text = (ROOT / "scripts" / "REPORT.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
