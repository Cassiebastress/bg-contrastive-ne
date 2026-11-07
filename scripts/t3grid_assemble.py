"""Assemble REPORT_mnist_grass_0to9_t3grid.md from scripts/REPORT_t3grid.template.md and the generated files in
results/mnist_grass_0to9_t3grid/. Each {{name}} marker is replaced verbatim by a generated file, so no table is typed by
hand.

Run with the cne-bg env (after scripts/t3grid_report.py):
    python scripts/t3grid_assemble.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mnist_grass_0to9_t3grid"


def body(path, demote=1):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def main():
    text = (ROOT / "scripts" / "REPORT_t3grid.template.md").read_text()
    text = text.replace("{{tables}}", body(OUT / "tables.md"))
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_mnist_grass_0to9_t3grid.md").write_text(text)


if __name__ == "__main__":
    main()
