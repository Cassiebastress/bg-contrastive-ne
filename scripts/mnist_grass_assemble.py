"""Assemble REPORT_mnist_grass.md from scripts/REPORT_mnist_grass.template.md and the generated files in
results/mnist_grass/. Each {{name}} marker is replaced verbatim by a generated file (or a table generated here from a
results JSON), so no table is typed by hand.

Run with the cne-bg env (after scripts/mnist_grass_report.py):  python scripts/mnist_grass_assemble.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mnist_grass"


def body(path, demote=2):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def build_table():
    b = json.loads((OUT / "build.json").read_text())
    out = [f"{b['n_grass_images']} grass images, split {b['seeds'][0]['n_target_pool']} / "
           f"{b['seeds'][0]['n_background_pool']} into target and background pools.", "",
           "| seed | digit 0 | digit 1 | distinct target sources | distinct background sources | build checks |",
           "|---|---|---|---|---|---|"]
    out += [f"| {s['seed']} | {s['digit_counts']['0']} | {s['digit_counts']['1']} | {s['distinct_target_sources']} "
            f"| {s['distinct_background_sources']} | {s['checks']} |" for s in b["seeds"]]
    return "\n".join(out)


def main():
    parts = {"build": build_table(), "prechecks": body(OUT / "prechecks.md"), "tables": body(OUT / "tables.md"),
             "validation": (OUT / "validation" / "validation.md").read_text().strip()}
    text = (ROOT / "scripts" / "REPORT_mnist_grass.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_mnist_grass.md").write_text(text)


if __name__ == "__main__":
    main()
