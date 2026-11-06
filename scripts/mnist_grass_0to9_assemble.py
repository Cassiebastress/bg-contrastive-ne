"""Assemble REPORT_mnist_grass_0to9.md from scripts/REPORT_mnist_grass_0to9.template.md and the generated files in
results/mnist_grass_0to9/. Each {{name}} marker is replaced verbatim by a generated file (or a table generated here
from build.json), so no table is typed by hand.

Run with the cne-bg env (after mnist_grass_0to9_prechecks.py --summarize and mnist_grass_0to9_report.py):
    python scripts/mnist_grass_0to9_assemble.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mnist_grass_0to9"


def body(path, demote=2):
    """A generated Markdown file without its top-level title, headings demoted to fit under the template's."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#" * demote + ln if ln.startswith("#") else ln for ln in lines).strip()


def build_table():
    b = json.loads((OUT / "build.json").read_text())
    reg = b["regression_0_1"]
    ref = b["xi_checks_refactor"]
    out = [f"{b['n_grass_images']} grass images, split {b['seeds'][0]['n_target_pool']} / "
           f"{b['seeds'][0]['n_background_pool']} into target and background pools.", "",
           "| seed | images per digit (min–max over digits 0–9) | distinct target sources "
           "| distinct background sources | build checks |", "|---|---|---|---|---|"]
    for s in b["seeds"]:
        counts = s["digit_counts"].values()
        out.append(f"| {s['seed']} | {min(counts)}–{max(counts)} | {s['distinct_target_sources']} "
                   f"| {s['distinct_background_sources']} | {s['checks']} |")
    ok = all(r["stored_file_matches_manifest"] and r["rebuild_identical"] for r in reg)
    out += ["", f"0/1 regression (stored 0/1 files match the manifest and rebuild bit-identically): "
            f"**{'passed' if ok else 'FAILED'}** on seeds {', '.join(str(r['seed']) for r in reg)}. "
            f"ξ_B check refactor (all {ref['fields']} fields of the stored 0/1 seed-0 pre-check reproduced exactly): "
            f"**{'passed' if ref['identical'] else 'FAILED'}**."]
    return "\n".join(out)


def main():
    parts = {"build": build_table(), "prechecks": body(OUT / "prechecks.md"), "tables": body(OUT / "tables.md")}
    text = (ROOT / "scripts" / "REPORT_mnist_grass_0to9.template.md").read_text()
    for name, content in parts.items():
        text = text.replace("{{" + name + "}}", content)
    assert "{{" not in text, "unfilled marker in the template"
    (ROOT / "REPORT_mnist_grass_0to9.md").write_text(text)


if __name__ == "__main__":
    main()
