"""The supervisor report's single-cell figure, generated from results/scrna/ (nothing is recomputed except the plot).

- docs/supervisor_report/figures/scrna_seed0.png: seed 0, every run colored by sample (pre blue, post orange), each
  titled with its seed-0 accuracy and ARI; the two linear projections drawn on the central 99% of cells, as in
  scripts/scrna_report.embedding_figures. Needs the embeddings and data/processed/ (cluster).

    python scripts/supervisor_scrna.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.scrna_report import FIG_SEED, LABEL_COLORS, LINEAR, MAIN_RUNS, read, scatter  # noqa: E402
from scripts.scrna_run import OUT, RUNS_DIR, load_data  # noqa: E402

DOC = ROOT / "docs" / "supervisor_report"
PLAIN = {"r_a": "BC-NE", "a": "plain Neg-t-SNE", "T1": "t-SNE", "T2": "PCA-50 → t-SNE", "U2": "PCA-50 → UMAP",
         "T3": "cPCA → t-SNE", "cPCA2D": "cPCA-2D", "PCA2D": "PCA-2D"}


def figure():
    rows = read(OUT / "runs.csv")
    rec = {r["run"]: r for r in rows if r["seed"] == FIG_SEED}
    y = load_data()["y"]
    order = np.random.default_rng(0).permutation(len(y))
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7))
        for ax, run in zip(axes.flat, MAIN_RUNS):
            scatter(ax, np.load(RUNS_DIR / f"seed{FIG_SEED}_{run}.npy"), LABEL_COLORS[y], order,
                    central=run in LINEAR)
            ax.set_title(f"{PLAIN[run]}{'*' if run in LINEAR else ''}\nacc {rec[run]['label_acc']:.3f} · "
                         f"ARI {np.round(rec[run]['ari'], 3) + 0.0:.3f}", fontsize=8.5)
        handles = [plt.Line2D([], [], marker="o", ls="", ms=5, color=c, label=lab)
                   for c, lab in zip(LABEL_COLORS, ("pre-transplant", "post-transplant"))]
        fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.2, w_pad=0.4)
        fig.savefig(DOC / "figures" / f"scrna_seed{FIG_SEED}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    figure()


if __name__ == "__main__":
    main()
