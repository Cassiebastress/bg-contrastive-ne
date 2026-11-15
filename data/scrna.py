"""Zheng et al. 2017 single-cell RNA-seq, as in the cPCA paper's Figure 3: bone marrow cells of AML patient 027 before
and after transplant (target; label 0 = pre, 1 = post) against a healthy donor's bone marrow (background).

Source: 10x Genomics "Single Cell 3' Paper: Zheng et al. 2017", filtered gene-barcode matrices (v1.1.0 URLs below);
each tarball's SHA-256 is pinned in TARBALL_SHA256. Matrices are read straight from the tarballs (nothing extracted).

Preprocessing (label-free; one transform shared by target and background):
1. each cell's counts scaled to sum to 10,000, then log(1 + x);
2. the 500 genes with the largest dispersion (variance / mean, both over target + background rows; genes with mean 0
   excluded; ties -> lower gene index);
3. each gene standardized with the target + background mean and standard deviation.
Donor 2 (the descriptive larger background) is put through the same transform (donor 1's genes and scaling), so the
target is identical in both background variants.

The paper's preprocessing, kept for the replication check only (experiments/Single-Cell RNA-seq (Figure 3).ipynb and
utils.Dataset in github.com/abidlabs/contrastive): the 500 genes with the largest dispersion of the raw counts
(target + background), then target and background each centered and standardized on their own.

    python data/scrna.py   # download (if needed), verify, build data/processed/scrna_aml027.npz, write the manifest
"""
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

URL = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/{s}/{s}_filtered_gene_bc_matrices.tar.gz"
TARBALL_SHA256 = {
    "aml027_pre_transplant": "5a16d744339faa8e26dd352d78be224a760a2822bd7c1717b2b802272e9205a7",
    "aml027_post_transplant": "74b4bf1380fb035ca11bc0c54098f8628be2de62dd467498f5661cc99252f9b2",
    "frozen_bmmc_healthy_donor1": "437678058acf9733826f2b953cd1fff58b2dd70076b171da87a63b0a0ddb732a",
    "frozen_bmmc_healthy_donor2": "6c0bba75430853cb59e6abf8282f7ba8f81c8292d85e167a9a65f92947534062",
}
TARGET = ("aml027_pre_transplant", "aml027_post_transplant")  # label = position
BACKGROUND = "frozen_bmmc_healthy_donor1"
EXTRA_BACKGROUND = "frozen_bmmc_healthy_donor2"
N_CELLS = {"aml027_pre_transplant": 3933, "aml027_post_transplant": 3965, "frozen_bmmc_healthy_donor1": 1985,
           "frozen_bmmc_healthy_donor2": 2472}
N_GENES = 500
COUNTS_PER_CELL = 1e4
MEMBER = "filtered_matrices_mex/hg19/{}"
NAME = "scrna_aml027"


def raw_dir(root):
    return Path(root) / "data" / "raw" / "zheng2017"


def processed_path(root):
    return Path(root) / "data" / "processed" / f"{NAME}.npz"


def manifest_path(root):
    return Path(root) / "data" / "MANIFEST_scrna.md"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(root):
    """Each tarball into data/raw/zheng2017/ unless present; stops on a SHA-256 mismatch."""
    d = raw_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    for s, want in TARBALL_SHA256.items():
        path = d / f"{s}.tar.gz"
        if not path.exists():  # the server refuses urllib's default user agent (403), so name the project instead
            req = urllib.request.Request(URL.format(s=s), headers={"User-Agent": "bg-contrastive-ne-data-download"})
            with urllib.request.urlopen(req) as r:
                path.write_bytes(r.read())
        if sha256(path) != want:
            raise RuntimeError(f"{path.name}: SHA-256 {sha256(path)} != pinned {want}")


def read_sample(root, s):
    """(cells x genes CSR int64 counts, gene ids, gene names, barcodes) read from the tarball without extracting."""
    with tarfile.open(raw_dir(root) / f"{s}.tar.gz", "r:gz") as tar:
        def member(name):
            return tar.extractfile(MEMBER.format(name)).read()
        counts = sp.csr_matrix(scipy.io.mmread(io.BytesIO(member("matrix.mtx")), spmatrix=True).T, dtype=np.int64)
        genes = [ln.split("\t") for ln in member("genes.tsv").decode().splitlines()]
        barcodes = member("barcodes.tsv").decode().splitlines()
    if counts.shape != (len(barcodes), len(genes)) or counts.shape[0] != N_CELLS[s]:
        raise RuntimeError(f"{s}: unexpected shape {counts.shape}")
    return counts, np.array([g[0] for g in genes]), np.array([g[1] for g in genes]), np.array(barcodes)


def log_normalize(counts):
    """log(1 + 10,000 * counts / the cell's total), sparse float64 (log(1 + 0) = 0 keeps the zeros)."""
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    if (totals <= 0).any():
        raise RuntimeError("a cell with no counts")
    L = sp.diags(COUNTS_PER_CELL / totals) @ counts.astype(np.float64)
    L = sp.csr_matrix(L)
    L.data = np.log1p(L.data)
    return L


def column_moments(X):
    """Per-column mean and population variance of a sparse matrix, without densifying it."""
    mean = np.asarray(X.mean(axis=0)).ravel()
    return mean, np.asarray(X.multiply(X).mean(axis=0)).ravel() - mean ** 2


def top_dispersion(X, n=N_GENES):
    """Indices (ascending) of the n columns of sparse X with the largest variance / mean; columns with mean 0 never
    chosen; ties broken toward the lower index (stable sort)."""
    mean, var = column_moments(X)
    disp = np.full(X.shape[1], -np.inf)
    pos = mean > 0
    disp[pos] = var[pos] / mean[pos]
    return np.sort(np.argsort(-disp, kind="stable")[:n])


def standardize(X):
    """Columns centered and scaled to unit standard deviation (population std), with NaN -> 0 as in the paper."""
    return np.nan_to_num((X - X.mean(axis=0)) / X.std(axis=0))


def build(root):
    samples = {s: read_sample(root, s) for s in (*TARGET, BACKGROUND, EXTRA_BACKGROUND)}
    gene_ids, gene_names = samples[TARGET[0]][1], samples[TARGET[0]][2]
    for s, (_, ids, _, _) in samples.items():
        if not np.array_equal(ids, gene_ids):
            raise RuntimeError(f"{s}: gene list differs")
    counts_T = sp.vstack([samples[s][0] for s in TARGET], format="csr")
    counts_B, counts_B2 = samples[BACKGROUND][0], samples[EXTRA_BACKGROUND][0]
    L_T, L_B, L_B2 = log_normalize(counts_T), log_normalize(counts_B), log_normalize(counts_B2)

    genes = top_dispersion(sp.vstack([L_T, L_B], format="csr"), N_GENES)
    T, B, B2 = (L[:, genes].toarray() for L in (L_T, L_B, L_B2))
    pooled = np.vstack([T, B])
    mu, sd = pooled.mean(axis=0), pooled.std(axis=0)
    if (sd == 0).any():
        raise RuntimeError("a selected gene is constant over target + background")

    genes_paper = top_dispersion(sp.vstack([counts_T, counts_B], format="csr").astype(np.float64), N_GENES)
    raw_T, raw_B = (c[:, genes_paper].toarray().astype(np.float64) for c in (counts_T, counts_B))

    def log_total(c):
        return np.log10(np.asarray(c.sum(axis=1), dtype=np.float64).ravel())

    return {
        "X_T": (T - mu) / sd, "X_B": (B - mu) / sd, "X_B2": (B2 - mu) / sd,
        "y": np.concatenate([np.full(samples[s][0].shape[0], i, dtype=np.int64) for i, s in enumerate(TARGET)]),
        "log10_total_T": log_total(counts_T), "log10_total_B": log_total(counts_B),
        "genes": genes, "gene_ids": gene_ids[genes], "gene_names": gene_names[genes],
        "barcodes_T": np.concatenate([samples[s][3] for s in TARGET]), "barcodes_B": samples[BACKGROUND][3],
        "X_T_paper": standardize(raw_T), "X_B_paper": standardize(raw_B),
        "genes_paper": genes_paper,
    }


def save(d, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def write_manifest(root):
    lines = ["# Single-cell RNA-seq data manifest (Zheng et al. 2017)", "",
             "Generated by `data/scrna.py` on the cluster. The files are not committed (`data/raw/` and "
             "`data/processed/` are gitignored).", "", "## Tarballs (`data/raw/zheng2017/`)", "",
             "| sample | cells | SHA-256 |", "|---|---|---|"]
    lines += [f"| `{s}` | {N_CELLS[s]} | `{h}` |" for s, h in TARBALL_SHA256.items()]
    lines += ["", "## Processed dataset (`data/processed/`)", "", "| file | SHA-256 |", "|---|---|",
              f"| `{processed_path(root).name}` | `{sha256(processed_path(root))}` |"]
    manifest_path(root).write_text("\n".join(lines) + "\n")


def for_embedding(d, background="donor1"):
    """X_T and X_B (donor 1, or donors 1 + 2) as float32, plus the evaluation-only truth."""
    X_B = d["X_B"] if background == "donor1" else np.vstack([d["X_B"], d["X_B2"]])
    truth = {"y": d["y"], "log10_total_T": d["log10_total_T"]}
    return d["X_T"].astype(np.float32), X_B.astype(np.float32), truth


def main():
    root = Path(__file__).resolve().parents[1]
    download(root)
    save(build(root), processed_path(root))
    write_manifest(root)
    print(manifest_path(root).read_text())


if __name__ == "__main__":
    sys.exit(main())
