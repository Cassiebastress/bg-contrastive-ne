# bg-contrastive-ne

Background-contrastive Neg-t-SNE: a neighbor-embedding analogue of contrastive PCA. Given a target
dataset and a background dataset in the same feature space, embed the target so that structure shared
with the background is suppressed, by adding extra negative samples drawn from pairs of target points
that look similar for background reasons.

See [`Initial_Project_Plan.md`](Initial_Project_Plan.md) for the full plan and
[`NOTES.md`](NOTES.md) for notes on the underlying package.

## Setup

```sh
conda env create -f env/environment.yml   # or: conda create -n cne-bg python=3.12
conda activate cne-bg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e third_party/contrastive-ne
```

## Third-party code

`third_party/contrastive-ne/` is a vendored copy of
[berenslab/contrastive-ne](https://github.com/berenslab/contrastive-ne) at v0.4.0 (`d5f0b53`),
MIT-licensed (Damrich, Böhm, Hamprecht, Kobak, *From t-SNE to UMAP with contrastive learning*,
ICLR 2023). See `third_party/contrastive-ne/UPSTREAM.md`.
