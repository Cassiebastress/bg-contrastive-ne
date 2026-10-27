"""Embedders run on a 10-D cPCA representation (Phase 5 prep runs (e2) and (e3); reused in Phase 5a).

(e2): the two-stage Neg-t-SNE of run (a), with its exact kNN graph (k = 15) and PCA init built in the 10-D space.
(e3): umap-learn defaults (n_neighbors = 15, min_dist = 0.1) with random_state = seed.
"""
import numpy as np
import umap

from common.embed import pca_init, two_stage
from common.knn import knn_graph


def embed_e2(Z10, seed):
    Z = Z10.astype(np.float32)
    return two_stage(pca_init(Z), knn_graph(Z, k=15), seed)[1]["embd"]


def embed_e3(Z10, seed):
    return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(Z10.astype(np.float32))
