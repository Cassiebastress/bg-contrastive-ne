"""Design K (Phase 5a): cluster-specific signal directions.

Target points:     x = B z_b + rho * A_{y_b} z_s + eps
Background points: x = B z_b' + eps
Each background cluster k has its own signal map A_k (D x d_s, orthonormal columns), drawn independently inside the
orthogonal complement of span(B). The latents, B, the noise and X_B are those of `make_dataset("easy", seed)`, bit for
bit; the A_k come from a fifth, independent stream of the same seed.
"""
import numpy as np

from data.synthetic import DEFAULTS, SyntheticDataset, make_dataset
from data.synthetic_j import extra_rng


def cluster_maps(seed, B, n_maps, d_s):
    """n_maps random D x d_s maps with orthonormal columns orthogonal to span(B) (QR with the sign fixed)."""
    rng = extra_rng(seed)
    maps = []
    for _ in range(n_maps):
        g = rng.standard_normal((B.shape[0], d_s))
        q, r = np.linalg.qr(g - B @ (B.T @ g))
        maps.append(q * np.sign(np.diag(r)))
    return np.stack(maps)


def make_k(seed=0, rho=1.0, n_T=5000, n_B=5000, D=50):
    """Generate one Design K dataset (see the module docstring); returns a SyntheticDataset."""
    base = make_dataset("easy", n_T=n_T, n_B=n_B, D=D, seed=seed, signal_to_background=1.0)
    t = base.truth
    A_k = cluster_maps(seed, t["B"], DEFAULTS["K_b"], DEFAULTS["d_s"])
    S_T = rho * np.einsum("nd,nDd->nD", t["z_s"], A_k[t["y_b"]])
    X_T = (S_T + t["Bg_T"] + t["eps_T"]).astype(np.float32)
    truth = {k: t[k] for k in ("y_s", "y_b", "y_b_background", "Bg_T", "z_s", "z_b", "B")}
    truth.update({"S_T": S_T, "signal_T": S_T, "A_k": A_k})
    overlap = [float(np.linalg.norm(A_k[i].T @ A_k[j]) ** 2 / DEFAULTS["d_s"])
               for i in range(len(A_k)) for j in range(i + 1, len(A_k))]
    params = {"design": "K", "n_T": n_T, "n_B": n_B, "D": D, "seed": seed, "rho": float(rho),
              "map_info": {"mean_pairwise_subspace_overlap": float(np.mean(overlap)),
                           "max_pairwise_subspace_overlap": float(np.max(overlap))}}
    return SyntheticDataset(X_T=X_T, X_B=base.X_B, truth=truth, params=params)
