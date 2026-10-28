"""Design L (Phase 5b): benchmarks where linear suppression is structurally limited.

Common to both variants: 6 background clusters with centers c_k on a regular simplex (edge 30) in a 5-D subspace U
(map B_U), within-cluster "nuisance" w ~ N(0, I_3) scaled by sigma_w, 4 signal groups z_s on a simplex (edge 6) in 3-D,
and isotropic noise eps (std 0.1). Cluster counts are exactly balanced (differ by at most 1) and identical in X_T and
X_B, so there is no composition shift between target and background.

L-cyclic: six mutually orthogonal 3-D blocks Q_0..Q_5, orthogonal to U. In cluster k (indices mod 6)
    target:     x = B_U c_k + sigma_w Q_{k+1} w + rho Q_k z_s + eps
    background: x = B_U c_k + sigma_w Q_{k+1} w + eps
so cluster k's signal block is cluster k-1's nuisance block.

L-between: a shared nuisance block V (orthogonal to U) and a signal subspace W = B_U R inside U (R a random 5 x 3
matrix with orthonormal columns):
    target:     x = B_U c_k + sigma_w V w + rho W z_s + eps
    background: x = B_U c_k + sigma_w V w + eps
Within a cluster, the spread along U is noise only.

All randomness comes from `seed`, via independent streams for geometry, target samples and background samples.
"""
import numpy as np

from data.synthetic import DEFAULTS, SyntheticDataset, _orthonormal, _simplex_centers

VARIANTS = ("cyclic", "between")
N_BLOCKS = 6
D_W = 3
SIGMA_W = float(np.sqrt(DEFAULTS["d_b"] * DEFAULTS["sigma_b"] ** 2 / D_W))  # easy's within-cluster variance, in 3-D


def balanced_labels(rng, K, n):
    """n labels in 0..K-1 with counts differing by at most 1, in random order."""
    counts = np.full(K, n // K)
    counts[: n % K] += 1
    return rng.permutation(np.repeat(np.arange(K), counts))


def make_l(variant, seed=0, rho=1.0, n_T=5000, n_B=5000, D=50):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    p = DEFAULTS
    K, d_b, d_s = p["K_b"], p["d_b"], p["d_s"]
    geo_rng, target_rng, bg_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    centers_b = _simplex_centers(geo_rng, K, d_b, p["edge_b"])
    centers_s = _simplex_centers(geo_rng, p["K_s"], d_s, p["edge_s"])
    Q = _orthonormal(geo_rng, D, d_b + N_BLOCKS * D_W)
    B_U = Q[:, :d_b]
    blocks = np.stack([Q[:, d_b + D_W * i: d_b + D_W * (i + 1)] for i in range(N_BLOCKS)])  # (6, D, 3)
    if variant == "cyclic":
        signal_maps = blocks
        nuisance_maps = blocks[(np.arange(K) + 1) % N_BLOCKS]
        extra = {"Q": blocks}
    else:
        V = blocks[0]
        W = B_U @ _orthonormal(geo_rng, d_b, d_s)
        signal_maps = np.repeat(W[None], K, axis=0)
        nuisance_maps = np.repeat(V[None], K, axis=0)
        extra = {"V": V, "W": W}

    def sample(rng, n, with_signal):
        y_b = balanced_labels(rng, K, n)
        w = rng.standard_normal((n, D_W))
        bg = centers_b[y_b] @ B_U.T + SIGMA_W * np.einsum("nd,nDd->nD", w, nuisance_maps[y_b])
        out = {"y_b": y_b, "Bg": bg}
        if with_signal:
            y_s = rng.integers(p["K_s"], size=n)
            z_s = centers_s[y_s] + p["sigma_s"] * rng.standard_normal((n, d_s))
            out.update({"y_s": y_s, "z_s": z_s, "S": rho * np.einsum("nd,nDd->nD", z_s, signal_maps[y_b])})
        out["eps"] = p["noise_std"] * rng.standard_normal((n, D))
        return out

    t, b = sample(target_rng, n_T, True), sample(bg_rng, n_B, False)
    X_T = (t["S"] + t["Bg"] + t["eps"]).astype(np.float32)
    X_B = (b["Bg"] + b["eps"]).astype(np.float32)
    truth = {"y_s": t["y_s"], "y_b": t["y_b"], "y_b_background": b["y_b"], "S_T": t["S"], "signal_T": t["S"],
             "Bg_T": t["Bg"], "z_s": t["z_s"], "B": B_U, **extra}
    params = {"design": f"L-{variant}", "variant": variant, "n_T": n_T, "n_B": n_B, "D": D, "seed": seed,
              "rho": float(rho), "sigma_w": SIGMA_W, "noise_std": p["noise_std"], "edge_b": p["edge_b"],
              "edge_s": p["edge_s"], "map_info": {}}
    return SyntheticDataset(X_T=X_T, X_B=X_B, truth=truth, params=params)
