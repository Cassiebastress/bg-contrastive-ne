"""Synthetic target/background datasets for background-contrastive embedding (Plan §2).

Target points:     x = rho * A z_s + B(z_b) + eps
Background points: x = B(z_b') + eps
with z_b, z_b' from the same Gaussian-mixture background latent, z_s from a Gaussian-mixture signal latent, and
eps isotropic Gaussian noise. Cluster centers of each latent are the K = d + 1 vertices of a regular simplex
(all pairs equidistant), randomly rotated per seed. Everything in `SyntheticDataset.truth` is ground truth for
evaluation only and must never be passed to the method.
"""
from dataclasses import dataclass

import numpy as np

LEVELS = ("easy", "medium", "hard")

DEFAULTS = {
    "K_b": 6, "d_b": 5, "edge_b": 30.0, "sigma_b": 1.0,  # background latent: clusters, dim, center distance, std
    "K_s": 4, "d_s": 3, "edge_s": 6.0, "sigma_s": 1.0,  # signal latent
    "noise_std": 0.1,  # isotropic noise in all D dims
    "overlap": 0.5,  # medium: fraction of A's column space inside span(B), ||B^T A||_F^2 / d_s
    "hidden": 64, "gain": 6.0, "bias_std": 0.1,  # hard: B(z) = W2 tanh(W1 z~ + b1), z~ standardized
    "n_ref": 20000,  # hard: reference sample size for matching the background variance
}


@dataclass
class SyntheticDataset:
    X_T: np.ndarray  # target, float32 (n_T, D)
    X_B: np.ndarray  # background, float32 (n_B, D)
    truth: dict  # evaluation-only ground truth
    params: dict  # generator settings and derived diagnostics


def _orthonormal(rng, D, d):
    """Random D x d matrix with orthonormal columns (Haar; QR with the sign fixed so it is unique)."""
    q, r = np.linalg.qr(rng.standard_normal((D, d)))
    return q * np.sign(np.diag(r))


def _simplex_centers(rng, K, d, edge):
    """K = d + 1 vertices of a regular simplex centered at 0 with the given edge length, randomly rotated."""
    if K != d + 1:
        raise ValueError(f"regular-simplex centers need K = d + 1, got K={K}, d={d}")
    V = np.eye(K) - 1.0 / K  # centered simplex in R^K with edge sqrt(2)
    U = np.linalg.svd(V)[0][:, :d]  # orthonormal basis of its d-dim span
    return (V @ U) * (edge / np.sqrt(2)) @ _orthonormal(rng, d, d).T


def _sample_mixture(rng, centers, sigma, n):
    """Equal-weight Gaussian mixture: labels, then points around their cluster centers."""
    y = rng.integers(len(centers), size=n)
    z = centers[y] + sigma * rng.standard_normal((n, centers.shape[1]))
    return y, z


def _min_center_distance(centers):
    d = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    return float(d[np.triu_indices(len(centers), k=1)].min())


def _make_maps(level, rng, D, p, centers_b):
    """Signal map A (D x d_s, orthonormal columns) and background map B (callable), plus map diagnostics."""
    d_b, d_s = p["d_b"], p["d_s"]
    if level == "easy":
        Q = _orthonormal(rng, D, d_b + d_s)
        B_mat, A = Q[:, :d_b], Q[:, d_b:]
    elif level == "medium":
        Q = _orthonormal(rng, D, d_b + d_s)
        Q1, Q2 = Q[:, :d_b], Q[:, d_b:]
        R = _orthonormal(rng, d_b, d_s)
        a = p["overlap"]
        B_mat, A = Q1, np.sqrt(a) * Q1 @ R + np.sqrt(1 - a) * Q2
    else:  # hard: nonlinear background, linear signal
        A = _orthonormal(rng, D, d_s)
        W1 = rng.normal(0.0, p["gain"] / np.sqrt(d_b), (p["hidden"], d_b))
        b1 = rng.normal(0.0, p["bias_std"], p["hidden"])
        W2 = rng.normal(0.0, 1.0 / np.sqrt(p["hidden"]), (D, p["hidden"]))
        # per-dim std of the background latent: simplex vertices have squared norm edge^2 (K-1) / (2K)
        latent_std = np.sqrt(p["edge_b"] ** 2 * (p["K_b"] - 1) / (2 * p["K_b"]) / d_b + p["sigma_b"] ** 2)

        def preact(z):
            return (z / latent_std) @ W1.T + b1

        # Rescale so the background's total variance equals that of an isometric linear map of the same latent.
        _, z_ref = _sample_mixture(rng, centers_b, p["sigma_b"], p["n_ref"])
        raw = np.tanh(preact(z_ref)) @ W2.T
        scale = np.sqrt(z_ref.var(axis=0).sum() / raw.var(axis=0).sum())
        pre_ref = np.abs(preact(z_ref))

        def B(z):
            return scale * np.tanh(preact(z)) @ W2.T

        info = {"output_scale": float(scale), "median_abs_preactivation": float(np.median(pre_ref)),
                "frac_abs_preactivation_gt_2": float((pre_ref > 2).mean())}
        return A, B, None, info

    info = {"overlap_A_B": float(np.linalg.norm(B_mat.T @ A) ** 2 / d_s)}
    return A, (lambda z: z @ B_mat.T), B_mat, info


def make_dataset(level, n_T=5000, n_B=5000, D=50, seed=0, signal_to_background=1.0, **overrides):
    """Generate one synthetic target/background dataset.

    All randomness comes from `seed`, via independent streams for centers, maps, target samples, and background
    samples, so `signal_to_background` (rho) changes no random draw and n_B does not affect X_T.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown generator parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **overrides}
    rho = float(signal_to_background)

    center_rng, map_rng, target_rng, bg_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))
    centers_b = _simplex_centers(center_rng, p["K_b"], p["d_b"], p["edge_b"])
    centers_s = _simplex_centers(center_rng, p["K_s"], p["d_s"], p["edge_s"])
    A, B, B_mat, map_info = _make_maps(level, map_rng, D, p, centers_b)

    y_b, z_b = _sample_mixture(target_rng, centers_b, p["sigma_b"], n_T)
    y_s, z_s = _sample_mixture(target_rng, centers_s, p["sigma_s"], n_T)
    eps_T = p["noise_std"] * target_rng.standard_normal((n_T, D))
    y_b_background, z_b_background = _sample_mixture(bg_rng, centers_b, p["sigma_b"], n_B)
    eps_B = p["noise_std"] * bg_rng.standard_normal((n_B, D))

    S_T = rho * (z_s @ A.T)
    Bg_T = B(z_b)
    X_T = (S_T + Bg_T + eps_T).astype(np.float32)
    X_B = (B(z_b_background) + eps_B).astype(np.float32)

    truth = {"y_s": y_s, "y_b": y_b, "y_b_background": y_b_background, "S_T": S_T, "Bg_T": Bg_T,
             "z_s": z_s, "z_b": z_b, "A": A, "B": B_mat}
    params = {"level": level, "n_T": n_T, "n_B": n_B, "D": D, "seed": seed, "signal_to_background": rho, **p,
              "min_center_distance_b": _min_center_distance(centers_b),
              "min_center_distance_s": _min_center_distance(centers_s), "map_info": map_info}
    return SyntheticDataset(X_T=X_T, X_B=X_B, truth=truth, params=params)
