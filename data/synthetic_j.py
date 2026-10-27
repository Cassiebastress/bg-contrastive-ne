"""Design J (Phase 5a): a jointly nonlinear signal whose effect depends on the background state.

Target points:     x = B z_b + rho * (cos(theta) * As~ + sin(theta) * Delta~) + eps
Background points: x = B z_b' + eps
with Delta = M(z_b, z_s) - M(z_b, 0) for a fixed random 2-layer tanh MLP M (input [z_b / std_b ; z_s / std_s],
hidden width 64), optionally projected off span(B), and As~ = A z_s. As~ and Delta~ are each scaled to unit total
variance over the dataset's own target samples, so theta mixes a background-independent and a background-dependent
signal at fixed total strength. s0 = 0 is the signal-simplex centroid, so X_B is exactly F(z_b', s0).

The latents, B, A, the noise and X_B are those of `make_dataset("easy", seed)`, bit for bit; M is drawn from a fifth,
independent stream of the same seed, and its weights do not depend on the gain (the gain scales them). At theta = 0,
Delta is never evaluated, so the dataset does not depend on `projected` or `gain`.
"""
import numpy as np

from data.synthetic import DEFAULTS, SyntheticDataset, make_dataset

THETAS = (0, 30, 60, 90)  # degrees
GAINS = (2.0, 6.0)
PROJECTIONS = (True, False)
HIDDEN, BIAS_STD = 64, 0.1
# analytic total variance of A z_s (A isometric): d_s (edge_s^2 (K_s - 1) / (2 K_s) / d_s + sigma_s^2) = 16.5
SIGNAL_TOTAL_VAR = (DEFAULTS["edge_s"] ** 2 * (DEFAULTS["K_s"] - 1) / (2 * DEFAULTS["K_s"])
                    + DEFAULTS["d_s"] * DEFAULTS["sigma_s"] ** 2)


def latent_std(K, d, edge, sigma):
    """Per-dim std of an equal-weight simplex mixture: vertices have squared norm edge^2 (K - 1) / (2K)."""
    return float(np.sqrt(edge ** 2 * (K - 1) / (2 * K) / d + sigma ** 2))


def extra_rng(seed):
    """The fifth stream of `seed`; the first four are make_dataset's (centers, maps, target, background)."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[4])


def _trig(theta):
    t = np.deg2rad(theta)
    c, s = np.cos(t), np.sin(t)
    return (0.0 if abs(c) < 1e-12 else float(c)), (0.0 if abs(s) < 1e-12 else float(s))


def _mlp(seed, D):
    """M's weights at unit gain: W1 (hidden x 8) ~ N(0, 1/8), b1 ~ N(0, BIAS_STD^2), W2 (D x hidden) ~ N(0, 1/64)."""
    rng = extra_rng(seed)
    d_in = DEFAULTS["d_b"] + DEFAULTS["d_s"]
    W1 = rng.normal(0.0, 1.0 / np.sqrt(d_in), (HIDDEN, d_in))
    b1 = rng.normal(0.0, BIAS_STD, HIDDEN)
    W2 = rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), (D, HIDDEN))
    return W1, b1, W2


def make_j(seed=0, rho=1.0, theta=0, projected=True, gain=2.0, n_T=5000, n_B=5000, D=50):
    """Generate one Design J dataset (see the module docstring); returns a SyntheticDataset."""
    base = make_dataset("easy", n_T=n_T, n_B=n_B, D=D, seed=seed, signal_to_background=1.0)
    t = base.truth
    p = DEFAULTS
    c, s = _trig(theta)
    As = t["z_s"] @ t["A"].T
    scale_A = float(np.sqrt(As.var(axis=0).sum()))
    signal = (rho * c / scale_A) * As
    info = {"scale_A": scale_A}
    if s != 0.0:
        W1, b1, W2 = _mlp(seed, D)
        std_b = latent_std(p["K_b"], p["d_b"], p["edge_b"], p["sigma_b"])
        std_s = latent_std(p["K_s"], p["d_s"], p["edge_s"], p["sigma_s"])

        def preact(z_b, z_s):
            return np.hstack([z_b / std_b, z_s / std_s]) @ (gain * W1).T + b1

        pre = preact(t["z_b"], t["z_s"])
        delta = np.tanh(pre) @ W2.T - np.tanh(preact(t["z_b"], np.zeros_like(t["z_s"]))) @ W2.T
        Bm = t["B"]
        centered = delta - delta.mean(axis=0)
        frac_in_B = float(((centered @ Bm) ** 2).sum() / (centered ** 2).sum())
        if projected:
            delta = delta - (delta @ Bm) @ Bm.T
        scale_D = float(np.sqrt(delta.var(axis=0).sum()))
        signal = signal + (rho * s / scale_D) * delta
        info.update({"scale_Delta": scale_D, "median_abs_preactivation": float(np.median(np.abs(pre))),
                     "frac_abs_preactivation_gt_2": float((np.abs(pre) > 2).mean()),
                     "frac_Delta_var_in_span_B": frac_in_B})
    X_T = (signal + t["Bg_T"] + t["eps_T"]).astype(np.float32)
    truth = {k: t[k] for k in ("y_s", "y_b", "y_b_background", "Bg_T", "z_s", "z_b", "A", "B")}
    truth["signal_T"] = signal
    params = {"design": "J", "n_T": n_T, "n_B": n_B, "D": D, "seed": seed, "rho": float(rho), "theta": theta,
              "projected": bool(projected), "gain": float(gain), "hidden": HIDDEN, "bias_std": BIAS_STD,
              "map_info": info}
    return SyntheticDataset(X_T=X_T, X_B=base.X_B, truth=truth, params=params)
