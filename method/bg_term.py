"""Background-negative term for Stage 2 of Neg-t-SNE (Plan §4), used as the trainer's `batch_hook`.

For each directed edge in a batch with head i, m_B background tails are drawn (from xi_B(i, .), uniformly from all
other target points, or, for the oracle control, uniformly from the other members of i's true background cluster via
`same_cluster_xi`) and repelled with the package's own Neg-t-SNE repulsion: the same Cauchy kernel, constant c
(`criterion.spec_param`), eps and clamps. The term joins the batch loss with the package's aggregation, so per edge

    L_ij = (existing attraction and uniform repulsion) - gamma * sum_b log(1 - phi_ib / (phi_ib + c)).

Every mode also logs the loss components per epoch (without gradients or random draws). The sampler uses its own
torch.Generator, so the global torch RNG that drives the edge shuffle and the package's negatives is never touched.
"""
import numpy as np
import torch

from method.xi_b import XiB

BG_MODES = ("none", "xi_b", "uniform_extra", "oracle_cluster")
CSR_MODES = ("xi_b", "oracle_cluster")  # modes that sample tails from a per-row distribution (an XiB)
LOG_KEYS = ("attraction", "uniform_repulsion", "bg_repulsion", "bg_weighted", "n_edges", "n_masked_heads",
            "n_batches", "n_nonfinite_batches")


class BackgroundTerm:
    def __init__(self, bg_mode="none", m_B=0, gamma=1.0, n=None, xi=None, seed=0):
        if bg_mode not in BG_MODES:
            raise ValueError(f"bg_mode must be one of {BG_MODES}, got {bg_mode!r}")
        if int(m_B) != m_B or m_B < 0:
            raise ValueError(f"m_B must be a non-negative integer, got {m_B}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if bg_mode != "none" and n is None:
            raise ValueError("n (number of target points) is required for a background term")
        if bg_mode in CSR_MODES:
            if xi is None:
                raise ValueError(f"bg_mode={bg_mode!r} needs xi (a method.xi_b.XiB)")
            if xi.n != n:
                raise ValueError(f"xi has {xi.n} rows but n={n}")
        self.bg_mode, self.m_B, self.gamma, self.n, self.seed = bg_mode, int(m_B), float(gamma), n, seed
        self.active = bg_mode != "none" and self.m_B > 0
        # derived seed: a generator seeded with the bare run seed would replay the global torch stream
        self.generator = torch.Generator().manual_seed(int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]))
        if bg_mode in CSR_MODES:
            lengths = xi.row_lengths()
            cdf = np.zeros(len(xi.probs), dtype=np.float64)
            for i in np.flatnonzero(lengths):
                lo, hi = xi.indptr[i], xi.indptr[i + 1]
                c = np.cumsum(xi.probs[lo:hi])
                c[-1] = 1.0  # rows occupy [i, i + 1] exactly, so a draw can never leave its row
                cdf[lo:hi] = i + c
            self.cdf = torch.from_numpy(cdf)
            self.tails_flat = torch.from_numpy(xi.indices.astype(np.int64))
            self.empty = torch.from_numpy(lengths == 0)
        self.log = {}

    def params(self):
        return {"bg_mode": self.bg_mode, "m_B": self.m_B, "gamma": self.gamma, "n": self.n, "seed": self.seed}

    def sample(self, heads):
        """Background tails for each head: (b, m_B) target indices, and a (b,) mask of heads that have tails."""
        heads = heads.long()
        b = len(heads)
        if self.bg_mode in CSR_MODES:
            u = torch.rand((b, self.m_B), generator=self.generator, dtype=torch.float64)
            pos = torch.searchsorted(self.cdf, heads[:, None].double() + u, right=True)
            valid = ~self.empty[heads]
            pos = torch.where(valid[:, None], pos, torch.zeros_like(pos))  # placeholder for empty rows, masked below
            return self.tails_flat[pos], valid
        t = torch.randint(0, self.n - 1, (b, self.m_B), generator=self.generator)
        return t + (t >= heads[:, None]).long(), torch.ones(b, dtype=torch.bool)

    @staticmethod
    def _repulsion(origs, others, criterion):
        """Package Neg-t-SNE per-entry repulsion -log(1 - est) and est itself, est = 1 / (1 + c (d^2 + eps))."""
        sq_dists = ((origs[:, None] - others) ** 2).sum(axis=2)
        est = 1 / (1 + criterion.spec_param * (sq_dists + criterion.eps))
        return -torch.log((1 - est).clamp(criterion.clamp_low, criterion.clamp_high)), est

    def __call__(self, item, neigh, features, model, criterion, epoch):
        if criterion.loss_mode != "neg" or criterion.metric != "euclidean":
            raise ValueError("the background term supports only loss_mode='neg' with metric='euclidean'")
        b = features.shape[0] // 2
        origs = features[:b]
        neigh_inds = criterion.neigh_inds  # the indices the package just used for this batch
        # mean aggregation divides every entry by b (m + 1); sum aggregation adds entries
        norm = 1.0 if criterion.loss_aggregation == "sum" else float(b * neigh_inds.shape[1])

        with torch.no_grad():
            rep, est = self._repulsion(origs, features[neigh_inds], criterion)
            attraction = -torch.log(est[:, 0].clamp(criterion.clamp_low, criterion.clamp_high)).sum() / norm
            uniform_rep = rep[:, 1:].sum() / norm

        extra, bg, n_masked = None, torch.zeros(()), 0
        if self.active:
            tails, valid = self.sample(item)
            bg_rep, _ = self._repulsion(origs, model(tails), criterion)
            bg = (bg_rep * valid[:, None]).sum() / norm
            extra = self.gamma * bg
            n_masked = int((~valid).sum())

        entry = self.log.setdefault(epoch, dict.fromkeys(LOG_KEYS, 0.0))
        bg_value = float(bg.detach())
        values = [float(attraction), float(uniform_rep), bg_value, self.gamma * bg_value]
        for key, v in zip(LOG_KEYS[:4], values):
            entry[key] += v
        entry["n_edges"] += b
        entry["n_masked_heads"] += n_masked
        entry["n_batches"] += 1
        entry["n_nonfinite_batches"] += int(not np.isfinite(values).all())
        return extra

    def epoch_log(self):
        """Per-epoch sums as lists in epoch order, plus the background share of the total loss."""
        epochs = sorted(self.log)
        out = {"epoch": epochs, **{k: [self.log[e][k] for e in epochs] for k in LOG_KEYS}}
        out["bg_share"] = [w / (a + u + w) if a + u + w > 0 else 0.0
                           for a, u, w in zip(out["attraction"], out["uniform_repulsion"], out["bg_weighted"])]
        return out


def same_cluster_xi(y_b):
    """Oracle control (labels!): an XiB whose row i is uniform over the other members of i's true background cluster."""
    y_b = np.asarray(y_b)
    members = {c: np.flatnonzero(y_b == c) for c in np.unique(y_b)}
    rows = [members[c][members[c] != i] for i, c in enumerate(y_b)]
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    probs = np.concatenate([np.full(len(r), 1.0 / len(r)) for r in rows])
    return XiB(indptr=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
               indices=np.concatenate(rows).astype(np.int64), jaccard=probs.copy(), probs=probs,
               params={"n_T": len(y_b), "kind": "oracle same-cluster (true y_b)",
                       "n_empty_rows": int((lengths == 0).sum())})
