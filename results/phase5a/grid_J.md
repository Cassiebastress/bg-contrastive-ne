# Design J calibration grid

## Pre-registered prediction (committed before any Design J dataset was evaluated)

**Design J.**
- Target: X_T = B z_b + ρ (cos θ · Ã z_s + sin θ · Δ̃) + ε, where Δ = M(z_b, z_s) − M(z_b, 0) for a fixed random 2-layer tanh MLP M.
- Background: X_B = B z_b′ + ε′.
- Ã z_s and Δ̃ are each normalized to unit total variance.
- **Grid:** θ ∈ {0°, 30°, 60°, 90°} × P ∈ {projected: Δ off span(B), unprojected} × MLP gain g ∈ {2, 6}, for 16 configurations.
- **ρ** is calibrated at θ = 0° with the Phase 1 rule and held fixed.

**Prediction.**
- **Projected configurations can fail V3 (cPCA → Neg-t-SNE does not solve the dataset) only where coherence is low (V4 fails).** When Δ is projected off span(B), the background enters X_T only through B z_b, which has the same distribution as in X_B. cPCA then removes exactly the background main effect, and what remains is the whole signal, interaction included. If the signal is still shared across background clusters (high coherence), the top cPCA directions carry it.
- **Unprojected configurations may fail V3 while coherence stays high.** There, part of the signal lies in background directions, which cPCA suppresses.

**How it is checked** (selection seeds 0–2):
- **Contradicted** if any projected configuration meets both V3 and V4 on every seed.
- **Second claim supported** if an unprojected configuration meets both V3 and V4 on every seed. It is a "may", so its absence does not contradict the prediction.
- Per-seed counts are reported as well.

**Criteria** (from the Phase 5a plan):

| Criterion | Meaning | Condition |
|---|---|---|
| V1 | background dominates | (a) signal accuracy ≤ 0.50 and (a) background accuracy ≥ 0.90 |
| V2 | signal present | high-D within-cluster signal accuracy ≥ 0.85 |
| V3 | cPCA does not solve it | (e2) fails at least one of: G1 ≥ 0.50, G2 ≤ 0.50, G3 "shares signal" ≥ 0.50, signal accuracy > 0.80 |
| V4 | signal still shared | coherence ≥ 0.5 |

The generated grid table is appended below this section. This section stays byte-identical to its first commit.

<!-- generated below -->
