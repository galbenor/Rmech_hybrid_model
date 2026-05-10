# The Value of Mechanistic Priors in Sequential Decision Making

Code accompanying the paper *The Value of Mechanistic Priors in Sequential Decision Making*. This repository reproduces the simulation results and sensitivity figures from Section 5 and Appendix H.

## Overview

The paper introduces **mechanistic information** `R_mech = I(π*; π̂)` — the mutual information between a hybrid mechanistic model's recommended policy `π̂` and the true optimal policy `π*` — and uses it to characterize the sample-complexity gain a mechanistic prior provides in a K-armed policy bandit. The framework is instantiated on adaptive 5-fluorouracil (5-FU) dosing under FOLFOX chemotherapy, where the policy class spans 8 dose levels and one round corresponds to one chemotherapy cycle.

The code in this repository:

- Implements Thompson Sampling with both an uninformed prior (`TS_uninf`) and a hybrid prior of exact entropy `H_mech = H(µ) − R_mech` (`TS_hyb`), plus a body-surface-area (BSA) standard-of-care baseline.
- Reproduces **Tables 1 and 2** (cumulative regret across `R_mech` values and across cycle horizons `N`).
- Reproduces the **sensitivity figures** (Figures 2 and 3) covering the model-quality certificate `B_µ / B_µ^crit` over the calibration parameters `(κ_µ, B_µ, d_F, K)`.

## Repository structure

```
.
├── rmech/
│   ├── __init__.py
│   └── finite_n.py             # Core Thompson Sampling simulator and the
│                               # mu_hyb prior construction (linear-mixture
│                               # PMF with target entropy H(µ) − R_mech).
├── create_table2_new_values.py # Reproduces Tables 1 and 2.
├── sensitivity_analysis.py     # Reproduces Figures 2 and 3 (sensitivity of
│                               # C(B_µ) and B_µ / B_µ^crit to the
│                               # calibration parameters).
├── test_entropy.py             # Small diagnostic: verifies that the
│                               # constructed mu_hyb prior has the target
│                               # entropy. Optional.
└── requirements.txt
```

## Installation

The code runs on CPU only and has no GPU or special hardware requirements. Tested with Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install matplotlib             # required for sensitivity_analysis.py
```

## Reproducing the results

All results are deterministic given the random seed (default `seed = 42`, `M = 10000` Monte Carlo trials per cell, `K = 8` dose levels).

### Tables 1 and 2 — cumulative regret

```bash
python create_table2_new_values.py
```

Prints two tables to standard output:

- **Table 1.** Cumulative regret at `N = 12` cycles for `R_mech ∈ {0.0, 0.3, 0.8, 1.4, 1.9}` nats, comparing `TS_hyb` against `TS_uninf` and the BSA fixed-dose baseline. Columns include the predicted lower-bound ratio `√(H(µ)/H_mech)` and the observed `Uninf/TS_hyb` and `BSA/TS_hyb` ratios.
- **Table 2.** Cumulative regret at `R_mech = 1.9` nats for `N ∈ {5, 10, 20, 50, 100}` cycles, separating the burn-in, transitional, and asymptotic regimes.

Approximate runtime: a few minutes on a modern laptop.

### Figures 2 and 3 — sensitivity analysis

```bash
python sensitivity_analysis.py
```

Writes two PDFs to the working directory:

- **`fig_sensitivity_full.pdf`** (Figure 2 in the paper): a 2×3 grid showing the channel capacity `C(B_µ)` as a function of `κ_µ`, `B_µ`, and `d_F` (top row) and heatmaps of the certificate ratio `B_µ / B_µ^crit` over each parameter pair (bottom row), with the calibrated 5-FU operating point marked.
- **`fig_sensitivity_K.pdf`** (Figure 3 in the paper): the phase-transition threshold `B_µ^crit(N=12)` and the asymptotic sample-complexity ratio `ρ = H(µ) / H_mech` plotted against the number of arms `K`.

Approximate runtime: under a minute.

### Diagnostic (optional)

```bash
python test_entropy.py
```

Prints the entropy of the constructed `mu_hyb` prior at `K = 8`, `R_mech = 0.87` and verifies it matches the closed-form target `H(µ) − R_mech`. Useful as a sanity check when modifying the prior construction.

## Calibration

The simulation parameters — `σ = 0.40`, `κ_µ = 1.8`, `d_F = 3`, `B_µ = 0.22`, dose grid `1600–3600 mg/m²`, target AUC window `20–30 mg·h/L`, infusion horizon 46 h — are calibrated to published 5-FU pharmacokinetic data. Full provenance and sensitivity sweeps are documented in Appendix H of the paper. The constants are exposed as named module-level variables near the top of `create_table2_new_values.py` and `sensitivity_analysis.py` and can be edited in place to explore alternative settings.

## License

To be added.

## Citation

Citation details will be added once the paper is published.
