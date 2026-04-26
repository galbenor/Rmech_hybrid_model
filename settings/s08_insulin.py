"""
settings/s08_insulin.py
─────────────────────────────────────────────────────────────────────────────
Setting 8: Insulin total daily dose (TDD) in type 1 diabetes — Failure mode 2

K=8 TDD arms (0.2–1.4 U/kg/day).
Insulin sensitivity (ISI) varies 5–10-fold; model explains 65% of variance.
Prescription: add CGM-derived ISI estimate + residual C-peptide.
"""

import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo

TDD_ARMS    = np.array([0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00, 1.40])
ISI_MEAN    = 1.0     # normalised insulin sensitivity index
SIGMA_ISI   = 0.55    # between-patient log-ISI SD (5–10× range ≈ 2 SD)
SIGMA_BG    = 0.12    # blood glucose noise
TARGET_LO   = 0.45   # normalised glucose target lower bound
TARGET_HI   = 0.65
R2_BASE     = 0.65


def _optimal(log_ISI):
    ISI  = np.exp(log_ISI)[:, np.newaxis]
    bg   = TDD_ARMS[np.newaxis, :] * ISI
    lbg  = np.log(np.clip(bg, 1e-9, None))
    p    = (norm.cdf((np.log(TARGET_HI) - lbg) / SIGMA_BG) -
            norm.cdf((np.log(TARGET_LO) - lbg) / SIGMA_BG))
    return p.argmax(axis=1)


def run(R2: float = R2_BASE, n_patients: int = 200_000, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)
    log_ISI_true = rng.normal(np.log(ISI_MEAN), SIGMA_ISI, n_patients)
    sigma_res    = np.sqrt(1.0 - R2) * SIGMA_ISI
    log_ISI_hat  = rng.normal(log_ISI_true, sigma_res)

    def sample_pistar(n): return _optimal(log_ISI_true[:n])
    def sample_pihat(p):  return _optimal(log_ISI_hat[:len(p)])

    result = rmech_monte_carlo(sample_pistar, sample_pihat,
                               n_patients=n_patients,
                               K=len(TDD_ARMS), seed=seed)
    result.update({"setting": "Insulin TDD in type 1 diabetes",
                   "failure_mode": 2, "R2": R2})
    return result


if __name__ == "__main__":
    r = run()
    print(f"Insulin  H(mu)={r['H_mu']:.3f}  R_mech={r['rmech']:.3f}")
