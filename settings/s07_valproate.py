"""
settings/s07_valproate.py
─────────────────────────────────────────────────────────────────────────────
Setting 7: Valproate dose individualisation in epilepsy — Failure mode 2

R² = 0.73 for body-weight + age + interacting medication model.
K=6 dose arms (500–2000 mg/day).
Prescription: add CYP2C9 + UGT genotype → R² > 0.90 → R_mech > 1.0 nat.
"""

import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo

DOSES_MG   = np.array([500., 750., 1000., 1250., 1500., 2000.])
TARGET_LO  = 50.0    # mg/L
TARGET_HI  = 100.0   # mg/L
CL_POP     = 0.93    # L/h  (typical valproate clearance)
SIGMA_CL   = 0.33    # between-patient log-CL SD
SIGMA_CONC = 0.10
R2_BASE    = 0.73


def _optimal(log_CL):
    CL   = np.exp(log_CL)[:, np.newaxis]
    conc = DOSES_MG[np.newaxis, :] / (CL * 24)  # daily dose → trough
    lc   = np.log(np.clip(conc, 1e-9, None))
    p    = (norm.cdf((np.log(TARGET_HI) - lc) / SIGMA_CONC) -
            norm.cdf((np.log(TARGET_LO) - lc) / SIGMA_CONC))
    return p.argmax(axis=1)


def run(R2: float = R2_BASE, n_patients: int = 200_000, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)
    log_CL_true = rng.normal(np.log(CL_POP), SIGMA_CL, n_patients)
    sigma_res   = np.sqrt(1.0 - R2) * SIGMA_CL
    log_CL_hat  = rng.normal(log_CL_true, sigma_res)

    def sample_pistar(n): return _optimal(log_CL_true[:n])
    def sample_pihat(p):  return _optimal(log_CL_hat[:len(p)])

    result = rmech_monte_carlo(sample_pistar, sample_pihat,
                               n_patients=n_patients,
                               K=len(DOSES_MG), seed=seed)
    result.update({"setting": "Valproate dose individualisation (epilepsy)",
                   "failure_mode": 2, "R2": R2})
    return result


if __name__ == "__main__":
    r = run()
    print(f"Valproate  H(mu)={r['H_mu']:.3f}  R_mech={r['rmech']:.3f}")
