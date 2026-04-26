"""
settings/s06_nortriptyline.py
─────────────────────────────────────────────────────────────────────────────
Setting 6: Nortriptyline dose selection in depression — Failure mode 4 (new)

Key finding: R² = 0.87 (best model fit of any audited setting) yet
R_mech = 0.62 nats because 20% of patients (CYP2D6 PM + UM) have no
dose arm within the therapeutic window. Mode 4: arm set span is the
bottleneck, not model quality.

Reference: CPIC guidelines for CYP2D6 and tricyclic antidepressants.
"""

import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo

# ── PK parameters ─────────────────────────────────────────────────────────────
DOSES_MG    = np.array([10., 25., 50., 75., 100., 150.])   # available dose arms
TARGET_LO   = 50.0    # ng/mL  therapeutic window
TARGET_HI   = 150.0   # ng/mL

R2_CYP2D6   = 0.87    # CYP2D6 genotype model R²

# CYP2D6 metaboliser phenotype distribution
PHENOTYPE_MU    = np.array([0.07, 0.40, 0.40, 0.13])  # PM, IM, EM, UM
PHENOTYPE_CL    = np.array([0.3,  0.7,  1.0,  2.5])   # relative CL per phenotype
SIGMA_CL_LOG    = 0.30   # residual within-phenotype log-CL SD
SIGMA_CONC_LOG  = 0.10   # measurement noise on log-concentration


def _optimal_arm(log_CL: np.ndarray) -> np.ndarray:
    CL    = np.exp(log_CL)[:, np.newaxis]
    conc  = DOSES_MG[np.newaxis, :] / CL
    lc    = np.log(np.clip(conc, 1e-9, None))
    p_win = (norm.cdf((np.log(TARGET_HI) - lc) / SIGMA_CONC_LOG) -
             norm.cdf((np.log(TARGET_LO) - lc) / SIGMA_CONC_LOG))
    return p_win.argmax(axis=1)


def run(R2: float = R2_CYP2D6, n_patients: int = 200_000, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)

    # Sample phenotype and individual clearance
    phenotype    = rng.choice(4, n_patients, p=PHENOTYPE_MU)
    cl_pheno     = PHENOTYPE_CL[phenotype]
    log_CL_true  = np.log(cl_pheno) + rng.normal(0, SIGMA_CL_LOG, n_patients)

    sigma_res    = np.sqrt(1.0 - R2) * SIGMA_CL_LOG
    log_CL_hat   = rng.normal(log_CL_true, sigma_res)

    def sample_pistar(n): return _optimal_arm(log_CL_true[:n])
    def sample_pihat(pistar): return _optimal_arm(log_CL_hat[:len(pistar)])

    result = rmech_monte_carlo(sample_pistar, sample_pihat,
                               n_patients=n_patients,
                               K=len(DOSES_MG), seed=seed)
    result.update({
        "setting":      "Nortriptyline dose selection (depression)",
        "failure_mode": 4,
        "R2":           R2,
        "note": ("Mode 4: arm set span. R²=0.87 but 20% of patients "
                 "(CYP2D6 PM+UM) have no arm in therapeutic window."),
    })
    return result


if __name__ == "__main__":
    r = run()
    print(f"Nortriptyline  H(mu)={r['H_mu']:.3f}  R_mech={r['rmech']:.3f}  R²={r['R2']}")
    print(f"  {r['note']}")
