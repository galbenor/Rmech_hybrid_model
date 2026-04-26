"""
settings/s01_5fu.py
─────────────────────────────────────────────────────────────────────────────
Setting 1: 5-FU FOLFOX dose individualisation — Failure mode 2

Population PK parameters:
  Kaldate et al. The Oncologist 2012 (N=187 patients, 307 cycle-pairs)
  Li et al. ESMO Open 2023 (multi-centre)
  Gamelin et al. JCO 2008 (French RCT cohort)

Therapeutic target: AUC 20–30 mg·h/L (Gamelin 2008)
Dose arms (mg/m²): 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3200
"""

import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo, convergence_factor

# ── Population PK parameters ──────────────────────────────────────────────────
DOSE_LEVELS   = np.array([1600., 1800., 2000., 2200., 2400., 2600., 2800., 3200.])
CL_POP_MEAN   = 96.0      # L/h  (population mean clearance)
SIGMA_CL_LOG  = 0.75      # log-scale between-patient SD of clearance
SIGMA_AUC_LOG = 0.08      # within-patient AUC measurement noise (log scale)
WIN_LO        = 20.0      # mg·h/L  therapeutic window lower bound
WIN_HI        = 30.0      # mg·h/L  therapeutic window upper bound

R2_BASE       = 0.51      # Kaldate 2012 population PK model R²
R2_RANGE      = {         # Published model R² values for sensitivity analysis
    "de_With_2016_EU":   0.38,
    "Gamelin_2008_FR":   0.44,
    "Kaldate_2012_US":   0.51,   # base case
    "Li_2023_multi":     0.58,
    "Peng_2022_Asian":   0.63,
    "DPYD_sex_enriched": 0.75,   # projected enrichment
}


def _optimal_arm(log_CL: np.ndarray) -> np.ndarray:
    """Return index of dose arm with highest probability of landing in window."""
    CL  = np.exp(log_CL)[:, np.newaxis]
    auc = DOSE_LEVELS[np.newaxis, :] / CL
    lau = np.log(np.clip(auc, 1e-9, None))
    p   = (norm.cdf((np.log(WIN_HI) - lau) / SIGMA_AUC_LOG) -
           norm.cdf((np.log(WIN_LO) - lau) / SIGMA_AUC_LOG))
    return p.argmax(axis=1)


def run(R2: float = R2_BASE, n_patients: int = 200_000, seed: int = 2024) -> dict:
    """
    Compute R_mech for 5-FU FOLFOX dose individualisation.

    Parameters
    ----------
    R2 : float
        Population PK model R² (fraction of between-patient CL variance explained).
    n_patients : int
        Monte Carlo sample size.
    seed : int

    Returns
    -------
    dict from rmech_monte_carlo, plus 'setting' and 'R2' keys.
    """
    rng = np.random.default_rng(seed)

    # Residual SD after model prediction
    sigma_res = np.sqrt(1.0 - R2) * SIGMA_CL_LOG

    # True patient clearances
    log_CL_true = rng.normal(np.log(CL_POP_MEAN), SIGMA_CL_LOG, n_patients)

    # Model predicts clearance with residual error
    log_CL_hat  = rng.normal(log_CL_true, sigma_res)

    def sample_pistar(n):
        return _optimal_arm(log_CL_true[:n])

    def sample_pihat(pistar):
        return _optimal_arm(log_CL_hat[:len(pistar)])

    result = rmech_monte_carlo(
        sample_pistar, sample_pihat,
        n_patients=n_patients, K=len(DOSE_LEVELS), seed=seed
    )
    result.update({
        "setting": "5-FU FOLFOX dose individualisation",
        "failure_mode": 2,
        "R2": R2,
    })
    return result


def sensitivity_sweep(n_patients: int = 100_000, seed: int = 2024) -> list:
    """Run across all published R² values."""
    results = []
    for model_name, R2 in R2_RANGE.items():
        r = run(R2=R2, n_patients=n_patients, seed=seed)
        r["model"] = model_name
        results.append(r)
        print(f"  {model_name:<26}  R²={R2:.2f}  R_mech={r['rmech']:.3f}  "
              f"{'✓ above' if r['threshold_passed'] else '✗ below'}")
    return results


if __name__ == "__main__":
    print("5-FU FOLFOX — base case (Kaldate 2012, R²=0.51)")
    r = run()
    print(f"  H(mu)  = {r['H_mu']:.3f} nats")
    print(f"  R_mech = {r['rmech']:.3f} nats  "
          f"({'ABOVE' if r['threshold_passed'] else 'below'} threshold)")
    print(f"  Factor = {convergence_factor(r['H_mu'], 1.0):.2f}× at threshold")
    print()
    print("Multi-model sensitivity:")
    sensitivity_sweep()
