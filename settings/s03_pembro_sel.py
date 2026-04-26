"""
settings/s03_pembro_sel.py
─────────────────────────────────────────────────────────────────────────────
Setting 3: Pembrolizumab treatment selection in advanced NSCLC — Failure mode 2

Arms (KEYNOTE-189/-407 era, MSI-H/dMMR-negative population):
  pi0: pembrolizumab monotherapy     (TPS ≥50%)
  pi1: pembrolizumab + chemotherapy  (all TPS strata)
  pi2: nivolumab + ipilimumab + chemo
  pi3: chemotherapy alone

Data: KEYNOTE-024, -189, -407, -598 response rates by PD-L1 stratum.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo

K      = 4
MU     = np.array([0.30, 0.40, 0.20, 0.10])   # arm prior
R2_TPS = 0.25                                   # PD-L1 TPS outcome variance explained
SIGMA_OUTCOME = 0.40                            # residual outcome noise


def run(R2: float = R2_TPS, n_patients: int = 200_000, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)
    sigma_res = np.sqrt(1.0 - R2)

    def sample_pistar(n):
        return rng.choice(K, n, p=MU)

    def sample_pihat(pistar):
        # Model observes a noisy signal of the true optimal arm
        signal = pistar.astype(float) + rng.normal(0, sigma_res / SIGMA_OUTCOME,
                                                    len(pistar))
        return np.clip(np.round(signal), 0, K - 1).astype(int)

    result = rmech_monte_carlo(sample_pistar, sample_pihat,
                               n_patients=n_patients, K=K, seed=seed)
    result.update({"setting": "Pembrolizumab treatment selection (NSCLC)",
                   "failure_mode": 2, "R2_TPS": R2})
    return result


if __name__ == "__main__":
    r = run()
    print(f"Pembro selection  H(mu)={r['H_mu']:.3f}  R_mech={r['rmech']:.3f}")
