"""
settings/s02_braf.py
─────────────────────────────────────────────────────────────────────────────
Setting 2: BRAF-mutated melanoma treatment sequencing — Failure mode 3

Data sources:
  DREAMseq (Atkins et al. NEJM 2022)
  SECOMBIT (Mandala et al. JCO 2022)
  COMBI-d/v (Long et al. Lancet Oncol 2015)
  CheckMate 067 (Wolchok et al. NEJM 2022)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo

# ── Arm definitions ───────────────────────────────────────────────────────────
# pi0: ICI-first  (BRAF+, normal LDH)
# pi1: targeted-first (BRAF+, elevated LDH)
# pi2: sandwich — 8 wk targeted then ICI (BRAF+, intermediate burden)
# pi3: ICI-only  (BRAF wild-type — only option)
K = 4

# Base-case arm proportions (DREAMseq/SECOMBIT mixed-prevalence cohort)
MU_BASE = np.array([0.25, 0.12, 0.08, 0.55])   # sums to 1.0

# BRAF PCR sensitivity and LDH model accuracy
BRAF_PCR_SENSITIVITY = 0.98
RECOMMENDATION_ACCURACY_BRAFPLUS = 0.87   # overall within BRAF+ subgroup


def run(
    wt_fraction:  float        = 0.55,
    accuracy:     float        = RECOMMENDATION_ACCURACY_BRAFPLUS,
    n_patients:   int          = 200_000,
    seed:         int          = 2024,
) -> dict:
    """
    Parameters
    ----------
    wt_fraction : float
        Fraction of patients who are BRAF wild-type (only arm pi3 available).
    accuracy : float
        Overall model recommendation accuracy within BRAF+ patients.
    """
    rng = np.random.default_rng(seed)

    # Scale BRAF+ arm proportions to match wt_fraction
    braf_plus = 1.0 - wt_fraction
    base_braf = np.array([0.25, 0.12, 0.08])
    scaled    = base_braf / base_braf.sum() * braf_plus
    mu        = np.append(scaled, wt_fraction)

    def sample_pistar(n):
        return rng.choice(K, n, p=mu)

    def sample_pihat(pistar):
        pihat    = np.zeros(len(pistar), dtype=int)
        # BRAF wild-type: identified with PCR sensitivity
        is_wt    = pistar == 3
        wt_idx   = np.where(is_wt)[0]
        corr_wt  = rng.random(len(wt_idx)) < BRAF_PCR_SENSITIVITY
        pihat[wt_idx[corr_wt]]  = 3
        pihat[wt_idx[~corr_wt]] = rng.integers(3, size=(~corr_wt).sum())
        # BRAF+: accuracy-based model
        bp_idx   = np.where(~is_wt)[0]
        corr_bp  = rng.random(len(bp_idx)) < accuracy
        pihat[bp_idx[corr_bp]]  = pistar[bp_idx[corr_bp]]
        pihat[bp_idx[~corr_bp]] = rng.integers(3, size=(~corr_bp).sum())
        return pihat

    result = rmech_monte_carlo(
        sample_pistar, sample_pihat,
        n_patients=n_patients, K=K, seed=seed
    )
    result.update({
        "setting":      "BRAF melanoma treatment sequencing",
        "failure_mode": 3,
        "wt_fraction":  wt_fraction,
    })
    return result


def sensitivity_sweep(n_patients: int = 100_000, seed: int = 2024) -> list:
    """R_mech across BRAF wild-type prevalence range 45%–65%."""
    results = []
    for wt in [0.45, 0.50, 0.55, 0.60, 0.65]:
        r = run(wt_fraction=wt, n_patients=n_patients, seed=seed)
        results.append(r)
        base = " ← base" if wt == 0.55 else ""
        print(f"  WT={wt:.0%}  H(mu)={r['H_mu']:.3f}  "
              f"R_mech={r['rmech']:.3f}  "
              f"{'✓ below' if not r['threshold_passed'] else '✗ ABOVE'}{base}")
    return results


if __name__ == "__main__":
    print("BRAF melanoma — base case (WT=55%)")
    r = run()
    print(f"  H(mu)  = {r['H_mu']:.3f} nats")
    print(f"  R_mech = {r['rmech']:.3f} nats")
    print()
    print("Arm proportion sensitivity:")
    sensitivity_sweep()
