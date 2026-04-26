"""
settings/s05_lithium.py
─────────────────────────────────────────────────────────────────────────────
Setting 5: Lithium dose individualisation in bipolar disorder — Failure mode 1

K=5 dose arms (300–1500 mg/day).
Narrow clearance distribution → 90% of patients optimally served by same dose.
H(mu) = 0.45 nats < 1.0 nat threshold.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo

K    = 5
# Narrow clearance distribution: ~90% of patients optimally on central dose.
# This gives H(mu) ≈ 0.46 nats (paper: 0.45 nats).
MU   = np.array([0.025, 0.025, 0.90, 0.025, 0.025])
DOSES = np.array([300., 600., 900., 1200., 1500.])  # mg/day


def run(n_patients: int = 200_000, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)

    def sample_pistar(n): return rng.choice(K, n, p=MU)
    def sample_pihat(pistar):
        # PK model for lithium: creatinine clearance + age — 75% accuracy
        correct = rng.random(len(pistar)) < 0.75
        return np.where(correct, pistar, rng.integers(K, size=len(pistar)))

    result = rmech_monte_carlo(sample_pistar, sample_pihat,
                               n_patients=n_patients, K=K, seed=seed)
    result.update({"setting": "Lithium dose individualisation (bipolar)",
                   "failure_mode": 1})
    return result


if __name__ == "__main__":
    r = run()
    print(f"Lithium  H(mu)={r['H_mu']:.3f}  R_mech={r['rmech']:.3f}  "
          f"[Mode 1: H(mu)={r['H_mu']:.2f} < 1.0 nat]")
