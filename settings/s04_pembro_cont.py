"""
settings/s04_pembro_cont.py
─────────────────────────────────────────────────────────────────────────────
Setting 4: Pembrolizumab continuation (Q3W → Q6W) — Failure mode 1

Binary decision: continue Q3W vs extend to Q6W after confirmed response.
Responder prevalence ~35% → H(mu) = 0.65 nats < 1.0 nat threshold.
No mechanistic model can achieve R_mech ≥ 1.0 nat when H(mu) < 1.0 nat.

References: Freshwater et al. 2017, Ohuchi et al. 2022
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.core import rmech_monte_carlo

K                   = 2
RESPONDER_PREVALENCE = 0.35    # fraction for whom Q6W is optimal


def run(prevalence: float = RESPONDER_PREVALENCE,
        model_accuracy: float = 0.85,
        n_patients: int = 200_000, seed: int = 2024) -> dict:
    rng  = np.random.default_rng(seed)
    mu   = np.array([1.0 - prevalence, prevalence])

    def sample_pistar(n): return rng.choice(K, n, p=mu)
    def sample_pihat(pistar):
        correct = rng.random(len(pistar)) < model_accuracy
        wrong   = rng.integers(K, size=len(pistar))
        return np.where(correct, pistar, wrong)

    result = rmech_monte_carlo(sample_pistar, sample_pihat,
                               n_patients=n_patients, K=K, seed=seed)
    result.update({"setting": "Pembrolizumab continuation (Q3W→Q6W)",
                   "failure_mode": 1})
    return result


if __name__ == "__main__":
    r = run()
    print(f"Pembro continuation  H(mu)={r['H_mu']:.3f}  R_mech={r['rmech']:.3f}  "
          f"[Mode 1: H(mu) < 1.0 nat]")
