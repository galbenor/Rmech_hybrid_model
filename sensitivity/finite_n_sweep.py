"""
sensitivity/finite_n_sweep.py — Finite-N regret validation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmech.finite_n import sweep
import numpy as np

if __name__ == "__main__":
    print("Finite-N regret validation (K=4, H(mu)~1.4 nats)\n")
    mu = np.array([0.30, 0.30, 0.25, 0.15])
    results = sweep(n_patients=5_000, mu_prior=mu)
    print(f"H(mu)={results['H_mu']:.3f} nats\n")
    print(f"{'Rmech':>8}  {'N=8':>8}  {'N=10':>8}  {'N=12':>8}  {'Red% N=10':>12}")
    print("-"*56)
    for i, rm in enumerate(results["rmech_vals"]):
        r = results["regret"]
        p = results["regret_reduction_pct"]
        tag = " <-- threshold" if abs(rm-1.0)<0.01 else ""
        print(f"{rm:>8.2f}  {r[i,0]:>8.3f}  {r[i,1]:>8.3f}  "
              f"{r[i,2]:>8.3f}  {p[i,1]:>11.1f}%{tag}")
