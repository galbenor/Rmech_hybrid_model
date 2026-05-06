"""
Reproduce Tables 1 and 2 of "Mechanistic Information and Regret"

"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from rmech.finite_n import run_thompson_sampling

# ============================================================================
#  Helper functions (used by both phases)
# ============================================================================

def conditional_entropy_symmetric_channel(q: float, K: int) -> float:
    """H(pi_hat | pi*) for a symmetric channel with self-correct prob q + (1-q)/K."""
    p_correct = q + (1.0 - q) / K
    p_other   = (1.0 - q) / K
    H = 0.0
    if p_correct > 0:
        H -= p_correct * np.log(p_correct)
    if p_other > 0:
        H -= (K - 1) * p_other * np.log(p_other)
    return H


def accuracy_for_target_mi(target_mi: float, K: int) -> float:
    """Solve for q in [0,1] s.t. I(pi*; pi_hat) = target_mi, pi* uniform."""
    if target_mi <= 0.0:
        return 0.0
    log_K = np.log(K)
    if target_mi >= log_K - 1e-9:
        return 1.0
    target_H_cond = log_K - target_mi
    f = lambda q: conditional_entropy_symmetric_channel(q, K) - target_H_cond
    return brentq(f, 1e-12, 1.0 - 1e-12)


def H_mu_hyb_exact(R_mech: float, K: int) -> float:
    """Exact entropy of mu_hyb (Eq. 8 of the paper)."""
    if R_mech <= 0.0:
        return np.log(K)
    Z = np.exp(R_mech) + K - 1
    return np.log(Z) - R_mech * np.exp(R_mech) / Z


def B_crit_corrected(sigma: float, kappa_mu: float, H_mu: float, d_F: int, N: int) -> float:
    """B_crit from Eq. 10 (corrected, NOT small-SNR approximation).
    Returns 0.0 if no value of B_µ can yield C(B_µ) > 1 (i.e., d_F too small)."""
    inside = (2.0 * H_mu / d_F) / (np.exp(2.0 * H_mu/ (d_F * N)) - 1.0) - 1.0
    if inside <= 0:
        return 0.0
    return (sigma / kappa_mu) * np.sqrt(inside)


def channel_capacity_C(B_mu: float, sigma: float, kappa_mu: float,
                        sigma_F: float, d_F: int) -> float:
    """C(B_mu) from Eq. 4."""
    num = kappa_mu ** 2 * sigma_F ** 2
    den = kappa_mu ** 2 * B_mu ** 2 + sigma ** 2
    return 0.5 * d_F * np.log1p(num / den)


def sample_pi_hat(pi_star: int, q: float, K: int, rng) -> int:
    """Sample pi_hat from symmetric channel: P(=pi*)=q+(1-q)/K, others (1-q)/K."""
    if rng.random() < q:
        return pi_star
    return int(rng.integers(0, K))


# ============================================================================
#  Phase 1: deterministic 0/1 reward, audit code prior
# ============================================================================

def run_phase1(K: int, N: int, M: int, R_mech: float, seed: int):
    """Phase 1 using the finite-N Thompson Sampling algorithm from rmech/finite_n.py."""
    mu_prior = np.full(K, 1.0 / K, dtype=float)
    mean_regret, ci_96 = run_thompson_sampling(
        N_cycles=N,
        K=K,
        mu_prior=mu_prior,
        rmech=R_mech,
        n_patients=M,
        seed=seed,
    )
    return mean_regret, ci_96


# ============================================================================
#  Phase 2: graded reward, mu_hyb-exact prior, Bayesian regret
# ============================================================================

# ---- Calibration constants (Kaldate 2012 The Oncologist 17(3):296-302) ----
KALDATE_SLOPE = 0.02063   # mg.h/L per mg/m^2, Kaldate verbatim
SIGMA_INTRA   = 4.9       # mg.h/L, derived: |dAUC|/sqrt(2) from Kaldate |dAUC|=6.9
TARGET_LO     = 20.0      # mg.h/L, Kaldate Discussion ("AUC in the desired range")
TARGET_HI     = 30.0      # mg.h/L
TARGET_MID    = 25.0      # mg.h/L; modeling assumption: optimal arm puts mean
                          # AUC at target center. NOT Kaldate's baseline mean (20.2)
DOSE_MIN      = 1600.0    # mg/m^2, Kaldate clinical adjustment range
DOSE_MAX      = 3600.0    # mg/m^2

RMECH = 1.9  # RMECH for Table 2


def reward_prob_at_distance(d_steps: int, K: int) -> float:
    """P(AUC in [20,30]) at d-step distance from optimal arm.
    By symmetry of the integrand around 25, shifting up or down gives the same value."""
    dose_grid = np.linspace(DOSE_MIN, DOSE_MAX, K)
    dauc_step = KALDATE_SLOPE * (dose_grid[1] - dose_grid[0])
    mu_AUC = TARGET_MID + d_steps * dauc_step
    return float(norm.cdf((TARGET_HI - mu_AUC) / SIGMA_INTRA) -
                 norm.cdf((TARGET_LO - mu_AUC) / SIGMA_INTRA))


# ============================================================================
#  Top-level configuration
# ============================================================================

K = 8
N_TABLE1 = 12
M = 10000
SEED = 42

# CHANGED: anchored on the new ceiling 0.87.  Rows above 0.87 are speculative
# (above the calibrated channel capacity); kept for sensitivity analysis.
R_MECH_VALUES = [0.0, 0.3, 0.7, 1.0, 1.4, 1.9]

N_VALUES_TABLE2 = [5, 10, 20, 30, 50, 100, 150]
log_K = np.log(K)


def print_phase1_table1():
    print()
    print("=" * 110)
    print(f"PHASE 1, Table 1 (paper-aligned, deterministic reward, audit-code prior)")
    print(f"   K={K}, N={N_TABLE1}, M={M}, mu = Uniform")
    print(f"   Calibrated R_mech ceiling = {RMECH}; rows above are speculative")
    print("=" * 110)
    hdr = "{:>8} {:>8} {:>10} {:>14} {:>14} {:>10} {:>10} {:>10}".format(
        "R_mech", "H_mech", "H(mu_hyb)", "TS_hyb", "Uninf_TS",
        "Obs/UnInf", "LB_pred", "ratio")
    print(hdr); print("-" * 110)
    base_mean, base_se = run_phase1(K, N_TABLE1, M, 0.0, SEED)
    for R in R_MECH_VALUES:
        H_mech = max(log_K - R, 1e-9)
        H_hyb = H_mu_hyb_exact(R, K)
        hyb_mean, hyb_se = run_phase1(K, N_TABLE1, M, R, SEED)
        obs = base_mean / hyb_mean if hyb_mean > 1e-9 else float("inf")
        LB = np.sqrt(log_K / H_mech)
        marker = ""
        if abs(R - RMECH) < 1e-6:
            marker = "  <- ceiling"
        elif R > RMECH:
            marker = "  (above ceiling)"
        print("{:>8.2f} {:>8.2f} {:>10.2f} {:>5.2f}+/-{:<6.2f} "
              "{:>5.2f}+/-{:<6.2f} {:>9.2f}x {:>9.2f}x {:>10.2f}{}".format(
            R, H_mech, H_hyb, hyb_mean, hyb_se, base_mean, base_se,
            obs, LB, obs/LB, marker))


def print_phase1_table2():
    print()
    print("=" * 88)
    print(f"PHASE 1, Table 2 (paper-aligned, varying N at R_mech = {RMECH})")
    print("=" * 88)
    hdr = "{:>6} {:>16} {:>16} {:>12} {:>24}".format(
        "N", "TS_hyb", "Uninf_TS", "Obs/UnInf", "Regime")
    print(hdr); print("-" * 88)
    for N in N_VALUES_TABLE2:
        bm, bs = run_phase1(K, N, M, 0.0, SEED + N)
        hm, hs = run_phase1(K, N, M, RMECH, SEED + N)
        obs = bm / hm if hm > 1e-9 else float("inf")
        if N <= 30: reg = "burn-in dominated"
        elif N < 100: reg = "transitional"
        else: reg = "asymptotic"
        print("{:>6d} {:>6.2f}+/-{:<8.2f} {:>6.2f}+/-{:<8.2f} {:>10.2f}x  {:<24}".format(
            N, hm, hs, bm, bs, obs, reg))


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    import sys
    # print_running_examples()
    # sys.stdout.flush()
    print_phase1_table1()
    sys.stdout.flush()
    print_phase1_table2()
    sys.stdout.flush()
    print()
    print("=" * 78)
    print("Done. All numbers above are reproducible (seed = {}).".format(SEED))
    print("=" * 78)
    sys.stdout.flush()