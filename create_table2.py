"""
Reproduce Tables 1 and 2 of "Mechanistic Information and Regret"
(mechanistic_information.pdf, the corrected manuscript).

Two phases run back-to-back:

PHASE 1 -- paper-aligned reproduction.
  Reward model:    deterministic 0/1 (audit code convention; what the
                   published Tables were generated from).
  Prior:           Beta(1+R_mech*K, 1) at pi_hat (audit code convention).
  Sample channel:  symmetric K-ary, with channel accuracy q solved
                   numerically so I(pi*; pi_hat) = R_mech EXACTLY.
  All formulas:    from mechanistic_information.pdf §3.3 (the corrected
                   manuscript version), including the corrected B_crit
                   formula (Eq. 10) which is NOT the small-SNR
                   approximation in hybrid_lb_submission.pdf.

PHASE 2 -- literature-calibrated.
  Reward model:    Bernoulli(p_arm) where p_arm depends on dose-step
                   distance from optimal, computed from the Gaussian
                   AUC integral with sigma_intra = 4.9 mg.h/L from
                   Kaldate 2012 and slope 0.02063 mg.h/L per mg/m^2
                   (Kaldate verbatim).  Gives:
                     p(d=0) = 0.6925  (analytic ceiling)
                     p(d=1) = 0.4145
                     p(d=2) = 0.0827  ...etc
  Regret:          Bayesian mean-reward gap (paper Definition 1).
  Prior:           Beta(exp(R_mech), 1) at pi_hat.  This makes
                   P(first pick = pi_hat) = exp(R_mech)/(exp(R_mech)+K-1)
                   = mu_hyb(pi_hat)   EXACTLY (closed-form derivation).
                   Persists across cycles via natural Bayesian updates.

USAGE:  python3 reproduce_tables.py
RUNTIME: ~60 seconds total.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

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


def B_crit_corrected(sigma: float, kappa_mu: float, H_mu: float, d_F: int) -> float:
    """B_crit from mechanistic_information.pdf Eq. 10 (corrected, NOT small-SNR)."""
    inside = (2.0 * H_mu / d_F) / (np.exp(2.0 / d_F) - 1.0) - 1.0
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
    """Sample pi_hat from symmetric channel: P(=pi*)=q, else Uniform(K)."""
    if rng.random() < q:
        return pi_star
    return int(rng.integers(0, K))


# ============================================================================
#  Phase 1: deterministic 0/1 reward, audit code prior
# ============================================================================

def ts_phase1_one_patient(K: int, N: int, R_mech: float, q: float, rng) -> float:
    """One patient, Phase 1. Deterministic 0/1 reward, audit code prior."""
    pi_star = int(rng.integers(0, K))
    alpha = np.ones(K, dtype=float)
    beta_ = np.ones(K, dtype=float)

    if R_mech > 0.0:
        pi_hat = sample_pi_hat(pi_star, q, K, rng)
        alpha[pi_hat] += R_mech * K  # audit code convention

    cum_regret = 0.0
    for _ in range(N):
        theta = rng.beta(alpha, beta_)
        arm = int(np.argmax(theta))
        reward = 1.0 if arm == pi_star else 0.0
        cum_regret += (1.0 - reward)
        alpha[arm] += reward
        beta_[arm] += (1.0 - reward)
    return cum_regret


def run_phase1(K: int, N: int, M: int, R_mech: float, seed: int):
    rng = np.random.default_rng(seed)
    q = accuracy_for_target_mi(R_mech, K)
    regrets = np.empty(M)
    for i in range(M):
        regrets[i] = ts_phase1_one_patient(K, N, R_mech, q, rng)
    return regrets.mean(), regrets.std(ddof=1) / np.sqrt(M)


# ============================================================================
#  Phase 2: graded reward, mu_hyb-exact prior, Bayesian regret
# ============================================================================

# Calibration constants (all from dossier v5)
KALDATE_SLOPE = 0.02063
SIGMA_INTRA   = 4.9
TARGET_LO     = 20.0
TARGET_HI     = 30.0
TARGET_MID    = 25.0
DOSE_MIN      = 1600.0
DOSE_MAX      = 3600.0
RMECH         = 0.37  # TODO: CHANGE HERE IF DESIRED (1.9 / 0.37...)


def reward_prob_at_distance(d_steps: int, K: int) -> float:
    """P(AUC in [20,30]) at d-step distance from optimal arm."""
    dose_grid = np.linspace(DOSE_MIN, DOSE_MAX, K)
    dauc_step = KALDATE_SLOPE * (dose_grid[1] - dose_grid[0])
    mu_AUC = TARGET_MID + d_steps * dauc_step
    return float(norm.cdf((TARGET_HI - mu_AUC) / SIGMA_INTRA) -
                 norm.cdf((TARGET_LO - mu_AUC) / SIGMA_INTRA))


def ts_phase2_one_patient(K: int, N: int, R_mech: float, q: float,
                           p_by_distance: np.ndarray, rng) -> float:
    """One patient, Phase 2."""
    pi_star = int(rng.integers(0, K))
    p_arms = np.array([p_by_distance[abs(k - pi_star)] for k in range(K)])
    p_opt = p_arms[pi_star]

    # Prior: Beta(exp(R_mech), 1) at pi_hat, Beta(1, 1) elsewhere.
    # This makes P(first pick = pi_hat) = mu_hyb(pi_hat) EXACTLY,
    # and persists across cycles via natural Bayesian updates.
    alpha = np.ones(K, dtype=float)
    beta_ = np.ones(K, dtype=float)
    if R_mech > 0.0:
        pi_hat = sample_pi_hat(pi_star, q, K, rng)
        alpha[pi_hat] = np.exp(R_mech)

    cum_regret = 0.0
    for _ in range(N):
        theta = rng.beta(alpha, beta_)
        arm = int(np.argmax(theta))
        # Bayesian regret = mean-reward gap (paper Definition 1)
        cum_regret += p_opt - p_arms[arm]
        # Bernoulli reward and posterior update
        reward = 1.0 if rng.random() < p_arms[arm] else 0.0
        alpha[arm] += reward
        beta_[arm] += (1.0 - reward)
    return cum_regret


def run_phase2(K: int, N: int, M: int, R_mech: float, seed: int):
    rng = np.random.default_rng(seed)
    q = accuracy_for_target_mi(R_mech, K)
    p_by_distance = np.array([reward_prob_at_distance(d, K) for d in range(K)])
    regrets = np.empty(M)
    for i in range(M):
        regrets[i] = ts_phase2_one_patient(K, N, R_mech, q, p_by_distance, rng)
    return regrets.mean(), regrets.std(ddof=1) / np.sqrt(M)


# ============================================================================
#  Top-level configuration
# ============================================================================

K = 8
N_TABLE1 = 12
M = 3000
SEED = 42
R_MECH_VALUES = [0.0, 0.3, 0.7, 1.0, 1.4, 1.7, 1.9]
N_VALUES_TABLE2 = [5, 10, 20, 30, 50, 100, 200]
log_K = np.log(K)


# ============================================================================
#  Reporting
# ============================================================================

def print_running_examples():
    """Recompute the paper's Running Examples 2 and 5 with the
    corrected B_crit formula from mechanistic_information.pdf §3.3."""
    print()
    print("=" * 78)
    print("Running Examples (mechanistic_information.pdf §5.1)")
    print("=" * 78)

    K_re, B_mu, sigma, kappa, d_F = 8, 0.22, 0.20, 1.8, 3
    H_mu = np.log(K_re)
    sigma_F_sq = 2.0 * sigma**2 * H_mu / (kappa**2 * d_F)
    sigma_F = np.sqrt(sigma_F_sq)
    C022 = channel_capacity_C(B_mu, sigma, kappa, sigma_F, d_F)
    B_crit_corr = B_crit_corrected(sigma, kappa, H_mu, d_F)
    B_crit_oldapprox = (sigma / kappa) * np.sqrt(H_mu - 1.0)
    H_hyb = H_mu_hyb_exact(RMECH, K_re)

    print(f"  Inputs: K={K_re}, B_mu={B_mu}, sigma={sigma}, kappa={kappa}, d_F={d_F}")
    print(f"  H(mu) = ln(K) = {H_mu:.4f} nats")
    print(f"  sigma_F^2 = 2*s^2*H/(k^2*d_F) = {sigma_F_sq:.5f}, sigma_F = {sigma_F:.4f}")
    print()
    print(f"  Channel capacity C({B_mu}) = {C022:.4f} nats   (paper says 0.37)  [Eq. 4]")
    print(f"  H(mu_hyb at R={RMECH}) = {H_hyb:.4f} nats        (paper says 1.69)  [Eq. 8]")
    print()
    print(f"  B_crit (corrected, Eq. 10):                  = {B_crit_corr:.4f}")
    print(f"  B_crit (small-SNR approximation, hybrid_lb):  = {B_crit_oldapprox:.4f}  (paper Ex. 5: 0.115)")
    print(f"  Calibrated B_mu / B_crit_corrected           = {B_mu/B_crit_corr:.3f} (well into baseline regime)")


def print_phase1_table1():
    print()
    print("=" * 102)
    print(f"PHASE 1, Table 1 (paper-aligned, deterministic reward, audit-code prior)")
    print(f"   K={K}, N={N_TABLE1}, M={M}, mu = Uniform")
    print("=" * 102)
    hdr = "{:>8} {:>8} {:>10} {:>14} {:>14} {:>10} {:>10} {:>10}".format(
        "R_mech", "H_mech", "H(mu_hyb)", "TS_hyb", "Uninf_TS",
        "Obs/UnInf", "LB_pred", "ratio")
    print(hdr); print("-" * 102)
    base_mean, base_se = run_phase1(K, N_TABLE1, M, 0.0, SEED)
    for R in R_MECH_VALUES:
        H_mech = max(log_K - R, 1e-9)
        H_hyb = H_mu_hyb_exact(R, K)
        hyb_mean, hyb_se = run_phase1(K, N_TABLE1, M, R, SEED)
        obs = base_mean / hyb_mean if hyb_mean > 1e-9 else float("inf")
        LB = np.sqrt(log_K / H_mech)
        print("{:>8.1f} {:>8.2f} {:>10.2f} {:>5.2f}+/-{:<6.2f} "
              "{:>5.2f}+/-{:<6.2f} {:>9.2f}x {:>9.2f}x {:>10.2f}".format(
            R, H_mech, H_hyb, hyb_mean, hyb_se, base_mean, base_se,
            obs, LB, obs/LB))


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
        elif N <= 100: reg = "transitional"
        else: reg = "asymptotic"
        print("{:>6d} {:>6.2f}+/-{:<8.2f} {:>6.2f}+/-{:<8.2f} {:>10.2f}x  {:<24}".format(
            N, hm, hs, bm, bs, obs, reg))


def print_phase2_calibration():
    print()
    print("=" * 78)
    print("PHASE 2 calibration (literature-derived, dossier v5 §1.1)")
    print("=" * 78)
    print(f"  Kaldate slope = {KALDATE_SLOPE} mg.h/L per mg/m^2")
    print(f"  sigma_intra   = {SIGMA_INTRA} mg.h/L  ( = |dAUC|/sqrt(2) from Kaldate )")
    print(f"  Target window = [{TARGET_LO}, {TARGET_HI}] mg.h/L")
    dose_grid = np.linspace(DOSE_MIN, DOSE_MAX, K)
    dose_step = dose_grid[1] - dose_grid[0]
    dauc_step = KALDATE_SLOPE * dose_step
    print(f"  Dose grid     = {[f'{d:.0f}' for d in dose_grid]}")
    print(f"  AUC step      = {dauc_step:.4f} mg.h/L per dose-step")
    print()
    print(f"  Reward probability by dose-step distance from optimal:")
    for d in range(K):
        p = reward_prob_at_distance(d, K)
        marker = ""
        if d == 0: marker = "    <- p_opt (analytic ceiling)"
        elif d == 1: marker = "    <- one step away (Delta_r ~ p(0)-p(1))"
        print(f"    d={d}: p = {p:.4f}{marker}")
    print()
    print(f"  Per-step Delta_r = p(0) - p(1) = {reward_prob_at_distance(0, K) - reward_prob_at_distance(1, K):.4f}")


def print_phase2_table1():
    print()
    print("=" * 102)
    print(f"PHASE 2, Table 1 (literature-calibrated, Bayesian regret, mu_hyb-exact prior)")
    print(f"   K={K}, N={N_TABLE1}, M={M}, mu = Uniform")
    print("=" * 102)
    hdr = "{:>8} {:>8} {:>10} {:>14} {:>14} {:>10} {:>10} {:>10}".format(
        "R_mech", "H_mech", "H(mu_hyb)", "TS_hyb", "Uninf_TS",
        "Obs/UnInf", "LB_pred", "ratio")
    print(hdr); print("-" * 102)
    base_mean, base_se = run_phase2(K, N_TABLE1, M, 0.0, SEED)
    for R in R_MECH_VALUES:
        H_mech = max(log_K - R, 1e-9)
        H_hyb = H_mu_hyb_exact(R, K)
        hyb_mean, hyb_se = run_phase2(K, N_TABLE1, M, R, SEED)
        obs = base_mean / hyb_mean if hyb_mean > 1e-9 else float("inf")
        LB = np.sqrt(log_K / H_mech)
        print("{:>8.1f} {:>8.2f} {:>10.2f} {:>5.2f}+/-{:<6.2f} "
              "{:>5.2f}+/-{:<6.2f} {:>9.2f}x {:>9.2f}x {:>10.2f}".format(
            R, H_mech, H_hyb, hyb_mean, hyb_se, base_mean, base_se,
            obs, LB, obs/LB))


def print_phase2_table2():
    print()
    print("=" * 88)
    print(f"PHASE 2, Table 2 (literature-calibrated, varying N at R_mech = {RMECH})")
    print("=" * 88)
    hdr = "{:>6} {:>16} {:>16} {:>12} {:>24}".format(
        "N", "TS_hyb", "Uninf_TS", "Obs/UnInf", "Regime")
    print(hdr); print("-" * 88)
    for N in N_VALUES_TABLE2:
        bm, bs = run_phase2(K, N, M, 0.0, SEED + N)
        hm, hs = run_phase2(K, N, M, RMECH , SEED + N)
        obs = bm / hm if hm > 1e-9 else float("inf")
        if N <= 30: reg = "burn-in dominated"
        elif N <= 100: reg = "transitional"
        else: reg = "asymptotic"
        print("{:>6d} {:>6.2f}+/-{:<8.2f} {:>6.2f}+/-{:<8.2f} {:>10.2f}x  {:<24}".format(
            N, hm, hs, bm, bs, obs, reg))


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    import sys
    print_running_examples()
    sys.stdout.flush()
    print_phase1_table1()
    sys.stdout.flush()
    print_phase1_table2()
    sys.stdout.flush()
    print_phase2_calibration()
    sys.stdout.flush()
    print_phase2_table1()
    sys.stdout.flush()
    print_phase2_table2()
    sys.stdout.flush()
    print()
    print("=" * 78)
    print("Done. All numbers above are reproducible (seed = {}).".format(SEED))
    print("=" * 78)
    sys.stdout.flush()