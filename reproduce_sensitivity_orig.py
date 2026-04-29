"""
Reproduce Tables 1 and 2 of "Mechanistic Information and Regret"
+ sensitivity analysis (heatmaps + tornado plots).

================================================================================
CALIBRATION SUMMARY
================================================================================
Reward in the bandit is Bernoulli (paper §2.1, §5.1) — a {0,1} indicator of
AUC ∈ [20, 30] mg.h/L. Therefore sigma is the std of Bernoulli noise, NOT the
std of the continuous AUC measurement.

  - At typical p ≈ 0.30 across arms (interpolating Li 2023's p_BSA=0.20 and
    Wilhelm 2016's p_opt=0.55):  sigma = sqrt(0.30·0.70) = 0.458 ≈ 0.46.
  - Hoeffding 1963 (Lemma 1) sub-Gaussian ceiling for [0,1]-bounded RV: sigma=0.50.

With sigma=0.46, B_µ=0.22, K=8, kappa=1.8, d_F=3:
  C(B_µ) = 0.879 nats   <- new ceiling, simulation RMECH=0.87 saturates this
  H_mech floor = 1.20 nats
  B_crit (Eq. 10) = 0.174  -> B_µ is 27% above phase transition
  Asymptotic LB ratio sqrt(H(µ)/H_mech) = 1.32x

================================================================================
TWO ORTHOGONAL KNOBS
================================================================================

  (A) PRIOR TYPE
      - "mu_hyb_exact" (DEFAULT, what Thm 2 analyzes):
            alpha[pi_hat] = exp(R_mech)
            yields P(first pick = pi_hat) = mu_hyb(pi_hat) EXACTLY.
      - "audit_code" (historical reproduction convention):
            alpha[pi_hat] = 1 + R_mech * K
            more concentrated than mu_hyb-exact at moderate-to-high R_mech.

  (B) REGRET TYPE
      - "bayesian" (DEFAULT, paper Definition 1):
            cum_regret += p_opt - p_arms[arm]
            mean-reward-gap regret. Smooth and what theorems analyze.
      - "sample" (realized 0/1):
            cum_regret += (1 - reward)  if reward is Bernoulli(p_arms[arm])
            sample regret. Higher variance but matches what a clinician sees.

ADVICE
------
  - For paper Tables 1 & 2 and asymptotic-bound validation:
        prior="mu_hyb_exact", regret="bayesian"  (default, recommended)
  - For ablation showing the prior choice matters:
        compare prior="mu_hyb_exact" vs prior="audit_code"
  - For matching what an oncologist would tally clinically:
        prior="mu_hyb_exact", regret="sample"
  - For reproducing historical published numbers exactly:
        prior="audit_code", regret="sample"

================================================================================
USAGE
================================================================================
  python3 reproduce_and_sensitivity.py            # full run with heatmaps
  python3 reproduce_and_sensitivity.py --notables # skip Tables 1, 2
  python3 reproduce_and_sensitivity.py --noplots  # skip heatmaps & tornados
  python3 reproduce_and_sensitivity.py --M 1000   # use M=1000 for regret heatmap

RUNTIME
  - Tables 1 + 2 (M=3000):                ~60 sec
  - Sensitivity heatmaps (analytical):    ~5 sec
  - Empirical regret heatmap (M=200):     ~30 sec
  - Tornado plots:                        ~2 sec
"""

import argparse
import os
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ============================================================================
#  Helper functions  (analytical, no Monte Carlo)
# ============================================================================

def conditional_entropy_symmetric_channel(q: float, K: int) -> float:
    """
    H(pi_hat | pi*) for the symmetric K-ary channel where:
        P(pi_hat = pi*) = q + (1-q)/K
        P(pi_hat = k != pi*) = (1-q)/K   for each of the K-1 other arms.
    """
    p_correct = q + (1.0 - q) / K
    p_other   = (1.0 - q) / K
    H = 0.0
    if p_correct > 0:
        H -= p_correct * np.log(p_correct)
    if p_other > 0:
        H -= (K - 1) * p_other * np.log(p_other)
    return H


def accuracy_for_target_mi(target_mi: float, K: int) -> float:
    """
    Numerically invert the symmetric channel: find q in [0, 1] such that
    I(pi*; pi_hat) = target_mi when pi* is uniform on K arms.
    Used to construct an exact channel that delivers the target R_mech.
    """
    if target_mi <= 0.0:
        return 0.0
    log_K = np.log(K)
    if target_mi >= log_K - 1e-9:
        return 1.0
    target_H_cond = log_K - target_mi
    f = lambda q: conditional_entropy_symmetric_channel(q, K) - target_H_cond
    return brentq(f, 1e-12, 1.0 - 1e-12)


def H_mu_hyb_exact(R_mech: float, K: int) -> float:
    """Exact entropy of mu_hyb (paper Eq. 8): H(mu_hyb) = ln(Z) - R*e^R/Z."""
    if R_mech <= 0.0:
        return np.log(K)
    Z = np.exp(R_mech) + K - 1
    return np.log(Z) - R_mech * np.exp(R_mech) / Z


def channel_capacity_C(B_mu, sigma, kappa_mu, d_F, H_mu=None):
    """
    Channel capacity bound on R_mech (paper Eq. 4, canonical parametrization):
        C(B_mu) = (d_F/2) * ln(1 + kappa^2 * sigma_F^2 / (kappa^2 * B_mu^2 + sigma^2))
    with sigma_F^2 = 2 * sigma^2 * H(mu) / (kappa^2 * d_F).
    Vectorized: any of B_mu, sigma, kappa_mu can be array-like.
    """
    B_mu     = np.asarray(B_mu, dtype=float)
    sigma    = np.asarray(sigma, dtype=float)
    kappa_mu = np.asarray(kappa_mu, dtype=float)
    if H_mu is None:
        # Default to ln K = ln 8
        H_mu = np.log(8)
    sigma_F_sq = 2.0 * sigma**2 * H_mu / (kappa_mu**2 * d_F)
    num = kappa_mu**2 * sigma_F_sq
    den = kappa_mu**2 * B_mu**2 + sigma**2
    return 0.5 * d_F * np.log1p(num / den)


def B_crit_corrected(sigma, kappa_mu, H_mu, d_F):
    """
    B_crit from paper Eq. 10 (the corrected, exact form — NOT the small-SNR approx):
        B_crit = (sigma/kappa) * sqrt[ (2H/d_F)/(e^(2/d_F) - 1) - 1 ]
    Returns 0 when no value of B_mu can yield C(B_mu) > 1 (i.e., d_F too small).
    Vectorized in sigma, kappa_mu.
    """
    sigma    = np.asarray(sigma, dtype=float)
    kappa_mu = np.asarray(kappa_mu, dtype=float)
    inside = (2.0 * H_mu / d_F) / (np.exp(2.0 / d_F) - 1.0) - 1.0
    if np.isscalar(inside):
        if inside <= 0:
            return np.zeros_like(sigma)
    return (sigma / kappa_mu) * np.sqrt(np.maximum(inside, 0))


# ============================================================================
#  Calibration constants for Phase 2 (literature-derived)
# ============================================================================

KALDATE_SLOPE = 0.02063   # mg.h/L per mg/m^2  (Kaldate 2012 verbatim)
SIGMA_INTRA   = 4.9       # mg.h/L  (= |dAUC|/sqrt(2) from Kaldate's |dAUC|=6.9)
TARGET_LO     = 20.0      # mg.h/L  (Kaldate Discussion: "AUC in the desired range")
TARGET_HI     = 30.0      # mg.h/L
TARGET_MID    = 25.0      # mg.h/L  (target window center; modeling assumption that
                          #          the optimal arm puts mean AUC at the center)
DOSE_MIN      = 1600.0    # mg/m^2  (Kaldate clinical adjustment range)
DOSE_MAX      = 3600.0    # mg/m^2

# Calibrated-ceiling default (saturates C(0.22) under sigma=0.46)
RMECH_DEFAULT = 0.87


def reward_prob_at_distance(d_steps: int, K: int) -> float:
    """
    P(AUC in [20, 30]) when the dose is d_steps away from the optimal arm.
    By symmetry of the Gaussian integrand around 25 (the window midpoint),
    shifting up or down by the same magnitude gives the same probability.
    """
    dose_grid = np.linspace(DOSE_MIN, DOSE_MAX, K)
    dauc_step = KALDATE_SLOPE * (dose_grid[1] - dose_grid[0])
    mu_AUC = TARGET_MID + d_steps * dauc_step
    return float(norm.cdf((TARGET_HI - mu_AUC) / SIGMA_INTRA) -
                 norm.cdf((TARGET_LO - mu_AUC) / SIGMA_INTRA))


def sample_pi_hat(pi_star: int, q: float, K: int, rng) -> int:
    """
    Symmetric K-ary channel: with probability q return pi_star,
    otherwise return a uniform random arm from {0, ..., K-1}.
    Total P(pi_hat = pi_star) = q + (1-q)/K.
    """
    if rng.random() < q:
        return pi_star
    return int(rng.integers(0, K))


# ============================================================================
#  Unified TS simulator with prior_type and regret_type as flags
# ============================================================================

def ts_one_patient(K, N, R_mech, q, p_by_distance, rng,
                    prior_type="mu_hyb_exact", regret_type="bayesian"):
    """
    One patient, configurable by prior_type and regret_type.

    prior_type:
        "mu_hyb_exact"  ->  alpha[pi_hat] = exp(R_mech)             [paper Thm 2]
        "audit_code"    ->  alpha[pi_hat] = 1 + R_mech * K          [historical]
    regret_type:
        "bayesian"  ->  cum_regret += p_opt - p_arms[arm]            [paper Def 1]
        "sample"    ->  cum_regret += (1 - realized_reward)          [observed]
    """
    pi_star = int(rng.integers(0, K))
    p_arms  = np.array([p_by_distance[abs(k - pi_star)] for k in range(K)])
    p_opt   = p_arms[pi_star]

    alpha = np.ones(K, dtype=float)
    beta_ = np.ones(K, dtype=float)
    if R_mech > 0.0:
        pi_hat = sample_pi_hat(pi_star, q, K, rng)
        if prior_type == "mu_hyb_exact":
            alpha[pi_hat] = np.exp(R_mech)
        elif prior_type == "audit_code":
            alpha[pi_hat] += R_mech * K
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")

    cum_regret = 0.0
    for _ in range(N):
        theta = rng.beta(alpha, beta_)
        arm = int(np.argmax(theta))
        reward = 1.0 if rng.random() < p_arms[arm] else 0.0
        if regret_type == "bayesian":
            cum_regret += p_opt - p_arms[arm]
        elif regret_type == "sample":
            cum_regret += 1.0 - reward
        else:
            raise ValueError(f"Unknown regret_type: {regret_type}")
        alpha[arm] += reward
        beta_[arm] += (1.0 - reward)
    return cum_regret


def run_simulation(K, N, M, R_mech, seed,
                    prior_type="mu_hyb_exact", regret_type="bayesian"):
    """
    Run M independent patients, return (mean regret, standard error of mean).
    """
    rng = np.random.default_rng(seed)
    q = accuracy_for_target_mi(R_mech, K)
    p_by_distance = np.array([reward_prob_at_distance(d, K) for d in range(K)])
    regrets = np.empty(M)
    for i in range(M):
        regrets[i] = ts_one_patient(K, N, R_mech, q, p_by_distance, rng,
                                      prior_type, regret_type)
    return regrets.mean(), regrets.std(ddof=1) / np.sqrt(M)


# ============================================================================
#  Tables 1 and 2 (using the unified simulator)
# ============================================================================

def print_running_examples(K, B_mu, sigma, kappa, d_F, R_mech_anchor):
    """Print the calibration block: C(B_mu), B_crit, H(mu_hyb)."""
    print("\n" + "=" * 78)
    print("RUNNING EXAMPLES — calibration block")
    print("=" * 78)

    H_mu = np.log(K)
    sigma_F_sq = 2.0 * sigma**2 * H_mu / (kappa**2 * d_F)
    sigma_F = np.sqrt(sigma_F_sq)
    C = channel_capacity_C(B_mu, sigma, kappa, d_F, H_mu=H_mu)
    Bcrit = B_crit_corrected(sigma, kappa, H_mu, d_F)
    H_hyb = H_mu_hyb_exact(R_mech_anchor, K)

    print(f"  Inputs: K={K}, B_mu={B_mu}, sigma={sigma}, kappa={kappa}, d_F={d_F}")
    print(f"  H(mu) = ln(K)               = {H_mu:.4f} nats")
    print(f"  sigma_F^2 (canonical)       = {sigma_F_sq:.5f}, sigma_F = {sigma_F:.4f}")
    print()
    print(f"  Channel capacity C({B_mu})    = {float(C):.4f} nats   [Eq. 4]")
    print(f"  H(mu_hyb at R={R_mech_anchor}) = {H_hyb:.4f} nats          [Eq. 8]")
    print(f"  B_crit (Eq. 10)               = {float(Bcrit):.4f}")
    ratio = B_mu / float(Bcrit) if float(Bcrit) > 0 else float('inf')
    print(f"  B_mu / B_crit                 = {ratio:.3f}", end="")
    print("   [baseline regime]" if ratio > 1.0 else "   [data-efficient regime]")


def print_table1(K, N, M, seed, R_mech_values, prior_type, regret_type):
    """Table 1 — sweep over R_mech at fixed N."""
    print("\n" + "=" * 110)
    print(f"TABLE 1 — sweep over R_mech at N={N}")
    print(f"   K={K}, M={M}, prior='{prior_type}', regret='{regret_type}'")
    print("=" * 110)
    log_K = np.log(K)
    print("{:>8} {:>8} {:>10} {:>16} {:>16} {:>10} {:>10}".format(
        "R_mech", "H_mech", "H(mu_hyb)", "Hyb regret", "Uninf regret",
        "Obs ratio", "LB pred"))
    print("-" * 110)
    base_mean, base_se = run_simulation(K, N, M, 0.0, seed,
                                          prior_type, regret_type)
    for R in R_mech_values:
        H_mech = max(log_K - R, 1e-9)
        H_hyb = H_mu_hyb_exact(R, K)
        hyb_mean, hyb_se = run_simulation(K, N, M, R, seed,
                                            prior_type, regret_type)
        obs = base_mean / hyb_mean if hyb_mean > 1e-9 else float("inf")
        LB_pred = np.sqrt(log_K / H_mech)
        print("{:>8.2f} {:>8.2f} {:>10.2f} {:>7.2f}+/-{:<6.2f} "
              "{:>7.2f}+/-{:<6.2f} {:>9.2f}x {:>9.2f}x".format(
                R, H_mech, H_hyb, hyb_mean, hyb_se, base_mean, base_se,
                obs, LB_pred))


def print_table2(K, M, seed, R_mech_anchor, N_values, prior_type, regret_type):
    """Table 2 — sweep over N at fixed R_mech."""
    print("\n" + "=" * 88)
    print(f"TABLE 2 — sweep over N at R_mech={R_mech_anchor}")
    print(f"   K={K}, M={M}, prior='{prior_type}', regret='{regret_type}'")
    print("=" * 88)
    print("{:>6} {:>16} {:>16} {:>12} {:>20}".format(
        "N", "Hyb regret", "Uninf regret", "Obs ratio", "Regime"))
    print("-" * 88)
    for N in N_values:
        bm, bs = run_simulation(K, N, M, 0.0, seed + N,
                                  prior_type, regret_type)
        hm, hs = run_simulation(K, N, M, R_mech_anchor, seed + N,
                                  prior_type, regret_type)
        obs = bm / hm if hm > 1e-9 else float("inf")
        if N <= 30:    reg = "burn-in dominated"
        elif N <= 100: reg = "transitional"
        else:          reg = "asymptotic"
        print("{:>6d} {:>7.2f}+/-{:<6.2f} {:>7.2f}+/-{:<6.2f} {:>10.2f}x  {:<20}".format(
            N, hm, hs, bm, bs, obs, reg))


# ============================================================================
#  SENSITIVITY ANALYSIS — Heatmaps
# ============================================================================
#
# All sensitivity is conducted around the BASELINE calibration:
#   sigma_0 = 0.46, kappa_0 = 1.8, B_mu_0 = 0.22, d_F_0 = 3, K_0 = 8.
# The parameters being swept (per the user's instruction): sigma, kappa, B_mu.
# For each 2D heatmap we hold the third sweep variable at multiple values
# (panels) so the reader can see how the picture changes across that variable.
# ----------------------------------------------------------------------------

# Sweep ranges
SIGMA_RANGE  = np.linspace(0.30, 0.55, 41)   # Bernoulli noise std (Hoeffding limit 0.50)
KAPPA_RANGE  = np.linspace(1.0, 3.0, 41)     # occupancy-Lipschitz constant
B_MU_RANGE   = np.linspace(0.05, 0.50, 41)   # ODE bias (normalized)

# Baseline values (all calibrated)
SIGMA_0 = 0.46
KAPPA_0 = 1.8
B_MU_0  = 0.22
D_F_0   = 3
K_0     = 8


def heatmap_C_of_B_mu(K=K_0, d_F=D_F_0,
                       kappa_panels=(1.0, 1.8, 2.5),
                       savepath=None, show=True):
    """
    HEATMAP 1.  C(B_mu) over (sigma, B_mu), with panels at three values of kappa.
    Shows how the R_mech ceiling depends on the noise model and ODE bias.
    """
    H_mu = np.log(K)
    fig, axes = plt.subplots(1, len(kappa_panels),
                              figsize=(6.0 * len(kappa_panels), 5.5),
                              sharey=True)
    if len(kappa_panels) == 1:
        axes = [axes]

    # Common color scale
    all_vals = []
    for kappa in kappa_panels:
        SIG, B = np.meshgrid(SIGMA_RANGE, B_MU_RANGE, indexing="ij")
        C = channel_capacity_C(B, SIG, kappa, d_F, H_mu=H_mu)
        all_vals.append(C)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    for ax, kappa, C in zip(axes, kappa_panels, all_vals):
        im = ax.imshow(C, origin="lower", aspect="auto",
                        extent=[B_MU_RANGE[0], B_MU_RANGE[-1],
                                SIGMA_RANGE[0], SIGMA_RANGE[-1]],
                        vmin=vmin, vmax=vmax, cmap="viridis")
        # Contours at C = 0.5, 1, 2, ...
        levels = [0.25, 0.5, 1.0, 1.5, 2.0]
        levels = [l for l in levels if vmin < l < vmax]
        if levels:
            cs = ax.contour(B_MU_RANGE, SIGMA_RANGE, C,
                              levels=levels, colors="white", linewidths=0.7)
            ax.clabel(cs, fmt="%.2f", fontsize=8, inline=True)
        # Mark calibrated point (sigma_0, B_mu_0)
        ax.plot(B_MU_0, SIGMA_0, "r*", markersize=14, markeredgecolor="white",
                  label=f"calibrated point\n($\\sigma$={SIGMA_0}, $B_\\mu$={B_MU_0})")
        ax.set_title(f"kappa = {kappa}")
        ax.set_xlabel(r"$B_\mu$  (ODE bias)")
    # Add legend to first panel only (showing what the star means)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85)

    axes[0].set_ylabel(r"$\sigma$  (reward noise std)")
    fig.suptitle(r"Heatmap 1: $C(B_\mu)$ — channel-capacity bound on $R_{\mathrm{mech}}$"
                 + "\n(white contours show iso-$C$ levels in nats)",
                 fontsize=12)
    plt.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.88, wspace=0.10)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.60])
    fig.colorbar(im, cax=cbar_ax, label=r"$C(B_\mu)$  (nats)")

    # Always save first (so we don't lose the figure if showing fails)
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    # Then optionally show, non-blocking, leaving the figure open
    # The caller is responsible for the final blocking plt.show() if interactive.
    if not show:
        plt.close(fig)


def heatmap_B_crit(K=K_0, d_F_panels=(2, 3, 5), savepath=None, show=True):
    """
    HEATMAP 2.  B_crit over (sigma, kappa), with panels at three values of d_F.
    Shows how the phase-transition certificate depends on the noise model
    and the sensitivity of the optimal policy.
    """
    H_mu = np.log(K)
    fig, axes = plt.subplots(1, len(d_F_panels),
                              figsize=(6.0 * len(d_F_panels), 5.5),
                              sharey=True)
    if len(d_F_panels) == 1:
        axes = [axes]

    # Common color scale
    all_vals = []
    for d_F in d_F_panels:
        SIG, K_ = np.meshgrid(SIGMA_RANGE, KAPPA_RANGE, indexing="ij")
        BC = B_crit_corrected(SIG, K_, H_mu, d_F)
        all_vals.append(BC)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    for ax, d_F, BC in zip(axes, d_F_panels, all_vals):
        im = ax.imshow(BC, origin="lower", aspect="auto",
                        extent=[KAPPA_RANGE[0], KAPPA_RANGE[-1],
                                SIGMA_RANGE[0], SIGMA_RANGE[-1]],
                        vmin=vmin, vmax=vmax, cmap="plasma")
        levels = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        levels = [l for l in levels if vmin < l < vmax]
        if levels:
            cs = ax.contour(KAPPA_RANGE, SIGMA_RANGE, BC,
                              levels=levels, colors="white", linewidths=0.7)
            ax.clabel(cs, fmt="%.2f", fontsize=8, inline=True)
        # Mark calibrated point (sigma_0, kappa_0); show B_mu_0 contour for "are we above?"
        ax.plot(KAPPA_0, SIGMA_0, "r*", markersize=14, markeredgecolor="white",
                  label=f"calibrated point\n($\\sigma$={SIGMA_0}, $\\kappa$={KAPPA_0})")
        # Plot the locus where B_crit = B_mu_0 (the phase-transition curve at the calibrated B_mu)
        cs2 = ax.contour(KAPPA_RANGE, SIGMA_RANGE, BC,
                          levels=[B_MU_0], colors="cyan", linewidths=2.0, linestyles="--")
        ax.clabel(cs2, fmt=f"B_crit = B_mu_0 = {B_MU_0}", fontsize=8)
        ax.set_title(f"d_F = {d_F}")
        ax.set_xlabel(r"$\kappa_\mu$  (occupancy-Lipschitz)")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.85)

    axes[0].set_ylabel(r"$\sigma$  (reward noise std)")
    fig.suptitle("Heatmap 2: $B_{\\mathrm{crit}}$ — phase-transition certificate\n"
                 "(cyan dashed = locus where $B_{\\mathrm{crit}}=B_\\mu^{\\mathrm{baseline}}$=0.22)",
                 fontsize=12)
    plt.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.88, wspace=0.10)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.60])
    fig.colorbar(im, cax=cbar_ax, label=r"$B_{\mathrm{crit}}$  (normalized)")

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


def heatmap_regret_ratio(M=200, K_panels=(4, 8, 16), seed=42,
                           prior_type="mu_hyb_exact", regret_type="bayesian",
                           savepath=None, show=True):
    """
    HEATMAP 3.  Empirical regret ratio (uninformed / hybrid) over (R_mech, N),
    with panels at three values of K. This is the only empirical heatmap;
    the other two are analytical and free.

    For each (K, R_mech, N) cell we run M Monte-Carlo patients twice (R=0 and R)
    and report the ratio of mean cumulative regret.
    """
    R_mech_grid = np.linspace(0.0, 1.5, 8)   # 0, 0.21, ..., 1.5 (saturates ceiling at 0.87)
    N_grid      = np.array([5, 10, 20, 30, 50, 75, 100, 150, 200])

    fig, axes = plt.subplots(1, len(K_panels),
                              figsize=(6.0 * len(K_panels), 5.5),
                              sharey=True)
    if len(K_panels) == 1:
        axes = [axes]

    # Compute all panels first to use a common color scale
    all_ratios = []
    for K in K_panels:
        ratios = np.full((len(R_mech_grid), len(N_grid)), np.nan)
        # Pre-compute uninformed-baseline regret per (K, N) — independent of R
        base_per_N = {}
        for j, N in enumerate(N_grid):
            base_per_N[N], _ = run_simulation(K, int(N), M, 0.0, seed + int(N),
                                                prior_type, regret_type)
        for i, R in enumerate(R_mech_grid):
            for j, N in enumerate(N_grid):
                hyb_mean, _ = run_simulation(K, int(N), M, float(R), seed + int(N),
                                               prior_type, regret_type)
                if hyb_mean > 1e-9:
                    ratios[i, j] = base_per_N[N] / hyb_mean
        all_ratios.append(ratios)
        print(f"  K={K} done")

    finite_vals = np.concatenate([r[np.isfinite(r)] for r in all_ratios])
    vmin = max(0.5, float(np.nanmin(finite_vals)))
    vmax = float(np.nanmax(finite_vals))

    # Use index-based axes for uniform cell sizes
    for ax, K, ratios in zip(axes, K_panels, all_ratios):
        im = ax.imshow(ratios, origin="lower", aspect="auto",
                        vmin=vmin, vmax=vmax, cmap="cividis")
        # Annotate each cell with its value (centered on the integer index)
        for i in range(len(R_mech_grid)):
            for j in range(len(N_grid)):
                v = ratios[i, j]
                if np.isfinite(v):
                    color = "white" if v < (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{v:.2f}",
                              ha="center", va="center", color=color, fontsize=9)
        # Outline the calibrated R_mech row in gold (no overlap with cell text)
        ceiling_idx = np.argmin(np.abs(R_mech_grid - RMECH_DEFAULT))
        from matplotlib.patches import Rectangle
        rect = Rectangle((-0.5, ceiling_idx - 0.5), len(N_grid), 1.0,
                          linewidth=2.5, edgecolor="#FFD700",
                          facecolor="none", clip_on=False)
        ax.add_patch(rect)
        ax.set_title(f"K = {K}")
        ax.set_xlabel("N  (number of cycles)")
        # Set tick labels to actual N and R_mech values
        ax.set_xticks(np.arange(len(N_grid)))
        ax.set_xticklabels([str(int(n)) for n in N_grid])
        ax.set_yticks(np.arange(len(R_mech_grid)))
        ax.set_yticklabels([f"{r:.2f}" for r in R_mech_grid])

    axes[0].set_ylabel(r"$R_{\mathrm{mech}}$  (nats)")
    fig.suptitle(f"Heatmap 3: empirical regret ratio (uninformed / hybrid)\n"
                  f"prior='{prior_type}', regret='{regret_type}', M={M}\n"
                  f"gold outline = calibrated $R_{{\\mathrm{{mech}}}}$ ceiling "
                  f"($\\sigma$=0.46, $B_\\mu$=0.22, $\\kappa$=1.8, $d_F$=3 → R≈{RMECH_DEFAULT})",
                  fontsize=12)
    plt.subplots_adjust(top=0.78, bottom=0.10, left=0.06, right=0.88, wspace=0.12)
    # Add colorbar in dedicated axes to avoid overlap
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.55])
    fig.colorbar(im, cax=cbar_ax, label="ratio")

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


# ============================================================================
#  TORNADO PLOTS — one-at-a-time sensitivity
# ============================================================================
#
# For each of three output quantities — C(B_mu), B_crit, asymptotic LB ratio —
# we vary each input parameter individually over a "low / baseline / high"
# bracket while holding all others at baseline. The resulting horizontal bars
# are sorted by sensitivity magnitude. This is the standard reviewer-friendly
# robustness check.
# ----------------------------------------------------------------------------

# Brackets for each parameter: (low, baseline, high)
PARAM_BRACKETS = {
    "sigma":  (0.40, 0.46, 0.50),    # Bernoulli noise band; Hoeffding ceiling
    "kappa":  (1.0,  1.8,  2.5),     # plausible occupancy-Lipschitz range
    "B_mu":   (0.10, 0.22, 0.40),    # well-calibrated to severely biased ODE
    "d_F":    (2,    3,    5),       # plausible residual-rank choices
    "K":      (4,    8,    16),      # dose-grid coarseness alternatives
}


def tornado_metric(metric_name):
    """
    Return a function (sigma, kappa, B_mu, d_F, K) -> scalar
    for the requested metric.

      "C"        -> C(B_mu) [nats]
      "B_crit"   -> phase-transition critical bias
      "LB_ratio" -> asymptotic LB ratio sqrt(H(mu)/H_mech)
                     where H_mech = H(mu) - C(B_mu)
    """
    def metric(sigma, kappa, B_mu, d_F, K):
        H_mu = np.log(K)
        if metric_name == "C":
            return float(channel_capacity_C(B_mu, sigma, kappa, d_F, H_mu=H_mu))
        if metric_name == "B_crit":
            return float(B_crit_corrected(sigma, kappa, H_mu, d_F))
        if metric_name == "LB_ratio":
            C = float(channel_capacity_C(B_mu, sigma, kappa, d_F, H_mu=H_mu))
            H_mech = max(H_mu - C, 1e-9)
            return float(np.sqrt(H_mu / H_mech))
        raise ValueError(f"Unknown metric: {metric_name}")
    return metric


def tornado_plot(metric_name, ax=None, show_baseline=True):
    """
    Tornado plot for one metric: bars sorted by sensitivity magnitude.
    """
    f = tornado_metric(metric_name)
    baseline_args = dict(sigma=SIGMA_0, kappa=KAPPA_0, B_mu=B_MU_0, d_F=D_F_0, K=K_0)
    baseline = f(**baseline_args)

    rows = []
    for param, (lo, mid, hi) in PARAM_BRACKETS.items():
        # Low value
        args = dict(baseline_args); args[param] = lo
        v_lo = f(**args)
        # High value
        args = dict(baseline_args); args[param] = hi
        v_hi = f(**args)
        rows.append((param, lo, hi, v_lo, v_hi))

    # Sort by total swing
    rows.sort(key=lambda r: abs(r[4] - r[3]), reverse=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        owns_fig = True
    else:
        owns_fig = False

    y_positions = np.arange(len(rows))
    for i, (param, lo, hi, v_lo, v_hi) in enumerate(rows):
        # Two bars: one for swing below baseline, one above
        v_min = min(v_lo, v_hi)
        v_max = max(v_lo, v_hi)
        ax.barh(i, v_max - baseline, left=baseline, height=0.6,
                  color="#3c8dad", alpha=0.85,
                  edgecolor="black", linewidth=0.5)
        ax.barh(i, v_min - baseline, left=baseline, height=0.6,
                  color="#f4a261", alpha=0.85,
                  edgecolor="black", linewidth=0.5)
        # Annotate end values JUST BEYOND each bar end
        # Low end (orange bar, on the left of baseline if v_lo < baseline)
        if v_lo < baseline:
            ax.text(v_lo, i, f"{param}={lo}: {v_lo:.3f} ",
                      va="center", ha="right", fontsize=8)
        else:
            ax.text(v_lo, i, f" {param}={lo}: {v_lo:.3f}",
                      va="center", ha="left", fontsize=8)
        # High end (blue bar, on the right of baseline if v_hi > baseline)
        if v_hi > baseline:
            ax.text(v_hi, i, f" {param}={hi}: {v_hi:.3f}",
                      va="center", ha="left", fontsize=8)
        else:
            ax.text(v_hi, i, f"{param}={hi}: {v_hi:.3f} ",
                      va="center", ha="right", fontsize=8)

    if show_baseline:
        ax.axvline(baseline, color="black", linewidth=1.5, label=f"baseline = {baseline:.3f}")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows])
    pretty = {"C": r"$C(B_\mu)$", "B_crit": r"$B_{\mathrm{crit}}$",
                "LB_ratio": r"asymptotic LB ratio"}
    ax.set_xlabel(pretty.get(metric_name, metric_name))
    ax.set_title(f"Tornado sensitivity: {pretty.get(metric_name, metric_name)}")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Add horizontal padding for label visibility
    cur_xlim = ax.get_xlim()
    span = cur_xlim[1] - cur_xlim[0]
    ax.set_xlim(cur_xlim[0] - 0.18 * span, cur_xlim[1] + 0.18 * span)

    if owns_fig:
        plt.tight_layout()


def tornado_panel(savepath=None, show=True):
    """Three tornado plots side by side: C, B_crit, LB_ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, metric in zip(axes, ["C", "B_crit", "LB_ratio"]):
        tornado_plot(metric, ax=ax)
    fig.suptitle("One-at-a-time sensitivity (low / baseline / high brackets)",
                  fontsize=13)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


# ============================================================================
#  SAMPLE-DENSITY ANALYSIS — Interpretation B (multi-fidelity intra-cycle)
# ============================================================================
#
# Clinical setup: one cycle = one dose, lasting 46 hours. Within that cycle we
# can take m plasma 5-FU samples to estimate the patient's PK parameters.
# More samples per cycle → better ODE fit → smaller ODE bias B_μ. With m=∞
# the ODE is "perfectly" calibrated to the patient (up to a structural floor
# from biological day-to-day variability that no amount of sampling removes).
#
# Two effects of m, in order of importance:
#
#   PRIMARY:  B_μ shrinks with m.
#             B_μ(m) = sqrt(B_inf² + (B_μ(m0) · sqrt(m0/m))²)
#             - statistical part: 1/sqrt(m) decay (standard PK-fit improvement)
#             - structural floor B_inf: biological variability the ODE family
#               cannot capture (clearance day-to-day, drug-drug interactions,
#               circadian effects, etc.) — even with continuous monitoring,
#               this floor remains.
#
#   SECONDARY:  σ inflates slightly at low m due to noisy AUC measurement.
#               σ(m) = σ0 · (1 + α/sqrt(m))
#               - at m=1 the AUC is itself noisily estimated; this blurs the
#                 binary in-window indicator and inflates the Bernoulli σ.
#               - α=0.10 is the default; α=0 disables the secondary effect.
#               - this is a 5-10% correction, not a 50% one — the dominant
#                 effect is on B_μ.
#
# The key clinical question this module answers: how many plasma samples per
# cycle are needed to cross the phase transition (B_μ < B_crit)?
# ----------------------------------------------------------------------------

# Default parameters for the m-dependent calibration
M0_DEFAULT      = 1       # baseline number of samples per cycle (single AUC draw)
B_MU_AT_M0      = 0.22    # calibrated B_μ at m=m0 (Kaldate-derived)
B_MU_FLOOR      = 0.05    # structural floor on B_μ (irreducible biological noise)
SIGMA_BASELINE  = 0.46    # asymptotic Bernoulli σ at m → ∞
ALPHA_INFLATION = 0.10    # secondary inflation: σ(m=1) is 10% larger than σ(∞)

# m values to sweep (∞ approximated by a large finite m)
M_SAMPLES_VALUES = [1, 2, 3, 5, 10, 25, 100]
M_INFTY_PROXY    = 100   # used as the "m → ∞" stand-in in simulation


def B_mu_of_m(m, m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0, B_inf=B_MU_FLOOR):
    """
    B_μ(m) — ODE bias as a function of intra-cycle samples.

      B_μ(m) = sqrt( B_inf² + (B_μ(m0) · sqrt(m0/m))² )

    Reasoning:
      - The "statistical" part B_μ(m0)·sqrt(m0/m) decays like 1/sqrt(m). This
        captures how more PK measurements improve the patient-specific fit.
      - The "structural" part B_inf is the floor from un-modeled biology.
        Even continuous monitoring cannot bring the bias below this floor.
    """
    m = np.asarray(m, dtype=float)
    statistical_part = B_mu_at_m0 * np.sqrt(m0 / m)
    return np.sqrt(B_inf**2 + statistical_part**2)


def sigma_of_m(m, sigma0=SIGMA_BASELINE, alpha=ALPHA_INFLATION):
    """
    σ(m) — Bernoulli reward noise, with a secondary inflation at low m.

      σ(m) = σ0 · (1 + α/sqrt(m))

    Reasoning:
      - Asymptotically (m → ∞), σ(m) → σ0 ≈ 0.46 (Bernoulli at p ≈ 0.30).
      - At small m, the AUC itself is poorly estimated, which "blurs" the
        binary in-window indicator and inflates the Bernoulli noise std.
      - This is a SECONDARY effect (5-10%); the primary effect is on B_μ.
      - Set α=0 to disable this effect entirely.
    """
    m = np.asarray(m, dtype=float)
    return sigma0 * (1.0 + alpha / np.sqrt(m))


def calibration_at_m(m, K=K_0, kappa=KAPPA_0, d_F=D_F_0,
                       m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                       B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                       alpha=ALPHA_INFLATION):
    """
    For a given m, compute all derived quantities:
      B_μ(m), σ(m), C(B_μ(m)), B_crit(m), B_μ/B_crit, asymptotic LB ratio.

    Returns dict with keys: m, B_mu, sigma, C, B_crit, ratio, regime,
    H_mech, LB_ratio.
    """
    H_mu = np.log(K)
    B_mu  = float(B_mu_of_m(m, m0, B_mu_at_m0, B_inf))
    sigma = float(sigma_of_m(m, sigma0, alpha))
    C     = float(channel_capacity_C(B_mu, sigma, kappa, d_F, H_mu=H_mu))
    Bc    = float(B_crit_corrected(sigma, kappa, H_mu, d_F))
    H_mech = max(H_mu - C, 1e-9)
    LB_ratio = float(np.sqrt(H_mu / H_mech))
    ratio = B_mu / Bc if Bc > 0 else float("inf")
    regime = "data-efficient" if ratio < 1 else "baseline"
    return dict(m=m, B_mu=B_mu, sigma=sigma, C=C, B_crit=Bc,
                ratio=ratio, regime=regime, H_mech=H_mech, LB_ratio=LB_ratio)


def find_m_crossing(K=K_0, kappa=KAPPA_0, d_F=D_F_0,
                      m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                      B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                      alpha=ALPHA_INFLATION, m_search=range(1, 101)):
    """
    Find the smallest integer m at which B_μ(m) < B_crit(m), i.e., the
    crossing into the data-efficient regime. Returns None if no crossing.
    """
    for m in m_search:
        cal = calibration_at_m(m, K=K, kappa=kappa, d_F=d_F,
                                 m0=m0, B_mu_at_m0=B_mu_at_m0, B_inf=B_inf,
                                 sigma0=sigma0, alpha=alpha)
        if cal["ratio"] < 1.0:
            return m, cal
    return None, None


# ----------------------------------------------------------------------------
#  Table 3 — sample density vs improvement
# ----------------------------------------------------------------------------

def cycles_to_target(K, M, R_mech, target_avg_regret, seed,
                       prior_type="mu_hyb_exact", regret_type="bayesian",
                       N_search=None):
    """
    Find the smallest N such that the empirical AVERAGE per-cycle regret
    drops below target_avg_regret.

      avg regret per cycle = cumulative_regret(N) / N

    This decreases monotonically with N as the algorithm converges, so
    "cycles to target" is well-posed.  A typical target is 0.1, meaning
    "average regret per cycle has dropped below 0.1 (i.e., on >= 90%
    of cycles the algorithm picks an arm with reward gap ≤ 0.1)."

    Returns the smallest N with avg_regret < target, or None if not found.
    """
    if N_search is None:
        # Geometric-ish search: try small first, then larger
        N_search = [3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200]
    for N in N_search:
        regret_mean, _ = run_simulation(K, int(N), M, float(R_mech), seed,
                                          prior_type, regret_type)
        avg_regret = regret_mean / N
        if avg_regret < target_avg_regret:
            return int(N)
    return None


def print_table3_sample_density(K=K_0, kappa=KAPPA_0, d_F=D_F_0,
                                  m_values=None,
                                  m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                                  B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                                  alpha=ALPHA_INFLATION,
                                  add_clinical_columns=True, M=200, seed=42,
                                  target_avg_regret=0.2,
                                  prior_type="mu_hyb_exact",
                                  regret_type="bayesian"):
    """
    Print Table 3: per-m calibration values + clinical-impact columns.

    Analytical columns (cheap, always shown):
        m, B_μ(m), σ(m), C(B_μ), B_crit, B_μ/B_crit, LB_ratio, regime

    Clinical columns (require Monte Carlo, flag-controlled):
        cycles_to_target — smallest N such that AVG per-cycle hybrid regret < target
        total_samples    — cycles_to_target × m  (total blood draws)
        regret_per_sample — regret saved per blood draw at calibrated ceiling

    target_avg_regret is the threshold on regret/N (per-cycle average).
    A typical clinical interpretation: target=0.1 means the algorithm picks
    a near-optimal arm on ~90% of cycles.
    """
    if m_values is None:
        m_values = M_SAMPLES_VALUES

    print("\n" + "=" * 130)
    print("TABLE 3 — sample density vs improvement")
    print(f"   K={K}, kappa={kappa}, d_F={d_F}, m0={m0}, "
          f"B_μ(m0)={B_mu_at_m0}, B_inf={B_inf}, σ0={sigma0}, α={alpha}")
    if add_clinical_columns:
        print(f"   Clinical metrics: target_avg_regret={target_avg_regret}, M={M}, "
              f"prior='{prior_type}', regret='{regret_type}'")
    print("=" * 130)

    if add_clinical_columns:
        print("{:>4}  {:>7}  {:>6}  {:>7}  {:>7}  {:>10}  {:>8}  {:>15}  {:>11}  {:>13}  {:<14}".format(
            "m", "B_μ(m)", "σ(m)", "C(B_μ)", "B_crit", "B_μ/B_crit",
            "LB_rat", "cycles→target", "tot.samples", "rgt.per.smpl", "Regime"))
    else:
        print("{:>4}  {:>7}  {:>6}  {:>7}  {:>7}  {:>10}  {:>8}  {:<14}".format(
            "m", "B_μ(m)", "σ(m)", "C(B_μ)", "B_crit", "B_μ/B_crit",
            "LB_rat", "Regime"))
    print("-" * 130)

    # Track for clinical-columns summary
    rows_for_summary = []

    for m in m_values:
        cal = calibration_at_m(m, K=K, kappa=kappa, d_F=d_F,
                                 m0=m0, B_mu_at_m0=B_mu_at_m0, B_inf=B_inf,
                                 sigma0=sigma0, alpha=alpha)
        marker = " ← phase transition" if 0.95 < cal["ratio"] < 1.05 else ""
        if add_clinical_columns:
            # Use the m-calibrated ceiling as R_mech for the simulation
            R_mech_at_m = cal["C"]
            # Cycles needed for hybrid to reach target regret
            N_target = cycles_to_target(K, M, R_mech_at_m, target_avg_regret, seed,
                                          prior_type, regret_type)
            # Cycles needed for uninformed to reach same target (for "regret saved")
            N_uninf = cycles_to_target(K, M, 0.0, target_avg_regret, seed,
                                          prior_type, regret_type)
            # Total samples = cycles × m
            total_samples = N_target * m if N_target is not None else None
            # Regret savings per blood draw at N=12 (a clinical reference point)
            base_at_12, _ = run_simulation(K, 12, M, 0.0, seed, prior_type, regret_type)
            hyb_at_12, _  = run_simulation(K, 12, M, R_mech_at_m, seed,
                                              prior_type, regret_type)
            regret_saved = max(base_at_12 - hyb_at_12, 0)
            regret_per_sample = regret_saved / (12 * m) if m > 0 else 0
            rows_for_summary.append((m, N_target, N_uninf, total_samples, regret_per_sample))

            N_target_str = str(N_target) if N_target else ">N_max"
            tot_str = str(total_samples) if total_samples else "—"
            print("{:>4d}  {:>7.4f}  {:>6.4f}  {:>7.4f}  {:>7.4f}  {:>10.3f}  {:>8.3f}  "
                  "{:>15}  {:>11}  {:>13.5f}  {:<14}{}".format(
                m, cal["B_mu"], cal["sigma"], cal["C"], cal["B_crit"],
                cal["ratio"], cal["LB_ratio"], N_target_str, tot_str,
                regret_per_sample, cal["regime"], marker))
        else:
            print("{:>4d}  {:>7.4f}  {:>6.4f}  {:>7.4f}  {:>7.4f}  {:>10.3f}  {:>8.3f}  {:<14}{}".format(
                m, cal["B_mu"], cal["sigma"], cal["C"], cal["B_crit"],
                cal["ratio"], cal["LB_ratio"], cal["regime"], marker))

    # m → ∞ limit (analytical only)
    cal_inf = calibration_at_m(1e9, K=K, kappa=kappa, d_F=d_F,
                                 m0=m0, B_mu_at_m0=B_mu_at_m0, B_inf=B_inf,
                                 sigma0=sigma0, alpha=alpha)
    if add_clinical_columns:
        print("{:>4}  {:>7.4f}  {:>6.4f}  {:>7.4f}  {:>7.4f}  {:>10.3f}  {:>8.3f}  "
              "{:>15}  {:>11}  {:>13}  {:<14}".format(
            "∞", cal_inf["B_mu"], cal_inf["sigma"], cal_inf["C"], cal_inf["B_crit"],
            cal_inf["ratio"], cal_inf["LB_ratio"], "—", "—", "—", cal_inf["regime"]))
    else:
        print("{:>4}  {:>7.4f}  {:>6.4f}  {:>7.4f}  {:>7.4f}  {:>10.3f}  {:>8.3f}  {:<14}".format(
            "∞", cal_inf["B_mu"], cal_inf["sigma"], cal_inf["C"], cal_inf["B_crit"],
            cal_inf["ratio"], cal_inf["LB_ratio"], cal_inf["regime"]))

    # Identify phase-transition m
    m_cross, _ = find_m_crossing(K=K, kappa=kappa, d_F=d_F,
                                   m0=m0, B_mu_at_m0=B_mu_at_m0,
                                   B_inf=B_inf, sigma0=sigma0, alpha=alpha)
    print()
    if m_cross is not None:
        print(f"   Phase-transition crossing: m* = {m_cross}")
        print(f"   Clinical interpretation: at least {m_cross} plasma sample(s) per cycle")
        print(f"   are needed to enter the data-efficient regime (B_μ < B_crit).")
    else:
        print("   No phase-transition crossing within m ∈ [1, 100].")

    # Best-(N, m) total-sample analysis for the Pareto story
    if add_clinical_columns and rows_for_summary:
        print()
        print("   Sample-budget analysis (smallest total blood draws to reach target):")
        valid = [(m, N, total) for (m, N, _, total, _) in rows_for_summary
                  if N is not None]
        if valid:
            best_m, best_N, best_total = min(valid, key=lambda x: x[2])
            print(f"   Optimal: m={best_m}, N={best_N}, total samples = {best_total}")


# ----------------------------------------------------------------------------
#  Plots for sample-density analysis
# ----------------------------------------------------------------------------

def plot_quantities_vs_m(K=K_0, kappa=KAPPA_0, d_F=D_F_0,
                           m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                           B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                           alpha=ALPHA_INFLATION,
                           savepath=None, show=True):
    """
    PLOT A.  Four panels showing how key quantities depend on m:
      (1) B_μ(m) and B_crit(m), with the calibrated B_μ(m=1) marked
      (2) σ(m), with asymptote at σ0
      (3) C(B_μ(m)) = R_mech ceiling, with horizontal asymptote at H(μ) = ln K
      (4) Asymptotic LB ratio sqrt(H(μ)/H_mech) and B_μ/B_crit ratio together
    """
    H_mu = np.log(K)
    m_grid = np.logspace(0, 2.5, 200)   # 1 → ~316
    B_mu_arr = B_mu_of_m(m_grid, m0, B_mu_at_m0, B_inf)
    sigma_arr = sigma_of_m(m_grid, sigma0, alpha)
    C_arr = np.array([channel_capacity_C(B, s, kappa, d_F, H_mu=H_mu)
                       for B, s in zip(B_mu_arr, sigma_arr)])
    Bc_arr = np.array([B_crit_corrected(s, kappa, H_mu, d_F) for s in sigma_arr])
    H_mech_arr = np.maximum(H_mu - C_arr, 1e-9)
    LB_ratio_arr = np.sqrt(H_mu / H_mech_arr)
    ratio_arr = B_mu_arr / Bc_arr

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: B_μ(m) and B_crit(m)
    ax = axes[0, 0]
    ax.semilogx(m_grid, B_mu_arr, color="#3c8dad", linewidth=2.5, label=r"$B_\mu(m)$  (ODE bias)")
    ax.semilogx(m_grid, Bc_arr, color="#f4a261", linewidth=2.5, label=r"$B_{\mathrm{crit}}(m)$")
    ax.axhline(y=B_inf, color="#3c8dad", linestyle=":", linewidth=1.5,
                label=f"$B_\\infty$ = {B_inf} (structural floor)")
    crossings = np.where(np.diff(np.sign(B_mu_arr - Bc_arr)))[0]
    if len(crossings) > 0:
        m_cross = m_grid[crossings[0]]
        ax.axvline(x=m_cross, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.text(m_cross * 1.1, ax.get_ylim()[1] * 0.85,
                f"crossing\n$m \\approx${m_cross:.1f}", fontsize=9)
    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel("normalized")
    ax.set_title(r"(a) $B_\mu$ and $B_{\mathrm{crit}}$ vs $m$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: σ(m)
    ax = axes[0, 1]
    ax.semilogx(m_grid, sigma_arr, color="#5e3c99", linewidth=2.5, label=r"$\sigma(m)$")
    ax.axhline(y=sigma0, color="#5e3c99", linestyle=":", linewidth=1.5,
                label=f"$\\sigma_0$ = {sigma0} (asymptote)")
    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel(r"$\sigma$  (Bernoulli noise)")
    ax.set_title(r"(b) $\sigma(m)$ — secondary effect")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: C(B_μ(m))
    ax = axes[1, 0]
    ax.semilogx(m_grid, C_arr, color="#2a9d8f", linewidth=2.5, label=r"$C(B_\mu(m))$")
    ax.axhline(y=H_mu, color="#2a9d8f", linestyle=":", linewidth=1.5,
                label=f"$H(\\mu) = \\ln K = {H_mu:.3f}$")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0,
                label="1 nat (phase-transition criterion)")
    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel(r"$R_{\mathrm{mech}}$ ceiling  (nats)")
    ax.set_title(r"(c) $C(B_\mu(m))$ — channel-capacity ceiling")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: ratios
    ax = axes[1, 1]
    ax.semilogx(m_grid, ratio_arr, color="#e63946", linewidth=2.5,
                  label=r"$B_\mu / B_{\mathrm{crit}}$")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
                label="phase transition")
    ax.fill_between(m_grid, 0, 1.0, alpha=0.15, color="green",
                      label="data-efficient regime")
    y_top = max(ax.get_ylim()[1], 2)
    ax.fill_between(m_grid, 1.0, y_top, alpha=0.15, color="red", label="baseline regime")
    ax.set_ylim(0, y_top)
    ax_twin = ax.twinx()
    ax_twin.semilogx(m_grid, LB_ratio_arr, color="#264653", linewidth=2.5,
                       linestyle="--", label=r"asymptotic LB ratio")
    ax_twin.set_ylabel(r"asymptotic LB ratio  $\sqrt{H(\mu)/H_{\mathrm{mech}}}$",
                          color="#264653")
    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel(r"$B_\mu / B_{\mathrm{crit}}$ (regime indicator)", color="#e63946")
    ax.set_title(r"(d) regime indicator and asymptotic LB ratio")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Sample-density analysis: how intra-cycle sampling affects the framework\n"
                  f"K={K}, κ={kappa}, $d_F$={d_F}, $B_\\mu$($m$={m0})={B_mu_at_m0}, "
                  f"$B_\\infty$={B_inf}, $\\sigma_0$={sigma0}, $\\alpha$={alpha}",
                  fontsize=12)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


def heatmap_regret_vs_m(N=12, K=K_0, M=200, seed=42,
                          m_grid=None, prior_type="mu_hyb_exact",
                          regret_type="bayesian",
                          m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                          B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                          alpha=ALPHA_INFLATION, kappa=KAPPA_0, d_F=D_F_0,
                          savepath=None, show=True,
                          N_panels=(5, 12, 30)):
    """
    PLOT B.  Empirical regret heatmap over (R_mech, m).  ** BUG FIXED **

    For each (R_mech, m) cell, the simulator is run with R_mech set to the
    USER-CHOSEN value, but we also display per-m ceiling lines for context.
    Each panel uses a different N to show how the picture changes.

    Each m gives a different calibrated R_mech ceiling C(B_μ(m)).  For each
    R_mech below the ceiling, the cell shows the empirical ratio.
    Cells above the ceiling are speculative (marked with *).
    """
    if m_grid is None:
        m_grid = np.array([1, 2, 3, 5, 10, 25])

    H_mu = np.log(K)
    R_mech_grid = np.linspace(0.0, 1.5, 8)

    # Per-m ceiling
    C_at_m = np.array([
        float(channel_capacity_C(B_mu_of_m(m, m0, B_mu_at_m0, B_inf),
                                    sigma_of_m(m, sigma0, alpha),
                                    kappa, d_F, H_mu=H_mu))
        for m in m_grid])

    fig, axes = plt.subplots(1, len(N_panels),
                              figsize=(6.5 * len(N_panels), 6),
                              sharey=True)
    if len(N_panels) == 1:
        axes = [axes]

    # Common color scale across panels
    all_ratios_panels = []
    for N in N_panels:
        # Pre-compute uninformed baseline at this N
        base_mean, _ = run_simulation(K, N, M, 0.0, seed,
                                        prior_type, regret_type)
        ratios = np.full((len(R_mech_grid), len(m_grid)), np.nan)
        # Each (R, m) cell: simulate at R_mech=R, but the m index just labels
        # which m's calibration corresponds to this R.  Same simulation regardless of m.
        # That's the correct semantics: m affects what's REACHABLE (via the ceiling),
        # not what the simulator does at a fixed R_mech.
        for i, R in enumerate(R_mech_grid):
            hyb_mean, _ = run_simulation(K, N, M, float(R), seed,
                                            prior_type, regret_type)
            if hyb_mean > 1e-9:
                ratios[i, :] = base_mean / hyb_mean
        all_ratios_panels.append(ratios)
        print(f"  N={N} done")

    finite_vals = np.concatenate([r[np.isfinite(r)] for r in all_ratios_panels])
    vmin = max(0.5, float(np.nanmin(finite_vals)))
    vmax = float(np.nanmax(finite_vals))

    for ax, N, ratios in zip(axes, N_panels, all_ratios_panels):
        im = ax.imshow(ratios, origin="lower", aspect="auto",
                        vmin=vmin, vmax=vmax, cmap="cividis")

        for i in range(len(R_mech_grid)):
            for j in range(len(m_grid)):
                v = ratios[i, j]
                if np.isfinite(v):
                    color = "white" if v < (vmin + vmax) / 2 else "black"
                    above_ceiling = R_mech_grid[i] > C_at_m[j]
                    txt = f"{v:.2f}*" if above_ceiling else f"{v:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                              color=color, fontsize=9)

        # Per-m ceiling bars
        for j, ceiling in enumerate(C_at_m):
            if R_mech_grid[0] <= ceiling <= R_mech_grid[-1]:
                y_frac = (ceiling - R_mech_grid[0]) / (R_mech_grid[-1] - R_mech_grid[0])
                y_idx = y_frac * (len(R_mech_grid) - 1)
                ax.plot([j - 0.4, j + 0.4], [y_idx, y_idx],
                          color="cyan", linewidth=2.5)

        ax.set_xticks(range(len(m_grid)))
        ax.set_xticklabels([str(m) for m in m_grid])
        ax.set_yticks(range(len(R_mech_grid)))
        ax.set_yticklabels([f"{r:.2f}" for r in R_mech_grid])
        ax.set_xlabel("samples per cycle  $m$")
        ax.set_title(f"N = {N} cycles")

    axes[0].set_ylabel(r"$R_{\mathrm{mech}}$  (nats)")
    fig.suptitle(f"Plot B: empirical regret ratio (uninformed / hybrid) vs ($R_{{mech}}$, $m$)\n"
                  f"K={K}, M={M}, prior='{prior_type}', regret='{regret_type}'\n"
                  "cyan bars = per-$m$ calibrated ceiling $C(B_\\mu(m))$;  "
                  "* = R above ceiling (speculative)",
                  fontsize=12)
    plt.subplots_adjust(top=0.80, bottom=0.10, left=0.08, right=0.88, wspace=0.10)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.60])
    fig.colorbar(im, cax=cbar_ax, label="ratio")

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


def plot_regret_at_ceiling_vs_m(K=K_0, N_values=(5, 12, 30, 100),
                                  M=200, seed=42,
                                  m_grid=None,
                                  prior_type="mu_hyb_exact",
                                  regret_type="bayesian",
                                  m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                                  B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                                  alpha=ALPHA_INFLATION, kappa=KAPPA_0, d_F=D_F_0,
                                  savepath=None, show=True):
    """
    PLOT C.  At each m, set R_mech = C(B_μ(m)) (saturate the calibrated ceiling)
    and run the simulator.  Two panels:
      (1) regret ratio per cycle vs m, several N curves
      (2) regret ratio per total sample (=N*m) vs m, normalized to N=12 baseline
    """
    if m_grid is None:
        m_grid = np.array([1, 2, 3, 5, 10, 25])

    H_mu = np.log(K)
    R_at_m = np.array([
        float(channel_capacity_C(B_mu_of_m(m, m0, B_mu_at_m0, B_inf),
                                    sigma_of_m(m, sigma0, alpha),
                                    kappa, d_F, H_mu=H_mu))
        for m in m_grid])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_values)))

    # Panel 1: regret ratio (cycle-based)
    ax = axes[0]
    ratio_data = {}   # for use in panel 2
    for color, N in zip(colors, N_values):
        ratios = []
        regrets_saved = []
        base_mean, _ = run_simulation(K, N, M, 0.0, seed, prior_type, regret_type)
        for m, R in zip(m_grid, R_at_m):
            hyb_mean, _ = run_simulation(K, N, M, float(R), seed,
                                            prior_type, regret_type)
            ratio = base_mean / hyb_mean if hyb_mean > 1e-9 else np.nan
            ratios.append(ratio)
            regrets_saved.append(max(base_mean - hyb_mean, 0))
        ratios = np.array(ratios)
        regrets_saved = np.array(regrets_saved)
        ratio_data[N] = (ratios, regrets_saved)
        ax.semilogx(m_grid, ratios, "o-", color=color, linewidth=2.0,
                      markersize=8, label=f"N = {N}")
        print(f"  N={N}: ratios at m={list(m_grid)}: {[f'{r:.2f}' for r in ratios]}")

    m_cross_int, _ = find_m_crossing(K=K, kappa=kappa, d_F=d_F,
                                        m0=m0, B_mu_at_m0=B_mu_at_m0,
                                        B_inf=B_inf, sigma0=sigma0, alpha=alpha)
    if m_cross_int is not None:
        ax.axvline(x=m_cross_int, color="black", linestyle="--", linewidth=1.5,
                    alpha=0.5, label=f"phase transition  $m^*$ = {m_cross_int}")

    ax.axhline(y=1.0, color="gray", linewidth=0.8)
    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel("regret ratio  (uninformed / hybrid)")
    ax.set_title("(a) Regret reduction per cycle")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: regret saved per total sample (= regret_saved / (N * m))
    # This shows whether more m is "worth it" given the additional sample cost.
    ax = axes[1]
    for color, N in zip(colors, N_values):
        regrets_saved = ratio_data[N][1]
        per_sample = regrets_saved / (N * m_grid)
        ax.semilogx(m_grid, per_sample, "o-", color=color, linewidth=2.0,
                      markersize=8, label=f"N = {N}")

    if m_cross_int is not None:
        ax.axvline(x=m_cross_int, color="black", linestyle="--", linewidth=1.5,
                    alpha=0.5)

    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel("regret saved per blood draw\n(= regret saved / (N × m))")
    ax.set_title("(b) Regret reduction per total sample\n(efficiency of blood-draw budget)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Plot C: clinical impact of more samples per cycle\n"
                  f"K={K}, M={M}, prior='{prior_type}', regret='{regret_type}'",
                  fontsize=12)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


def plot_cycles_to_target(K=K_0, M=200, seed=42, m_grid=None,
                            target_avg_regrets=(0.15, 0.20, 0.30),
                            prior_type="mu_hyb_exact", regret_type="bayesian",
                            m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                            B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                            alpha=ALPHA_INFLATION, kappa=KAPPA_0, d_F=D_F_0,
                            savepath=None, show=True):
    """
    PLOT D.  Cycles needed for AVERAGE per-cycle regret to fall below target.
    For each m, R_mech = C(B_μ(m)), and we sweep N until regret/N < target.
    Lower curve = faster convergence.

    Targets are in units of average per-cycle regret (units of [0,1]):
      0.15  =  near-optimal (~85% of cycles pick a near-optimal arm)
      0.20  =  good
      0.30  =  acceptable
    Note: these are reachable within N ≤ 200 for typical K=8 calibrated runs.
    More aggressive targets (e.g., 0.05) require N > 200.
    """
    if m_grid is None:
        m_grid = np.array([1, 2, 3, 5, 10, 25])

    H_mu = np.log(K)
    R_at_m = np.array([
        float(channel_capacity_C(B_mu_of_m(m, m0, B_mu_at_m0, B_inf),
                                    sigma_of_m(m, sigma0, alpha),
                                    kappa, d_F, H_mu=H_mu))
        for m in m_grid])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(target_avg_regrets)))

    # Panel 1: cycles needed
    ax = axes[0]
    cycles_table = {}   # target -> array of cycles per m
    for color, target in zip(colors, target_avg_regrets):
        cycles_hyb = []
        cycles_uninf = []
        for m, R in zip(m_grid, R_at_m):
            N_hyb = cycles_to_target(K, M, float(R), float(target), seed,
                                        prior_type, regret_type)
            N_uninf = cycles_to_target(K, M, 0.0, float(target), seed,
                                          prior_type, regret_type)
            cycles_hyb.append(N_hyb if N_hyb else 200)
            cycles_uninf.append(N_uninf if N_uninf else 200)
        cycles_hyb = np.array(cycles_hyb)
        cycles_uninf = np.array(cycles_uninf)
        cycles_table[target] = (cycles_hyb, cycles_uninf)
        ax.semilogx(m_grid, cycles_hyb, "o-", color=color, linewidth=2.0,
                      markersize=8, label=f"target avg regret = {target}")
        # Uninformed line (constant in m, so just mark mean)
        ax.axhline(y=cycles_uninf.mean(), color=color, linestyle=":",
                    alpha=0.6, linewidth=1.5)
        print(f"  target={target}: cycles_hyb={list(cycles_hyb)}, "
              f"cycles_uninf={list(cycles_uninf)}")

    m_cross_int, _ = find_m_crossing(K=K, kappa=kappa, d_F=d_F,
                                        m0=m0, B_mu_at_m0=B_mu_at_m0,
                                        B_inf=B_inf, sigma0=sigma0, alpha=alpha)
    if m_cross_int is not None:
        ax.axvline(x=m_cross_int, color="black", linestyle="--", linewidth=1.5,
                    alpha=0.5, label=f"phase transition  $m^*$ = {m_cross_int}")

    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel("cycles to reach target avg regret")
    ax.set_title("(a) Cycles needed (solid: hybrid; dotted: uninformed)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: total samples needed = cycles × m
    ax = axes[1]
    for color, target in zip(colors, target_avg_regrets):
        cycles_hyb, _ = cycles_table[target]
        total_hyb = cycles_hyb * m_grid
        ax.semilogx(m_grid, total_hyb, "o-", color=color, linewidth=2.0,
                      markersize=8, label=f"target avg regret = {target}")

    if m_cross_int is not None:
        ax.axvline(x=m_cross_int, color="black", linestyle="--", linewidth=1.5,
                    alpha=0.5)

    ax.set_xlabel("samples per cycle  $m$")
    ax.set_ylabel("total blood draws  (= cycles × m)")
    ax.set_title("(b) Total blood draws needed\n(sum of all monitoring across cycles)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Plot D: cycles- and samples-to-target\n"
                  f"K={K}, M={M}, prior='{prior_type}', regret='{regret_type}'",
                  fontsize=12)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


def plot_pareto_frontier(K=K_0, M=200, seed=42, m_grid=None, N_grid=None,
                           prior_type="mu_hyb_exact", regret_type="bayesian",
                           m0=M0_DEFAULT, B_mu_at_m0=B_MU_AT_M0,
                           B_inf=B_MU_FLOOR, sigma0=SIGMA_BASELINE,
                           alpha=ALPHA_INFLATION, kappa=KAPPA_0, d_F=D_F_0,
                           savepath=None, show=True):
    """
    PLOT E.  Pareto frontier in (total blood draws, cumulative regret) space.
    Each (m, N) pair is a point.  The Pareto front shows the most
    sample-efficient (m, N) combinations.

    Lower-left = better (fewer samples, less regret).
    """
    if m_grid is None:
        m_grid = np.array([1, 2, 3, 5, 10, 25])
    if N_grid is None:
        N_grid = np.array([3, 5, 8, 12, 20, 30, 50, 100])

    H_mu = np.log(K)
    R_at_m = np.array([
        float(channel_capacity_C(B_mu_of_m(m, m0, B_mu_at_m0, B_inf),
                                    sigma_of_m(m, sigma0, alpha),
                                    kappa, d_F, H_mu=H_mu))
        for m in m_grid])

    points = []   # (total_samples, regret, m, N)
    print(f"  computing Pareto grid: {len(m_grid)} m × {len(N_grid)} N = {len(m_grid)*len(N_grid)} points")
    for m, R in zip(m_grid, R_at_m):
        for N in N_grid:
            regret_mean, _ = run_simulation(K, int(N), M, float(R), seed,
                                              prior_type, regret_type)
            points.append((int(N) * int(m), regret_mean, int(m), int(N)))

    # Identify Pareto-front points (no other point dominates them)
    points_arr = np.array([(p[0], p[1]) for p in points])
    pareto_mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j: continue
            if (points_arr[j, 0] <= points_arr[i, 0] and
                points_arr[j, 1] <= points_arr[i, 1] and
                (points_arr[j, 0] < points_arr[i, 0] or
                 points_arr[j, 1] < points_arr[i, 1])):
                pareto_mask[i] = False
                break

    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot all points colored by m
    cmap = plt.cm.viridis
    m_min, m_max = float(m_grid.min()), float(m_grid.max())
    for total, regret, m, N in points:
        color = cmap((np.log(m) - np.log(m_min)) / (np.log(m_max) - np.log(m_min) + 1e-9))
        ax.scatter(total, regret, color=color, s=80, alpha=0.6,
                     edgecolor="black", linewidth=0.5)

    # Pareto front
    pareto_pts = sorted([points[i] for i in range(len(points)) if pareto_mask[i]],
                          key=lambda p: p[0])
    pareto_x = [p[0] for p in pareto_pts]
    pareto_y = [p[1] for p in pareto_pts]
    ax.plot(pareto_x, pareto_y, "r-", linewidth=2.5, alpha=0.8, label="Pareto front")
    for total, regret, m, N in pareto_pts:
        ax.annotate(f"m={m},N={N}",
                      (total, regret),
                      textcoords="offset points",
                      xytext=(8, 5),
                      fontsize=8, color="darkred")

    # Colorbar for m
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                  norm=plt.Normalize(vmin=np.log(m_min), vmax=np.log(m_max)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("$m$  (samples per cycle, log scale)")

    ax.set_xlabel("total blood draws  (= N × m)")
    ax.set_ylabel("cumulative regret")
    ax.set_xscale("log")
    ax.set_title(f"Plot E: Pareto frontier — sample budget vs cumulative regret\n"
                  f"K={K}, M={M}, prior='{prior_type}', regret='{regret_type}'\n"
                  "each point = one (m, N) combination; red = Pareto-optimal")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  saved: {savepath}")
    if not show:
        plt.close(fig)


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notables",  action="store_true", help="skip Tables 1, 2, 3")
    parser.add_argument("--noplots",   action="store_true", help="skip heatmaps & tornados")
    parser.add_argument("--noshow",    action="store_true", help="save plots but don't display")
    parser.add_argument("--no_m_analysis", action="store_true",
                        help="skip the sample-density (m) analysis")
    parser.add_argument("--M",         type=int, default=200,
                        help="number of patients for the regret-ratio heatmap (default 200)")
    parser.add_argument("--M_tables",  type=int, default=3000,
                        help="number of patients for Tables 1, 2 (default 3000)")
    parser.add_argument("--prior",     choices=["mu_hyb_exact", "audit_code"],
                        default="mu_hyb_exact",
                        help="prior type (default mu_hyb_exact)")
    parser.add_argument("--regret",    choices=["bayesian", "sample"],
                        default="bayesian",
                        help="regret type (default bayesian)")
    parser.add_argument("--outdir",    type=str, default="./sensitivity_outputs",
                        help="directory for saved figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    show = not args.noshow

    # ----- Calibration block -----
    print_running_examples(K=K_0, B_mu=B_MU_0, sigma=SIGMA_0,
                              kappa=KAPPA_0, d_F=D_F_0,
                              R_mech_anchor=RMECH_DEFAULT)

    # ----- Tables 1 and 2 -----
    if not args.notables:
        R_mech_values  = [0.0, 0.3, 0.5, 0.7, 0.87, 1.0, 1.4, 1.9]
        N_values       = [5, 10, 20, 30, 50, 100, 200]
        print_table1(K=K_0, N=12, M=args.M_tables, seed=42,
                       R_mech_values=R_mech_values,
                       prior_type=args.prior, regret_type=args.regret)
        print_table2(K=K_0, M=args.M_tables, seed=42,
                       R_mech_anchor=RMECH_DEFAULT,
                       N_values=N_values,
                       prior_type=args.prior, regret_type=args.regret)

    # ----- Heatmaps -----
    if not args.noplots:
        print("\n" + "=" * 78)
        print("HEATMAPS")
        print("=" * 78)

        print("\nHeatmap 1: C(B_mu) over (sigma, B_mu), panels at kappa in {1.0, 1.8, 2.5}")
        heatmap_C_of_B_mu(savepath=os.path.join(args.outdir, "heatmap_C.png"), show=show)

        print("\nHeatmap 2: B_crit over (sigma, kappa), panels at d_F in {2, 3, 5}")
        heatmap_B_crit(savepath=os.path.join(args.outdir, "heatmap_Bcrit.png"), show=show)

        print(f"\nHeatmap 3: empirical regret ratio (M={args.M}, "
              f"prior='{args.prior}', regret='{args.regret}')")
        heatmap_regret_ratio(M=args.M,
                              prior_type=args.prior,
                              regret_type=args.regret,
                              savepath=os.path.join(args.outdir, "heatmap_regret_ratio.png"),
                              show=show)

        # ----- Tornado -----
        print("\nTornado plots: one-at-a-time sensitivity for C, B_crit, LB_ratio")
        tornado_panel(savepath=os.path.join(args.outdir, "tornado.png"), show=show)

    # ----- Sample-density (m) analysis -----
    if not args.no_m_analysis:
        if not args.notables:
            print_table3_sample_density(K=K_0, kappa=KAPPA_0, d_F=D_F_0,
                                          m_values=M_SAMPLES_VALUES,
                                          add_clinical_columns=True,
                                          M=args.M, seed=42,
                                          target_avg_regret=0.2,
                                          prior_type=args.prior,
                                          regret_type=args.regret)
        if not args.noplots:
            print("\n" + "=" * 78)
            print("SAMPLE-DENSITY ANALYSIS — plots")
            print("=" * 78)

            print("\nPlot A: B_μ(m), σ(m), C(m), B_crit(m), regime indicator")
            plot_quantities_vs_m(
                savepath=os.path.join(args.outdir, "m_analysis_A_quantities.png"),
                show=show)

            print(f"\nPlot B: empirical regret heatmap (R_mech, m) at multiple N, M={args.M}")
            heatmap_regret_vs_m(
                M=args.M, prior_type=args.prior, regret_type=args.regret,
                savepath=os.path.join(args.outdir, "m_analysis_B_heatmap.png"),
                show=show)

            print(f"\nPlot C: regret ratio at calibrated ceiling vs m (per-cycle and per-sample), M={args.M}")
            plot_regret_at_ceiling_vs_m(
                M=args.M, prior_type=args.prior, regret_type=args.regret,
                savepath=os.path.join(args.outdir, "m_analysis_C_clinical.png"),
                show=show)

            print(f"\nPlot D: cycles-to-target as a function of m, M={args.M}")
            plot_cycles_to_target(
                M=args.M, prior_type=args.prior, regret_type=args.regret,
                savepath=os.path.join(args.outdir, "m_analysis_D_cycles_to_target.png"),
                show=show)

            print(f"\nPlot E: Pareto frontier (total samples vs cumulative regret), M={args.M}")
            plot_pareto_frontier(
                M=args.M, prior_type=args.prior, regret_type=args.regret,
                savepath=os.path.join(args.outdir, "m_analysis_E_pareto.png"),
                show=show)

    print("\n" + "=" * 78)
    print(f"Done. Plots saved to: {os.path.abspath(args.outdir)}")
    print("=" * 78)

    # If interactive (show=True), display all figures at once.
    # (Each plot function leaves its figure open; this single blocking call
    # shows them all together. Closing any window unblocks the script.)
    if show and not args.noplots:
        plt.show()


if __name__ == "__main__":
    main()
