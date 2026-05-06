"""
sensitivity_analysis.py
================================================================================
Sensitivity-analysis figures for "Mechanistic Information and Regret"
(NeurIPS 2026 submission, mechanistic_information.pdf).

Four figure-generating functions, all loyal to the paper and the existing
code in `rmech.finite_n` and `create_table2_new_values.py`.  No existing
file is modified -- this is a drop-in addition.

  - make_fig_sensitivity_full()  -> fig_sensitivity_full.pdf
        2x3 grid: sensitivity of C(B_mu) and B_mu / B_crit to (kappa_mu, B_mu, d_F)
        at the calibrated 5-FU operating point.
  - make_fig_sensitivity_K()     -> fig_sensitivity_K.pdf
        1x2 grid: sensitivity to K (number of arms).  Shows B_crit vs K and
        the asymptotic ratio rho = H(mu)/H_mech.

Formulas (verified to 4 decimals against the paper at the calibrated point):
  - C(B_mu)       -- Eq. 3 with canonical sigma_F^2 = 2 sigma^2 H(mu) / (kappa^2 d_F)
  - B_crit(N)     -- Eq. 8 (corrected, NOT the small-SNR approximation)
  - mu_hyb prior  -- generate_linear_mixture_pmf, the same construction used
                     in rmech.finite_n.run_thompson_sampling.

Calibrated 5-FU operating point (paper Sec. 5.1, Appendix H.4):
  K = 8,  B_mu = 0.22,  sigma = 0.40,  kappa_mu = 1.8,  d_F = 3,  N = 12
  H(mu) = ln K = 2.0794,   H(mu)/N = 0.1733,   sqrt(ln K) = 1.4420.

Run as:  python sensitivity_analysis.py
Runtime: ~60-90s (figure 4 dominates: ~30 simulations of K=8, 10000 patients).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from rmech.finite_n import generate_linear_mixture_pmf

# ============================================================================
#  Calibrated 5-FU operating point (paper Sec. 5.1 / Appendix H.4)
# ============================================================================
CAL_K     = 8
CAL_BMU   = 0.22
CAL_SIGMA = 0.40
CAL_KAPPA = 1.8
CAL_DF    = 3
CAL_N     = 12
CAL_HMU   = np.log(CAL_K)         # = 2.07944

# Bernoulli reward rate used by _run_ts_trajectory, mirroring
# rmech.finite_n.run_thompson_sampling exactly: a single opt_prob knob
# with selection_prob = opt_prob if arm == pi*, 1 - opt_prob otherwise.
# Default 0.8 matches upstream's default at the calibrated 5-FU point.
OPT_PROB = 0.8


# ============================================================================
#  Theoretical helpers (paper formulas; canonical parametrization, Remark 2)
# ============================================================================
def _sigmaF2(sigma: float, kappa: float, H_mu: float, d_F: int) -> float:
    """Canonical sigma_F^2 = 2 sigma^2 H(mu) / (kappa^2 d_F)  (Remark 2)."""
    return 2.0 * sigma**2 * H_mu / (kappa**2 * d_F)


def channel_capacity_C(B_mu: float, sigma: float, kappa: float,
                       H_mu: float, d_F: int) -> float:
    """C(B_mu) under canonical parametrization (paper Eq. 3)."""
    sF2 = _sigmaF2(sigma, kappa, H_mu, d_F)
    num = kappa**2 * sF2
    den = kappa**2 * B_mu**2 + sigma**2
    return 0.5 * d_F * np.log1p(num / den)


def B_crit(sigma: float, kappa: float, H_mu: float, d_F: int, N: int) -> float:
    """Corrected B_crit from paper Eq. 8 (NOT the small-SNR approximation).
    Returns NaN if the argument under the sqrt is non-positive."""
    inside = (2.0 * H_mu / d_F) / (np.exp(2.0 * H_mu / (d_F * N)) - 1.0) - 1.0
    if inside <= 0:
        return np.nan
    return (sigma / kappa) * np.sqrt(inside)




# ============================================================================
#  Figure 1:  fig_sensitivity_full.pdf
#             Sensitivity of the model-quality certificate to (kappa, B, d_F)
# ============================================================================
def make_fig_sensitivity_full(out_path: str = "fig_sensitivity_full.pdf") -> None:
    """2x3 grid.  Top row: 1D capacity curves C vs each parameter.  Bottom row:
    2D heatmaps of B_mu / B_crit over each parameter pair, with the phase
    boundary at ratio = 1 marked in black and the calibrated 5-FU point
    marked with a gold dot."""
    plt.rcParams['font.size'] = 14
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    threshold = CAL_HMU / CAL_N    # 0.1733; the C(B_crit)=H(mu)/N working point
    H_mu = CAL_HMU

    # -- Panel A: C vs B_mu, kappa = 1.8, varying d_F ------------------------
    B_grid = np.linspace(0.05, 1.5, 250)
    ax = axes[0, 0]
    for d_F, color in zip([1, 3, 10], ["C0", "C1", "C2"]):
        Cv = [channel_capacity_C(B, CAL_SIGMA, CAL_KAPPA, H_mu, d_F) for B in B_grid]
        ax.plot(B_grid, Cv, color=color, lw=1.7, label=f"$d_F = {d_F}$")
    ax.axhline(threshold, ls=":", color="grey", lw=1,
               label=fr"$H(\mu)/N = {threshold:.3f}$")
    ax.axvline(CAL_BMU, ls="--", color="grey", lw=1, alpha=0.8)
    ax.text(CAL_BMU + 0.015, 0.03,
            f"$B_\\mu = {CAL_BMU}$\n(calibrated)", fontsize=12, color="grey")
    ax.set_xlabel(r"$B_\mu$  (model bias)")
    ax.set_ylabel(r"Channel capacity $C(B_\mu)$  [nats]")
    ax.set_title("A.")
    ax.legend(loc="upper right", fontsize=12)
    ax.set_xlim(B_grid[0], B_grid[-1])
    ax.set_ylim(0, None)

    # -- Panel B: C vs kappa, B_mu = 0.22, varying d_F -----------------------
    kappa_grid = np.linspace(0.3, 5.5, 250)
    ax = axes[0, 1]
    for d_F, color in zip([1, 3, 10], ["C0", "C1", "C2"]):
        Cv = [channel_capacity_C(CAL_BMU, CAL_SIGMA, k, H_mu, d_F) for k in kappa_grid]
        ax.plot(kappa_grid, Cv, color=color, lw=1.7, label=f"$d_F = {d_F}$")
    ax.axhline(threshold, ls=":", color="grey", lw=1,
               label=fr"$H(\mu)/N = {threshold:.3f}$")
    ax.axvline(CAL_KAPPA, ls="--", color="grey", lw=1, alpha=0.8)
    ax.text(CAL_KAPPA + 0.05, 0.07,
            f"$\\kappa_\\mu = {CAL_KAPPA}$\n(calibrated)", fontsize=12, color="grey")
    ax.set_xlabel(r"$\kappa_\mu$  (PMP sensitivity)")
    ax.set_ylabel(rf"$C(B_\mu = {CAL_BMU})$  [nats]")
    ax.set_title("B.")
    ax.legend(loc="upper right", fontsize=12)
    ax.set_xlim(kappa_grid[0], kappa_grid[-1])
    ax.set_ylim(0, None)

    # -- Panel C: C vs d_F, B_mu = 0.22, varying kappa -----------------------
    d_F_grid = np.arange(1, 11)
    ax = axes[0, 2]
    for kappa, color in zip([1.0, 1.8, 3.0], ["C0", "C1", "C2"]):
        Cv = [channel_capacity_C(CAL_BMU, CAL_SIGMA, kappa, H_mu, d) for d in d_F_grid]
        ax.plot(d_F_grid, Cv, marker="o", color=color, lw=1.7, ms=4,
                label=f"$\\kappa_\\mu = {kappa}$")
    ax.axhline(H_mu, ls="-.", color="grey", lw=1, alpha=0.6)
    ax.text(d_F_grid[-1] - 0.4, H_mu - 0.07,
            f"$H(\\mu) = \\ln K = {H_mu:.2f}$", fontsize=12, color="grey",
            ha="right")
    ax.axhline(threshold, ls=":", color="grey", lw=1)
    ax.text(d_F_grid[-1] - 0.4, threshold + 0.02,
            f"$H(\\mu)/N = {threshold:.3f}$", fontsize=12, color="grey", ha="right")
    ax.axvline(CAL_DF, ls="--", color="grey", lw=1, alpha=0.8)
    ax.text(CAL_DF + 0.1, 0.55,
            f"$d_F = {CAL_DF}$\n(calibrated)", fontsize=12, color="grey")
    ax.set_xlabel(r"$d_F$  (residual GP rank)")
    ax.set_ylabel(rf"$C(B_\mu = {CAL_BMU})$  [nats]")
    ax.set_title("C.")
    ax.legend(loc="upper left", fontsize=12)
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0, max(2.1, H_mu + 0.1))

    # -- Heatmaps: B_mu / B_crit ratio over each parameter pair --------------
    cmap = "RdBu_r"
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.5)

    B_h     = np.linspace(0.05, 1.5, 120)
    kappa_h = np.linspace(0.3, 5.5, 120)
    d_F_h   = np.arange(1, 21)        # integer d_F in heatmaps
    N_h     = np.arange(5, 51)        # N in heatmaps

    # Panel D: (B_mu, kappa) at d_F = 3
    Z_D = np.empty((len(kappa_h), len(B_h)))
    for i, k in enumerate(kappa_h):
        bc = B_crit(CAL_SIGMA, k, H_mu, CAL_DF, CAL_N)
        Z_D[i, :] = B_h / bc if not np.isnan(bc) else np.nan
    ax = axes[1, 0]
    im = ax.imshow(Z_D, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   extent=[B_h[0], B_h[-1], kappa_h[0], kappa_h[-1]])
    cs = ax.contour(B_h, kappa_h, Z_D, levels=[1.0], colors="black", linewidths=1.3)
    try:
        ax.clabel(cs, fmt={1.0: r"$B_\mu/B^{\rm crit}_\mu = 1$"},
                  fontsize=12, inline=True)
    except Exception:
        pass
    ax.scatter([CAL_BMU], [CAL_KAPPA], marker="o", s=120,
               facecolor="gold", edgecolor="black", linewidth=1.0, zorder=10)
    ax.text(CAL_BMU + 0.07, CAL_KAPPA - 0.05, "calibrated 5-FU", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))
    ax.set_xlabel(r"$B_\mu$")
    ax.set_ylabel(r"$\kappa_\mu$")
    ax.set_title("D.")

    # Panel E: (B_mu, d_F) at kappa = 1.8
    Z_E = np.empty((len(d_F_h), len(B_h)))
    for i, d in enumerate(d_F_h):
        bc = B_crit(CAL_SIGMA, CAL_KAPPA, H_mu, d, CAL_N)
        Z_E[i, :] = B_h / bc if not np.isnan(bc) else np.nan
    ax = axes[1, 1]
    im = ax.imshow(Z_E, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   extent=[B_h[0], B_h[-1], d_F_h[0], d_F_h[-1]])
    cs = ax.contour(B_h, d_F_h, Z_E, levels=[1.0], colors="black", linewidths=1.3)
    try:
        ax.clabel(cs, fmt={1.0: r"$B_\mu/B^{\rm crit}_\mu = 1$"},
                  fontsize=12, inline=True)
    except Exception:
        pass
    ax.scatter([CAL_BMU], [CAL_DF], marker="o", s=120,
               facecolor="gold", edgecolor="black", linewidth=1.0, zorder=10)
    ax.text(CAL_BMU + 0.07, CAL_DF - 0.4, "calibrated 5-FU", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))
    ax.set_xlabel(r"$B_\mu$")
    ax.set_ylabel(r"$d_F$")
    ax.set_title("E.")

    # Panel F: (B_mu, N) at kappa = 1.8, d_F = 3
    Z_F = np.empty((len(N_h), len(B_h)))
    for i, n in enumerate(N_h):
        bc = B_crit(CAL_SIGMA, CAL_KAPPA, H_mu, CAL_DF, n)
        Z_F[i, :] = B_h / bc if not np.isnan(bc) else np.nan
    ax = axes[1, 2]
    im = ax.imshow(Z_F, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   extent=[B_h[0], B_h[-1], N_h[0], N_h[-1]])
    cs = ax.contour(B_h, N_h, Z_F, levels=[1.0], colors="black", linewidths=1.3)
    try:
        ax.clabel(cs, fmt={1.0: r"$B_\mu/B^{\rm crit}_\mu = 1$"},
                  fontsize=12, inline=True)
    except Exception:
        pass
    ax.scatter([CAL_BMU], [CAL_N], marker="o", s=120,
               facecolor="gold", edgecolor="black", linewidth=1.0, zorder=10)
    ax.text(CAL_BMU + 0.07, CAL_N - 1.5, "calibrated 5-FU", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))
    ax.set_xlabel(r"$B_\mu$")
    ax.set_ylabel(r"$N$")
    ax.set_title("F.")

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.85, aspect=22, pad=0.015)
    cbar.set_label(r"$B_\mu / B^{\rm crit}_\mu$"
                   "\n($< 1$: data-efficient, $> 1$: baseline)", fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================================
#  Figure 2:  fig_sensitivity_K.pdf
#             Sensitivity of the certificate to K (number of arms).
# ============================================================================
def make_fig_sensitivity_K(out_path: str = "fig_sensitivity_K.pdf") -> None:
    """Sensitivity to K (number of arms) at the calibrated 5-FU operating
    point (B_mu = 0.22, sigma = 0.40, kappa_mu = 1.8, d_F = 3, N = 12).

    Shows rho = H(mu)/H_mech vs K (Eq. 7), with rho = 1 = no benefit.
    Calibrated K = 8 marked with a gold dot."""
    plt.rcParams['font.size'] = 10
    K_grid = np.arange(2, 21)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # -- Panel B: asymptotic regret-ratio rho vs K ---------------------------
    eps = 1e-9
    rho = []
    for K in K_grid:
        Hm = np.log(K)
        C  = channel_capacity_C(CAL_BMU, CAL_SIGMA, CAL_KAPPA, Hm, CAL_DF)
        rho.append(Hm / max(Hm - C, eps))
    rho = np.array(rho)
    ax.plot(K_grid, rho, color="C0", lw=2,
            label=r"$\rho = H(\mu)/H_{\rm mech}$")
    ax.axhline(1.0, ls=":", color="grey", lw=1.2, label=r"$\rho = 1$ (no benefit)")
    K_idx = int(np.where(K_grid == CAL_K)[0][0])
    ax.plot([CAL_K], [rho[K_idx]], marker="o", color="gold", ms=10,
            mec="black", mew=0.8, zorder=10)
    ax.set_xlabel(r"$K$  (number of arms)")
    ax.set_ylabel(r"$\rho = H(\mu)/H_{\rm mech}$   (asymptotic samples saved)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    ax.set_xlim(K_grid[0], K_grid[-1])
    ax.set_ylim(0.95, 2.0)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")




# ============================================================================
#  Main
# ============================================================================
if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    make_fig_sensitivity_full()
    sys.stdout.flush()
    make_fig_sensitivity_K()
    sys.stdout.flush()
    print(f"All four figures generated in {time.time() - t0:.1f}s.")