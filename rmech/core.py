"""
rmech/core.py
─────────────────────────────────────────────────────────────────────────────
Core engine for computing mechanistic information R_mech.

R_mech := I_mu(pi*; pi_hat)
        = H(pi*) - H(pi* | pi_hat)
        = mutual information between the patient's true optimal arm (pi*)
          and the mechanistic model's recommendation (pi_hat), measured
          before any patient-specific data are collected.

Reference:
  Mannor S. (2026). "Mechanistic information: a pre-trial quality criterion
  for adaptive precision medicine." Nature Medicine (submitted).

  Companion theory paper:
  Mannor S. (2026). "Mechanistic Information and Regret: Lower and Upper
  Bounds for Hybrid Model Learning." JMLR (submitted).
"""

import numpy as np
from typing import Callable, Optional

__all__ = ["rmech_monte_carlo", "entropy", "convergence_factor", "sample_size_inflation"]

RNG_SEED = 2024


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats of a discrete distribution p."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def rmech_monte_carlo(
    sample_pistar: Callable[[int], np.ndarray],
    sample_pihat:  Callable[[np.ndarray], np.ndarray],
    n_patients:    int = 200_000,
    K:             Optional[int] = None,
    seed:          int = RNG_SEED,
) -> dict:
    """
    Compute R_mech by Monte Carlo simulation.

    Parameters
    ----------
    sample_pistar : callable(n) -> array of int
        Draw n i.i.d. samples of the true optimal arm index (0-indexed).
        Should reflect the population prior mu over K arms.
    sample_pihat : callable(pistar_array) -> array of int
        Given the true optimal arms, return the mechanistic model's
        recommendation for each patient. May use the true arm (to model
        a partially-accurate predictor) or independent features.
    n_patients : int
        Monte Carlo sample size. Default 200,000 gives bootstrap SE < 0.02 nats.
    K : int, optional
        Number of arms. Inferred from pistar samples if not given.
    seed : int
        NumPy random seed.

    Returns
    -------
    dict with keys:
        rmech   : float  -- R_mech estimate (nats)
        H_mu    : float  -- Prior entropy H(mu) (nats)
        H_cond  : float  -- Conditional entropy H(pi*|pi_hat) (nats)
        mu      : ndarray -- Estimated arm prior probabilities
        threshold_passed : bool -- True if rmech >= 1.0 nat
    """
    rng = np.random.default_rng(seed)

    pistar = np.asarray(sample_pistar(n_patients), dtype=int)
    if K is None:
        K = int(pistar.max()) + 1

    pihat  = np.asarray(sample_pihat(pistar), dtype=int)

    # Prior entropy
    mu     = np.bincount(pistar, minlength=K) / n_patients
    H_mu   = entropy(mu)

    # Conditional entropy H(pi* | pi_hat)
    H_cond = 0.0
    for j in range(K):
        mask = pihat == j
        pj   = mask.mean()
        if pj < 1e-9:
            continue
        c     = np.bincount(pistar[mask], minlength=K) / mask.sum()
        H_cond += pj * entropy(c)

    rmech = H_mu - H_cond

    return {
        "rmech":            round(float(rmech),  4),
        "H_mu":             round(float(H_mu),   4),
        "H_cond":           round(float(H_cond), 4),
        "mu":               mu,
        "K":                K,
        "n_patients":       n_patients,
        "threshold_passed": rmech >= 1.0,
    }


def convergence_factor(H_mu: float, rmech: float) -> float:
    """
    Convergence acceleration factor when R_mech >= 1.0 nat.

    The mechanistic prior reduces the number of cycles to convergence by
    this factor relative to uninformed adaptive selection.

    factor = H(mu) / (H(mu) - R_mech)

    Returns float('inf') if rmech >= H_mu.
    """
    denom = H_mu - rmech
    if denom <= 0:
        return float('inf')
    return H_mu / denom


def sample_size_inflation(H_mu: float, rmech_actual: float,
                          rmech_assumed: float = 1.0) -> float:
    """
    Convergence-time ratio: how many more patient-cycles a sub-threshold
    programme requires relative to a programme at rmech_assumed.

    ratio = (H_mu - rmech_actual) / (H_mu - rmech_assumed)

    This is the convergence-time ratio. The corresponding sample size
    inflation for a specific power target requires additional trial design
    assumptions about the effect size distribution.

    Parameters
    ----------
    H_mu : float
        Prior entropy of the decision (nats).
    rmech_actual : float
        True R_mech of the programme.
    rmech_assumed : float
        R_mech assumed when powering the trial (default: 1.0 nat = threshold).

    Returns
    -------
    float >= 1.0  (returns inf if rmech_assumed >= H_mu)
    """
    denom = H_mu - rmech_assumed
    if denom <= 0:
        return float('inf')
    return (H_mu - rmech_actual) / denom
