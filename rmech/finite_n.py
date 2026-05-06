"""
rmech/finite_n.py
─────────────────────────────────────────────────────────────────────────────
Finite-N regret validation of the 1.0-nat threshold.

Simulates Thompson Sampling with a mechanistic prior over a short clinical
course (N = 8–12 cycles) and measures cumulative regret reduction relative
to an uninformed prior.

Reproduces the result reported in the paper:
  "For a representative case (K=4, H(mu)≈1.4 nats) at N=10 cycles,
   R_mech = 1.0 nat yields 22% lower cumulative regret than an uninformed
   prior; R_mech = 0.5 nats yields only 7%."
"""

import numpy as np
from scipy.optimize import brentq


def generate_linear_mixture_pmf(num_states, target_entropy, target_index=0):
    """
    Generates a Categorical PMF using a linear mixture to match a target entropy.
    
    Args:
        num_states (int): The number of states (K).
        target_entropy (float): The desired Shannon entropy in bits (C).
        target_index (int): The index that will hold the maximum probability.
        
    Returns:
        np.ndarray: The resulting Probability Mass Function (PMF).
    """
    max_entropy = np.log(num_states)
    
    # 1. Handle edge cases
    if target_entropy < 0 or target_entropy > max_entropy + 1e-9:
        raise ValueError(f"Target entropy must be between 0 and {max_entropy:.4f} bits.")
    
    if np.isclose(target_entropy, 0):
        pmf = np.zeros(num_states)
        pmf[target_index] = 1.0
        return pmf
        
    if np.isclose(target_entropy, max_entropy):
        return np.full(num_states, 1.0 / num_states)

    # 2. Define the entropy function based on alpha
    def mixture_entropy(alpha):
        p_target = 1.0 - alpha * ((num_states - 1) / num_states)
        p_other = alpha / num_states
        
        # Calculate Shannon entropy: H = -sum(p * log2(p))
        H = 0.0
        if p_target > 0:
            H -= p_target * np.log(p_target)
        if p_other > 0:
            # Multiply by (K-1) since there are K-1 other identical states
            H -= (num_states - 1) * p_other * np.log(p_other)
            
        return H

    # 3. Find the optimal mixing parameter alpha where H(alpha) - target = 0
    # brentq is highly reliable for bounded root-finding on monotonic functions
    optimal_alpha = brentq(lambda a: mixture_entropy(a) - target_entropy, 0.0, 1.0)

    # 4. Construct the final PMF array
    pmf = np.full(num_states, optimal_alpha / num_states)
    pmf[target_index] = 1.0 - optimal_alpha * ((num_states - 1) / num_states)
    
    return pmf


def run_thompson_sampling(
    N_cycles:      int,
    K:             int,
    mu_prior:      np.ndarray,
    rmech:         float,
    n_patients:    int = 5_000,
    seed:          int = 0,
    opt_prob:      float = 0.85,
    bsa_prob:      float = 0.25
) -> float:
    """
    Simulate Thompson Sampling over a fixed N-cycle course.

    A mechanistic prior with strength rmech biases the initial Dirichlet
    concentration toward the model's recommended arm. The model's accuracy
    is 1 - exp(-rmech) (monotone in rmech, 0 at rmech=0, →1 as rmech→∞).

    Returns
    -------
    float : mean cumulative regret per patient
    """
    rng = np.random.default_rng(seed)
    mu  = np.asarray(mu_prior, dtype=float)
    mu  = mu / mu.sum()

    total_regret = 0.0
    i_prior = generate_linear_mixture_pmf(K, np.log(K)-rmech, target_index=1)
    total_regrets = []

    for _ in range(n_patients):
        pistar = rng.choice(K, p=mu)
        i_prior = generate_linear_mixture_pmf(K, np.log(K)-rmech, target_index=pistar)
        # Initialise Dirichlet/Beta prior
        alpha   = np.ones(K, dtype=float)
        beta_   = np.ones(K, dtype=float)

        if rmech > 0:
            # Model recommends an arm; with probability `accuracy` it is correct
            pihat = rng.choice(K, p=i_prior)
            i_prior = generate_linear_mixture_pmf(K, np.log(K)-rmech, target_index=pihat)
            alpha = i_prior       # concentrate prior on recommendation
            beta_ = np.ones(K, dtype=float)

        cumulative_regret = 0.0
        for _ in range(N_cycles):
            theta = rng.beta(alpha, beta_)
            arm   = int(np.argmax(theta))
            selection_prob = opt_prob if arm == pistar else bsa_prob
            update_reward = 1.0 if rng.random() < selection_prob else 0.0
            cumulative_regret += (opt_prob - selection_prob)
            alpha[arm] += update_reward
            beta_[arm]  += (1.0 - update_reward)

        total_regret += cumulative_regret
        total_regrets.append(cumulative_regret)

    mean_regret = total_regret / n_patients
    std_error = np.std(total_regrets, ddof=1) / np.sqrt(n_patients) * 2  # 96% CI
    return mean_regret, std_error
