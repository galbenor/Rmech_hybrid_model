import numpy as np
from scipy.special import digamma
from rmech.finite_n import generate_linear_mixture_pmf
from scipy.special import betaln
from scipy.optimize import root_scalar

def exact_p1(alpha_0, beta_0, K):
    """Calculates exact p1 using Log-Beta for numerical stability."""
    return np.exp(betaln(alpha_0 + K - 1, beta_0) - betaln(alpha_0, beta_0))

def calculate_entropy(p1, K):
    """Calculates Shannon entropy in nats."""
    if p1 >= 1.0 or p1 <= 0.0: 
        return 0.0
    p_rest = (1 - p1) / (K - 1)
    return -p1 * np.log(p1) - (1 - p1) * np.log(p_rest)

def find_prior_generalized(target_entropy_nats, K, beta_0=1.0):
    """Finds the required alpha_0 for ANY chosen beta_0."""
    max_entropy = np.log(K)
    if not (0 < target_entropy_nats < max_entropy):
        raise ValueError(f"Target entropy must be between 0 and {max_entropy:.4f} nats.")

    def objective(alpha):
        p1 = exact_p1(alpha, beta_0, K)
        return calculate_entropy(p1, K) - target_entropy_nats

    # Bracket set wide to handle beta_0 > 1 which requires massive alpha_0 values
    res = root_scalar(objective, bracket=[0.01, 1e6])
    
    if res.converged:
        alpha_0 = res.root
        return alpha_0, exact_p1(alpha_0, beta_0, K)
    else:
        raise ValueError("Optimization failed to converge.")
    
def categorical_entropy_from_beta(alpha, beta):
    """Approximate categorical entropy from Beta(alpha, beta) samples."""
    rng = np.random.default_rng(42)
    K = len(alpha)
    num_samples = 100_000
    # 1. Sample theta values for all arms simultaneously
    samples = rng.beta(alpha, beta, size=(num_samples, K))
    
    # 2. Simulate Thompson Sampling: Which arm has the highest theta?
    best_arms = np.argmax(samples, axis=1)
    
    # 3. Calculate the empirical probability of each arm winning (p_i)
    counts = np.bincount(best_arms, minlength=K)
    p = counts / num_samples
    
    # 4. Compute Shannon entropy (filtering out 0 probabilities to avoid log(0))
    p_nonzero = p[p > 0]
    H = -np.sum(p_nonzero * np.log(p_nonzero))
    
    return H

K = 8
rmech = 0.87
target_entropy = np.log(K) - rmech

print(f"K = {K}, rmech = {rmech}")
print(f"Target entropy: log(K) - rmech = {target_entropy:.6f}")
print()

# Generate the mechanistic prior PMF (this has exactly the target entropy)
i_prior = generate_linear_mixture_pmf(K, target_entropy, target_index=0)
print(f"i_prior entropy: {-np.sum(i_prior[i_prior > 0] * np.log(i_prior[i_prior > 0])):.6f}")
print()
