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

# Method 1: alpha = i_prior, beta = 1 (your current change)
alpha1 = i_prior
beta1 = np.ones(K)
H1 = categorical_entropy_from_beta(alpha1, beta1)
print(f"Method 1: alpha = i_prior, beta = 1")
print(f"  Categorical entropy: {H1:.6f}")
print(f"  Error from target: {abs(H1 - target_entropy):.6f}")
print()

# Method 2: alpha = 1.0 + rmech * i_prior, beta = 1 (my suggestion)
alpha2 = 1.0 + rmech * i_prior
beta2 = np.ones(K)
H2 = categorical_entropy_from_beta(alpha2, beta2)
print(f"Method 2: alpha = 1.0 + rmech * i_prior, beta = 1")
print(f"  Categorical entropy: {H2:.6f}")
print(f"  Error from target: {abs(H2 - target_entropy):.6f}")
print()

# Method 3: alpha = rmech * i_prior, beta = rmech * (1 - i_prior)
alpha3 = rmech * i_prior
beta3 = rmech * (1.0 - i_prior)
H3 = categorical_entropy_from_beta(alpha3, beta3)
print(f"Method 3: alpha = rmech * i_prior, beta = rmech * (1 - i_prior)")
print(f"  Categorical entropy: {H3:.6f}")
print(f"  Error from target: {abs(H3 - target_entropy):.6f}")
print()

alpha4 = np.ones(K)
beta4 = np.ones(K)
alpha_0, p_1 = find_prior_generalized(target_entropy, K, 1.0)
alpha4[0] = alpha_0
H4 = categorical_entropy_from_beta(alpha4, beta4)
print(f"Method 4: alpha = {alpha_0:.6f}, beta = 1.0")
print(f"  Categorical entropy: {H4:.6f}")
print(f"  Error from target: {abs(H4 - target_entropy):.6f}")
print()

print(f"Which is closer to target?")
print(f"  Method 1 error: {abs(H1 - target_entropy):.6f}")
print(f"  Method 2 error: {abs(H2 - target_entropy):.6f}")
print(f"  Method 3 error: {abs(H3 - target_entropy):.6f}")
best_method = min(
    (abs(H1 - target_entropy), "Method 1"),
    (abs(H2 - target_entropy), "Method 2"),
    (abs(H3 - target_entropy), "Method 3"),
)
print(f"  → {best_method[1]} is closest with error {best_method[0]:.6f}")

if best_method[0] < 0.1:
    print(f"\n✓ Error is below 0.1 threshold - ready for application")
else:
    print(f"\n✗ Error is above 0.1 threshold - more testing needed")
