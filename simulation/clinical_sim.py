"""
Simulation study: TS_hyb vs. BSA dosing vs. uninformed TS
for 5-FU dose individualisation in FOLFOX.

Population PK parameters from:
  Gamelin et al. JCO 2008 (N=186 RCT)
  Kaldate et al. The Oncologist 2012 (N=187, 307 cycle-pairs)
  Li et al. ESMO Open 2023 (N=434, 37 sites)
  Fleming et al. 2015 (5-FU PK/PD model)
"""

import numpy as np
import json
from scipy.stats import beta as beta_dist

rng = np.random.default_rng(42)

# ── Population parameters ────────────────────────────────────────────────
# 8 dose levels (mg/m²), spanning clinical adjustment range
DOSE_LEVELS = np.array([1600, 1800, 2000, 2200, 2400, 2600, 2800, 3200])
K = len(DOSE_LEVELS)
BSA_DOSE_IDX = 4          # index of standard BSA dose (2400 mg/m²)

# AUC ~ Dose / CL; target AUC = 20-30 mg·h/L
# CL ~ LogNormal(mu, sigma) with population median such that
# AUC(2400) ~ median 25 mg·h/L and CV ~55% (from Gamelin/Kaldate)
# Equivalently: CL(L/h) ~ LogNormal with median 96 mg·h/L (= 2400/25)
CL_MEDIAN = 96.0    # normalised units: dose_unit / AUC_unit
CL_SIGMA  = 0.55    # log-scale SD (55% CV gives 20% in target window at BSA dose)
TARGET_LO, TARGET_HI = 20.0, 30.0

def simulate_patient():
    """Draw a patient's true clearance and return reward probabilities per dose."""
    cl = CL_MEDIAN * np.exp(rng.normal(0, CL_SIGMA))
    # cycle-to-cycle intra-patient AUC noise (CV ~20%)
    sigma_auc = 5.0   # mg·h/L (20% of 25 mg·h/L)
    def p_hit(dose):
        """P(AUC in target | dose, CL) averaged over intra-patient noise."""
        auc_mean = dose / cl * 25.0   # scaled so median CL → AUC = dose/96*25
        # P(target lo < N(auc_mean, sigma_auc) < target hi)
        from scipy.stats import norm
        return norm.cdf(TARGET_HI, auc_mean, sigma_auc) - \
               norm.cdf(TARGET_LO, auc_mean, sigma_auc)
    probs = np.array([p_hit(d) for d in DOSE_LEVELS])
    optimal_arm = int(np.argmax(probs))
    return probs, optimal_arm

def optimal_dose_ode(cl_est, noise_frac=0.22):
    """ODE recommendation: best dose under noisy clearance estimate."""
    cl_noisy = cl_est * np.exp(rng.normal(0, noise_frac))
    auc_est = np.array([d / cl_noisy * 25.0 for d in DOSE_LEVELS])
    # score: negative squared distance from target midpoint (25 mg·h/L)
    score = -(auc_est - 25.0)**2
    return int(np.argmax(score))

def run_trial(N_cycles, patient_probs, algorithm, pihat=None, rmech=1.9):
    """
    Simulate N_cycles of adaptive dosing.
    algorithm: 'bsa' | 'uninformed_ts' | 'ts_hyb'
    Returns cumulative regret sequence.
    """
    K = len(patient_probs)
    p_star = np.max(patient_probs)

    if algorithm == 'bsa':
        total_regret = 0.0
        regrets = []
        for _ in range(N_cycles):
            reward = rng.binomial(1, patient_probs[BSA_DOSE_IDX])
            total_regret += p_star - patient_probs[BSA_DOSE_IDX]
            regrets.append(total_regret)
        return np.array(regrets)

    # Beta-Bernoulli Thompson Sampling
    if algorithm == 'uninformed_ts':
        alpha = np.ones(K)
        beta  = np.ones(K)
    elif algorithm == 'ts_hyb':
        # Hybrid prior: exponential tilt toward pihat
        Z = np.exp(rmech) + K - 1
        w = np.array([np.exp(rmech)/Z if k == pihat else 1.0/Z for k in range(K)])
        # Map to Beta(a,b) with matching mean and approximately correct concentration
        # Use pseudo-count κ = 10 (clinically realistic prior strength)
        kappa = 10.0
        alpha = 1.0 + kappa * w
        beta  = 1.0 + kappa * (1.0 - w)

    total_regret = 0.0
    regrets = []
    for _ in range(N_cycles):
        samples = rng.beta(alpha, beta)
        arm = int(np.argmax(samples))
        reward = rng.binomial(1, patient_probs[arm])
        alpha[arm] += reward
        beta[arm]  += 1 - reward
        total_regret += p_star - patient_probs[arm]
        regrets.append(total_regret)
    return np.array(regrets)

# ── Main simulation ─────────────────────────────────────────────────────
N_PATIENTS = 5000
N_CYCLES   = 50
RMECH      = 1.9

results = {a: np.zeros((N_PATIENTS, N_CYCLES))
           for a in ['bsa', 'uninformed_ts', 'ts_hyb']}
target_achieved_by_cycle = {a: np.zeros((N_PATIENTS, N_CYCLES))
                             for a in ['uninformed_ts', 'ts_hyb']}
first_target_cycle = {a: np.full(N_PATIENTS, N_CYCLES+1, dtype=float)
                      for a in ['uninformed_ts', 'ts_hyb']}

for i in range(N_PATIENTS):
    probs, opt_arm = simulate_patient()
    # ODE recommends based on a noisy CL estimate (population mean = median)
    pihat = optimal_dose_ode(CL_MEDIAN)

    for alg in ['bsa', 'uninformed_ts', 'ts_hyb']:
        kw = {'pihat': pihat, 'rmech': RMECH} if alg == 'ts_hyb' else {}
        reg = run_trial(N_CYCLES, probs, alg, **kw)
        results[alg][i] = reg

    # Track first cycle at target dose
    for alg in ['uninformed_ts', 'ts_hyb']:
        kw = {'pihat': pihat, 'rmech': RMECH} if alg == 'ts_hyb' else {}
        # Re-run tracking arm selections
        if alg == 'uninformed_ts':
            alpha = np.ones(K); bta = np.ones(K)
        else:
            Z = np.exp(RMECH)+K-1
            w = np.array([np.exp(RMECH)/Z if k==pihat else 1.0/Z for k in range(K)])
            alpha = 1.0+10.0*w; bta = 1.0+10.0*(1.0-w)
        for cyc in range(N_CYCLES):
            arm = int(np.argmax(rng.beta(alpha, bta)))
            reward = rng.binomial(1, probs[arm])
            alpha[arm]+=reward; bta[arm]+=1-reward
            if probs[arm] >= 0.70 and first_target_cycle[alg][i] == N_CYCLES+1:
                first_target_cycle[alg][i] = cyc+1

# ── Summarise ────────────────────────────────────────────────────────────
summary = {}
for alg in ['bsa', 'uninformed_ts', 'ts_hyb']:
    reg = results[alg]
    summary[alg] = {
        'mean_regret_by_cycle': reg.mean(axis=0).tolist(),
        'se_regret_by_cycle':  (reg.std(axis=0)/np.sqrt(N_PATIENTS)).tolist(),
    }

for alg in ['uninformed_ts', 'ts_hyb']:
    fc = first_target_cycle[alg]
    fc_valid = fc[fc <= N_CYCLES]
    summary[alg]['pct_reached_target'] = float(len(fc_valid)/N_PATIENTS)
    summary[alg]['median_cycles_to_target'] = float(np.median(fc_valid)) if len(fc_valid)>0 else float('nan')
    summary[alg]['p25_cycles'] = float(np.percentile(fc_valid,25)) if len(fc_valid)>0 else float('nan')
    summary[alg]['p75_cycles'] = float(np.percentile(fc_valid,75)) if len(fc_valid)>0 else float('nan')

# BSA target attainment
bsa_auc_hits = []
for i in range(N_PATIENTS):
    probs, _ = simulate_patient()
    bsa_auc_hits.append(probs[BSA_DOSE_IDX] >= 0.70)
summary['bsa']['pct_at_target'] = float(np.mean(bsa_auc_hits))

with open('/home/claude/clinical_sim_results.json', 'w') as f:
    json.dump(summary, f)

print("Simulation complete.")
print(f"BSA target attainment: {summary['bsa']['pct_at_target']:.1%}")
for alg in ['uninformed_ts','ts_hyb']:
    print(f"{alg}: {summary[alg]['pct_reached_target']:.1%} reach target, "
          f"median {summary[alg]['median_cycles_to_target']:.1f} cycles "
          f"(IQR {summary[alg]['p25_cycles']:.0f}–{summary[alg]['p75_cycles']:.0f})")
