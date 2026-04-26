"""
Bootstrap 95% CIs for R_mech across all nine settings.
Uses the Monte Carlo estimate approach: resample virtual patients,
recompute R_mech, report 2.5th–97.5th percentile over B=500 replicates.
"""
import numpy as np
from scipy.stats import norm
np.random.seed(2024)

N_SIM  = 80_000   # primary pool size
N_BOOT = 500      # bootstrap replicates

def entropy(counts, total):
    p = counts[counts>0] / total
    return -np.sum(p * np.log(p))

def rmech_from_arrays(pistar, pihat, K):
    N = len(pistar)
    mu = np.bincount(pistar, minlength=K) / N
    H_mu = entropy(mu, 1.0)      # already probabilities
    H_cond = 0.0
    for j in range(K):
        m = pihat == j; pj = m.mean()
        if pj < 1e-9: continue
        c = np.bincount(pistar[m], minlength=K) / m.sum()
        H_cond += pj * entropy(c, 1.0)
    return H_mu, H_mu - H_cond

def bootstrap_ci(pistar, pihat, K, B=N_BOOT):
    N = len(pistar)
    samples = []
    for _ in range(B):
        idx = np.random.randint(0, N, N)
        _, rm = rmech_from_arrays(pistar[idx], pihat[idx], K)
        samples.append(rm)
    return np.percentile(samples, [2.5, 97.5])

def best_arm_vec(auc_mat, win_lo, win_hi, sigma_noise):
    # auc_mat: N x K
    log_auc = np.log(np.clip(auc_mat, 1e-9, None))
    p = (norm.cdf((np.log(win_hi) - log_auc) / sigma_noise) -
         norm.cdf((np.log(win_lo) - log_auc) / sigma_noise))
    return p.argmax(axis=1)

results = {}

# ── Setting 1: 5-FU ───────────────────────────────────────────────────────
sigma_CL, R2, sigma_AUC = 0.75, 0.51, 0.08
doses = np.array([1600, 1800, 2000, 2200, 2400, 2600, 2800, 3200])
CL_pop, win_lo, win_hi = 96.0, 20.0, 30.0
log_CL = np.random.normal(np.log(CL_pop), sigma_CL, N_SIM)
CL = np.exp(log_CL)
auc_true = doses[None,:] / CL[:,None]
pistar = best_arm_vec(auc_true, win_lo, win_hi, sigma_AUC)
sig_res = np.sqrt(1-R2)*sigma_CL
auc_hat = doses[None,:] / np.exp(np.random.normal(log_CL, sig_res))[:,None]
pihat = best_arm_vec(auc_hat, win_lo, win_hi, sigma_AUC)
H_mu, rm = rmech_from_arrays(pistar, pihat, len(doses))
ci = bootstrap_ci(pistar, pihat, len(doses))
results['5-FU'] = (H_mu, rm, ci)
print(f"5-FU:     H(µ)={H_mu:.2f}  Rmech={rm:.3f}  95%CI=[{ci[0]:.3f},{ci[1]:.3f}]")

# ── Setting 2: BRAF ───────────────────────────────────────────────────────
# Prior: (0.25, 0.12, 0.08, 0.55); LDH as mechanistic signal
# LDH distributions by arm: arm1~LN(180,55²), arm2~LN(520,130²), arm3~LN(310,90²), arm4~LN(220,55²)
# ODE boundary: LDH>350 → arm2; BRAF-: arm4; else arm1/3
prior = np.array([0.25, 0.12, 0.08, 0.55])
K = 4
pistar_b = np.random.choice(K, N_SIM, p=prior)
# Mechanistic signal: LDH sampled from conditional distributions
LDH_params = {0:(180,55), 1:(520,130), 2:(310,90), 3:(220,55)}
LDH = np.zeros(N_SIM)
for k, (mu_ldh, s_ldh) in LDH_params.items():
    idx = pistar_b == k
    LDH[idx] = np.random.lognormal(np.log(mu_ldh), s_ldh/mu_ldh, idx.sum())
# BRAF test: arm4 correctly identified with 98% sensitivity
braf_neg = pistar_b == 3
braf_detected = braf_neg & (np.random.rand(N_SIM) < 0.98)
# LDH boundary
high_ldh = LDH > 350
# Recommendations
pihat_b = np.zeros(N_SIM, dtype=int)
pihat_b[braf_detected]                    = 3  # correctly identified wild-type
pihat_b[~braf_detected & high_ldh]        = 1  # high LDH → targeted
pihat_b[~braf_detected & ~high_ldh]       = 0  # normal LDH → ICI-first
# Some noise: ~13% uncertainty on non-arm4
flip = (np.random.rand(N_SIM) < 0.13) & ~braf_detected
pihat_b[flip] = np.random.choice([0,1,2], flip.sum())
H_mu, rm = rmech_from_arrays(pistar_b, pihat_b, K)
ci = bootstrap_ci(pistar_b, pihat_b, K)
results['BRAF'] = (H_mu, rm, ci)
print(f"BRAF:     H(µ)={H_mu:.2f}  Rmech={rm:.3f}  95%CI=[{ci[0]:.3f},{ci[1]:.3f}]")

# ── Settings 3-4: Pembrolizumab ────────────────────────────────────────────
# S3: arm selection, prior (0.30,0.40,0.20,0.10), TPS R²=0.25
prior3 = np.array([0.30, 0.40, 0.20, 0.10])
K3 = 4
pistar3 = np.random.choice(K3, N_SIM, p=prior3)
# TPS mediocre predictor
tps_means = {0:30, 1:50, 2:70, 3:15}  # TPS by optimal arm
tps = np.zeros(N_SIM)
for k, mu_t in tps_means.items():
    idx = pistar3==k
    tps[idx] = np.clip(np.random.normal(mu_t, 25, idx.sum()), 0, 100)
# Recommendation: simple TPS threshold rule (noisy)
pihat3 = np.zeros(N_SIM, dtype=int)
pihat3[tps < 25]  = 3   # chemo alone
pihat3[(tps >= 25) & (tps < 50)] = 0   # mono
pihat3[(tps >= 50) & (tps < 75)] = 1   # combo
pihat3[tps >= 75] = 2   # dual
# Add noise
noise_mask = np.random.rand(N_SIM) < 0.55
pihat3[noise_mask] = np.random.choice(K3, noise_mask.sum(), p=prior3)
H_mu3, rm3 = rmech_from_arrays(pistar3, pihat3, K3)
ci3 = bootstrap_ci(pistar3, pihat3, K3)
results['Pembro (selection)'] = (H_mu3, rm3, ci3)
print(f"Pembro sel: H(µ)={H_mu3:.2f}  Rmech={rm3:.3f}  95%CI=[{ci3[0]:.3f},{ci3[1]:.3f}]")

# S4: continuation, K=2, p=0.35
prior4 = np.array([0.35, 0.65])
pistar4 = np.random.choice(2, N_SIM, p=prior4)
# CL ratio as signal (noisy)
cl_ratio = np.where(pistar4==0, 
    np.random.normal(0.75, 0.15, N_SIM),
    np.random.normal(1.10, 0.20, N_SIM))
pihat4 = (cl_ratio < 0.92).astype(int)   # threshold decision rule
H_mu4, rm4 = rmech_from_arrays(pistar4, pihat4, 2)
ci4 = bootstrap_ci(pistar4, pihat4, 2)
results['Pembro (cont.)'] = (H_mu4, rm4, ci4)
print(f"Pembro cont: H(µ)={H_mu4:.2f}  Rmech={rm4:.3f}  95%CI=[{ci4[0]:.3f},{ci4[1]:.3f}]")

# ── Setting 5: Lithium ─────────────────────────────────────────────────────
sigma_CL_li, R2_li = 0.33, 0.848
CL_pop_li = 0.909  # L/h
doses_li = np.array([600, 900, 1200, 1500, 1800]) / 24  # mg/h
win_li = (0.4, 0.8)  # mmol/L
V_li, ke_li_base = 42.0, 0.909/42.0
log_CL_li = np.random.normal(np.log(CL_pop_li), sigma_CL_li, N_SIM)
CL_li = np.exp(log_CL_li)
def trough_li(D, CL):
    ke = CL / V_li
    return (D/V_li) * np.exp(-ke*12) / (1 - np.exp(-ke*24))
auc_li = np.stack([trough_li(d, CL_li) for d in doses_li], axis=1)
pistar_li = best_arm_vec(auc_li, win_li[0], win_li[1], 0.05)
sig_res_li = np.sqrt(1-R2_li)*sigma_CL_li
CL_li_hat = np.exp(np.random.normal(log_CL_li, sig_res_li))
auc_li_hat = np.stack([trough_li(d, CL_li_hat) for d in doses_li], axis=1)
pihat_li = best_arm_vec(auc_li_hat, win_li[0], win_li[1], 0.05)
H_mu_li, rm_li = rmech_from_arrays(pistar_li, pihat_li, 5)
ci_li = bootstrap_ci(pistar_li, pihat_li, 5)
results['Lithium'] = (H_mu_li, rm_li, ci_li)
print(f"Lithium:   H(µ)={H_mu_li:.2f}  Rmech={rm_li:.3f}  95%CI=[{ci_li[0]:.3f},{ci_li[1]:.3f}]")

# ── Setting 6: Nortriptyline ───────────────────────────────────────────────
# CYP2D6 mixture: (PM=7%, IM=15%, NM=65%, UM=13%)
# CL/F by group: 6, 18, 35, 100 L/h
cyp_probs = np.array([0.07, 0.15, 0.65, 0.13])
cyp_cl    = np.array([6.0, 18.0, 35.0, 100.0])
cyp_group = np.random.choice(4, N_SIM, p=cyp_probs)
sigma_within = 0.25
log_CL_nort = np.log(cyp_cl[cyp_group]) + np.random.normal(0, sigma_within, N_SIM)
CL_nort = np.exp(log_CL_nort)
doses_nort = np.array([10, 25, 50, 75, 100, 150])
win_nort = (50, 150)  # ng/mL
F, V_nort = 0.50, 1300.0
def trough_nort(D, CL):
    kf = CL*F/V_nort
    return (D*F/V_nort) * np.exp(-kf*24) / (1 - np.exp(-kf*24)) * 1000
auc_nort = np.stack([trough_nort(d, CL_nort) for d in doses_nort], axis=1)
pistar_nort = best_arm_vec(auc_nort, win_nort[0], win_nort[1], 0.10)
sigma_CL_nort = np.sqrt(np.log(1+(np.std(np.log(cyp_cl[cyp_group]+1e-3))**2)))
R2_nort = 1 - (sigma_within/(0.70))**2
sig_res_nort = np.sqrt(1-R2_nort)*0.70
CL_nort_hat = np.exp(log_CL_nort + np.random.normal(0, sig_res_nort, N_SIM))
auc_nort_hat = np.stack([trough_nort(d, CL_nort_hat) for d in doses_nort], axis=1)
pihat_nort = best_arm_vec(auc_nort_hat, win_nort[0], win_nort[1], 0.10)
H_mu_nort, rm_nort = rmech_from_arrays(pistar_nort, pihat_nort, 6)
ci_nort = bootstrap_ci(pistar_nort, pihat_nort, 6)
results['Nortriptyline'] = (H_mu_nort, rm_nort, ci_nort)
print(f"Nortrip:   H(µ)={H_mu_nort:.2f}  Rmech={rm_nort:.3f}  95%CI=[{ci_nort[0]:.3f},{ci_nort[1]:.3f}]")

# ── Setting 7: Valproate ───────────────────────────────────────────────────
sigma_CL_v, R2_v = 0.33, 0.73
CL_pop_v = 0.464
doses_v  = np.array([250, 375, 500, 625, 750, 1000]) * 2  # mg/day (twice daily -> daily total)
win_v = (50, 100)
V_v = 23.3
log_CL_v = np.random.normal(np.log(CL_pop_v), sigma_CL_v, N_SIM)
CL_v = np.exp(log_CL_v)
def trough_v(D, CL):
    ke = CL/V_v
    d_dose = D/2  # per-dose
    return (d_dose/V_v) * np.exp(-ke*12) / (1-np.exp(-ke*12))
auc_v = np.stack([trough_v(d, CL_v) for d in doses_v], axis=1)
pistar_v = best_arm_vec(auc_v, win_v[0], win_v[1], 0.10)
sig_res_v = np.sqrt(1-R2_v)*sigma_CL_v
CL_v_hat = np.exp(np.random.normal(log_CL_v, sig_res_v))
auc_v_hat = np.stack([trough_v(d, CL_v_hat) for d in doses_v], axis=1)
pihat_v = best_arm_vec(auc_v_hat, win_v[0], win_v[1], 0.10)
H_mu_v, rm_v = rmech_from_arrays(pistar_v, pihat_v, 6)
ci_v = bootstrap_ci(pistar_v, pihat_v, 6)
results['Valproate'] = (H_mu_v, rm_v, ci_v)
print(f"Valproate: H(µ)={H_mu_v:.2f}  Rmech={rm_v:.3f}  95%CI=[{ci_v[0]:.3f},{ci_v[1]:.3f}]")

# ── Setting 8: Insulin T1D ─────────────────────────────────────────────────
sigma_ISI, R2_ins = 0.55, 0.65
doses_ins = np.array([0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00, 1.40])
ISI_pop = 1.0
log_ISI = np.random.normal(np.log(ISI_pop), sigma_ISI, N_SIM)
ISI = np.exp(log_ISI)
TDD_opt = 0.55 / ISI
sigma_reward = 0.18
rewards = np.exp(-0.5*((doses_ins[None,:] - TDD_opt[:,None])/sigma_reward)**2)
pistar_ins = rewards.argmax(axis=1)
sig_res_ins = np.sqrt(1-R2_ins)*sigma_ISI
ISI_hat = np.exp(np.random.normal(log_ISI, sig_res_ins))
TDD_hat = 0.55 / ISI_hat
rewards_hat = np.exp(-0.5*((doses_ins[None,:] - TDD_hat[:,None])/sigma_reward)**2)
pihat_ins = rewards_hat.argmax(axis=1)
H_mu_ins, rm_ins = rmech_from_arrays(pistar_ins, pihat_ins, 8)
ci_ins = bootstrap_ci(pistar_ins, pihat_ins, 8)
results['Insulin T1D'] = (H_mu_ins, rm_ins, ci_ins)
print(f"Insulin:   H(µ)={H_mu_ins:.2f}  Rmech={rm_ins:.3f}  95%CI=[{ci_ins[0]:.3f},{ci_ins[1]:.3f}]")

print("\n=== SUMMARY FOR TABLE ===")
for name, (hmu, rm, ci) in results.items():
    print(f"{name:22} H(µ)={hmu:.2f}  Rmech={rm:.2f} [{ci[0]:.2f},{ci[1]:.2f}]")
