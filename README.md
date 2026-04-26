# Mechanistic Information Audit — Code Repository

Simulation code for:

> Mannor S. (2026). **Mechanistic information: a pre-trial quality criterion
> for adaptive precision medicine.** *Nature Medicine* (submitted).

---

## What this repository contains

| Directory | Contents |
|---|---|
| `rmech/` | Core engine: R_mech Monte Carlo, entropy utilities, finite-N validation |
| `settings/` | One module per audited setting (S1–S9); `run_all.py` reproduces Table 1 |
| `sensitivity/` | Sensitivity analyses: 5-FU multi-model, BRAF arm proportions, finite-N sweep |
| `figures/` | Figure generation (`gen_figures.py` → reproduces Figs 1–5) |
| `simulation/` | 5-FU clinical simulation (TS_hyb vs Gamelin rule) and bootstrap CIs |
| `data/` | Data provenance: all parameters sourced from published literature |

---

## Quick start

```bash
pip install -r requirements.txt

# Reproduce Table 1 (all nine R_mech values)
python settings/run_all.py

# Reproduce all five figures
python figures/gen_figures.py

# Run sensitivity analyses
python sensitivity/s5fu_multimodel.py
python sensitivity/braf_arms.py
python sensitivity/finite_n_sweep.py
```

---

## Key result

Eight of nine precision medicine settings fall below the **1.0 nat** threshold
for mechanistic information. The one positive case (busulfan HSCT conditioning)
clears the threshold only when the pharmacogenomic prior (R_mech = 0.72 nats)
is combined with mandatory therapeutic drug monitoring (combined ≈ 1.3 nats).

Predicted improvement factor: **1.69×** — observed: **1.77×**.

---

## R_mech: definition and computation

```python
from rmech.core import rmech_monte_carlo, entropy, convergence_factor

# Define your setting
def sample_pistar(n):
    """Draw true optimal arms from population prior."""
    return np.random.choice(K, n, p=mu)

def sample_pihat(pistar):
    """Mechanistic model recommendation (may use pistar as a proxy for
    the true patient features that drive optimal treatment)."""
    correct = np.random.rand(len(pistar)) < accuracy
    return np.where(correct, pistar, np.random.randint(K, size=len(pistar)))

result = rmech_monte_carlo(sample_pistar, sample_pihat, n_patients=200_000)
print(f"R_mech = {result['rmech']:.3f} nats")
print(f"Threshold passed: {result['threshold_passed']}")
```

### Sample size inflation for sub-threshold programmes

```python
from rmech.core import sample_size_inflation

# How many more patient-cycles are required relative to a threshold programme?
factor = sample_size_inflation(H_mu=1.62, rmech_actual=0.32)
# → 2.1× for 5-FU FOLFOX
# This is the convergence-time ratio; sample size for a specific power
# target requires additional trial design assumptions.
```

---

## Finite-N validation

The 1.0 nat threshold is validated at clinically relevant finite N:

```
R_mech    N=8    N=10    N=12   Reduction% (N=10)
  0.00   3.61    3.84    3.99             0%
  0.50   3.31    3.60    3.86             7%
  1.00   2.72    3.02    3.25            22%   ← threshold
  1.30   2.38    2.73    2.87            29%
  1.62   2.14    2.39    2.57            38%
```

---

## Failure mode taxonomy

| Mode | Cause | Settings |
|---|---|---|
| 1 | Near-binary decision (H(µ) ≤ 1.0 nat) | Pembrolizumab continuation, Lithium |
| 2 | ODE too imprecise | 5-FU, Pembrolizumab selection, Valproate, Insulin |
| 3 | One biomarker resolves most patients | BRAF melanoma |
| 4 | Arm set too narrow for pharmacogenomic distribution | Nortriptyline |

---

## Citation

```bibtex
@article{mannor2026rmech,
  author  = {Mannor, Shie},
  title   = {Mechanistic information: a pre-trial quality criterion
             for adaptive precision medicine},
  journal = {Nature Medicine},
  year    = {2026},
  note    = {Submitted}
}
```

Companion theory paper (regret bounds):
```bibtex
@article{mannor2026bounds,
  author  = {Mannor, Shie},
  title   = {Mechanistic Information and Regret: Lower and Upper Bounds
             for Hybrid Model Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2026},
  note    = {Submitted. arXiv:[TBD]}
}
```

---

## Licence

MIT. If you use this code, please cite the paper above.

## Contact

Shie Mannor — shie@technion.ac.il  
Technion – Israel Institute of Technology
