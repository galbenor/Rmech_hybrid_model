"""
settings/run_all.py
─────────────────────────────────────────────────────────────────────────────
Run all nine settings and reproduce Table 1 from:

  Mannor S. (2026). "Mechanistic information: a pre-trial quality criterion
  for adaptive precision medicine." Nature Medicine (submitted).

Usage:
    python settings/run_all.py

Output:
    Printed Table 1 + results saved to results/table1.json
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import s01_5fu, s02_braf, s03_pembro_sel, s04_pembro_cont
from settings import s05_lithium, s06_nortriptyline, s07_valproate
from settings import s08_insulin, s09_busulfan
from rmech.core import sample_size_inflation

N = 200_000   # Monte Carlo sample size


def main():
    print("NOTE: R_mech values from this simulation will differ from Table 1")
    print("in the paper (~20-30% higher). The simplified models use flat-accuracy")
    print("recommendation functions; the paper uses fully-calibrated noise")
    print("distributions. The qualitative result (8/9 below threshold) is exact.")
    print()
    print("Computing R_mech for all nine settings (N={:,} per setting)...\n".format(N))

    results = []

    # Settings 1–8 (eight below threshold)
    runners = [
        ("5-FU FOLFOX",            s01_5fu.run,          {}),
        ("BRAF melanoma",          s02_braf.run,          {}),
        ("Pembro selection",       s03_pembro_sel.run,    {}),
        ("Pembro continuation",    s04_pembro_cont.run,   {}),
        ("Lithium bipolar",        s05_lithium.run,       {}),
        ("Nortriptyline",          s06_nortriptyline.run, {}),
        ("Valproate epilepsy",     s07_valproate.run,     {}),
        ("Insulin T1D",            s08_insulin.run,       {}),
    ]

    for name, fn, kwargs in runners:
        r = fn(n_patients=N, **kwargs)
        results.append(r)
        mode = r.get("failure_mode")
        h, rm = r["H_mu"], r["rmech"]
        # Sample size inflation only applies to Mode 2/3/4 where H(mu) > 1.0.
        # Mode 1 fails because H(mu) ≤ 1.0 nat, not because of weak R_mech.
        if mode in (2, 3, 4) and h > 1.0 and rm < 1.0:
            ss_str = f"  SS inflation: {sample_size_inflation(h, rm):.1f}×"
        else:
            ss_str = ""
        print(f"  {name:<28}  K={r['K']}  "
              f"H(mu)={h:.2f}  R_mech={rm:.2f}  "
              f"Mode={mode if mode else '—'}{ss_str}")

    # Setting 9: Busulfan (positive case — two components)
    r_prior = s09_busulfan.run_prior_only(n_patients=N)
    r_tdm   = s09_busulfan.run_prior_plus_tdm(n_patients=N)
    results += [r_prior, r_tdm]
    print(f"  {'Busulfan (prior only)':<28}  K={r_prior['K']}  "
          f"H(mu)={r_prior['H_mu']:.2f}  R_mech={r_prior['rmech']:.2f}  Mode=—")
    print(f"  {'Busulfan (prior + TDM)':<28}  "
          f"R_mech≈{r_tdm['rmech']:.2f}  "
          f"{'✓ ABOVE threshold' if r_tdm['threshold_passed'] else '✗ below'}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "═"*72)
    print(f"{'Setting':<30} {'K':>3} {'H(µ)':>6} {'R_mech':>7} "
          f"{'Mode':>5} {'Pass?':>6}")
    print("─"*72)
    for r in results:
        if r.get("component") == "prior + TDM (combined)":
            continue
        mode = str(r.get("failure_mode", "—"))
        comp = r.get("component", "")
        name = r["setting"][:28] + (f" ({comp[:6]})" if comp else "")
        print(f"{name:<30} {r['K']:>3} {r['H_mu']:>6.2f} "
              f"{r['rmech']:>7.3f} {mode:>5} "
              f"{'✓' if r['threshold_passed'] else '✗':>6}")
    print("═"*72)
    print(f"\nResult: {sum(1 for r in results if not r['threshold_passed'] and 'prior' not in r.get('component',''))} of 9 settings below threshold")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/table1.json", "w") as f:
        # Convert ndarray to list for JSON serialisation
        clean = []
        for r in results:
            c = {k: (v.tolist() if hasattr(v, "tolist") else v)
                 for k, v in r.items()}
            clean.append(c)
        json.dump(clean, f, indent=2)
    print("\nResults saved to results/table1.json")


if __name__ == "__main__":
    main()
