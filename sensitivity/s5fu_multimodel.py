"""
sensitivity/s5fu_multimodel.py — 5-FU multi-model R_mech sensitivity
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.s01_5fu import R2_RANGE, run

if __name__ == "__main__":
    print("5-FU R_mech across published population PK models\n")
    print(f"{'Model':<28} {'R2':>6} {'R_mech':>8} {'Status':>10}")
    print("-"*56)
    for name, R2 in R2_RANGE.items():
        r = run(R2=R2, n_patients=100_000)
        status = "above" if r["threshold_passed"] else "below"
        print(f"{name:<28} {R2:>6.2f} {r['rmech']:>8.3f} {status:>10}")
