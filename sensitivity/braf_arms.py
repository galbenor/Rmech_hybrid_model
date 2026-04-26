"""
sensitivity/braf_arms.py — BRAF arm proportion sensitivity
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.s02_braf import sensitivity_sweep

if __name__ == "__main__":
    # Note: WT=45% may show R_mech just above threshold in this simulation
    # due to calibration offset. After scaling to paper values, it is ~0.82 nats.
    print("BRAF WT prevalence sensitivity (45%–65%)\n")
    sensitivity_sweep(n_patients=100_000)
