"""
validate_sio2.py — SiO2 데이터 검증 (3주차 ARC 준비)
  (1) CSV 보간값 = Malitson 1965 Sellmeier 식 (800–1000 nm)
  (2) 940 nm 굴절률과 1/4파장 ARC 두께 출력
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from sio2_data import SiO2OpticalConstants, sellmeier_n

sio2 = SiO2OpticalConstants()
lam = np.arange(800, 1000.1, 1.0)
n_csv = np.array([sio2.N(l).real for l in lam])
err = np.abs(n_csv - sellmeier_n(lam)).max()
ok = err < 1e-5
print(f"[{'PASS' if ok else 'FAIL'}] CSV 보간 vs Sellmeier 식 (800–1000 nm)  최대 차이 {err:.1e}")
n940 = sio2.N(940).real
print(f"  n(940 nm) = {n940:.5f},  1/4파장 두께 = 940/(4n) = {940/(4*n940):.1f} nm")
sys.exit(0 if ok else 1)
