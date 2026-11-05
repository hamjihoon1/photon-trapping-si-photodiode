"""
conv_time_diag.py — 그레이팅 A 가 계산 시간에 따라 안정되는지 확인 (5주차 체크 2 원인 진단)

배경 (2026-11-05 결과):
  - 평면 Si: 해상도·시간 상한에 관계없이 TMM 과 일치 (해상도 400 에서 -0.014%p). → 셀 설정은 정상
  - 그레이팅 (940 nm, Λ=0.8): 시간 상한 200 과 400 에서 A 가 크게 다름 (해상도 400: 37.0% vs 25.7%)
    → 빛이 오래 갇히는 공진이 있어 t=200~400 안에 다 빠져나가지 않는 것으로 보임

이 스크립트: 해상도 200 고정, 시간 상한 200 / 400 / 800 / 1600 에서 그레이팅 A 를 본다.
  - A 가 어느 시간 이후 값이 안정되면 → 그 시간을 상한으로 쓰면 된다
  - 계속 변하면 → 시간 기반 방법으로는 한계가 있음 (RCWA 같은 주파수 영역 방법으로 교차 확인 필요)

평면 Si 는 시간 상한과 무관하므로 돌리지 않는다.

결과: results/week5_conv_time_diag.csv (git 추적 안 함)
실행 (WSL2, photontrap-meep). 전체 약 1시간 안팎 예상:
  python -u scripts/conv_time_diag.py 2>&1 | tee /mnt/c/projects/conv_time_log.txt
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from si_data import SiOpticalConstants
from meep_1d import medium_from_N
from meep_2d import LamellarGrating2D

ROOT = os.path.join(os.path.dirname(__file__), "..")
LAM = 940.0
P, F, H, T_UM = 0.8, 0.6, 0.5, 2.0
RES = 200
CAPS = [200.0, 400.0, 800.0, 1600.0]


def main():
    si = SiOpticalConstants(verbose=False)
    mat = medium_from_N(si.N(LAM), LAM / 1000.0)
    freqs = np.array([1000.0 / LAM])
    out_path = os.path.join(ROOT, "results", "week5_conv_time_diag.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    t_all = time.time()

    print(f"그레이팅 Λ={P}, f={F}, h={H}, 940 nm, 해상도 {RES}", flush=True)
    for cap in CAPS:
        t0 = time.time()
        g = LamellarGrating2D(P, F, H, T_UM, freqs, RES, t_cap=cap)
        R, Tt, A = g.rta(mat)
        R, Tt, A = float(R[0]), float(Tt[0]), float(A[0])
        dt = time.time() - t0
        rows.append({"resolution": RES, "t_cap": cap, "R": R, "T": Tt, "A": A,
                     "runtime_s": round(dt, 1)})
        print(f"t_cap {cap:6.0f}:  R {100*R:.3f}%  T {100*Tt:.3f}%  A {100*A:.4f}%  ({dt:.0f} s)",
              flush=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n총 실행 시간 {time.time()-t_all:.0f} s")
    print(f"저장: {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
