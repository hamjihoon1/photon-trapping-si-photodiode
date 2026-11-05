"""
conv_940_diag.py — 5주차 체크 2 실패 원인 찾기 (940 nm 한 점)

질문 두 가지를 분리한다:
  (a) 해상도 문제인가?   → 해상도 100/200/300/400 에서 결과가 수렴하는가
  (b) 시간 문제인가?     → 시간 상한 t_cap 200 vs 400 에서 결과가 같은가
                            (공진이 강하면 t=200 안에 빛이 다 빠져나가지 못해 플럭스가 잘릴 수 있다)

각 조건마다 그레이팅(Λ=0.8, f=0.6, h=0.5)과 평면 Si(f=1, h=0)를 같이 돌린다.
평면 Si 는 TMM 값(940 nm, 2 μm, 2.527%)과 비교해서 계산 자체가 맞는지 확인하는 기준이다.

결과: results/week5_conv940_diag.csv (git 추적 안 함 — 진단용)
실행 (WSL2, photontrap-meep). 시간이 오래 걸리므로 로그를 파일로 남긴다:
  python -u scripts/conv_940_diag.py 2>&1 | tee /mnt/c/projects/conv_log.txt
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
RESES = [100, 200, 300, 400]
CAPS = [200.0, 400.0]

# TMM 기준 (평면 2 μm, 940 nm): 진행 노트 4절
TMM_FLAT_A = 0.02527


def main():
    si = SiOpticalConstants(verbose=False)
    mat = medium_from_N(si.N(LAM), LAM / 1000.0)
    freqs = np.array([1000.0 / LAM])
    out_path = os.path.join(ROOT, "results", "week5_conv940_diag.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    t_all = time.time()

    print(f"TMM 평면 기준 A = {100*TMM_FLAT_A:.3f}%")
    for res in RESES:
        for cap in CAPS:
            for label, (f, h) in (("grating", (F, H)), ("flat", (1.0, 0.0))):
                t0 = time.time()
                g = LamellarGrating2D(P, f, h, T_UM, freqs, res, t_cap=cap)
                R, Tt, A = g.rta(mat)
                R, Tt, A = float(R[0]), float(Tt[0]), float(A[0])
                dt = time.time() - t0
                rows.append({"structure": label, "resolution": res, "t_cap": cap,
                             "R": R, "T": Tt, "A": A, "runtime_s": round(dt, 1)})
                extra = f"  (TMM 대비 {100*(A-TMM_FLAT_A):+.3f}%p)" if label == "flat" else ""
                print(f"[{label:7s}] res {res:4d}  t_cap {cap:5.0f}:  "
                      f"R {100*R:.3f}%  T {100*Tt:.3f}%  A {100*A:.4f}%  ({dt:.0f} s){extra}",
                      flush=True)
                # 중간 저장: 긴 계산 도중에 멈춰도 결과가 남도록
                with open(out_path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    print(f"\n총 실행 시간 {time.time()-t_all:.0f} s")
    print(f"저장: {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
