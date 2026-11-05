"""
conv_res_t800.py — 5주차 체크 2: 그레이팅 A 의 해상도 수렴 (시간 상한 800 고정)

배경 (2026-11-05 결과, 해상도 200 기준):
  - t_cap 200 → 29.05%, 400 → 17.15%, 800 → 19.74%, 1600 → 19.55%
  - 800 과 1600 이 거의 같으므로 시간 상한은 800 으로 고정한다.
  - 해상도 수렴은 아직 확인되지 않았다 (이전 표는 t_cap 200/400 기준이라 무효).

이 스크립트가 하는 일:
  1) 그레이팅 (Λ=0.8, f=0.6, h=0.5, 940 nm, t=2 μm) 을 t_cap 800 에서 해상도 100/200/300/400 으로 계산
  2) 해상도 200 에서 PML 을 2 μm 로 바꾼 PML 2배 테스트
  3) 평면 Si (기준) 를 해상도 200, t_cap 800 에서 계산 (TMM 2.527% 와 비교)
  4) 각 실행에서 "시간 상한 시점에 구조 안에 남은 전자기 에너지 비율" 을 기록한다.
     비율 = (t_cap 시점 에너지) / (실행 중 최대 에너지).
     이 비율이 작으면 1−R−T 가 빠져나가지 않은 에너지가 아니라 흡수 손실이라고 볼 근거가 된다.

결과: results/week5_conv_res_t800.csv (git 추적 안 함)
재시작: 이미 CSV 에 있는 조건은 건너뛴다. 중간에 멈춰도 같은 명령으로 이어서 돈다.
실행 (WSL2, photontrap-meep, 총 수 시간 예상 — 해상도 400 이 대부분을 차지):
  python -u scripts/conv_res_t800.py 2>&1 | tee /mnt/c/projects/conv_res_t800_log.txt

※ 이 파일은 Claude 작업 환경에 MEEP 이 없어 실행하지 못했다. 사용자 WSL 에서 처음 돌릴 때는
  해상도 100 한 점만 먼저 확인하는 것을 권한다 (RESES 를 [100] 으로 줄여서 실행).
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import meep as mp
from si_data import SiOpticalConstants
from meep_1d import medium_from_N
from meep_2d import LamellarGrating2D

ROOT = os.path.join(os.path.dirname(__file__), "..")
LAM = 940.0
P, F, H, T_UM = 0.8, 0.6, 0.5, 2.0
CAP = 800.0
RESES = [100, 200, 300, 400]
RECORD_DT = 5.0          # 잔류 에너지 기록 간격 (MEEP 시간 단위)
TMM_FLAT_A = 0.02527     # 평면 2 μm, 940 nm (진행 노트 4절)
OUT = os.path.join(ROOT, "results", "week5_conv_res_t800.csv")
FIELDS = ["label", "resolution", "t_cap", "dpml_um", "R", "T", "A",
          "residual_ratio", "runtime_s"]


def cell_energy(sim, g):
    """구조 + 패드 영역 안의 전자기 에너지 (ε·|E|² 합, 상대값). PML 은 제외."""
    half_y = g.t / 2 + g.pad
    center = mp.Vector3(0, 0, 0)
    size = mp.Vector3(g.P, 2 * half_y, 0)
    eps = sim.get_array(center=center, size=size, component=mp.Dielectric)
    w = 0.0
    for comp in (mp.Ex, mp.Ey, mp.Ez):
        a = sim.get_array(center=center, size=size, component=comp)
        w += np.sum(eps * np.abs(a) ** 2)
    return float(w)


def run_case(g, mat):
    """g.rta() 와 같은 절차 + 시간에 따른 잔류 에너지 기록."""
    inc, refl_data = g.normalize()
    sim, refl, tran = g._build(mat)
    sim.load_minus_flux_data(refl, refl_data)
    series = []

    def record(s):
        series.append(cell_energy(s, g))

    sim.run(mp.at_every(RECORD_DT, record), until_after_sources=g.t_cap)
    R = float(-np.array(mp.get_fluxes(refl))[0] / inc[0])
    T = float(np.array(mp.get_fluxes(tran))[0] / inc[0])
    sim.reset_meep()
    series = np.array(series)
    ratio = float(series[-1] / series.max()) if len(series) and series.max() > 0 else float("nan")
    return R, T, 1.0 - R - T, ratio


def load_done():
    if not os.path.exists(OUT):
        return {}, []
    with open(OUT, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    done = {(r["label"], int(r["resolution"]), float(r["dpml_um"])): r for r in rows}
    return done, rows


def save(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    si = SiOpticalConstants(verbose=False)
    mat = medium_from_N(si.N(LAM), LAM / 1000.0)
    freqs = np.array([1000.0 / LAM])

    # (라벨, 해상도, 구조 f, 구조 h, PML 두께)
    cases = [("grating", r, F, H, 1.0) for r in RESES]
    cases.append(("grating_pml2", 200, F, H, 2.0))
    cases.append(("flat", 200, 1.0, 0.0, 1.0))

    done, rows = load_done()
    t_all = time.time()
    print(f"그레이팅 Λ={P}, f={F}, h={H}, 940 nm, t_cap={CAP:.0f}")
    print(f"평면 기준 TMM A = {100*TMM_FLAT_A:.3f}%", flush=True)

    for label, res, f, h, dpml in cases:
        key = (label, res, dpml)
        if key in done:
            print(f"[건너뜀] {label} res {res} dpml {dpml}: 이미 계산됨", flush=True)
            continue
        t0 = time.time()
        g = LamellarGrating2D(P, f, h, T_UM, freqs, res, dpml_um=dpml, t_cap=CAP)
        R, T, A, ratio = run_case(g, mat)
        dt = time.time() - t0
        row = {"label": label, "resolution": res, "t_cap": CAP, "dpml_um": dpml,
               "R": R, "T": T, "A": A, "residual_ratio": ratio, "runtime_s": round(dt, 1)}
        rows.append(row)
        done[key] = row
        extra = f"  (TMM 대비 {100*(A-TMM_FLAT_A):+.3f}%p)" if label == "flat" else ""
        print(f"[{label:12s}] res {res:4d} dpml {dpml:.0f}:  R {100*R:.3f}%  T {100*T:.3f}%  "
              f"A {100*A:.4f}%  잔류에너지 비율 {ratio:.2e}  ({dt:.0f} s){extra}", flush=True)
        save(rows)

    print(f"\n총 실행 시간 {time.time()-t_all:.0f} s")
    print(f"저장: {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
