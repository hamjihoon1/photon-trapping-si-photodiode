"""
validate_nanohole_3d_flat.py — 6주차 착수 전 필수: 3D 단위셀(src/meep_3d.py) 평면 환원 검증

계획서 6주차: "얇은 Si 층(예: 2 μm)에 원기둥 나노홀 단위셀 구성" — 그 전에 3D 코드가 평면 Si 를 맞게 푸는지 확인한다.
진행 노트 6절 6번: "홀 반지름 0일 때 평면 Si와 일치해야 함"

체크 1a — 평면 환원 vs coherent TMM (해상도 100/200/400)
  홀 반경 0 이면 수직입사·주기 경계(k_point = 0)에서 필드가 x, y 로 균일하다 → 결과는 단위셀 크기 P 와 무관하다.
  그래서 계산을 가볍게 하려고 P = 4 격자 (4/res μm) 로 줄여서 돈다. 이 대체가 맞는지는 체크 1b 로 확인.
  기준: 5주차 2D 체크 1 과 같음 — 해상도↑ 시 max|ΔA| 감소, 최종 해상도에서 ≤ 0.1 %p (1e-3).
체크 1b — 단위셀 크기 무관성 (해상도 20, 940 nm)
  P = 4/res (작은 셀) 와 P = 1.0 μm (6주차 실제 후보 주기) 의 R, T, A 차이. 기준 |ΔA| < 1e-6.
체크 1c — 홀 기하 확인 (r = 0.35 μm, h = 0.5 μm, P = 1.0 μm, 해상도 40)
  MEEP 이 실제로 만든 ε 분포를 꺼내서
    - 홀은 Si 윗면(z = +t/2) 깊이 h 구간에만 있어야 하고
    - 그 구간의 공기 면적비가 π r² / P² 와 맞아야 하며 (격자 계단 근사 오차 수 % 허용)
    - 홀 아래 Si 는 꽉 차 있어야 한다.
  ※ meep_3d.py 의 "윗면" = z = +t/2 쪽. 광원은 z 음수 쪽(z_src < 0)에 있으므로 빛은 아래(−z)에서 들어와
    홀 없는 면을 먼저 만나고, 홀은 투과 쪽 면에 있다. 이 방향이 의도와 맞는지는 출력에 기록하고 노트에서 판단한다.

시간 상한: t_cap = 800 (진행 노트 3절 확정값, 2026-11-05)

실행 (WSL2 photontrap-meep):
  python scripts/validate_nanohole_3d_flat.py           # 전체: 해상도 400 × 5파장 포함, 약 1.5시간 이상 (밤에)
  python scripts/validate_nanohole_3d_flat.py --quick   # 해상도 50/100, 940 nm 한 점 (1~2분)
  python scripts/validate_nanohole_3d_flat.py --only 1a --res 400 --lams 940 --tag res400_940   # 940 nm 한 점 (Claude 환경 약 17분)
  Claude 환경 실제 시간 (2026-11-06): 1c 1 s, 1b 49 s, 1a 해상도 100/200 × 5파장 949 s, 해상도 400 × 940 nm 1001 s
출력: results/week6_3d_flat_check.csv (--quick 이면 _quick)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import meep as mp

from tmm import tmm_normal_incidence
from si_data import SiOpticalConstants
from meep_1d import medium_from_N
from meep_3d import NanoHoleCell3D

ROOT = os.path.join(os.path.dirname(__file__), "..")
T_UM = 2.0
T_CAP = 800.0
LAM_TARGET = 940.0

mp.verbosity(0)


def tmm_flat(si, lam_nm):
    return tmm_normal_incidence(lam_nm, [si.N(lam_nm)], [T_UM * 1000.0], 1.0, 1.0)


def flat_sweep(si, period, lams_nm, res, dpml=1.0):
    """홀 없는 3D 셀 (radius 0, depth 0). 파장마다 재료를 맞춰 본 런, 정규화 런은 한 번."""
    lams_nm = np.asarray(lams_nm, dtype=float)
    cell = NanoHoleCell3D(period, 0.0, 0.0, T_UM, 1000.0 / lams_nm, res, dpml_um=dpml, t_cap=T_CAP)
    out = np.zeros((len(lams_nm), 3))
    for i, l in enumerate(lams_nm):
        R, T, A = cell.rta(medium_from_N(si.N(l), l / 1000.0))
        out[i] = R[i], T[i], A[i]
    return out


def check_1a(si, res_list, lams):
    print("\n[체크 1a] 3D 평면 환원 (r = 0, P = 4 격자) vs coherent TMM, t_cap 800")
    rows = []
    for res in res_list:
        t0 = time.time()
        m = flat_sweep(si, 4.0 / res, lams, res)
        dt = time.time() - t0
        for k, l in enumerate(lams):
            Rt, Tt, At = tmm_flat(si, l)
            R, T, A = m[k]
            rows.append({"check": "1a", "resolution_per_um": res, "period_um": 4.0 / res, "lambda_nm": l,
                         "R": R, "T": T, "A": A, "R_tmm": Rt, "T_tmm": Tt, "A_tmm": At,
                         "dA_pp": 100 * (A - At), "runtime_s": round(dt, 1)})
        dA = np.array([r["A"] - r["A_tmm"] for r in rows if r["resolution_per_um"] == res])
        dR = np.array([r["R"] - r["R_tmm"] for r in rows if r["resolution_per_um"] == res])
        i940 = list(lams).index(LAM_TARGET)
        print(f"  res {res:4d}: max|ΔA| {100*np.abs(dA).max():.3f}%p  max|ΔR| {100*np.abs(dR).max():.3f}%p"
              f"  | 940 nm A {100*m[i940,2]:.3f}% (TMM {100*tmm_flat(si, LAM_TARGET)[2]:.3f}%)  ({dt:.0f} s)")
    final = max(res_list)
    worst = max(abs(r["A"] - r["A_tmm"]) for r in rows if r["resolution_per_um"] == final)
    errs = [max(abs(r["A"] - r["A_tmm"]) for r in rows if r["resolution_per_um"] == s) for s in res_list]
    mono = all(b < a for a, b in zip(errs, errs[1:]))   # 해상도 1개만 주면 True (단독 실행 시 다른 실행과 합쳐 판정)
    ok = mono and worst <= 1e-3
    print(f"  판정: 해상도↑ 오차 감소 {mono}, 최종 res {final} max|ΔA| {100*worst:.3f}%p ≤ 0.1%p → {'통과' if ok else '미달'}")
    return rows, ok


def check_1b(si, res=20):
    print(f"\n[체크 1b] 단위셀 크기 무관성 (res {res}, 940 nm): P = 4/res vs P = 1.0 μm")
    rows = []
    vals = {}
    for P in (4.0 / res, 1.0):
        t0 = time.time()
        R, T, A = flat_sweep(si, P, [LAM_TARGET], res)[0]
        vals[P] = (R, T, A)
        rows.append({"check": "1b", "resolution_per_um": res, "period_um": P, "lambda_nm": LAM_TARGET,
                     "R": R, "T": T, "A": A, "runtime_s": round(time.time() - t0, 1)})
        print(f"  P {P:.3f} μm: R {100*R:.5f}%  T {100*T:.5f}%  A {100*A:.5f}%  ({rows[-1]['runtime_s']} s)")
    a, b = vals.values()
    d = max(abs(x - y) for x, y in zip(a, b))
    ok = d < 1e-6
    print(f"  max 차이 {d:.2e} → 기준 < 1e-6 {'통과' if ok else '미달'}")
    return rows, ok


def check_1c(si, res=40, P=1.0, r=0.35, h=0.5):
    print(f"\n[체크 1c] 홀 기하: P {P}, r {r}, h {h} μm, res {res}")
    cell = NanoHoleCell3D(P, r, h, T_UM, [1000.0 / LAM_TARGET], res, t_cap=T_CAP)
    eps_si = (si.N(LAM_TARGET).real ** 2 - si.N(LAM_TARGET).imag ** 2)
    sim, _, _ = cell._build(medium_from_N(si.N(LAM_TARGET), LAM_TARGET / 1000.0))
    sim.init_sim()
    eps = np.real(sim.get_array(component=mp.Dielectric, center=mp.Vector3(), size=mp.Vector3(P, P, cell.L)))
    sim.reset_meep()
    nz = eps.shape[2]
    z = np.linspace(-cell.L / 2, cell.L / 2, nz)
    is_si = eps > 0.5 * (1 + eps_si)
    si_frac = is_si.reshape(-1, nz).mean(axis=0)          # z 단면별 Si 면적비
    air_hole = 1 - np.pi * r * r / (P * P)                 # 홀 구간의 기대 Si 면적비
    inner = 1.5 / res                                      # 경계면 격자 1~2개는 섞일 수 있어 제외
    m_hole = (z > T_UM / 2 - h + inner) & (z < T_UM / 2 - inner)
    m_bulk = (z > -T_UM / 2 + inner) & (z < T_UM / 2 - h - inner)
    m_out = (np.abs(z) > T_UM / 2 + inner)
    f_hole, f_bulk, f_out = si_frac[m_hole].mean(), si_frac[m_bulk].min(), si_frac[m_out].max()
    print(f"  ε 배열 크기 {eps.shape}, ε_Si(940) = {eps_si:.3f}")
    print(f"  홀 구간 z ∈ ({T_UM/2-h:.2f}, {T_UM/2:.2f}) Si 면적비 {f_hole:.4f} (기대 1 − πr²/P² = {air_hole:.4f})")
    print(f"  홀 아래 Si 구간 최소 Si 면적비 {f_bulk:.4f} (기대 1),  필름 밖 최대 Si 면적비 {f_out:.4f} (기대 0)")
    # 홀이 어느 쪽 면에 있는지: 광원(z_src) 쪽인지 투과 모니터 쪽인지
    side = "투과 쪽(+z, 광원 반대편)" if T_UM / 2 > 0 and cell.z_src < 0 else "광원 쪽"
    print(f"  홀 위치: z = +t/2 면 = {side}. 광원 z = {cell.z_src:.2f}")
    ok = abs(f_hole - air_hole) < 0.03 and f_bulk > 0.999 and f_out < 1e-3
    print(f"  판정 (면적비 오차 < 0.03, 바닥 꽉 참, 밖은 공기) → {'통과' if ok else '미달'}")
    return [{"check": "1c", "resolution_per_um": res, "period_um": P, "lambda_nm": LAM_TARGET,
             "si_frac_hole": f_hole, "si_frac_hole_expected": air_hole, "si_frac_bulk_min": f_bulk,
             "si_frac_outside_max": f_out}], ok, (z, si_frac)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {os.path.relpath(path, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["1a", "1b", "1c"], default=None)
    ap.add_argument("--res", type=int, nargs="+", default=None, help="체크 1a 해상도 목록 직접 지정")
    ap.add_argument("--lams", type=float, nargs="+", default=None, help="체크 1a 파장 목록 [nm] 직접 지정")
    ap.add_argument("--tag", default="", help="출력 파일 이름 뒤에 붙일 꼬리표 (병렬 실행용)")
    args = ap.parse_args()
    print(f"MEEP {mp.__version__}, t_cap {T_CAP}")
    si = SiOpticalConstants(verbose=False)
    if args.quick:
        res_list, lams = [50, 100], [LAM_TARGET]
    else:
        res_list, lams = [100, 200, 400], [800.0, 850.0, 900.0, 940.0, 1000.0]
    res_list = args.res or res_list
    lams = args.lams or lams
    rows, verdict = [], {}
    t0 = time.time()
    if args.only in (None, "1c"):
        r, verdict["1c"], _ = check_1c(si)
        rows += r
    if args.only in (None, "1b"):
        r, verdict["1b"] = check_1b(si)
        rows += r
    if args.only in (None, "1a"):
        r, verdict["1a"] = check_1a(si, res_list, lams)
        rows += r
    suffix = "_quick" if args.quick else ""
    name = f"week6_3d_flat_check{('_' + args.only) if args.only else ''}{('_' + args.tag) if args.tag else ''}{suffix}.csv"
    write_csv(os.path.join(ROOT, "results", name), rows)
    print("\n요약:", ", ".join(f"체크 {k} {'통과' if v else '미달'}" for k, v in sorted(verdict.items())),
          f"/ 총 {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
