"""
nanohole_3d.py — 6주차: 3D 원기둥 나노홀 단위셀 첫 흡수 스펙트럼

계획서 6주차: "얇은 Si 층(예: 2 μm)에 원기둥 나노홀 단위셀 구성 — 주기 700–1000 nm 범위에서 시작"
              결과물 "첫 3D 흡수 스펙트럼, 평면 Si 대비 향상 확인"
구조 (notes/06_nanohole_3d_prep.md 첫 후보): 정사각 격자 P = 1.0 μm, 홀 지름 0.7 μm (r = 0.35), 깊이 h = 0.5 μm,
  Si 2 μm, 홀 = 광원 쪽 면 (진행 노트 확정 2026-11-06), 뒷면 공기, Ex 편광 수직입사, t_cap 800

모드:
  --symcheck [RES]      : 대칭(1/4 셀) 켠 것과 끈 것 비교 (940 nm). 거친 해상도에서는 이산화 차이가 있으므로
                          해상도를 올릴 때 차이가 줄어드는지로 판정 (2026-11-06: res 20 9.4%p, 30 0.26%p)
  --conv 20 30 40 ...   : 940 nm 해상도별 R/T/A + 잔류 에너지 비율 + 시간 (대칭 사용)
  --spectrum RES        : 800–1000 nm 20 nm 간격 그레이팅 스펙트럼 (대칭 사용). --lams 로 일부만 (병렬 실행)
  --broadband RES       : Si 구간별 분산 재료(data/si_broadband_fit.json)로 구간당 한 번 실행, 2 nm 간격 (--bands 0 1 2 3)
  평면 기준선은 작은 셀(P = 4/res, 6주차 체크 1b)로 같은 해상도에서 계산

출력 (git 미추적): results/week6_nanohole_{symcheck|conv|spectrum}[_tag].csv
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
P_UM, R_UM, H_UM, T_UM = 1.0, 0.35, 0.5, 2.0
T_CAP = 800.0
LAM_TARGET = 940.0
RECORD_DT = 20.0

mp.verbosity(0)


def cell_energy(sim, c):
    """필름 + 패드 영역의 전자기 에너지 (ε|E|² 합, 상대값). PML 제외. 대칭 사용 시에도 전체 셀 배열로 받음."""
    center = mp.Vector3()
    size = mp.Vector3(c.P, c.P, c.t + 2 * c.pad)
    eps = sim.get_array(center=center, size=size, component=mp.Dielectric)
    w = 0.0
    for comp in (mp.Ex, mp.Ey, mp.Ez):
        a = sim.get_array(center=center, size=size, component=comp)
        w += np.sum(eps * np.abs(a) ** 2)
    return float(w)


def run_one(c, mat, i, record=False):
    """c.rta() 와 같은 절차. i 번째 주파수 값만 돌려준다. record=True 면 잔류 에너지 비율도."""
    inc, refl_data = c.normalize()
    sim, refl, tran = c._build(mat)
    sim.load_minus_flux_data(refl, refl_data)
    series = []
    if record:
        sim.run(mp.at_every(RECORD_DT, lambda s: series.append(cell_energy(s, c))), until_after_sources=c.t_cap)
    else:
        sim.run(until_after_sources=c.t_cap)
    R = float(-np.array(mp.get_fluxes(refl))[i] / inc[i])
    T = float(np.array(mp.get_fluxes(tran))[i] / inc[i])
    sim.reset_meep()
    ratio = float(series[-1] / max(series)) if series and max(series) > 0 else float("nan")
    return R, T, 1.0 - R - T, ratio


def flat_A(si, lam, res):
    c = NanoHoleCell3D(4.0 / res, 0.0, 0.0, T_UM, [1000.0 / lam], res, t_cap=T_CAP)
    return run_one(c, medium_from_N(si.N(lam), lam / 1000.0), 0)


def symcheck(si, res=20, which=(False, True)):
    print(f"[대칭 확인] res {res}, 940 nm, P {P_UM}, r {R_UM}, h {H_UM}")
    rows = []
    mat = medium_from_N(si.N(LAM_TARGET), LAM_TARGET / 1000.0)
    for sym in which:
        t0 = time.time()
        c = NanoHoleCell3D(P_UM, R_UM, H_UM, T_UM, [1000.0 / LAM_TARGET], res, t_cap=T_CAP, use_symmetry=sym)
        R, T, A, _ = run_one(c, mat, 0)
        dt = time.time() - t0
        rows.append({"mode": "symcheck", "symmetry": sym, "resolution": res, "lambda_nm": LAM_TARGET,
                     "R": R, "T": T, "A": A, "runtime_s": round(dt, 1)})
        print(f"  대칭 {'켬' if sym else '끔'}: R {100*R:.5f}%  T {100*T:.5f}%  A {100*A:.5f}%  ({dt:.0f} s)", flush=True)
    if len(rows) == 2:
        d = max(abs(rows[0][k] - rows[1][k]) for k in "RTA")
        print(f"  최대 차이 {d:.2e}  /  속도 {rows[0]['runtime_s']/rows[1]['runtime_s']:.1f}배")
    return rows


def conv(si, res_list):
    rows = []
    mat = medium_from_N(si.N(LAM_TARGET), LAM_TARGET / 1000.0)
    At = tmm_normal_incidence(LAM_TARGET, [si.N(LAM_TARGET)], [T_UM * 1000.0], 1.0, 1.0)[2]
    print(f"[해상도별 940 nm] 나노홀 P {P_UM}, r {R_UM}, h {H_UM}, 대칭 사용. 평면 TMM A {100*At:.3f}%")
    for res in res_list:
        t0 = time.time()
        c = NanoHoleCell3D(P_UM, R_UM, H_UM, T_UM, [1000.0 / LAM_TARGET], res, t_cap=T_CAP, use_symmetry=True)
        R, T, A, ratio = run_one(c, mat, 0, record=True)
        dt = time.time() - t0
        Af = flat_A(si, LAM_TARGET, res)[2]
        rows.append({"mode": "conv", "resolution": res, "lambda_nm": LAM_TARGET, "R": R, "T": T, "A": A,
                     "A_flat_meep": Af, "A_flat_tmm": At, "enhancement": A / Af,
                     "residual_ratio": ratio, "runtime_s": round(dt, 1)})
        print(f"  res {res:3d}: R {100*R:.3f}%  T {100*T:.3f}%  A {100*A:.3f}%  평면 {100*Af:.3f}%  "
              f"배율 {A/Af:.2f}  잔류 {ratio:.1e}  ({dt:.0f} s)", flush=True)
    return rows


def spectrum(si, res, lams):
    lams = np.asarray(lams, dtype=float)
    print(f"[스펙트럼] res {res}, {len(lams)} 점, 대칭 사용", flush=True)
    c = NanoHoleCell3D(P_UM, R_UM, H_UM, T_UM, 1000.0 / lams, res, t_cap=T_CAP, use_symmetry=True)
    cf = NanoHoleCell3D(4.0 / res, 0.0, 0.0, T_UM, 1000.0 / lams, res, t_cap=T_CAP)
    rows = []
    for i, l in enumerate(lams):
        t0 = time.time()
        mat = medium_from_N(si.N(l), l / 1000.0)
        R, T, A, _ = run_one(c, mat, i)
        Af = run_one(cf, mat, i)[2]
        At = tmm_normal_incidence(l, [si.N(l)], [T_UM * 1000.0], 1.0, 1.0)[2]
        rows.append({"mode": "spectrum", "resolution": res, "lambda_nm": l, "R": R, "T": T, "A": A,
                     "A_flat_meep": Af, "A_flat_tmm": At, "enhancement": A / Af, "runtime_s": round(time.time() - t0, 1)})
        print(f"  {l:6.1f} nm: A {100*A:6.2f}%  (R {100*R:5.1f}, T {100*T:5.1f})  평면 {100*Af:5.2f}%  "
              f"배율 {A/Af:5.2f}  ({rows[-1]['runtime_s']} s)", flush=True)
    return rows


def broadband(res, band_ids, nfreq=26):
    """
    넓은 대역 모드: Si 를 구간별 분산 재료(data/si_broadband_fit.json)로 넣고 구간당 한 번 실행.
    구간당 nfreq 주파수 (기본 26 = 2 nm 간격). 평면 기준선도 같은 재료·해상도로 작은 셀에서 한 번.
    """
    import json
    from si_broadband import meep_medium
    with open(os.path.join(ROOT, "data", "si_broadband_fit.json"), encoding="utf-8") as fh:
        bands = json.load(fh)["bands"]
    si = SiOpticalConstants(verbose=False)
    rows = []
    for bi in band_ids:
        b = bands[bi]
        b["poles"] = [tuple(p) for p in b["poles"]]
        lam = np.linspace(b["work_nm"][0], b["work_nm"][1], nfreq)
        mat = meep_medium(b)
        t0 = time.time()
        c = NanoHoleCell3D(P_UM, R_UM, H_UM, T_UM, 1000.0 / lam, res, t_cap=T_CAP, use_symmetry=True)
        R, T, A = c.rta(mat)
        t1 = time.time()
        cf = NanoHoleCell3D(4.0 / res, 0.0, 0.0, T_UM, 1000.0 / lam, res, t_cap=T_CAP)
        Af = cf.rta(mat)[2]
        At = np.array([tmm_normal_incidence(l, [si.N(l)], [T_UM * 1000.0], 1.0, 1.0)[2] for l in lam])
        for i, l in enumerate(lam):
            rows.append({"mode": "broadband", "resolution": res, "band": bi, "lambda_nm": l,
                         "R": R[i], "T": T[i], "A": A[i], "A_flat_meep": Af[i], "A_flat_tmm": At[i],
                         "runtime_s": round(t1 - t0, 1)})
        print(f"  res {res} 구간 {b['work_nm'][0]:.0f}–{b['work_nm'][1]:.0f} nm: 평균 A {100*A.mean():.2f}%  "
              f"평면 MEEP {100*Af.mean():.2f}% (TMM {100*At.mean():.2f}%)  ({t1-t0:.0f} s)", flush=True)
    return rows


def write_csv(name, rows):
    path = os.path.join(ROOT, "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"저장: results/{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symcheck", type=int, nargs="?", const=20, default=None, help="해상도 (기본 20)")
    ap.add_argument("--sym-only", choices=["on", "off"], default=None, help="--symcheck 에서 한쪽만 (병렬 실행용)")
    ap.add_argument("--conv", type=int, nargs="+")
    ap.add_argument("--spectrum", type=int, help="스펙트럼 해상도")
    ap.add_argument("--lams", type=float, nargs="+", default=None)
    ap.add_argument("--broadband", type=int, help="넓은 대역 모드 해상도 (구간당 1회 실행)")
    ap.add_argument("--bands", type=int, nargs="+", default=[0, 1, 2, 3], help="--broadband 구간 번호 (0: 800–850 … 3: 950–1000)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    si = SiOpticalConstants(verbose=False)
    tag = f"_{a.tag}" if a.tag else ""
    print(f"MEEP {mp.__version__}, t_cap {T_CAP}, 홀 = 광원 쪽", flush=True)
    if a.symcheck:
        which = {None: (False, True), "on": (True,), "off": (False,)}[a.sym_only]
        write_csv(f"week6_nanohole_symcheck_res{a.symcheck}{tag}.csv", symcheck(si, a.symcheck, which))
    if a.conv:
        write_csv(f"week6_nanohole_conv{tag}.csv", conv(si, a.conv))
    if a.broadband:
        write_csv(f"week6_nanohole_broadband_res{a.broadband}{tag}.csv", broadband(a.broadband, a.bands))
    if a.spectrum:
        lams = a.lams or list(np.arange(800.0, 1000.0 + 1e-9, 20.0))
        write_csv(f"week6_nanohole_spectrum_res{a.spectrum}{tag}.csv", spectrum(si, a.spectrum, lams))


if __name__ == "__main__":
    main()
