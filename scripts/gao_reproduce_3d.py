"""
gao_reproduce_3d.py — 7주차 (체크포인트 2): Gao 2017 구조 파라미터로 흡수 향상 경향 재현

계획서 7주차: "Gao 2017 [1]의 구조 파라미터로 흡수 향상 경향 재현, 정량 비교"
범위 (2026-11-06 결정): 단계적으로. 1단계 = 지금 Si 슬랩(뒷면 공기) 모델에 Gao 의 격자·d/p·깊이·형상을 넣음.
  BOX/SOI 등 층 구조 차이는 "차이 원인 분석"으로 2단계.

Gao 2017 Fig. 3c 조건 (논문 PDF 확인): 육각 격자, d/p = 700/1000 · 1300/2000 nm, 깔때기(75°, 66°, −75°)와 원기둥,
  800–900 nm, A = 1 − R − T. 홀은 바닥 p층까지 (보충 Fig. S7 깊이 2000 nm). 층: n 0.2 / i 2.0 / p 0.2 + SOI 0.25 μm.

케이스 (한 번에 하나씩 바꿈):
  w6sq      : 6주차 기준 구조 — 정사각, d 0.7, P 1.0, h 0.5, t 2.0
  hex05     : 격자만 육각으로 — 육각, d 0.7, P 1.0, h 0.5, t 2.0
  gao700cyl : Gao 등가 슬랩 — 육각, d 0.7, P 1.0, h 2.0, t 2.65 (n+i+p 합, 도핑 무시), 원기둥
지표: 800–900 nm 대역 평균 흡수 (구간 0·1, 0.5 nm 간격, 구간별 사다리꼴 평균의 평균). 평면 기준 = 같은 두께 coherent TMM.

모드:
  --geomcheck RES             : ε 분포로 홀 위치·면적비 확인 (MEEP 실행 없음, 수 초)
  --flatcheck RES             : 육각 셀 r = 0 vs 작은 정사각 셀 (850 nm, 같은 값이어야 함)
  --symcheck RES              : 육각 원기둥(gao700cyl) 대칭 켬 vs 끔 (850 nm)
  --broadband RES --case C    : 넓은 대역 재료로 구간당 1회 실행 (--bands 0 1)
출력: results/week7_*.csv
"""
import argparse
import csv
import json
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
T_CAP = 800.0
LAM_CHECK = 850.0

# name: (lattice, P, r_top, r_bottom(None=원기둥), h, t)
CASES = {
    "w6sq":      ("square", 1.0, 0.35, None, 0.5, 2.0),
    "hex05":     ("hex",    1.0, 0.35, None, 0.5, 2.0),
    "gao700cyl": ("hex",    1.0, 0.35, None, 2.0, 2.65),
    "gao1300cyl": ("hex",   2.0, 0.65, None, 2.0, 2.65),   # 셀 2 × 3.46 μm (700/1000 의 4배 면적) → PC 실행용
}

mp.verbosity(0)


def make_cell(case, freqs, res, sym=True, **kw):
    lat, P, r, rb, h, t = CASES[case]
    return NanoHoleCell3D(P, r, h, t, freqs, res, t_cap=T_CAP, use_symmetry=sym,
                          lattice=lat, bottom_radius_um=rb, **kw)


def geomcheck(res, case="gao700cyl"):
    """ε 분포: 홀 깊이 중간 단면의 Si 면적비, z 방향 홀 구간."""
    lat, P, r, rb, h, t = CASES[case]
    c = make_cell(case, [1.0], res, sym=False)
    si = SiOpticalConstants(verbose=False)
    sim, _, _ = c._build(medium_from_N(si.N(LAM_CHECK), LAM_CHECK / 1000.0))
    sim.init_sim()
    eps = sim.get_array(center=mp.Vector3(), size=mp.Vector3(c.P, c.Py, c.t + 1.0), component=mp.Dielectric)
    z = np.linspace(-(c.t + 1.0) / 2, (c.t + 1.0) / 2, eps.shape[2])
    is_si = eps > 2.0
    frac_z = is_si.mean(axis=(0, 1))
    zin = z[(frac_z > 0.01)]
    zmid = -t / 2 + h / 2
    k = np.argmin(abs(z - zmid))
    n_holes = 1 if lat == "square" else 2
    r_mid = r if rb is None else 0.5 * (r + rb)
    expect = 1 - n_holes * np.pi * r_mid ** 2 / (c.P * c.Py)
    hole_z = z[(frac_z > 0.01) & (frac_z < 0.99)]
    print(f"[기하 확인] {case}, res {res}, 셀 {c.P:.3f} × {c.Py:.3f} μm, ε 배열 {eps.shape}")
    print(f"  Si 구간 z {zin.min():+.3f} ~ {zin.max():+.3f} μm (기대 {-t/2:+.3f} ~ {t/2:+.3f})")
    if hole_z.size:
        print(f"  홀 구간 z {hole_z.min():+.3f} ~ {hole_z.max():+.3f} μm (기대 {-t/2:+.3f} ~ {-t/2 + h:+.3f})")
    print(f"  홀 중간 깊이 Si 면적비 {frac_z[k]:.3f} (기대 {expect:.3f})")
    # 모서리 홀 확인: 단면에서 네 모서리가 공기인지
    sl = is_si[:, :, k]
    print(f"  단면 가운데/모서리 Si 여부: 가운데 {bool(sl[sl.shape[0]//2, sl.shape[1]//2])}, "
          f"모서리 {bool(sl[0, 0])} {bool(sl[-1, -1])} (육각이면 모두 False)")
    sim.reset_meep()
    return [{"mode": "geomcheck", "case": case, "resolution": res, "si_frac_mid": float(frac_z[k]),
             "si_frac_expect": float(expect)}]


def rta1(c, mat):
    R, T, A = c.rta(mat)
    return float(R[0]), float(T[0]), float(A[0])


def flatcheck(res):
    si = SiOpticalConstants(verbose=False)
    mat = medium_from_N(si.N(LAM_CHECK), LAM_CHECK / 1000.0)
    f = [1000.0 / LAM_CHECK]
    rows = []
    for name, c in [("hex P1.0 r0", NanoHoleCell3D(1.0, 0.0, 0.0, 2.65, f, res, t_cap=T_CAP, lattice="hex")),
                    ("square P=4/res", NanoHoleCell3D(4.0 / res, 0.0, 0.0, 2.65, f, res, t_cap=T_CAP))]:
        t0 = time.time()
        R, T, A = rta1(c, mat)
        rows.append({"mode": "flatcheck", "cell": name, "resolution": res, "lambda_nm": LAM_CHECK, "R": R, "T": T, "A": A,
                     "runtime_s": round(time.time() - t0, 1)})
        print(f"  {name:16s}: R {100*R:.6f}%  T {100*T:.6f}%  A {100*A:.6f}%  ({rows[-1]['runtime_s']} s)", flush=True)
    At = tmm_normal_incidence(LAM_CHECK, [si.N(LAM_CHECK)], [2650.0], 1.0, 1.0)[2]
    print(f"  차이 {max(abs(rows[0][k]-rows[1][k]) for k in 'RTA'):.1e}, TMM A {100*At:.4f}%")
    return rows


def symcheck(res, which=(False, True), case="gao700cyl"):
    si = SiOpticalConstants(verbose=False)
    mat = medium_from_N(si.N(LAM_CHECK), LAM_CHECK / 1000.0)
    rows = []
    for sym in which:
        t0 = time.time()
        R, T, A = rta1(make_cell(case, [1000.0 / LAM_CHECK], res, sym=sym), mat)
        rows.append({"mode": "symcheck", "case": case, "symmetry": sym, "resolution": res, "lambda_nm": LAM_CHECK,
                     "R": R, "T": T, "A": A, "runtime_s": round(time.time() - t0, 1)})
        print(f"  {case} 대칭 {'켬' if sym else '끔'}: R {100*R:.4f}%  T {100*T:.4f}%  A {100*A:.4f}%  "
              f"({rows[-1]['runtime_s']} s)", flush=True)
    return rows


def broadband(case, res, band_ids, nfreq=101, flat=True):
    from si_broadband import meep_medium
    with open(os.path.join(ROOT, "data", "si_broadband_fit.json"), encoding="utf-8") as fh:
        bands = json.load(fh)["bands"]
    si = SiOpticalConstants(verbose=False)
    t_um = CASES[case][5]
    rows = []
    for bi in band_ids:
        b = bands[bi]
        b["poles"] = [tuple(p) for p in b["poles"]]
        lam = np.linspace(b["work_nm"][0], b["work_nm"][1], nfreq)
        mat = meep_medium(b)
        t0 = time.time()
        R, T, A = make_cell(case, 1000.0 / lam, res).rta(mat)
        t1 = time.time()
        Af = (NanoHoleCell3D(4.0 / res, 0.0, 0.0, t_um, 1000.0 / lam, res, t_cap=T_CAP).rta(mat)[2]
              if flat else np.full_like(lam, np.nan))
        At = np.array([tmm_normal_incidence(l, [si.N(l)], [t_um * 1000.0], 1.0, 1.0)[2] for l in lam])
        for i, l in enumerate(lam):
            rows.append({"mode": "broadband", "case": case, "resolution": res, "band": bi, "nfreq": nfreq,
                         "lambda_nm": l, "R": R[i], "T": T[i], "A": A[i], "A_flat_meep": Af[i], "A_flat_tmm": At[i],
                         "runtime_s": round(t1 - t0, 1)})
        avg = np.trapezoid(A, lam) / (lam[-1] - lam[0])
        avg_t = np.trapezoid(At, lam) / (lam[-1] - lam[0])
        print(f"  {case} res {res} 구간 {b['work_nm'][0]:.0f}–{b['work_nm'][1]:.0f} nm: 평균 A {100*avg:.2f}%  "
              f"평면 TMM {100*avg_t:.2f}%  배율 {avg/avg_t:.2f}  최대 A {100*A.max():.1f}%  ({t1-t0:.0f} s)", flush=True)
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
    ap.add_argument("--geomcheck", type=int)
    ap.add_argument("--flatcheck", type=int)
    ap.add_argument("--symcheck", type=int)
    ap.add_argument("--sym-only", choices=["on", "off"], default=None)
    ap.add_argument("--broadband", type=int)
    ap.add_argument("--case", default="gao700cyl", choices=list(CASES))
    ap.add_argument("--bands", type=int, nargs="+", default=[0, 1], help="0: 800–850, 1: 850–900 (Gao Fig. 3c 대역)")
    ap.add_argument("--nfreq", type=int, default=101)
    ap.add_argument("--no-flat", action="store_true", help="평면 MEEP 기준선 생략 (TMM 만)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""
    print(f"MEEP {mp.__version__}, t_cap {T_CAP}, 홀 = 광원 쪽", flush=True)
    if a.geomcheck:
        geomcheck(a.geomcheck, a.case)
    if a.flatcheck:
        write_csv(f"week7_flatcheck_res{a.flatcheck}{tag}.csv", flatcheck(a.flatcheck))
    if a.symcheck:
        which = {None: (False, True), "on": (True,), "off": (False,)}[a.sym_only]
        write_csv(f"week7_symcheck_{a.case}_res{a.symcheck}{tag}.csv", symcheck(a.symcheck, which, a.case))
    if a.broadband:
        write_csv(f"week7_bb_{a.case}_res{a.broadband}_nf{a.nfreq}{tag}.csv",
                  broadband(a.case, a.broadband, a.bands, a.nfreq, flat=not a.no_flat))


if __name__ == "__main__":
    main()
