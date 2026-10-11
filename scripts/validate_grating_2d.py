"""
validate_grating_2d.py — 5주차: 2D 라멜라 그레이팅 흡수 스펙트럼 + 수렴 테스트

계획서 5주차 결과물: "그레이팅 흡수 스펙트럼 + 수렴 테스트 그래프"
진행 노트 5절 체크리스트를 그대로 따른다:
  - 해상도 수렴 (1D 표 기준)           → 체크 2
  - 광원: 평면파 + 주기 경계, 가장자리 세기 → 스펙트럼 실행 시 확인 (출력에 기록)
  - PML 두께 2배 테스트                  → 체크 3
  - flux: 정규화 런 + load_minus_flux_data → src/meep_2d.py 에 구현

체크 1 — 평면 환원 (구조를 믿기 전에 먼저 통과해야 함)
  f = 1, h = 0 (평면 Si 필름) 이면 1D 슬랩(2 μm)과 같아야 한다. TMM 과 비교. 기준은 1D 와 같은 개념:
  각 해상도에서 max|ΔA| 가 해상도 증가에 따라 줄어드는지 확인하고, 최종 해상도에서 1D 수준(≈0.1 %p) 이어야 한다.
체크 2 — 해상도 수렴 (940 nm, 그레이팅 Λ=0.8 μm, f=0.6, h=0.5 μm)
체크 3 — PML 두께 2배 (dpml 1.0 → 2.0 μm), 같은 해상도에서 ΔA, ΔR, ΔT
스펙트럼 — Λ 별 그레이팅 A(λ) 와 같은 해상도의 평면 Si A(λ) 비교 (향상 배율 = A_grating / A_flat)

출력 (--quick 이면 파일 이름 끝에 _quick, git 미추적):
  results/week5_grating_flat_check.csv
  results/week5_grating_convergence.csv
  results/week5_grating_spectrum.csv
  figures/fig_week5_grating_spectrum.png

실행 (WSL2, photontrap-meep 환경):
  python scripts/validate_grating_2d.py --quick   # 해상도 100, 20 nm 간격, Λ=0.8 한 개. 수 분 목표
  python scripts/validate_grating_2d.py           # 전체. 해상도 100/200/400, 10 nm 간격, Λ=0.7/0.85/1.0. 수 시간 가능 — 실행 전 --quick 먼저

※ 이 스크립트는 Claude 작업 환경(MEEP 없음)에서 실행하지 못했다. 실행 결과를 확인한 뒤에만 결과로 인정한다.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tmm import tmm_normal_incidence
from si_data import SiOpticalConstants

try:
    import meep as mp
    from meep_1d import medium_from_N
    from meep_2d import LamellarGrating2D
except ImportError:
    mp = None

ROOT = os.path.join(os.path.dirname(__file__), "..")
T_UM = 2.0                       # 베이스라인 필름 두께 (진행 노트 3절)
LAM_TARGET = 940.0
LAM_MIN, LAM_MAX = 800.0, 1000.0
F_SI, H_UM = 0.6, 0.5            # 그레이팅 예시 값 (5주차 탐색용. 8~10주차에 스윕으로 결정)
LAM_CONV = 0.8                   # 수렴 테스트에 쓰는 주기 (μm)


def tmm_flat(si, lam_nm):
    """평면 Si 필름의 coherent TMM (공기 / Si 2μm / 공기, 수직입사)."""
    N = si.N(lam_nm)
    R, T, A = tmm_normal_incidence(lam_nm, [N], [T_UM * 1000.0], 1.0, 1.0)
    return R, T, A


def run_point(period, f, h, lam_nm, res, si, dpml=1.0, freqs=None):
    """한 파장(lam_nm)의 재료로 그레이팅 한 번 계산 → 그 파장의 (R, T, A)."""
    if freqs is None:
        freqs = np.array([1000.0 / lam_nm])
    g = LamellarGrating2D(period, f, h, T_UM, freqs, res, dpml_um=dpml)
    mat = medium_from_N(si.N(lam_nm), lam_nm / 1000.0)
    R, T, A = g.rta(mat)
    return float(R[0]), float(T[0]), float(A[0])


def check_flat(si, res_list, lam_list):
    print("\n[체크 1] 평면 환원: f=1, h=0 (2D 셀 안의 평면 Si) vs coherent TMM")
    rows = []
    for res in res_list:
        t0 = time.time()
        errs = []
        for lam in lam_list:
            R, T, A = run_point(LAM_CONV, 1.0, 0.0, lam, res, si)
            Rt, Tt, At = tmm_flat(si, lam)
            errs.append((abs(R - Rt), abs(T - Tt), abs(A - At)))
        e = np.array(errs)
        rows.append({"resolution_per_um": res,
                     "max_abs_dR": e[:, 0].max(), "max_abs_dT": e[:, 1].max(), "max_abs_dA": e[:, 2].max(),
                     "runtime_s": round(time.time() - t0, 1)})
        print(f"  res {res:4d}: max|ΔR| {e[:,0].max():.2e}  max|ΔT| {e[:,1].max():.2e}"
              f"  max|ΔA| {e[:,2].max():.2e}  ({rows[-1]['runtime_s']} s)")
    return rows


def check_convergence(si, res_list):
    print(f"\n[체크 2] 해상도 수렴: 940 nm, Λ={LAM_CONV} μm, f={F_SI}, h={H_UM} μm")
    rows = []
    for res in res_list:
        t0 = time.time()
        R, T, A = run_point(LAM_CONV, F_SI, H_UM, LAM_TARGET, res, si)
        rows.append({"test": "resolution", "resolution_per_um": res, "dpml_um": 1.0,
                     "R": R, "T": T, "A": A, "runtime_s": round(time.time() - t0, 1)})
        print(f"  res {res:4d}: R {100*R:.3f}%  T {100*T:.3f}%  A {100*A:.4f}%  ({rows[-1]['runtime_s']} s)")
    return rows


def check_pml(si, res):
    print(f"\n[체크 3] PML 두께 2배 (res {res}, 940 nm)")
    rows = []
    for dpml in (1.0, 2.0):
        t0 = time.time()
        R, T, A = run_point(LAM_CONV, F_SI, H_UM, LAM_TARGET, res, si, dpml=dpml)
        rows.append({"test": "pml", "resolution_per_um": res, "dpml_um": dpml,
                     "R": R, "T": T, "A": A, "runtime_s": round(time.time() - t0, 1)})
        print(f"  dpml {dpml:.1f} μm: R {100*R:.4f}%  T {100*T:.4f}%  A {100*A:.4f}%")
    d = {k: abs(rows[1][k] - rows[0][k]) for k in ("R", "T", "A")}
    print(f"  차이: |ΔR| {d['R']:.2e}  |ΔT| {d['T']:.2e}  |ΔA| {d['A']:.2e}"
          f"  → 기준 |ΔA| < 0.001 이면 통과 (진행 노트 5절 체크리스트)")
    return rows


def spectrum(si, periods, res, step_nm):
    lam = np.round(np.arange(LAM_MIN, LAM_MAX + 1e-9, step_nm), 3)
    if LAM_TARGET not in lam:
        lam = np.sort(np.append(lam, LAM_TARGET))
    rows = []
    for P in periods:
        print(f"\n[스펙트럼] Λ = {P} μm, f = {F_SI}, h = {H_UM} μm, res {res}, {len(lam)} 점")
        t0 = time.time()
        for i, l in enumerate(lam):
            R, T, A = run_point(P, F_SI, H_UM, l, res, si)
            Rf, Tf, Af = run_point(P, 1.0, 0.0, l, res, si)   # 같은 셀·해상도의 평면 Si (MEEP 기준선)
            At = tmm_flat(si, l)[2]
            rows.append({"period_um": P, "lambda_nm": l, "R": R, "T": T, "A": A,
                         "A_flat_meep": Af, "A_flat_tmm": At,
                         "enhancement": A / Af if Af > 0 else float("nan")})
            if i % max(1, len(lam) // 8) == 0:
                print(f"  {l:7.1f} nm: A grating {100*A:.3f}%  flat(MEEP) {100*Af:.3f}%  "
                      f"flat(TMM) {100*At:.3f}%  ({i+1}/{len(lam)}, {time.time()-t0:.0f} s)")
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  저장: {os.path.relpath(path, ROOT)}")


def _set_korean_font():
    # WSL2 에서는 Windows 폰트가 matplotlib 에 안 보이므로 맑은 고딕을 직접 등록 (validate_meep_1d.py 와 같은 방식)
    if os.path.exists("/mnt/c/Windows/Fonts/malgun.ttf"):
        matplotlib.font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/malgun.ttf")
    names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK", "AppleGothic"):
        hit = sorted(n for n in names if n.startswith(fam))
        if hit:
            plt.rcParams["font.family"] = hit[0]
            break
    plt.rcParams["axes.unicode_minus"] = False


def make_figure(spec_rows, conv_rows, path):
    _set_korean_font()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    periods = sorted({r["period_um"] for r in spec_rows})
    lam = np.array([r["lambda_nm"] for r in spec_rows if r["period_um"] == periods[0]])
    for P in periods:
        sub = [r for r in spec_rows if r["period_um"] == P]
        ax[0].plot(lam, [100 * r["A"] for r in sub], label=f"그레이팅 Λ={P} μm")
    flat = [r for r in spec_rows if r["period_um"] == periods[0]]
    ax[0].plot(lam, [100 * r["A_flat_meep"] for r in flat], "k--", label="평면 Si (MEEP)")
    ax[0].plot(lam, [100 * r["A_flat_tmm"] for r in flat], "k:", label="평면 Si (TMM)")
    ax[0].set_xlabel("파장 (nm)")
    ax[0].set_ylabel("흡수율 A (%)")
    ax[0].set_title("2D 라멜라 그레이팅 흡수 스펙트럼 (f=0.6, h=0.5 μm, t=2 μm)")
    ax[0].legend(fontsize=8)

    res = [r["resolution_per_um"] for r in conv_rows if r["test"] == "resolution"]
    A = [100 * r["A"] for r in conv_rows if r["test"] == "resolution"]
    ax[1].plot(res, A, "o-")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("해상도 (격자점/μm)")
    ax[1].set_ylabel("940 nm 흡수율 A (%)")
    ax[1].set_title("해상도 수렴 (Λ=0.8 μm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  그림: {os.path.relpath(path, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if mp is None:
        sys.exit("MEEP(pymeep)을 import 할 수 없습니다. WSL2의 photontrap-meep 환경에서 실행하세요.")

    si = SiOpticalConstants(verbose=False)
    sfx = "_quick" if args.quick else ""
    if args.quick:
        res_flat, res_conv, res_spec, periods, step = [100], [100, 200], 100, [LAM_CONV], 20.0
        lam_flat = [800.0, 940.0, 1000.0]
    else:
        res_flat, res_conv, res_spec, periods, step = [100, 200, 400], [100, 200, 400], 200, [0.7, 0.85, 1.0], 10.0
        lam_flat = [800.0, 870.0, 940.0, 1000.0]

    t0 = time.time()
    flat_rows = check_flat(si, res_flat, lam_flat)
    conv_rows = check_convergence(si, res_conv)
    conv_rows += check_pml(si, res_conv[-1] if args.quick else 200)
    spec_rows = spectrum(si, periods, res_spec, step)

    write_csv(os.path.join(ROOT, "results", f"week5_grating_flat_check{sfx}.csv"), flat_rows)
    write_csv(os.path.join(ROOT, "results", f"week5_grating_convergence{sfx}.csv"), conv_rows)
    write_csv(os.path.join(ROOT, "results", f"week5_grating_spectrum{sfx}.csv"), spec_rows)
    make_figure(spec_rows, conv_rows, os.path.join(ROOT, "figures", f"fig_week5_grating_spectrum{sfx}.png"))
    print(f"\n총 실행 시간 {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
