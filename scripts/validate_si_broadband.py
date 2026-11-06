"""
validate_si_broadband.py — 6주차: Si 구간별 분산 피팅 만들기 + 검증

구간 (작업 대역 4개, 피팅은 양쪽 5 nm 여유, 이웃 구간 경계 파장 850/900/950 은 두 구간에서 모두 계산 → 일관성 확인):
  800–850 (피팅 795–855), 850–900 (845–905), 900–950 (895–955), 950–1000 (945–1005), 각 Lorentz 4항, 세기 상한 0.2
  제약: ε_∞ ≥ 1 (없으면 ε_∞ = −16 해가 나와 MEEP 발산), 공진 f_n ≥ 0.8, 항 세기 σ_n ≤ 0.2 (아래 SIGMA_MAX 근거)
  한 구간을 전체 800–1000 으로 맞추면 4항이어도 Im ε 오차 18% (간접 밴드갭 흡수 꼬리 모양을 Lorentz 꼬리가 못 따라감)

검증
  A. ε 오차: Im ε 최대 상대오차, Re ε 최대 절대오차 (Green 2008 n 은 소수 셋째 자리 → Re ε 반올림 수준 ≈ 3e-3)
  B. TMM (2 μm, 공기/Si/공기, coherent) 흡수율: 피팅 N vs Green N, 1 nm 간격. 기준 max|ΔA| < 1e-3 (4주차 기준과 같은 값)
  C. MEEP 1D 한 번 실행(구간당) vs TMM(Green): 해상도 800, 구간당 51 주파수. 기준 max|ΔA| < 1e-3
     (파장별 conductivity 방식의 4주차 결과 5.9e-4 와 같은 수준이어야 함)

출력: data/si_broadband_fit.json (피팅값, git 추적 — 재피팅 없이 같은 재료를 쓰기 위해)
      results/week6_si_broadband_check.csv (미추적)
실행: python scripts/validate_si_broadband.py [--refit] [--no-meep]
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np

from si_data import SiOpticalConstants
from si_broadband import fit, eps_model, eps_target, N_from_fit
from tmm import tmm_normal_incidence

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIT_PATH = os.path.join(ROOT, "data", "si_broadband_fit.json")
BANDS = [  # (작업 구간, 피팅 구간, Lorentz 항 수)
    ((800.0, 850.0), (795.0, 855.0), 4),
    ((850.0, 900.0), (845.0, 905.0), 4),
    ((900.0, 950.0), (895.0, 955.0), 4),
    ((950.0, 1000.0), (945.0, 1005.0), 4),
]
SIGMA_MAX = 0.2   # Lorentz 항 세기 상한 → ε 대부분을 ε_∞ 에 둠 (MEEP 경계면 평활화는 ε_∞ 에만 적용)
# 근거 (800–850 nm, 1D 해상도 200/400/800, max|ΔA| vs TMM(같은 피팅 재료)):
#   상한 없음(ε_∞ 1, 고주파 항 σ 10.5): 3.34 / 1.30 / 0.56 %p (1차 수렴)
#   상한 1.0 (ε_∞ 11.9):                0.92 / 0.25 / 0.086 %p
#   상한 0.2 (ε_∞ 12.9):                0.71 / 0.17 / 0.048 %p  ← 파장별 conductivity 모델(res 800 0.038 %p)과 비슷
T_UM = 2.0


def make_fits(si):
    out = []
    for work, rng, npole in BANDS:
        t0 = time.time()
        p = fit(si, npole=npole, lam_range=rng, n_trials=40, n_pts=61, seed=0, sigma_max=SIGMA_MAX)
        p["work_nm"] = work
        out.append(p)
        print(f"  피팅 {work[0]:.0f}–{work[1]:.0f} nm (맞춤 {rng[0]:.0f}–{rng[1]:.0f}, {npole}항): "
              f"Im ε 최대 상대오차 {100*p['max_rel_err_im']:.3f}%, Re ε 최대 {p['max_abs_err_re']:.1e} ({time.time()-t0:.0f} s)")
    with open(FIT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"source": "data/Si_Green2008.csv (Green 2008), MEEP 규약 Im ε > 0",
                   "model": "eps_inf + sum sigma*f0^2/(f0^2 - f^2 - i f gamma), f = 1/lambda[um]",
                   "bands": out}, fh, indent=2, ensure_ascii=False)
    print(f"  저장: data/si_broadband_fit.json")
    return out


def load_fits():
    with open(FIT_PATH, encoding="utf-8") as fh:
        d = json.load(fh)
    for b in d["bands"]:
        b["poles"] = [tuple(p) for p in b["poles"]]
    return d["bands"]


def check_tmm(si, fits):
    print("\n[B] TMM 흡수율: 피팅 N vs Green N (2 μm, coherent, 1 nm 간격)")
    rows, worst = [], 0.0
    for b in fits:
        lam = np.arange(b["work_nm"][0], b["work_nm"][1] + 0.5, 1.0)
        Nf = N_from_fit(b, lam)
        dA = [tmm_normal_incidence(l, [nf], [T_UM * 1000], 1.0, 1.0)[2] - tmm_normal_incidence(l, [si.N(l)], [T_UM * 1000], 1.0, 1.0)[2]
              for l, nf in zip(lam, Nf)]
        m = float(np.max(np.abs(dA)))
        worst = max(worst, m)
        rows.append({"check": "B_tmm", "band": f"{b['work_nm'][0]:.0f}-{b['work_nm'][1]:.0f}", "max_abs_dA": m})
        print(f"  {b['work_nm'][0]:.0f}–{b['work_nm'][1]:.0f} nm: max|ΔA| {m:.2e}")
    print(f"  판정: 최대 {worst:.2e} → 기준 < 1e-3 {'통과' if worst < 1e-3 else '미달'}")
    return rows


def check_meep_1d(si, fits, res=800, nfreq=51):
    import meep as mp
    from si_broadband import meep_medium
    from meep_1d import Slab1D
    mp.verbosity(0)
    print(f"\n[C] MEEP 1D 한 번 실행(구간당) vs TMM(Green), 해상도 {res}, 구간당 {nfreq} 주파수")
    rows, worst = [], 0.0
    for b in fits:
        lam = np.linspace(b["work_nm"][0], b["work_nm"][1], nfreq)
        t0 = time.time()
        slab = Slab1D(T_UM, 1000.0 / lam, res)
        R, T, A = slab.rta(meep_medium(b))
        At = np.array([tmm_normal_incidence(l, [si.N(l)], [T_UM * 1000], 1.0, 1.0)[2] for l in lam])
        m = float(np.max(np.abs(A - At)))
        worst = max(worst, m)
        rows.append({"check": "C_meep1d", "band": f"{b['work_nm'][0]:.0f}-{b['work_nm'][1]:.0f}", "max_abs_dA": m,
                     "runtime_s": round(time.time() - t0, 1)})
        print(f"  {b['work_nm'][0]:.0f}–{b['work_nm'][1]:.0f} nm: max|ΔA| {m:.2e}  ({time.time()-t0:.0f} s)", flush=True)
    print(f"  판정: 최대 {worst:.2e} → 기준 < 1e-3 {'통과' if worst < 1e-3 else '미달'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="피팅을 다시 해서 data/si_broadband_fit.json 덮어쓰기")
    ap.add_argument("--no-meep", action="store_true")
    a = ap.parse_args()
    si = SiOpticalConstants(verbose=False)
    if a.refit or not os.path.exists(FIT_PATH):
        print("[A] 구간별 피팅")
        fits = make_fits(si)
    else:
        fits = load_fits()
        print("[A] 저장된 피팅 사용: data/si_broadband_fit.json")
        for b in fits:
            print(f"  {b['work_nm'][0]:.0f}–{b['work_nm'][1]:.0f} nm: Im ε 최대 상대오차 {100*b['max_rel_err_im']:.3f}%, Re ε 최대 {b['max_abs_err_re']:.1e}")
    rows = check_tmm(si, fits)
    if not a.no_meep:
        rows += check_meep_1d(si, fits)
    path = os.path.join(ROOT, "results", "week6_si_broadband_check.csv")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("저장: results/week6_si_broadband_check.csv")


if __name__ == "__main__":
    main()
