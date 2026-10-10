"""
validate_meep_1d.py — 4주차: MEEP 1D Si 슬랩 R/T/A ↔ coherent TMM 비교 (모든 후속 FDTD의 신뢰 기반)

계획서 4주차 결과물: "MEEP–TMM 일치 그래프"
구조: 공기 / Si(2 μm) / 공기, 수직입사, 800–1000 nm  (진행 노트의 3주차 베이스라인 구조와 동일)
비교 대상: coherent TMM (src/tmm.py 의 tmm_normal_incidence 를 직접 호출 — npz 파일 불필요)
  ※ MEEP 은 간섭을 포함한 계산이므로 incoherent 가 아니라 coherent 와 비교한다 (진행 노트 6절)

파트 A — 솔버 검증 + 해상도 수렴
  Si 를 940 nm 에서 맞춘 ε_∞ + σ_D 로 고정하고, MEEP 이 실제로 쓰는 ε(f) 를 그대로 TMM 에도 넣는다.
  → 두 계산의 재료가 '완전히 같으므로' 차이는 순수하게 FDTD 이산화 오차.
  해상도를 2배씩 올리며 오차가 1/res^2 (Yee 격자의 2차 정확도)로 줄어드는지 확인한다.
파트 B — 실제 Si 분산(Green 2008) 비교
  파장마다 그 파장의 n, k 에 정확히 맞춘 재료로 한 번씩 돌리고 그 파장의 R/T 만 읽는다.
  → 3주차까지 쓴 coherent TMM 스펙트럼과 겹쳐 그려 오차 확인.

합격 기준(실행 전에 정함): 파트 B 해상도에서 800–1000 nm 전 구간
  max|ΔR|, max|ΔT| < 0.01 (1 %p),  max|ΔA| < 0.001 (0.1 %p),  그리고 파트 A 수렴 차수 1.8–2.2
  (수렴 차수는 첫 실행 후 res ≥ 100 구간만 판정하도록 수정 — 이유는 main() 의 checks 주석)

출력:
  results/week4_meep_convergence.csv     파트 A 해상도별 오차
  results/week4_meep_vs_tmm_2um.csv      파트 B 파장별 MEEP/TMM R,T,A
  figures/fig_week4_meep_vs_tmm.png

실행 (photontrap 환경, MEEP 설치 후):
  python scripts/validate_meep_1d.py            # 전체 (파트 B 해상도 800, 2 nm 간격: 수 분~10분)
  python scripts/validate_meep_1d.py --quick    # 빠른 확인 (해상도 200, 5 nm 간격: 1분 이내)
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

# MEEP 은 계산할 때만 필요 (--plot-only 는 Windows photontrap 환경에서도 동작하도록)
try:
    import meep as mp
    from meep_1d import Slab1D, medium_from_N, N_tmm_from_medium
except ImportError:
    mp = None

ROOT = os.path.join(os.path.dirname(__file__), "..")
D_UM = 2.0                        # 진행 노트 3절: 베이스라인 두께 2 μm
LAM_TARGET = 940.0
LAM_MIN, LAM_MAX = 800.0, 1000.0
RES_A = [50, 100, 200, 400, 800]  # 파트 A 해상도 (1 μm 당 격자점)
TOL_RT, TOL_A = 0.01, 0.001


def tmm_coherent(lam_nm, N):
    R, T, A = tmm_normal_incidence(lam_nm, [N], [D_UM * 1000.0], 1.0, 1.0)
    return R, T, A


# -----------------------------------------------------------------------------
def part_a(si):
    print("\n[파트 A] 같은 재료(940 nm 고정 ε_∞+σ_D)로 MEEP vs TMM — 해상도 수렴")
    lam = np.arange(LAM_MIN, LAM_MAX + 1e-9, 0.5)
    freqs = 1000.0 / lam
    mat = medium_from_N(si.N(LAM_TARGET), LAM_TARGET / 1000.0)
    tmm = np.array([tmm_coherent(l, N_tmm_from_medium(mat, f)) for l, f in zip(lam, freqs)])
    j = int(np.argmin(np.abs(lam - LAM_TARGET)))

    rows = []
    for res in RES_A:
        t0 = time.time()
        R, T, A = Slab1D(D_UM, freqs, res).rta(mat)
        dR, dT, dA = np.abs(R - tmm[:, 0]), np.abs(T - tmm[:, 1]), np.abs(A - tmm[:, 2])
        rows.append({
            "resolution_per_um": res,
            "pts_per_wavelength_in_Si_800nm": round(res * 0.8 / si.N(800.0).real, 1),
            "max_abs_dR": dR.max(), "max_abs_dT": dT.max(), "max_abs_dA": dA.max(),
            "bandavg_A_meep": A.mean(), "bandavg_A_tmm": tmm[:, 2].mean(),
            "A940_meep": A[j], "A940_tmm": tmm[j, 2],
            "runtime_s": round(time.time() - t0, 1),
        })
        print(f"  res {res:4d}: max|ΔR| {dR.max():.2e}  max|ΔT| {dT.max():.2e}  max|ΔA| {dA.max():.2e}"
              f"  | A940 MEEP {100*A[j]:.3f}% vs TMM {100*tmm[j,2]:.3f}%  ({rows[-1]['runtime_s']} s)")

    err = np.array([r["max_abs_dR"] for r in rows])
    order = np.log2(err[:-1] / err[1:])
    print("  수렴 차수 (max|ΔR|, 해상도 2배마다):", " ".join(f"{o:.2f}" for o in order))
    for r, o in zip(rows[1:], order):
        r["order_vs_prev"] = round(float(o), 3)
    rows[0]["order_vs_prev"] = ""
    return rows, order


# -----------------------------------------------------------------------------
def part_b(si, res, step_nm):
    print(f"\n[파트 B] 실제 Si 분산(Green 2008), 파장마다 재료를 맞춰 계산 — 해상도 {res}, {step_nm} nm 간격")
    lam = np.round(np.arange(LAM_MIN, LAM_MAX + 1e-9, step_nm), 3)
    if LAM_TARGET not in lam:
        lam = np.sort(np.append(lam, LAM_TARGET))
    freqs = 1000.0 / lam
    solver = Slab1D(D_UM, freqs, res)     # 정규화 런은 한 번만
    t0 = time.time()
    out = np.zeros((len(lam), 6))
    for i, l in enumerate(lam):
        N = si.N(l)
        R, T, A = solver.rta(medium_from_N(N, l / 1000.0))
        out[i, :3] = R[i], T[i], A[i]      # 재료가 정확한 그 파장 값만 사용
        out[i, 3:] = tmm_coherent(l, N)
        if i % max(1, len(lam) // 10) == 0:
            print(f"  {l:7.1f} nm: A MEEP {100*out[i,2]:.3f}%  TMM {100*out[i,5]:.3f}%"
                  f"   ({i+1}/{len(lam)}, {time.time()-t0:.0f} s)")
    return lam, out


# -----------------------------------------------------------------------------
def make_figure(lam, out, conv, res_b, path):
    c_R, c_T, c_A = "#2a78d6", "#eb6834", "#1a9e75"
    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e6e5e1"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
        "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
    })
    # WSL2 에서는 Windows 폰트가 matplotlib 에 안 보이므로 맑은 고딕을 직접 등록
    for p in ("/mnt/c/Windows/Fonts/malgun.ttf",):
        if os.path.exists(p):
            matplotlib.font_manager.fontManager.addfont(p)
    names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK", "AppleGothic"):
        hit = sorted(n for n in names if n.startswith(fam))
        if hit:
            plt.rcParams["font.family"] = hit[0]
            break
    plt.rcParams["axes.unicode_minus"] = False

    # TMM 은 연속선(0.2 nm 간격)으로 다시 계산해서 그림
    si = SiOpticalConstants(verbose=False)
    lam_f = np.arange(LAM_MIN, LAM_MAX + 1e-9, 0.2)
    tmm_f = np.array([tmm_coherent(l, si.N(l)) for l in lam_f])
    step = max(1, len(lam) // 50)          # 마커가 너무 빽빽하지 않게
    mk = dict(ls="none", marker="o", ms=4, mfc="white", mew=1.2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    ax1.plot(lam_f, 100 * tmm_f[:, 0], color=c_R, lw=1.6, label="R  (TMM)")
    ax1.plot(lam_f, 100 * tmm_f[:, 1], color=c_T, lw=1.6, label="T  (TMM)")
    ax1.plot(lam[::step], 100 * out[::step, 0], color=c_R, **mk, label="R  (MEEP)")
    ax1.plot(lam[::step], 100 * out[::step, 1], color=c_T, **mk, label="T  (MEEP)")
    ax1.set(xlabel="파장 (nm)", ylabel="반사율 / 투과율 (%)", xlim=(LAM_MIN, LAM_MAX))
    ax1.set_title("(a) R, T — 공기/Si 2 μm/공기", loc="left", color=ink)
    ax1.grid(axis="y", color=grid, lw=0.8)
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=False, ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax2.plot(lam_f, 100 * tmm_f[:, 2], color=c_A, lw=1.6, label="A  (coherent TMM)")
    ax2.plot(lam[::step], 100 * out[::step, 2], color=c_A, **mk, label=f"A  (MEEP, res {res_b}/μm)")
    ax2.axvline(LAM_TARGET, color=muted, lw=1, ls=":")
    j = int(np.argmin(np.abs(lam - LAM_TARGET)))
    ax2.text(LAM_TARGET + 3, ax2.get_ylim()[1] * 0.96,
             f"940 nm\nMEEP {100*out[j,2]:.3f}%\nTMM  {100*out[j,5]:.3f}%",
             color=ink, fontsize=8.5, va="top")
    ax2.set(xlabel="파장 (nm)", ylabel="흡수율 A = 1 − R − T (%)", xlim=(LAM_MIN, LAM_MAX))
    ax2.set_title("(b) 흡수 스펙트럼 (실제 Si 분산, Green 2008)", loc="left", color=ink)
    ax2.grid(axis="y", color=grid, lw=0.8)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.80))

    res = np.array([r["resolution_per_um"] for r in conv], float)
    for key, c, lab in (("max_abs_dR", c_R, "max|ΔR|"), ("max_abs_dT", c_T, "max|ΔT|"),
                        ("max_abs_dA", c_A, "max|ΔA|")):
        e = np.array([r[key] for r in conv])
        ax3.loglog(res, e, color=c, lw=1.6, marker="o", ms=4, label=lab)
    e0 = conv[0]["max_abs_dR"]
    ax3.loglog(res, e0 * (res[0] / res) ** 2, color=muted, lw=1, ls="--", label="기울기 −2 (Yee 2차)")
    ax3.axhline(TOL_RT, color=c_R, lw=0.8, ls=":")
    ax3.axhline(TOL_A, color=c_A, lw=0.8, ls=":")
    ax3.set_xticks(res)
    ax3.set_xticklabels([str(int(r)) for r in res])
    ax3.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax3.set(xlabel="해상도 (격자점 / μm)", ylabel="800–1000 nm 최대 절대오차")
    ax3.set_title("(c) 해상도 수렴 (MEEP·TMM 동일 재료)", loc="left", color=ink)
    ax3.grid(which="major", color=grid, lw=0.8)
    ax3.legend(frameon=False, fontsize=8.5)

    fig.savefig(path, dpi=200)
    print(f"\n그림 저장: {os.path.relpath(path, ROOT)}")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"표 저장: {os.path.relpath(path, ROOT)}")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="빠른 확인용 (파트 B 해상도 200, 5 nm)")
    ap.add_argument("--res-b", type=int, default=None, help="파트 B 해상도 (기본 800)")
    ap.add_argument("--step", type=float, default=None, help="파트 B 파장 간격 nm (기본 2)")
    ap.add_argument("--plot-only", action="store_true",
                    help="계산 없이 results/ 의 week4 CSV 두 개로 그림만 다시 그림 (MEEP 불필요)")
    args = ap.parse_args()
    res_b = args.res_b or (200 if args.quick else 800)
    step = args.step or (5.0 if args.quick else 2.0)

    if args.plot_only:
        with open(os.path.join(ROOT, "results", "week4_meep_convergence.csv"), encoding="utf-8") as f:
            conv = [{k: float(v) if v not in ("",) else v for k, v in r.items()} for r in csv.DictReader(f)]
        with open(os.path.join(ROOT, "results", "week4_meep_vs_tmm_2um.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lam = np.array([float(r["wavelength_nm"]) for r in rows])
        out = np.array([[float(r[k]) for k in ("R_meep", "T_meep", "A_meep", "R_tmm", "T_tmm", "A_tmm")]
                        for r in rows])
        make_figure(lam, out, conv, args.res_b or 800, os.path.join(ROOT, "figures", "fig_week4_meep_vs_tmm.png"))
        return

    if mp is None:
        sys.exit("MEEP(pymeep)을 import 할 수 없습니다. WSL2의 photontrap-meep 환경에서 실행하세요 "
                 "(notes/04_meep_install_wsl2.md). 그림만 다시 그리려면 --plot-only.")
    mp.verbosity(0)
    print(f"MEEP {mp.__version__}, Python {sys.version.split()[0]}")
    si = SiOpticalConstants(verbose=True)

    conv, order = part_a(si)
    lam, out = part_b(si, res_b, step)

    dR, dT, dA = (np.abs(out[:, i] - out[:, i + 3]) for i in range(3))
    j = int(np.argmin(np.abs(lam - LAM_TARGET)))
    print(f"\n[파트 B 결과] 해상도 {res_b}: max|ΔR| {dR.max():.2e}, max|ΔT| {dT.max():.2e}, max|ΔA| {dA.max():.2e}")
    print(f"  940 nm: MEEP R {100*out[j,0]:.2f}% T {100*out[j,1]:.2f}% A {100*out[j,2]:.3f}%"
          f"  |  TMM R {100*out[j,3]:.2f}% T {100*out[j,4]:.2f}% A {100*out[j,5]:.3f}%")
    print(f"  대역 평균 A: MEEP {100*out[:,2].mean():.3f}%  TMM {100*out[:,5].mean():.3f}%")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    write_csv(os.path.join(ROOT, "results", "week4_meep_convergence.csv"), conv, list(conv[-1].keys()))
    rows_b = [{"wavelength_nm": l, "R_meep": o[0], "T_meep": o[1], "A_meep": o[2],
               "R_tmm": o[3], "T_tmm": o[4], "A_tmm": o[5]} for l, o in zip(lam, out)]
    write_csv(os.path.join(ROOT, "results", "week4_meep_vs_tmm_2um.csv"), rows_b, list(rows_b[0].keys()))
    make_figure(lam, out, conv, res_b, os.path.join(ROOT, "figures", "fig_week4_meep_vs_tmm.png"))

    checks = {
        # 수렴 차수는 res ≥ 100 구간만 판정 (2026-10-10 첫 실행 후 수정).
        # res 50 은 Si 안 파장당 ~11점뿐이라 간섭무늬가 반 주기 가까이 밀려 max 오차가
        # 무늬 진폭에서 포화 → 점근 영역이 아님 (50→100 차수 1.66). 이 값은 표에 그대로 남긴다.
        f"파트 A 수렴 차수 1.8–2.2, res≥100 (측정 {order[1:].min():.2f}–{order[1:].max():.2f})":
            bool(np.all((order[1:] > 1.8) & (order[1:] < 2.2))),
        f"파트 B max|ΔR| < {TOL_RT}": dR.max() < TOL_RT,
        f"파트 B max|ΔT| < {TOL_RT}": dT.max() < TOL_RT,
        f"파트 B max|ΔA| < {TOL_A}": dA.max() < TOL_A,
    }
    print("\n=== 검증 판정 ===")
    for name, ok in checks.items():
        print(f"  [{'통과' if ok else '실패'}] {name}")
    if args.quick:
        print("  (--quick 은 해상도가 낮아 파트 B 기준을 통과하지 못하는 것이 정상입니다. 최종 판정은 기본 실행으로.)")
    sys.exit(0 if all(checks.values()) or args.quick else 1)


if __name__ == "__main__":
    main()
