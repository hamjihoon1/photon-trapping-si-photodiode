"""
sweep_thickness.py — 3주차: 평면 Si 슬랩 두께 스윕 (베이스라인 정량화)

계획서 3주차: "두께 1–10 μm Si 슬랩의 800–1000 nm 흡수 스펙트럼 계산, 940 nm 흡수율 표 정리"

구조: 공기 / Si(d) / 공기   (뒷면 매질은 아직 미확정 → 2주차와 같은 공기로 둠)
계산 두 가지:
  - coherent  : tmm_normal_incidence  (간섭무늬 포함, 나중에 MEEP과 비교할 값)
  - incoherent: incoherent_slab       (간섭무늬 평균, 실제 소자에 가까운 대표값)
  - Beer-Lambert(반사 무시) 1-exp(-alpha d) 도 참고용으로 저장

출력:
  results/week3_thickness_sweep.npz   스펙트럼 원본 + 메타데이터
  results/week3_940nm_table.csv       940 nm 흡수율 표
  figures/fig_week3_thickness_sweep.png
"""
import sys
import os
import csv
import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tmm import tmm_normal_incidence, incoherent_slab
from si_data import SiOpticalConstants

ROOT = os.path.join(os.path.dirname(__file__), "..")
THICKNESSES_UM = [1, 2, 3, 5, 7, 10]
LAM = np.round(np.arange(800.0, 1000.0 + 1e-9, 0.1), 1)  # 0.1 nm 간격 (10um 무늬간격 ~12nm)
LAM_TARGET = 940.0
BAND = (930.0, 950.0)   # 940 nm 주변 간섭 변동폭을 볼 구간
N_AIR = 1.0


def main():
    si = SiOpticalConstants(verbose=True)
    N_si = np.array([si.N(l) for l in LAM])

    nT, nL = len(THICKNESSES_UM), len(LAM)
    A_coh = np.zeros((nT, nL)); R_coh = np.zeros((nT, nL)); T_coh = np.zeros((nT, nL))
    A_inc = np.zeros((nT, nL)); R_inc = np.zeros((nT, nL)); T_inc = np.zeros((nT, nL))
    A_bl = np.zeros((nT, nL))

    for i, d_um in enumerate(THICKNESSES_UM):
        d = d_um * 1000.0
        for j, (lam, N) in enumerate(zip(LAM, N_si)):
            R_coh[i, j], T_coh[i, j], A_coh[i, j] = tmm_normal_incidence(lam, [N], [d], N_AIR, N_AIR)
            R_inc[i, j], T_inc[i, j], A_inc[i, j] = incoherent_slab(lam, N, d, N_AIR, N_AIR)
            alpha = 4 * np.pi * (-N.imag) / lam
            A_bl[i, j] = 1 - np.exp(-alpha * d)
        print(f"  d = {d_um:>2} um 완료")

    # 에너지 보존 확인 (스윕 전체)
    err = max(np.abs(R_coh + T_coh + A_coh - 1).max(), np.abs(R_inc + T_inc + A_inc - 1).max())
    print(f"  에너지 보존 최대 오차: {err:.1e}")

    # ---- 저장: npz ----
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    np.savez(
        os.path.join(ROOT, "results", "week3_thickness_sweep.npz"),
        wavelength_nm=LAM, thickness_um=np.array(THICKNESSES_UM, float),
        A_coh=A_coh, R_coh=R_coh, T_coh=T_coh,
        A_inc=A_inc, R_inc=R_inc, T_inc=T_inc, A_beer_lambert=A_bl,
        meta=np.array(str({
            "structure": "air / Si(d) / air, normal incidence",
            "si_data": os.path.basename(si.source_path),
            "sign_convention": "N = n - ik",
            "created": datetime.date.today().isoformat(),
            "script": "scripts/sweep_thickness.py",
        })),
    )

    # ---- 940 nm 표 ----
    j940 = int(np.argmin(np.abs(LAM - LAM_TARGET)))
    band = (LAM >= BAND[0]) & (LAM <= BAND[1])
    rows = []
    for i, d_um in enumerate(THICKNESSES_UM):
        rows.append({
            "thickness_um": d_um,
            "A_incoherent_pct": 100 * A_inc[i, j940],
            "A_coherent_940_pct": 100 * A_coh[i, j940],
            "A_coherent_930_950_min_pct": 100 * A_coh[i, band].min(),
            "A_coherent_930_950_max_pct": 100 * A_coh[i, band].max(),
            "A_beer_lambert_pct": 100 * A_bl[i, j940],
        })
    with open(os.path.join(ROOT, "results", "week3_940nm_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    print("\n940 nm 흡수율 표 (공기/Si/공기)")
    print(f"{'두께':>6} | {'비간섭(대표값)':>12} | {'coherent@940':>12} | {'930-950 변동폭':>16} | {'Beer-Lambert':>12}")
    for r in rows:
        print(f"{r['thickness_um']:>4} um | {r['A_incoherent_pct']:>11.2f}% | {r['A_coherent_940_pct']:>11.2f}% | "
              f"{r['A_coherent_930_950_min_pct']:>6.2f}–{r['A_coherent_930_950_max_pct']:>6.2f}% | {r['A_beer_lambert_pct']:>11.2f}%")

    # ---- 940 nm 흡수율 vs 두께 (촘촘한 두께 격자) ----
    d_fine_um = np.round(np.arange(1.0, 10.0 + 1e-9, 0.02), 2)
    lam_band = np.arange(BAND[0], BAND[1] + 1e-9, 0.05)
    N_band = [si.N(l) for l in lam_band]
    N940 = si.N(LAM_TARGET)
    A_inc_fine = np.array([incoherent_slab(LAM_TARGET, N940, d * 1000, 1, 1)[2] for d in d_fine_um])
    band_min = np.zeros_like(d_fine_um); band_max = np.zeros_like(d_fine_um)
    for k, d in enumerate(d_fine_um):
        a = [tmm_normal_incidence(l, [N], [d * 1000], 1, 1)[2] for l, N in zip(lam_band, N_band)]
        band_min[k], band_max[k] = min(a), max(a)

    make_figure(A_inc, A_coh, d_fine_um, A_inc_fine, band_min, band_max, rows)


def make_figure(A_inc, A_coh, d_fine_um, A_inc_fine, band_min, band_max, rows):
    # 두께 = 순서가 있는 크기 → 단일 색상(파랑) 밝은→진한 순차 팔레트
    blues = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e6e5e1"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
        "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
    })
    # 한글 폰트: Windows=맑은 고딕, 없으면 나눔/Noto CJK/Mac 순으로 탐색
    names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK", "AppleGothic"):
        hit = sorted(n for n in names if n.startswith(fam))
        if hit:
            plt.rcParams["font.family"] = hit[0]
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)

    # (a) 비간섭 스펙트럼 (두께별)
    for i, d_um in enumerate(THICKNESSES_UM):
        ax1.plot(LAM, 100 * A_inc[i], color=blues[i], lw=2)
        # 라벨은 선이 잘 벌어져 있는 왼쪽(800 nm 부근)에 둔다
        ax1.text(806, 100 * A_inc[i, 60] + 0.8, f"{d_um} μm", color=ink, va="bottom", fontsize=9)
    ax1.axvline(LAM_TARGET, color=muted, lw=1, ls=":")
    ax1.text(LAM_TARGET + 2, ax1.get_ylim()[1] * 0.97, "940 nm", color=muted, fontsize=9, va="top")
    ax1.set_xlim(800, 1000)
    ax1.set_ylim(0, None)
    ax1.set_xlabel("파장 (nm)")
    ax1.set_ylabel("흡수율 (%)")
    ax1.set_title("(a) 평면 Si 흡수 스펙트럼 (간섭 평균)", loc="left", color=ink)
    ax1.grid(axis="y", color=grid, lw=0.8)

    # (b) 940 nm 흡수율 vs 두께
    ax2.fill_between(d_fine_um, 100 * band_min, 100 * band_max, color="#cde2fb", lw=0,
                     label="간섭 포함 시 변동폭 (930–950 nm)")
    ax2.plot(d_fine_um, 100 * A_inc_fine, color="#256abf", lw=2, label="간섭 평균 (대표값)")
    ax2.plot([r["thickness_um"] for r in rows], [r["A_coherent_940_pct"] for r in rows], "o",
             ms=6, color="#0d366b", mec="white", mew=1.5, label="간섭 포함, 940 nm 한 점")
    last = A_inc_fine[-1] * 100
    ax2.annotate(f"10 μm에서도 {last:.1f}%", (10, last), (8.4, last - 4.5), color=ink, fontsize=9,
                 ha="center", arrowprops=dict(arrowstyle="-", color=muted, lw=0.8))
    ax2.set_xlim(0.8, 10.2)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("Si 두께 (μm)")
    ax2.set_ylabel("940 nm 흡수율 (%)")
    ax2.set_title("(b) 940 nm 흡수율 vs 두께", loc="left", color=ink)
    ax2.grid(axis="y", color=grid, lw=0.8)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    out = os.path.join(ROOT, "figures", "fig_week3_thickness_sweep.png")
    fig.savefig(out, dpi=200)
    print(f"\n그림 저장: {out}")


if __name__ == "__main__":
    main()
