"""
plot_nanohole_3d_broadband.py — 6주차: 3D 나노홀 넓은 대역 스펙트럼 + 대역 평균 해상도 수렴 (6주차 최종 그림)

입력 (git 미추적, scripts/nanohole_3d.py 로 재생성):
  0.5 nm (구간당 101점, PML 1 μm): results/week6_nanohole_bbtest_res{30,40,50}_nf101_pml1*.csv  ← 기본 지표
  2 nm (구간당 26점):              results/week6_nanohole_broadband_res*.csv                     ← 샘플링 비교용
  PML 2 μm (res 40, 900–950 nm):   results/week6_nanohole_bbtest_res40_nf101_pml2*.csv
  940 nm 한 점 (파장별 재료):       results/week6_nanohole_conv_*.csv, week6_nanohole_symcheck_res*_on.csv
대역 평균 = 구간마다 사다리꼴 적분 평균 → 4구간 단순 평균 (구간 폭이 모두 50 nm)
  샘플링 확인 (res 40): 2 nm 35.08%, 1 nm 35.36%, 0.5 nm 35.36% → 0.5 nm 사용
출력: figures/fig_week6_nanohole_broadband.png, results/week6_nanohole_band_average.csv (미추적)
실행: python scripts/plot_nanohole_3d_broadband.py   (MEEP 불필요)
"""
import collections
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
BAND_NAMES = ["800–850", "850–900", "900–950", "950–1000"]


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def set_font():
    if os.path.exists("/mnt/c/Windows/Fonts/malgun.ttf"):
        matplotlib.font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/malgun.ttf")
    names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Serif CJK KR"):
        if fam in names:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False


def load(pattern):
    by_res = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results", pattern)):
        for r in read(f):
            by_res[int(r["resolution"])].append(r)
    return by_res


def band_mean(rows, bi, key="A"):
    s = sorted([r for r in rows if int(r["band"]) == bi], key=lambda r: float(r["lambda_nm"]))
    lam = np.array([float(r["lambda_nm"]) for r in s])
    y = np.array([float(r[key]) for r in s])
    return float(np.trapezoid(y, lam) / (lam[-1] - lam[0]))


def summarize(by_res):
    table = []
    for res in sorted(by_res):
        rows = by_res[res]
        if sorted({int(r["band"]) for r in rows}) != [0, 1, 2, 3]:
            continue   # 구간이 다 모이지 않은 해상도는 건너뜀
        rec = {"resolution": res}
        for bi in range(4):
            rec[f"A_band{bi}"] = band_mean(rows, bi)
            rec[f"Aflat_tmm_band{bi}"] = band_mean(rows, bi, "A_flat_tmm")
        rec["A_800_1000"] = np.mean([rec[f"A_band{i}"] for i in range(4)])
        rec["Aflat_tmm_800_1000"] = np.mean([rec[f"Aflat_tmm_band{i}"] for i in range(4)])
        rec["enhancement_vs_tmm"] = rec["A_800_1000"] / rec["Aflat_tmm_800_1000"]
        table.append(rec)
    return table


def main():
    set_font()
    fine = load("week6_nanohole_bbtest_res*_nf101_pml1*.csv")
    coarse = load("week6_nanohole_broadband_res*.csv")
    t_fine, t_coarse = summarize(fine), summarize(coarse)
    for lab, t in (("0.5 nm", t_fine), ("2 nm", t_coarse)):
        for rec in t:
            print(f"[{lab}] res {rec['resolution']}: 대역 평균 A {100*rec['A_800_1000']:.2f}%  (구간 " +
                  " / ".join(f"{100*rec[f'A_band{i}']:.2f}" for i in range(4)) +
                  f")  평면 TMM {100*rec['Aflat_tmm_800_1000']:.2f}%  배율 {rec['enhancement_vs_tmm']:.2f}")
    with open(os.path.join(ROOT, "results", "week6_nanohole_band_average.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sampling"] + list(t_fine[0].keys()))
        w.writeheader()
        for lab, t in (("0.5nm", t_fine), ("2nm", t_coarse)):
            for rec in t:
                w.writerow({"sampling": lab, **rec})

    single = {}
    for pat in ("week6_nanohole_symcheck_res*_on.csv", "week6_nanohole_conv_*.csv"):
        for f in glob.glob(os.path.join(ROOT, "results", pat)):
            for r in read(f):
                single[int(r["resolution"])] = 100 * float(r["A"])

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    reses = [rec["resolution"] for rec in t_fine]
    for k, res in enumerate(reses):
        for bi in range(4):
            s = sorted([r for r in fine[res] if int(r["band"]) == bi], key=lambda r: float(r["lambda_nm"]))
            ax[0].plot([float(r["lambda_nm"]) for r in s], [100 * float(r["A"]) for r in s],
                       color=f"C{k}", lw=0.9, label=f"나노홀 해상도 {res}" if bi == 0 else None)
    s = sorted(fine[reses[-1]], key=lambda r: float(r["lambda_nm"]))
    ax[0].plot([float(r["lambda_nm"]) for r in s], [100 * float(r["A_flat_tmm"]) for r in s], "k:", label="평면 Si (TMM)")
    for x in (850, 900, 950):
        ax[0].axvline(x, color="0.85", lw=0.8, zorder=0)
    ax[0].set_xlabel("파장 (nm)")
    ax[0].set_ylabel("흡수율 A (%)")
    ax[0].set_title("3D 나노홀 스펙트럼 (P 1.0, d 0.7, h 0.5 μm, 0.5 nm 간격)")
    ax[0].legend(fontsize=8)

    ax[1].plot(reses, [100 * t["A_800_1000"] for t in t_fine], "o-", lw=2, color="k", label="800–1000 nm 평균 (0.5 nm)")
    if t_coarse:
        ax[1].plot([t["resolution"] for t in t_coarse], [100 * t["A_800_1000"] for t in t_coarse], "s--",
                   color="0.4", lw=1, label="800–1000 nm 평균 (2 nm)")
    for i in range(4):
        ax[1].plot(reses, [100 * t[f"A_band{i}"] for t in t_fine], "o--", lw=1, label=f"{BAND_NAMES[i]} nm")
    if single:
        sr = sorted(r for r in single if r >= 30)
        ax[1].plot(sr, [single[r] for r in sr], "x:", color="0.6", label="940 nm 한 점 (참고)")
    ax[1].set_xlabel("해상도 (격자점/μm)")
    ax[1].set_ylabel("흡수율 A (%)")
    ax[1].set_title("대역 평균의 해상도 수렴 (t_cap 800, 대칭, PML 1 μm)")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_week6_nanohole_broadband.png")
    fig.savefig(out, dpi=150)
    print(f"그림: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
