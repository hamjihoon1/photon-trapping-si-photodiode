"""
plot_nanohole_3d_broadband.py — 6주차: 3D 나노홀 넓은 대역 스펙트럼 + 대역 평균 해상도 수렴

입력 (git 미추적, scripts/nanohole_3d.py --broadband RES 로 재생성): results/week6_nanohole_broadband_res*.csv
      (선택) results/week6_nanohole_conv_*.csv, week6_nanohole_symcheck_res*_on.csv — 940 nm 한 점 비교용
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


def main():
    set_font()
    by_res = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results", "week6_nanohole_broadband_res*.csv")):
        for r in read(f):
            by_res[int(r["resolution"])].append(r)
    reses = sorted(by_res)

    # 구간 평균 (구간 경계 파장은 두 구간에 모두 있으므로 구간별로 평균한 뒤 4구간 단순 평균 = 같은 폭 가중)
    table = []
    for res in reses:
        rec = {"resolution": res}
        for bi in range(4):
            rs = [r for r in by_res[res] if int(r["band"]) == bi]
            rec[f"A_band{bi}"] = np.mean([float(r["A"]) for r in rs])
            rec[f"Aflat_meep_band{bi}"] = np.mean([float(r["A_flat_meep"]) for r in rs])
            rec[f"Aflat_tmm_band{bi}"] = np.mean([float(r["A_flat_tmm"]) for r in rs])
        rec["A_800_1000"] = np.mean([rec[f"A_band{i}"] for i in range(4)])
        rec["Aflat_meep_800_1000"] = np.mean([rec[f"Aflat_meep_band{i}"] for i in range(4)])
        rec["Aflat_tmm_800_1000"] = np.mean([rec[f"Aflat_tmm_band{i}"] for i in range(4)])
        rec["enhancement_vs_tmm"] = rec["A_800_1000"] / rec["Aflat_tmm_800_1000"]
        table.append(rec)
        print(f"res {res}: 대역 평균 A {100*rec['A_800_1000']:.2f}%  (구간 " +
              " / ".join(f"{100*rec[f'A_band{i}']:.2f}" for i in range(4)) +
              f")  평면 TMM {100*rec['Aflat_tmm_800_1000']:.2f}%  배율 {rec['enhancement_vs_tmm']:.2f}")
    out_csv = os.path.join(ROOT, "results", "week6_nanohole_band_average.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # 940 nm 한 점 (파장별 conductivity 계산) — 비교용
    single = {}
    for pat in ("week6_nanohole_symcheck_res*_on.csv", "week6_nanohole_conv_*.csv"):
        for f in glob.glob(os.path.join(ROOT, "results", pat)):
            for r in read(f):
                single[int(r["resolution"])] = 100 * float(r["A"])

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for res in reses:
        for bi in range(4):
            rs = sorted([r for r in by_res[res] if int(r["band"]) == bi], key=lambda r: float(r["lambda_nm"]))
            ax[0].plot([float(r["lambda_nm"]) for r in rs], [100 * float(r["A"]) for r in rs],
                       color=f"C{reses.index(res)}", lw=1.2, label=f"나노홀 해상도 {res}" if bi == 0 else None)
    rs = sorted([r for r in by_res[reses[-1]]], key=lambda r: float(r["lambda_nm"]))
    ax[0].plot([float(r["lambda_nm"]) for r in rs], [100 * float(r["A_flat_tmm"]) for r in rs], "k:", label="평면 Si (TMM)")
    for x in (850, 900, 950):
        ax[0].axvline(x, color="0.85", lw=0.8, zorder=0)
    ax[0].set_xlabel("파장 (nm)")
    ax[0].set_ylabel("흡수율 A (%)")
    ax[0].set_title("3D 나노홀 스펙트럼 (P 1.0, d 0.7, h 0.5 μm, 2 nm 간격)")
    ax[0].legend(fontsize=8)

    ax[1].plot(reses, [100 * t["A_800_1000"] for t in table], "o-", lw=2, color="k", label="800–1000 nm 평균")
    for i in range(4):
        ax[1].plot(reses, [100 * t[f"A_band{i}"] for t in table], "o--", lw=1, label=f"{BAND_NAMES[i]} nm 평균")
    if single:
        sr = sorted(r for r in single if r >= 30)
        ax[1].plot(sr, [single[r] for r in sr], "x:", color="0.5", label="940 nm 한 점 (참고)")
    ax[1].set_xlabel("해상도 (격자점/μm)")
    ax[1].set_ylabel("흡수율 A (%)")
    ax[1].set_title("대역 평균의 해상도 수렴 (t_cap 800, 대칭)")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_week6_nanohole_broadband.png")
    fig.savefig(out, dpi=150)
    print(f"그림: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
