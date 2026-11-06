"""
plot_gao_reproduce.py — 7주차 비교 그림: 6주차 구조 → 육각 격자 → Gao 등가 슬랩 (800–900 nm)

입력: results/week6_nanohole_bbtest_res{R}_nf101_pml1*.csv (w6sq), results/week7_bb_{case}_res{R}_nf101*.csv
출력: figures/fig_week7_gao_reproduce.png, results/week7_band_average.csv
MEEP 불필요 (Windows photontrap 환경에서도 실행 가능)
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.join(os.path.dirname(__file__), "..")
for f in ("Noto Sans CJK KR", "Malgun Gothic", "NanumGothic"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

LABEL = {"w6sq": "6주차: 정사각, h 0.5, t 2.0", "hex05": "육각, h 0.5, t 2.0",
         "gao700cyl": "Gao 등가: 육각, h 2.0, t 2.65", "gao1300cyl": "Gao 1300/2000 원기둥"}
T_OF = {"w6sq": 2.0, "hex05": 2.0, "gao700cyl": 2.65, "gao1300cyl": 2.65}


def load(case, res):
    pat = (f"week6_nanohole_bbtest_res{res}_nf101_pml1*.csv" if case == "w6sq"
           else f"week7_bb_{case}_res{res}_nf101*.csv")   # PC 에서 구간별로 나눠 돌린 _a, _b 포함
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", pat))):
        if "regress" in f:
            continue
        rows += [r for r in csv.DictReader(open(f, encoding="utf-8")) if int(r["band"]) in (0, 1)]
    bands = {}
    for r in rows:
        bands.setdefault(int(r["band"]), []).append((float(r["lambda_nm"]), float(r["A"]), float(r["A_flat_tmm"])))
    if sorted(bands) != [0, 1]:
        return None
    return {b: np.array(sorted(v)) for b, v in bands.items()}


def band_avg(arr, col):
    lam = arr[:, 0]
    return np.trapezoid(arr[:, col], lam) / (lam[-1] - lam[0])


def main():
    cases = ["w6sq", "hex05", "gao700cyl", "gao1300cyl"]
    table, best = [], {}
    for c in cases:
        for res in (30, 40, 45, 50, 60):
            d = load(c, res)
            if d is None:
                continue
            a = [band_avg(d[b], 1) for b in (0, 1)]
            t = [band_avg(d[b], 2) for b in (0, 1)]
            A, At = np.mean(a), np.mean(t)
            table.append({"case": c, "resolution": res, "A_800_850": a[0], "A_850_900": a[1], "A_800_900": A,
                          "A_flat_tmm_800_900": At, "enhancement": A / At,
                          "A_max": max(d[0][:, 1].max(), d[1][:, 1].max())})
            best[c] = (res, d)
    if not table:
        sys.exit("결과 CSV 가 없습니다.")
    out = os.path.join(ROOT, "results", "week7_band_average.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"{'케이스':12s} res  800–850  850–900  800–900  평면TMM  배율  최대A")
    for r in table:
        print(f"{r['case']:12s} {r['resolution']:3d}  {100*r['A_800_850']:6.2f}  {100*r['A_850_900']:6.2f}  "
              f"{100*r['A_800_900']:6.2f}  {100*r['A_flat_tmm_800_900']:6.2f}  {r['enhancement']:4.2f}  {100*r['A_max']:5.1f}")

    fig, axes = plt.subplots(len(best), 1, figsize=(8, 2.3 * len(best) + 0.6), sharex=True, squeeze=False)
    for ax, (c, (res, d)) in zip(axes[:, 0], best.items()):
        for b in (0, 1):
            ax.plot(d[b][:, 0], 100 * d[b][:, 1], color="C0", lw=0.9, label="나노홀 (MEEP)" if b == 0 else None)
            ax.plot(d[b][:, 0], 100 * d[b][:, 2], color="0.4", lw=1.2, ls="--",
                    label=f"평면 Si {T_OF[c]:g} μm (TMM)" if b == 0 else None)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        r = [x for x in table if x["case"] == c and x["resolution"] == res][0]
        ax.set_title(f"{LABEL[c]}  (해상도 {res}) — 800–900 nm 평균 {100*r['A_800_900']:.1f}% "
                     f"(평면 대비 {r['enhancement']:.1f}배)", fontsize=9.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("흡수 A (%)")
        ax.axhline(80, color="C3", lw=0.7, ls=":")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("파장 (nm)")
    axes[-1, 0].text(801, 82, "Gao Fig. 3c: 최고 흡수 > 80%", color="C3", fontsize=8)
    fig.suptitle("7주차: Gao 2017 구조 파라미터 재현 (Si 슬랩, 뒷면 공기, Ex 수직입사)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(ROOT, "figures", "fig_week7_gao_reproduce.png")
    fig.savefig(path, dpi=200)
    print(f"저장: {path}\n저장: {out}")


if __name__ == "__main__":
    main()
