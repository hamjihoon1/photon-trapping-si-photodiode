"""
plot_grating_side_compare.py — 5주차 재계산: 홈 위치(광원 쪽 vs 뒷면) 비교 그림

입력 (git 미추적, 재계산 필요하면 conv_res_t800.py --side source / validate_grating_2d.py --spectrum-only):
  results/week5_conv_res_t800_source_a.csv, results/week5_conv_res_t800_source_b.csv
  results/week5_grating_spectrum_source.csv
뒷면(back) 수렴값은 2026-11-06 진행 노트 확정값 (사용자 WSL2 실행, 파일은 사용자 PC 에만 있음).
출력: figures/fig_week5_grating_side_compare.png

실행: python scripts/plot_grating_side_compare.py   (MEEP 불필요, Windows photontrap 에서도 됨)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
BACK_CONV = {100: 9.54, 200: 19.74, 300: 23.67, 400: 25.37}   # %, 진행 노트 4절 (홈 뒷면)


def read(name):
    path = os.path.join(ROOT, "results", name)
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
    conv = [r for f in ("week5_conv_res_t800_source_a.csv", "week5_conv_res_t800_source_b.csv")
            for r in read(f) if r["label"] == "grating"]
    src = {int(r["resolution"]): 100 * float(r["A"]) for r in conv}
    spec = read("week5_grating_spectrum_source.csv")
    lam = np.array([float(r["lambda_nm"]) for r in spec])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(lam, [100 * float(r["A"]) for r in spec], "o-", label="그레이팅 (홈 = 광원 쪽)")
    ax[0].plot(lam, [100 * float(r["A_flat_meep"]) for r in spec], "k--", label="평면 Si (MEEP)")
    ax[0].plot(lam, [100 * float(r["A_flat_tmm"]) for r in spec], "k:", label="평면 Si (TMM)")
    ax[0].set_xlabel("파장 (nm)")
    ax[0].set_ylabel("흡수율 A (%)")
    ax[0].set_title("2D 그레이팅 스펙트럼 (Λ=0.8, f=0.6, h=0.5 μm, 해상도 200, 잠정)")
    ax[0].legend(fontsize=8)

    for d, lab, mk in ((src, "홈 = 광원 쪽 (재계산)", "o-"), (BACK_CONV, "홈 = 뒷면 (이전 5주차)", "s--")):
        rs = sorted(d)
        ax[1].plot(rs, [d[r] for r in rs], mk, label=lab)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks([100, 200, 300, 400])
    ax[1].set_xticklabels(["100", "200", "300", "400"])
    ax[1].set_xlabel("해상도 (격자점/μm)")
    ax[1].set_ylabel("940 nm 흡수율 A (%)")
    ax[1].set_title("940 nm 해상도 수렴: 홈 위치 비교 (t_cap 800)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_week5_grating_side_compare.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"그림: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
