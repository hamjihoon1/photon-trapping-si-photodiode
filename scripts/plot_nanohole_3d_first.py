"""
plot_nanohole_3d_first.py — 6주차 첫 3D 나노홀 그림 (잠정)

입력 (git 미추적, scripts/nanohole_3d.py 로 재생성):
  results/week6_nanohole_spectrum_res40.csv
  results/week6_nanohole_conv_*.csv, results/week6_nanohole_symcheck_res*_on.csv (해상도별 940 nm)
출력: figures/fig_week6_nanohole_first.png
실행: python scripts/plot_nanohole_3d_first.py   (MEEP 불필요)
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")


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
    spec = read(os.path.join(ROOT, "results", "week6_nanohole_spectrum_res40.csv"))
    lam = [float(r["lambda_nm"]) for r in spec]

    conv = {}
    for f in glob.glob(os.path.join(ROOT, "results", "week6_nanohole_symcheck_res*_on.csv")):
        for r in read(f):
            conv[int(r["resolution"])] = 100 * float(r["A"])
    for f in glob.glob(os.path.join(ROOT, "results", "week6_nanohole_conv_*.csv")):
        for r in read(f):
            conv[int(r["resolution"])] = 100 * float(r["A"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(lam, [100 * float(r["A"]) for r in spec], "o-", label="나노홀 (해상도 40)")
    ax[0].plot(lam, [100 * float(r["A_flat_meep"]) for r in spec], "k--", label="평면 Si (MEEP, 해상도 40)")
    ax[0].plot(lam, [100 * float(r["A_flat_tmm"]) for r in spec], "k:", label="평면 Si (TMM)")
    ax[0].set_xlabel("파장 (nm)")
    ax[0].set_ylabel("흡수율 A (%)")
    ax[0].set_title("3D 나노홀 첫 스펙트럼 (P 1.0, d 0.7, h 0.5 μm) — 미수렴 잠정")
    ax[0].legend(fontsize=8)

    rs = sorted(conv)
    ax[1].plot(rs, [conv[r] for r in rs], "o-")
    ax[1].set_xlabel("해상도 (격자점/μm)")
    ax[1].set_ylabel("940 nm 흡수율 A (%)")
    ax[1].set_title("940 nm 해상도 의존성 (대칭 사용, t_cap 800)")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_week6_nanohole_first.png")
    fig.savefig(out, dpi=150)
    print(f"그림: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
