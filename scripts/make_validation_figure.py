"""
make_validation_figure.py — 2주차 결과물: "검증을 통과한 TMM 코드와 검증 그래프"

그림 2개를 생성한다.
  Fig A: 반무한 흡수 매질 극한 수렴 테스트
         두께가 커질수록 R -> 단일계면 Fresnel 반사율로 수렴하고 T -> 0,
         A -> (1 - R_front)로 수렴함을 보여준다 (Beer-Lambert 극한과의 정합성).
  Fig B: 공기/Si(2um)/공기 구조의 800-1000nm 흡수 스펙트럼 (TMM, 앞뒤 반사 포함)
         vs 1주차에 계산한 순수 Beer-Lambert 근사(반사 무시, 940nm 단일점)와 비교.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import glob
for _p in glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic*.ttf"):
    fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from tmm import tmm_normal_incidence
from si_data import SiOpticalConstants

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Fig A: 두께 증가에 따른 R, T, A 수렴 (반무한 극한 검증)
# ---------------------------------------------------------------
si = SiOpticalConstants()
lam = 940.0
N_si = si.N(lam)
n_air = 1.0 + 0j

r_front = (n_air - N_si) / (n_air + N_si)
R_front_limit = abs(r_front) ** 2

thicknesses = np.logspace(np.log10(5), np.log10(200000), 60)  # nm, 5nm ~ 200um
Rs, Ts, As = [], [], []
for d in thicknesses:
    R, T, A = tmm_normal_incidence(lam, [N_si], [d], n_air, n_air)  # 앞뒤 다 공기(freestanding)
    Rs.append(R); Ts.append(T); As.append(A)
Rs, Ts, As = map(np.array, (Rs, Ts, As))

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.semilogx(thicknesses / 1000, Rs, label="R (TMM)", color="tab:blue")
ax.semilogx(thicknesses / 1000, Ts, label="T (TMM)", color="tab:orange")
ax.semilogx(thicknesses / 1000, As, label="A (TMM)", color="tab:green")
ax.semilogx(thicknesses / 1000, Rs + Ts + As, "--", color="k", lw=1, label="R+T+A (에너지 보존)")
ax.axhline(R_front_limit, color="tab:blue", ls=":", lw=1)
ax.text(thicknesses[-1] / 1000 * 0.6, R_front_limit + 0.02,
        f"단일계면 Fresnel R = {R_front_limit:.3f}", fontsize=8, color="tab:blue")
ax.set_xlabel("Si 두께 (μm, log scale)")
ax.set_ylabel("R, T, A")
ax.set_title(f"검증: 두께 증가에 따른 R/T/A 수렴 @ {lam:.0f} nm\n(공기/Si/공기, 반무한 극한에서 R → 단일계면 Fresnel값)")
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
figA_path = os.path.join(FIG_DIR, "figA_validation_convergence.png")
fig.savefig(figA_path, dpi=150)
print("저장:", figA_path)
print(f"두께 200um에서 R={Rs[-1]:.6f} (해석적 극한 {R_front_limit:.6f}), "
      f"T={Ts[-1]:.2e}, max|R+T+A-1|={np.max(np.abs(Rs+Ts+As-1)):.2e}")

# ---------------------------------------------------------------
# Fig B: 공기/Si(2um)/공기 스펙트럼, TMM vs 1주차 Beer-Lambert 근사
# ---------------------------------------------------------------
wavelengths = np.linspace(800, 1000, 201)
d_slab = 2000.0  # nm = 2 um, 1주차 진행 노트의 예비값과 동일 두께

R_spec, T_spec, A_spec = [], [], []
alpha_spec = []  # Beer-Lambert 비교용 흡수계수 [1/cm]
for lam_i in wavelengths:
    N = si.N(lam_i)
    R, T, A = tmm_normal_incidence(lam_i, [N], [d_slab], n_air, n_air)
    R_spec.append(R); T_spec.append(T); A_spec.append(A)
    k = -N.imag  # N = n - ik 이므로 k = -Im(N)
    alpha_cm = 4 * np.pi * k / (lam_i * 1e-7)  # nm -> cm
    alpha_spec.append(alpha_cm)
R_spec, T_spec, A_spec, alpha_spec = map(np.array, (R_spec, T_spec, A_spec, alpha_spec))

# 순수 Beer-Lambert(반사 무시): A_BL = 1 - exp(-alpha * d)
d_cm = d_slab * 1e-7
A_beer_lambert = 1 - np.exp(-alpha_spec * d_cm)

fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
ax2.plot(wavelengths, A_spec * 100, label="TMM 흡수율 (앞뒤 반사 포함)", color="tab:green")
ax2.plot(wavelengths, A_beer_lambert * 100, "--", label="Beer-Lambert 근사 (반사 무시, 1주차 방식)", color="tab:gray")
ax2.plot(wavelengths, R_spec * 100, ":", label="TMM 반사율 R", color="tab:blue", lw=1)
ax2.axvline(940, color="k", lw=0.8, ls=":")
ax2.text(941, 3, "940 nm", fontsize=8)
ax2.set_xlabel("파장 (nm)")
ax2.set_ylabel("비율 (%)")
ax2.set_title(f"평면 Si 슬랩 (두께 {d_slab/1000:.0f} μm) 흡수 스펙트럼\nTMM vs 1주차 Beer-Lambert 근사")
ax2.legend(fontsize=8)
fig2.tight_layout()
figB_path = os.path.join(FIG_DIR, "figB_slab_TMM_vs_BeerLambert.png")
fig2.savefig(figB_path, dpi=150)
print("저장:", figB_path)

idx_940 = np.argmin(np.abs(wavelengths - 940))
print(f"\n940nm, 두께 2um 결과:")
print(f"  TMM: R={R_spec[idx_940]*100:.2f}%, T={T_spec[idx_940]*100:.2f}%, A={A_spec[idx_940]*100:.2f}%")
print(f"  Beer-Lambert만(반사 무시): A={A_beer_lambert[idx_940]*100:.2f}%")
print(f"  1주차 진행노트 기록값: A≈3.59%")
