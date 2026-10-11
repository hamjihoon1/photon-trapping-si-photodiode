"""
explore_fdtd_concepts.py — 4주차: FDTD 기본 개념 4가지를 우리 1D 코드로 직접 확인

계획서 4주차 "기본 개념: Yee 격자, 광원, PML, flux monitor"
개념 설명은 notes/04_fdtd_concepts.md. 이 스크립트는 각 개념마다 실험 하나씩:

  (a) Yee 격자  : 수치분산 이론식으로 예측한 간섭무늬 이동량 vs MEEP 에서 관측한 이동량
  (b) 광원      : 가우시안 펄스의 스펙트럼 (MEEP 소스코드의 정의식) vs 실제 입사 플럭스
  (c) PML       : PML 두께를 바꿔 가며 빈 셀에서 생기는 '가짜 반사' 크기
  (d) flux 모니터: 입사장을 빼지 않은 반사 플럭스 vs 빼서 얻은 진짜 반사율

구조·재료는 validate_meep_1d.py 파트 A 와 같다 (공기/Si 2 μm/공기, Si = 940 nm 에서 맞춘 ε_∞+σ_D).
실행 (WSL2 photontrap-meep 환경): python scripts/explore_fdtd_concepts.py   (약 1–2분)
출력: figures/fig_week4_fdtd_concepts.png, 콘솔 숫자 (노트에 기록)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

from tmm import tmm_normal_incidence
from si_data import SiOpticalConstants
from meep_1d import Slab1D, medium_from_N, N_tmm_from_medium

ROOT = os.path.join(os.path.dirname(__file__), "..")
D_UM = 2.0
COURANT = 0.5          # MEEP 기본값 (Python_User_Interface.md: Courant=0.5)


# -----------------------------------------------------------------------------
# (a) Yee 격자 — 수치분산
# -----------------------------------------------------------------------------
def yee_numerical_k(lam_um, n, res, S=COURANT):
    """
    1D Yee 격자(중앙차분, 공간·시간 2차 정확도)의 수치 분산관계 (c = 1, Δt = S·Δx):
        sin(ω Δt / 2) / Δt = (1/n) · sin(k Δx / 2) / (Δx / 2)
    → k_num = (2/Δx) · arcsin[ (n/S) · sin(ω S Δx / 2) ]
    격자가 촘촘해지면 (Δx→0) k_num → n ω = 정확한 k.  오차 ≈ (kΔx)²/24 · (1 − (S/n)²).
    근거: Taflove & Hagness, Computational Electrodynamics (3rd ed.) 4장 수치분산 (표준 결과)
    """
    dx = 1.0 / res
    w = 2 * np.pi / lam_um
    return 2.0 / dx * np.arcsin((n / S) * np.sin(w * S * dx / 2.0))


def peak_position(lam, y):
    """최댓값 주변 3점에 포물선을 맞춰 피크 위치를 격자 간격보다 정밀하게 찾는다."""
    i = int(np.argmax(y))
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    return lam[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (lam[1] - lam[0])


def exp_a(si):
    print("\n(a) Yee 격자: 수치분산 → 간섭무늬 이동")
    mat = medium_from_N(si.N(940.0), 0.94)
    lam = np.arange(945.0, 970.0 + 1e-9, 0.02)        # 958.6 nm 투과 피크 하나를 포함
    f = 1000.0 / lam
    T_tmm = np.array([tmm_normal_incidence(l, [N_tmm_from_medium(mat, ff)], [D_UM * 1000], 1, 1)[1]
                      for l, ff in zip(lam, f)])
    peak_tmm = peak_position(lam, T_tmm)
    n_si = np.sqrt(complex(np.asarray(mat.epsilon(1000.0 / peak_tmm))[0, 0])).real
    rows = []
    for res in (100, 200, 400, 800):
        _, T, _ = Slab1D(D_UM, f, res).rta(mat)
        peak = peak_position(lam, T)
        k_exact = 2 * np.pi * n_si / (peak_tmm / 1000)
        rel = yee_numerical_k(peak_tmm / 1000, n_si, res) / k_exact - 1
        # Fabry–Pérot 공진: 2·k·d = 2πm. k 가 rel 만큼 크면 같은 m 이 λ·(1+rel) 에서 생김
        pred = rel * peak_tmm
        rows.append((res, round(res * peak_tmm / 1000 / n_si, 1), peak - peak_tmm, pred, rel))
        print(f"  res {res:4d} (Si 안 파장당 {rows[-1][1]:6.1f}점): 관측 이동 {peak-peak_tmm:+.3f} nm,"
              f" 이론 예측 {pred:+.3f} nm  (수치 k 상대오차 {rel:.2e})")
    return rows, peak_tmm


# -----------------------------------------------------------------------------
# (b) 광원 — 가우시안 펄스 스펙트럼
# -----------------------------------------------------------------------------
def exp_b():
    print("\n(b) 광원: 가우시안 펄스 스펙트럼")
    lam = np.arange(700.0, 1150.0 + 1e-9, 2.0)
    f = 1000.0 / lam
    s = Slab1D(D_UM, 1000.0 / np.arange(800.0, 1000.0 + 1e-9, 2.0), 200)   # 실제 계산과 같은 펄스
    fcen, fwidth = s.fcen, s.fwidth
    s2 = Slab1D(D_UM, f, 200)
    s2.fcen, s2.fwidth = fcen, fwidth                    # 넓은 대역에서 같은 펄스를 관측
    inc, _ = s2.normalize()
    # MEEP src/sources.cpp gaussian_src_time::fourier_transform: |J(f)| ∝ exp(-½(2π(f−fcen)·w)²), w = 1/fwidth
    model = np.exp(-(2 * np.pi * (f - fcen) / fwidth) ** 2)          # 세기 = 진폭²
    model *= inc.max() / model.max()
    band = (lam >= 800) & (lam <= 1000)
    edge = inc[band].min() / inc.max()
    print(f"  fcen = {fcen:.4f} (λ = {1000/fcen:.1f} nm), fwidth = {fwidth:.4f}")
    print(f"  800–1000 nm 가장자리 입사 세기 = 최대의 {100*edge:.2f}%  (정규화로 나누므로 결과는 정확)")
    return lam, inc / inc.max(), model / inc.max(), fcen, edge


# -----------------------------------------------------------------------------
# (c) PML — 가짜 반사
# -----------------------------------------------------------------------------
def exp_c():
    print("\n(c) PML: 두께별 가짜 반사 (빈 셀, 해상도 200)")
    res, pad = 200, 1.0
    f = 1000.0 / np.arange(800.0, 1000.0 + 1e-9, 2.0)
    fcen = 0.5 * (f.min() + f.max())
    fwidth = max(1.5 * (f.max() - f.min()), 0.2 * fcen)

    def build(dpml):
        L = 2 * (pad + dpml)
        src = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth), component=mp.Ex,
                         center=mp.Vector3(z=-pad + 0.1))]
        sim = mp.Simulation(cell_size=mp.Vector3(0, 0, L), dimensions=1, resolution=res,
                            boundary_layers=[mp.PML(dpml)], sources=src)
        r = sim.add_flux(f, mp.FluxRegion(center=mp.Vector3(z=-0.5)))
        t = sim.add_flux(f, mp.FluxRegion(center=mp.Vector3(z=+0.5)))
        return sim, r, t

    def run(sim):
        sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ex, mp.Vector3(z=0.5), 1e-11))

    # 기준: 아주 두꺼운 PML(4 μm). 같은 위치의 입사장을 빼면 남는 것 = PML 에서 되돌아온 파동
    sim, r, t = build(4.0); run(sim)
    inc = np.array(mp.get_fluxes(t)); ref = sim.get_flux_data(r); sim.reset_meep()
    rows = []
    for dpml in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        sim, r, t = build(dpml); sim.load_minus_flux_data(r, ref); run(sim)
        Rsp = np.abs(-np.array(mp.get_fluxes(r)) / inc).max()
        rows.append((dpml, round(dpml * res), Rsp)); sim.reset_meep()
        print(f"  PML {dpml:4.2f} μm ({round(dpml*res):3d} 격자): 최대 가짜 반사 {Rsp:.1e}")
    return rows


# -----------------------------------------------------------------------------
# (d) flux 모니터 — 입사장 빼기
# -----------------------------------------------------------------------------
def exp_d(si):
    print("\n(d) flux 모니터: 반사 모니터에서 입사장을 빼야 하는 이유 (해상도 400)")
    mat = medium_from_N(si.N(940.0), 0.94)
    lam = np.arange(800.0, 1000.0 + 1e-9, 1.0)
    f = 1000.0 / lam
    s = Slab1D(D_UM, f, 400)
    inc, _ = s.normalize()
    sim, refl, _ = s._build(mat)            # load_minus_flux_data 를 '하지 않은' 런
    s._run(sim)
    R_naive = -np.array(mp.get_fluxes(refl)) / inc
    sim.reset_meep()
    R, T, _ = s.rta(mat)
    R_tmm = np.array([tmm_normal_incidence(l, [N_tmm_from_medium(mat, ff)], [D_UM * 1000], 1, 1)[0]
                      for l, ff in zip(lam, f)])
    print(f"  빼지 않음: 'R' 범위 {R_naive.min():+.3f} ~ {R_naive.max():+.3f}  (음수 = 입사+반사가 섞인 순 플럭스)")
    print(f"  뺌      : R 범위 {R.min():.3f} ~ {R.max():.3f},  TMM 과 최대차 {np.abs(R-R_tmm).max():.1e}")
    print(f"  확인: 빼지 않은 값 = −(1 − R) 이어야 함 → 최대차 {np.abs(R_naive + (1 - R)).max():.1e}")
    return lam, R_naive, R, R_tmm


# -----------------------------------------------------------------------------
def make_figure(a, b, c, d, path):
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1a9e75"
    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e6e5e1"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
        "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
    })
    for p in ("/mnt/c/Windows/Fonts/malgun.ttf",):
        if os.path.exists(p):
            matplotlib.font_manager.fontManager.addfont(p)
    names = {fn.name for fn in matplotlib.font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK", "AppleGothic"):
        hit = sorted(n for n in names if n.startswith(fam))
        if hit:
            plt.rcParams["font.family"] = hit[0]
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)

    rows_a, peak_tmm = a
    res = np.array([r[0] for r in rows_a], float)
    obs = np.array([r[2] for r in rows_a]); pred = np.array([r[3] for r in rows_a])
    x = ax[0, 0]
    x.loglog(res, pred, color=muted, lw=1.4, ls="--", label="이론 예측 (Yee 수치분산식)")
    x.loglog(res, obs, color=c1, lw=0, marker="o", ms=7, label="MEEP 관측")
    for r_, o_ in zip(res, obs):
        x.annotate(f"{o_:.2f} nm", (r_, o_), textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=ink)
    x.set_xticks(res); x.set_xticklabels([str(int(r)) for r in res])
    x.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    x.set(xlabel="해상도 (격자점 / μm)", ylabel="간섭무늬 이동량 (nm, 긴 파장 쪽)")
    x.set_title(f"(a) Yee 격자: 수치분산이 무늬를 민다 ({peak_tmm:.1f} nm 피크)", loc="left", color=ink)
    x.grid(which="major", color=grid, lw=0.8); x.legend(frameon=False, fontsize=9)

    lam, inc, model, fcen, edge = b
    x = ax[0, 1]
    x.axvspan(800, 1000, color="#cde2fb", lw=0, label="측정 대역 800–1000 nm")
    x.plot(lam, model, color=muted, lw=1.4, ls="--", label="정의식 exp[−(2πΔf/fwidth)²]")
    x.plot(lam, inc, color=c2, lw=2, label="MEEP 입사 플럭스")
    x.set(xlabel="파장 (nm)", ylabel="입사 세기 (최대 = 1)", ylim=(0, 1.08))
    x.set_title(f"(b) 광원: 가우시안 펄스 (가장자리 = 최대의 {100*edge:.1f}%)", loc="left", color=ink)
    x.grid(axis="y", color=grid, lw=0.8); x.legend(frameon=False, fontsize=9, loc="upper right")

    x = ax[1, 0]
    dp = np.array([r[0] for r in c]); Rs = np.array([r[2] for r in c])
    x.semilogy(dp, Rs, color=c3, lw=1.6, marker="o", ms=6)
    x.axvline(1.0, color=muted, lw=1, ls=":")
    x.text(1.03, Rs.max(), "validate_meep_1d.py\n사용값 1 μm", fontsize=8.5, color=ink, va="top")
    x.set(xlabel="PML 두께 (μm)", ylabel="최대 가짜 반사율")
    x.set_title("(c) PML: 1D 수직입사에선 얇아도 반사가 거의 없음", loc="left", color=ink)
    x.grid(which="major", color=grid, lw=0.8)

    lam, Rn, R, Rt = d
    x = ax[1, 1]
    x.plot(lam, 100 * Rt, color=c1, lw=1.6, label="R (TMM)")
    x.plot(lam[::5], 100 * R[::5], color=c1, ls="none", marker="o", ms=4, mfc="white", mew=1.2,
           label="R (MEEP, 입사장 뺌)")
    x.plot(lam, 100 * Rn, color=c2, lw=1.6, label="입사장 안 뺀 반사 모니터 값")
    x.axhline(0, color=muted, lw=0.8)
    x.set(xlabel="파장 (nm)", ylabel="반사율 (%)")
    x.set_title("(d) flux 모니터: 입사장을 빼야 반사율이 된다", loc="left", color=ink)
    x.grid(axis="y", color=grid, lw=0.8)
    x.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    fig.savefig(path, dpi=200)
    print(f"\n그림 저장: {os.path.relpath(path, ROOT)}")


def main():
    mp.verbosity(0)
    print(f"MEEP {mp.__version__}, Python {sys.version.split()[0]}")
    si = SiOpticalConstants(verbose=False)
    a = exp_a(si)
    b = exp_b()
    c = exp_c()
    d = exp_d(si)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    make_figure(a, b, c, d, os.path.join(ROOT, "figures", "fig_week4_fdtd_concepts.png"))


if __name__ == "__main__":
    main()
