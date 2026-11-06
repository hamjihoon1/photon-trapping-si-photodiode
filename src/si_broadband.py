"""
si_broadband.py — 6주차: Si(Green 2008)를 800–1000 nm 전체에서 맞춘 MEEP 분산 재료 (Lorentz 항 합)

왜 필요한가
  meep_1d.medium_from_N()는 파장 한 점에서만 정확한 재료(ε_∞ + D_conductivity)라서, 스펙트럼을 얻으려면
  파장마다 따로 실행해야 했다 (3D 에서 11점 = 11회 실행). 넓은 대역에서 ε(f)를 맞춘 분산 재료를 쓰면
  한 번의 실행으로 모니터에 기록한 모든 주파수의 R, T 를 얻는다 (MEEP 의 표준 방식).

모델 (MEEP 규약, exp(−iωt), Im ε > 0 이 손실). MEEP 문서 doc/docs/Materials.md "Material Dispersion":
  ε(f) = ε_∞ + Σ_n σ_n f_n² / (f_n² − f² − i f γ_n)
  mp.LorentzianSusceptibility(frequency=f_n, gamma=γ_n, sigma=σ_n)
  f = 1/λ[μm] (MEEP 단위 a = 1 μm)

피팅 대상: data/Si_Green2008.csv (si_data.SiOpticalConstants), 같은 Si 를 MEEP 규약으로 ε = (n² − k²) + i·2nk
  - 흡수는 Im ε 에 비례하므로 Im ε 는 상대오차로 맞춘다 (800 nm 0.040 → 1000 nm 0.0036, 약 11배 차이)
  - 물리 모델이 아니라 이 대역 안에서만 쓰는 수치 피팅이다. 대역 밖(특히 더 짧은 파장)에서는 맞지 않는다.
  - 결과 검증: scripts/validate_si_broadband.py (ε 오차, TMM 흡수 비교, MEEP 1D 한 번 실행 vs TMM)
"""
import numpy as np
from scipy.optimize import least_squares

from si_data import SiOpticalConstants

LAM_MIN_NM, LAM_MAX_NM = 790.0, 1010.0   # 피팅 대역 (작업 대역 800–1000 nm 보다 양쪽 10 nm 넓게)


def eps_target(si, lam_nm):
    """Green 2008 의 ε (MEEP 규약, Im > 0 = 손실)."""
    N = np.array([si.N(l) for l in np.atleast_1d(lam_nm)])
    n, k = N.real, -N.imag
    return (n * n - k * k) + 1j * (2 * n * k)


def eps_model(f, eps_inf, poles):
    """poles = [(f_n, γ_n, σ_n), ...]. MEEP LorentzianSusceptibility 정의와 같은 식."""
    f = np.asarray(f, dtype=float)
    e = np.full(f.shape, eps_inf, dtype=complex)
    for fn, gn, sn in poles:
        e += sn * fn * fn / (fn * fn - f * f - 1j * f * gn)
    return e


def _unpack(x, npole):
    eps_inf = x[0]
    poles = [(np.exp(x[1 + 3 * i]), np.exp(x[2 + 3 * i]), np.exp(x[3 + 3 * i])) for i in range(npole)]
    return eps_inf, poles


def fit(si=None, npole=3, n_pts=111, seed=0, n_trials=150, lam_range=(LAM_MIN_NM, LAM_MAX_NM), sigma_max=None):
    """
    Lorentz 항 npole 개로 피팅. 반환: dict(eps_inf, poles, max_rel_err_im, max_abs_err_re, lam_nm)
    파라미터는 log 로 풀어서 f_n, γ_n, σ_n > 0 (수동 매질, MEEP 안정 조건) 을 보장한다.
    """
    si = si or SiOpticalConstants(verbose=False)
    lam = np.linspace(lam_range[0], lam_range[1], n_pts)
    f = 1000.0 / lam
    tgt = eps_target(si, lam)

    def resid(x):
        e = eps_model(f, *_unpack(x, npole))
        r_re = (e.real - tgt.real) / 0.002          # Re ε 0.002 ≈ n 오차 3e-4
        r_im = (e.imag - tgt.imag) / (0.003 * tgt.imag)   # Im ε 상대 0.3%
        return np.concatenate([r_re, r_im])

    # 제약 (MEEP 안정 조건 + 의미 없는 항 방지, 2026-11-06):
    #   ε_∞ ≥ 1 — 제약 없이 맞추면 ε_∞ = −16 + 큰 고주파 항으로 상쇄되는 해가 나오고, MEEP 실행이 NaN 으로 발산했다
    #   f_n ≥ 0.8 (λ ≤ 1250 nm) — f ≈ 0.01 같은 사실상 정적인 항 방지
    #   sigma_max (선택): 모든 Lorentz 항의 세기 상한. ε 의 대부분을 ε_∞ 에 두기 위해 사용 —
    #     MEEP 의 경계면 평활화(subpixel averaging)는 ε_∞ 에만 적용되고 분산 항에는 적용되지 않으므로,
    #     큰 분산 항이 있으면 해상도에 대해 1차 수렴으로 떨어진다 (1D 해상도 200–1600 에서 확인, 2026-11-06)
    smax = np.log(sigma_max) if sigma_max else np.log(100.0)
    lb = [1.0] + [np.log(0.8), np.log(1e-9), np.log(1e-9)] * npole
    ub = [14.0] + [np.log(10.0), np.log(10.0), smax] * npole
    rng = np.random.default_rng(seed)
    best = None
    # 초기값: 흡수 꼬리를 만드는 항은 대역 바로 위(f 1.26–2.0, λ 500–790 nm)에 공진, 폭은 좁게~넓게.
    # 마지막 항은 실수부(n ≈ 3.6)를 만드는 높은 주파수 항 (Si 직접 밴드갭 쪽, f ~ 2.5–4).
    for trial in range(n_trials):
        x0 = [rng.uniform(1.5, 8.0) if not sigma_max else rng.uniform(12.0, 13.9)]
        for i in range(npole):
            if sigma_max:
                fn = (1000.0 / lam_range[0]) * rng.uniform(1.01, 2.5)
                gn, sn = 10 ** rng.uniform(-2.5, 0.5), 10 ** rng.uniform(-4, np.log10(sigma_max) - 0.01)
            elif i < npole - 1:
                fn, gn, sn = (1000.0 / lam_range[0]) * rng.uniform(1.01, 1.6), 10 ** rng.uniform(-2.5, 0), 10 ** rng.uniform(-4, -1)
            else:
                fn, gn, sn = rng.uniform(2.5, 4.0), 10 ** rng.uniform(-3, -1), rng.uniform(3.0, 11.0)
            x0 += [np.log(fn), np.log(gn), np.log(sn)]
        try:
            r = least_squares(resid, x0, method="trf", max_nfev=5000, bounds=(lb, ub))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    eps_inf, poles = _unpack(best.x, npole)
    e = eps_model(f, eps_inf, poles)
    return {"eps_inf": float(eps_inf), "poles": [tuple(map(float, p)) for p in poles],
            "max_rel_err_im": float(np.max(np.abs(e.imag - tgt.imag) / tgt.imag)),
            "max_abs_err_re": float(np.max(np.abs(e.real - tgt.real))),
            "lam_nm": tuple(lam_range)}


def N_from_fit(params, lam_nm):
    """피팅 ε → TMM 규약 N = n − ik (TMM 으로 피팅 품질을 흡수율로 확인할 때)."""
    e = eps_model(1000.0 / np.atleast_1d(lam_nm), params["eps_inf"], params["poles"])
    Nm = np.sqrt(e)                 # 주값: Im ≥ 0 (MEEP 규약 손실)
    return np.conj(Nm)              # TMM 규약 n − ik


def meep_medium(params):
    import meep as mp
    return mp.Medium(epsilon=params["eps_inf"],
                     E_susceptibilities=[mp.LorentzianSusceptibility(frequency=fn, gamma=gn, sigma=sn)
                                         for fn, gn, sn in params["poles"]])
