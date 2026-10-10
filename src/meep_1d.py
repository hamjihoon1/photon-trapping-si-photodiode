"""
meep_1d.py — 4주차: MEEP 1D 평면파로 슬랩의 R/T/A 계산 (TMM 검증용)

계획서 4주차: "1D 평면파 → Si 슬랩의 R/T/A 계산, 2–3주차 TMM 결과와 겹쳐 그려 오차 확인"

=== 단위 ===
MEEP 길이 단위 a = 1 μm 로 둔다. 그러면
  - 진공 파장 λ[μm] ↔ MEEP 주파수 f = 1/λ   (800 nm → 1.25, 940 nm → 1.0638, 1000 nm → 1.0)
  - 두께 d[μm] 를 그대로 MEEP 길이로 사용

=== 부호 규약 (TMM과 반대이므로 주의) ===
  - 이 프로젝트 TMM(src/tmm.py): N = n - ik,  시간 의존 exp(+iωt)
  - MEEP: 시간 의존 exp(-iωt), Im ε > 0 이 흡수(손실)
    근거: MEEP 공식 문서 doc/docs/Materials.md — "Meep's definition of ω uses a sign
    convention exp(-iωt)", "imaginary part ... associated with absorption loss ... if it is positive"
  → 같은 물리적 Si 를 MEEP 에 넣으려면  ε_MEEP = (n + ik)^2 = (n^2 - k^2) + i·2nk
    (k ≥ 0 은 그대로 쓰고 부호만 MEEP 규약으로 바꾼다)

=== 손실을 넣는 방법: D_conductivity (σ_D) ===
MEEP 문서(Materials.md, "Conductivity and Complex ε"):
  Im ε = ε_∞ · σ_D / ω = ε_∞ · σ_D / (2π f)
  → 원하는 주파수 f0 에서 ε = ε_r + i ε_i 를 맞추려면
     ε_∞ = ε_r,   σ_D = 2π f0 · ε_i / ε_r
주의: 이 모델의 Im ε 는 1/f 로 변하므로 "f0 한 점에서만" 정확하다. 실제 Si 의 n, k 는
800–1000 nm 에서 크게 변하므로(k 는 한 자릿수 이상), 실제 분산을 쓰는 비교는
파장마다 따로 맞춘 재료로 한 번씩 돌리고 그 파장의 값만 읽는다(validate_meep_1d.py 파트 B).
넓은 대역을 한 번에 돌리는 Lorentz 피팅은 5–6주차(2D/3D) 때 필요하면 별도로 만든다.

=== 계산 방식 (MEEP 표준 R/T 정규화 절차) ===
  1) 정규화 런: 슬랩 없이 같은 셀 → 입사 플럭스, 반사 모니터의 입사파 DFT 저장
  2) 슬랩 런: 반사 모니터에서 입사파 DFT 를 빼서(load_minus_flux_data) 반사파만 남김
  R = -Φ_refl / Φ_inc,  T = Φ_trans / Φ_inc,  A = 1 - R - T  (계획서 5주차의 A = 1 − R − T 와 동일)
  ※ MEEP 은 conductivity 손실 에너지를 직접 추적하지 않는다(Materials.md). 그래서 A 는
    반드시 1 - R - T 로 구한다.

=== 셀 구성 (z 방향 1D) ===
  PML | 공기(소스) | 공기(반사 모니터) | 슬랩 d | 공기(투과 모니터) | PML
"""
import numpy as np
import meep as mp


# -----------------------------------------------------------------------------
# 재료 변환
# -----------------------------------------------------------------------------
def medium_from_N(N_tmm: complex, wavelength_um: float) -> mp.Medium:
    """
    TMM 규약의 복소 굴절률 N = n - ik 를, 파장 wavelength_um 에서 정확히 같은 ε 를 갖는
    MEEP Medium(ε_∞ + D_conductivity) 으로 변환한다.
    """
    N_tmm = complex(N_tmm)
    n, k = N_tmm.real, -N_tmm.imag
    if k < 0:
        raise ValueError(f"k = {k} < 0 (이득 매질). N = n - ik 규약에서 k ≥ 0 이어야 합니다.")
    eps_r = n * n - k * k
    eps_i = 2.0 * n * k                      # MEEP 규약(exp(-iωt))의 Im ε, 손실이면 양수
    f0 = 1.0 / wavelength_um
    sigma_d = 2.0 * np.pi * f0 * eps_i / eps_r
    return mp.Medium(epsilon=eps_r, D_conductivity=sigma_d)


def N_tmm_from_medium(medium: mp.Medium, freq: float) -> complex:
    """
    MEEP 이 주파수 freq 에서 실제로 쓰는 ε(f) 를 MEEP 자신에게 물어보고(Medium.epsilon),
    TMM 규약 N = n - ik 로 되돌린다. MEEP 과 TMM 에 '완전히 같은 재료'를 넣어
    솔버만 비교할 때(파트 A) 사용.
    """
    eps = complex(np.asarray(medium.epsilon(freq))[0, 0])
    N_meep = np.sqrt(eps)                    # 주값: Im ≥ 0 (MEEP 규약에서 손실)
    return complex(np.conj(N_meep))          # TMM 규약 n - ik


# -----------------------------------------------------------------------------
# 시뮬레이션
# -----------------------------------------------------------------------------
class Slab1D:
    """
    공기 / 슬랩(d) / 공기, 수직입사 평면파의 1D MEEP 계산기.

    정규화 런은 (해상도, 주파수 목록, 셀) 이 같으면 한 번만 돌리고 재사용한다
    (파트 B 처럼 파장마다 재료만 바꿔 여러 번 돌릴 때 시간을 절반으로 줄임).

    Parameters
    ----------
    d_um        : 슬랩 두께 [μm]
    freqs       : 플럭스를 기록할 MEEP 주파수 배열 (= 1/λ[μm])
    resolution  : 1 μm 당 격자점 수
    dpml_um     : PML 두께 [μm]
    pad_um      : 슬랩과 PML 사이 공기 두께 [μm] (소스·모니터가 이 안에 들어감)
    decay_tol   : 필드가 이 비율까지 줄면 종료 (stop_when_fields_decayed)
    """

    def __init__(self, d_um, freqs, resolution, dpml_um=1.0, pad_um=1.0, decay_tol=1e-9):
        self.d = float(d_um)
        self.freqs = np.asarray(freqs, dtype=float)
        self.res = resolution
        self.dpml = float(dpml_um)
        self.pad = float(pad_um)
        self.decay_tol = decay_tol

        self.L = 2 * self.dpml + 2 * self.pad + self.d
        self.z_src = -self.L / 2 + self.dpml + 0.1 * self.pad
        self.z_refl = -self.d / 2 - 0.5 * self.pad
        self.z_tran = +self.d / 2 + 0.5 * self.pad
        self.z_probe = self.z_tran              # 종료 판정용 필드 관측점

        fmin, fmax = self.freqs.min(), self.freqs.max()
        self.fcen = 0.5 * (fmin + fmax)
        # 가우시안 펄스 대역을 측정 대역보다 넓게 (가장자리 주파수도 세기가 충분하도록)
        self.fwidth = max(1.5 * (fmax - fmin), 0.2 * self.fcen)

        self._norm = None   # (입사 플럭스, 반사 모니터 DFT 데이터)

    def _build(self, material):
        geometry = []
        if material is not None:
            geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, self.d),
                                 center=mp.Vector3(), material=material)]
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.fwidth),
                             component=mp.Ex, center=mp.Vector3(z=self.z_src))]
        sim = mp.Simulation(cell_size=mp.Vector3(0, 0, self.L), dimensions=1,
                            resolution=self.res, boundary_layers=[mp.PML(self.dpml)],
                            sources=sources, geometry=geometry)
        refl = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(z=self.z_refl)))
        tran = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(z=self.z_tran)))
        return sim, refl, tran

    def _run(self, sim):
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ex, mp.Vector3(z=self.z_probe), self.decay_tol))

    def normalize(self):
        if self._norm is None:
            sim, refl, tran = self._build(None)
            self._run(sim)
            self._norm = (np.array(mp.get_fluxes(tran)), sim.get_flux_data(refl))
            sim.reset_meep()
        return self._norm

    def rta(self, material):
        """슬랩 재료 material 에 대한 (R, T, A) 배열 — self.freqs 순서."""
        inc, refl_data = self.normalize()
        sim, refl, tran = self._build(material)
        sim.load_minus_flux_data(refl, refl_data)
        self._run(sim)
        R = -np.array(mp.get_fluxes(refl)) / inc
        T = np.array(mp.get_fluxes(tran)) / inc
        sim.reset_meep()
        return R, T, 1.0 - R - T
