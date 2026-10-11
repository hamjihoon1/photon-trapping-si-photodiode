"""
meep_2d.py — 5주차: 2D 라멜라(선형 홈) 그레이팅의 수직입사 R/T/A 계산 (MEEP, x 방향 주기 경계)

계획서 5주차: "2D 주기 구조 — 그레이팅 흡수 스펙트럼 + 수렴 테스트 그래프"

=== 구조 (x–z 평면, y 방향은 무한히 균일) ===
  z 위쪽: 공기
  Si 필름: 두께 t (z ∈ [−t/2, t/2]), 윗면에 폭 w = (1−f)·Λ, 깊이 h 인 공기 홈이 주기 Λ 로 반복
  f = Si 점유율(fill fraction).  f = 1 이고 h = 0 이면 평면 필름 → 1D 슬랩(meep_1d.py)과 같아야 함
  → 이것이 이 모듈의 첫 번째 검증 (validate_grating_2d.py 의 '평면 환원' 체크)

  ※ 2D 홈 그레이팅은 계획서의 "원기둥 나노홀"(3D)로 가는 중간 단계다. 홈은 x 방향으로만 주기적이다.

=== 경계·광원·모니터 ===
  x : 주기 경계 (MEEP 기본값: PML 을 두지 않은 방향은 주기). 수직입사이므로 Bloch k_x = 0
  z : PML(dpml) 양쪽. 광원은 면 광원(size = Λ) 으로 x 전체에 같은 위상의 평면파를 만든다
  모니터: 플럭스 영역을 x 전체 주기(size = Λ)로 잡는다 → 회절된 모든 차수의 합이 자동으로 들어간다.
          (Λ < λ 이면 공기 중 ±1차 이상은 evanescent 라 0차만 전파; Λ > λ 이면 전파 차수가 생김)

=== 계산 방식 (1D 와 동일한 MEEP 표준 R/T 절차) ===
  1) 정규화 런: 구조 없이 같은 셀 → 입사 플럭스, 반사 모니터의 입사장 DFT 저장
  2) 본 런: 반사 모니터에서 입사장 DFT 를 빼서(load_minus_flux_data) 반사파만 남김
  R = −Φ_refl/Φ_inc,  T = Φ_trans/Φ_inc,  A = 1 − R − T
  ※ 2D/3D 에서는 입사장 빼기가 필수 (진행 노트 4절 실험 d 에서 1D 에서만 교차항이 사라진다는 것을 확인)

=== 재료 ===
  Si: 1D 와 같은 medium_from_N (파장 한 점에서 맞춘 ε_∞ + D_conductivity). 파장마다 따로 계산한다.
  넓은 대역을 한 번에 돌리는 Lorentz 피팅은 아직 없음 (진행 노트 6절 7번).
  홈 안은 mp.air.

=== 주의 (미검증 항목) ===
  이 파일은 Claude 작업 환경에 MEEP 이 없어 실행 검증을 하지 못했다 (conda-forge 접속 불가).
  사용자 WSL2 환경에서 validate_grating_2d.py 의 '평면 환원' 체크를 먼저 통과시켜야 한다.
"""
import numpy as np
import meep as mp

from meep_1d import medium_from_N


class LamellarGrating2D:
    """
    Parameters
    ----------
    period_um   : 주기 Λ [μm]
    si_fill     : Si 점유율 f (0 < f ≤ 1). 홈 폭 w = (1 − f)·Λ
    depth_um    : 홈 깊이 h [μm] (0 이면 평면 필름). 홈은 필름 윗면에서 아래로 판다
    film_um     : Si 필름 두께 t [μm]
    freqs       : 플럭스를 기록할 MEEP 주파수 배열 (= 1/λ[μm])
    resolution  : 1 μm 당 격자점 수
    dpml_um     : z 방향 PML 두께 [μm]. 5주차 PML 2배 테스트를 위해 바꿔 쓸 수 있게 열어둠
    pad_um      : 구조와 PML 사이 공기 두께 [μm] (소스·모니터가 이 안에 들어감)
    decay_tol   : stop_when_fields_decayed 의 기준 (1D 와 같은 1e-9)
    max_time    : 안전장치: 최대 시뮬레이션 시간 (MEEP 단위, a=1μm). 공진이 강해 필드가 안 죽을 때 무한정 돌지 않도록
    """

    def __init__(self, period_um, si_fill, depth_um, film_um, freqs, resolution,
                 dpml_um=1.0, pad_um=1.0, decay_tol=1e-9, max_time=2000.0):
        if not (0.0 < si_fill <= 1.0):
            raise ValueError(f"si_fill = {si_fill} 는 (0, 1] 범위여야 합니다.")
        if depth_um < 0 or depth_um >= film_um:
            raise ValueError("홈 깊이는 0 이상이고 필름 두께보다 작아야 합니다 (바닥 Si 가 남아야 함).")
        self.P = float(period_um)
        self.f = float(si_fill)
        self.h = float(depth_um)
        self.t = float(film_um)
        self.freqs = np.asarray(freqs, dtype=float)
        self.res = resolution
        self.dpml = float(dpml_um)
        self.pad = float(pad_um)
        self.decay_tol = decay_tol
        self.max_time = max_time

        self.L = 2 * self.dpml + 2 * self.pad + self.t          # z 방향 셀 길이
        self.z_src = -self.L / 2 + self.dpml + 0.1 * self.pad
        self.z_refl = -self.t / 2 - 0.5 * self.pad
        self.z_tran = +self.t / 2 + 0.5 * self.pad
        self.z_probe = self.z_tran

        fmin, fmax = self.freqs.min(), self.freqs.max()
        self.fcen = 0.5 * (fmin + fmax)
        self.fwidth = max(1.5 * (fmax - fmin), 0.2 * self.fcen)

        self._norm = None

    # -------------------------------------------------------------------------
    def geometry(self, material):
        """Si 필름(material) + 윗면 공기 홈. 나중에 쓴 객체가 앞의 것을 덮는다."""
        if material is None:
            return []
        geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, self.t),
                         center=mp.Vector3(), material=material)]
        if self.h > 0 and self.f < 1.0:
            w = (1.0 - self.f) * self.P
            # 홈 위쪽으로 0.2 μm 더 뻗게 해서 필름 윗면과 면이 딱 겹치는 것을 피한다
            ext = 0.2
            geom.append(mp.Block(size=mp.Vector3(w, mp.inf, self.h + ext),
                                 center=mp.Vector3(0, 0, self.t / 2 - self.h / 2 + ext / 2),
                                 material=mp.air))
        return geom

    def _build(self, material):
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.fwidth),
                             component=mp.Ex,
                             center=mp.Vector3(0, 0, self.z_src),
                             size=mp.Vector3(self.P, 0, 0))]
        sim = mp.Simulation(cell_size=mp.Vector3(self.P, 0, self.L), dimensions=2,
                            resolution=self.res,
                            boundary_layers=[mp.PML(self.dpml, direction=mp.Z)],
                            sources=sources, geometry=self.geometry(material))
        refl = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_refl),
                                                      size=mp.Vector3(self.P, 0, 0)))
        tran = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_tran),
                                                      size=mp.Vector3(self.P, 0, 0)))
        return sim, refl, tran

    def _run(self, sim):
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ex, mp.Vector3(0, 0, self.z_probe), self.decay_tol))

    # -------------------------------------------------------------------------
    def normalize(self):
        if self._norm is None:
            sim, refl, tran = self._build(None)
            self._run(sim)
            self._norm = (np.array(mp.get_fluxes(tran)), sim.get_flux_data(refl))
            sim.reset_meep()
        return self._norm

    def rta(self, material):
        """재료 material 에 대한 (R, T, A) 배열 — self.freqs 순서."""
        inc, refl_data = self.normalize()
        sim, refl, tran = self._build(material)
        sim.load_minus_flux_data(refl, refl_data)
        self._run(sim)
        R = -np.array(mp.get_fluxes(refl)) / inc
        T = np.array(mp.get_fluxes(tran)) / inc
        sim.reset_meep()
        return R, T, 1.0 - R - T


def si_medium_at(N_tmm_value: complex, wavelength_nm: float):
    """nm 단위 입력을 받는 얇은 래퍼 (스크립트에서 편하게 쓰도록)."""
    return medium_from_N(N_tmm_value, wavelength_nm / 1000.0)
