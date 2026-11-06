"""
meep_2d.py — 5주차: 2D 라멜라(선형 홈) 그레이팅의 수직입사 R/T/A 계산 (MEEP, x 방향 주기 경계)

계획서 5주차: "2D 주기 구조 — 그레이팅 흡수 스펙트럼 + 수렴 테스트 그래프"

=== 좌표 규약 (중요) ===
  MEEP 의 dimensions=2 는 xy 평면 시뮬레이션이다. 세 번째 축(z)은 불변 방향이다.
  이 코드는 물리적으로 "x = 주기 방향, z = 전파 방향(깊이)" 으로 생각한다.
  따라서 아래의 z 좌표는 모두 MEEP 의 y 좌표로 들어간다 (_pt(x, z) = Vector3(x, z, 0)).
  셀 크기는 Vector3(Λ, L, 0), PML 방향은 mp.Y.
  (처음 버전은 (Λ, 0, L) 로 넣어서 전파 방향 길이가 버려졌고, 입사 플럭스가 0 이 되었다 — 2026-11-03 수정)

=== 구조 (x–z 평면, 홈은 불변 방향으로 무한히 뻗음) ===
  공기 / Si 필름: 두께 t (z ∈ [−t/2, t/2]), 한쪽 면에 폭 w = (1−f)·Λ, 깊이 h 인 공기 홈이 주기 Λ 로 반복
  groove_side (2026-11-06 추가):
    "source" (기본값): 홈이 광원 쪽 면(z = −t/2)에 있다 → 빛이 홈 면으로 들어감 (계획서·Gao 2017 과 같은 방향)
    "back"           : 홈이 광원 반대편 면(z = +t/2). 2026-11-06 이전 5주차 결과는 모두 이 방향
  f = Si 점유율(fill fraction).  f = 1 이고 h = 0 이면 평면 필름 → 1D 슬랩(meep_1d.py)과 같아야 함
  → 이것이 이 모듈의 첫 번째 검증 (validate_grating_2d.py 의 '평면 환원' 체크)

=== 경계·광원·모니터 ===
  x : 주기 경계. k_point = Vector3(0,0,0) 을 반드시 넣는다 (수직입사 Bloch k_x = 0).
      k_point 를 빼면 MEEP 기본값이 금속벽(PEC)이다. 2026-11-06 이전 코드는 k_point 가 없었으나,
      대칭 홈 + 수직입사에서는 PEC 벽 = 주기 경계라서 값 차이는 3.3e-5 (해상도 100, 940 nm) 뿐이었다.
  z : PML(dpml) 양쪽 (MEEP 에서는 y 방향 PML). 광원은 면 광원(size = Λ) 으로 x 전체에 같은 위상의 평면파
  모니터: 플럭스 영역을 x 전체 주기(size = Λ)로 잡는다 → 전파하는 모든 회절 차수의 합이 들어간다.

=== 계산 방식 (1D 와 동일한 MEEP 표준 R/T 절차) ===
  1) 정규화 런: 구조 없이 같은 셀 → 입사 플럭스, 반사 모니터의 입사장 DFT 저장
  2) 본 런: 반사 모니터에서 입사장 DFT 를 빼서(load_minus_flux_data) 반사파만 남김
  R = −Φ_refl/Φ_inc,  T = Φ_trans/Φ_inc,  A = 1 − R − T
  ※ 2D/3D 에서는 입사장 빼기가 필수 (진행 노트 4절 실험 d 에서 1D 에서만 교차항이 사라진다는 것을 확인)

=== 재료 ===
  Si: 1D 와 같은 medium_from_N (파장 한 점에서 맞춘 ε_∞ + D_conductivity). 파장마다 따로 계산한다.
  홈 안은 mp.air.

=== 검증 상태 ===
  5주차 체크 1–3 통과 (사용자 WSL2, 2026-11-06, groove_side="back" 방향, k_point 없음).
  groove_side="source" 재계산은 진행 노트 참고.
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
    dpml_um     : 전파 방향 PML 두께 [μm]. 5주차 PML 2배 테스트에서 바꿔 쓴다
    pad_um      : 구조와 PML 사이 공기 두께 [μm] (소스·모니터가 이 안에 들어감)
    t_cap       : 소스 종료 후 계산 시간 상한 (MEEP 단위, a=1μm). 이 시간까지만 돈다. 기본값 800 = 확정값
    groove_side : "source" (홈이 광원 쪽 면, 기본값) 또는 "back" (광원 반대편 면, 2026-11-06 이전 방향)
    """

    def __init__(self, period_um, si_fill, depth_um, film_um, freqs, resolution,
                 dpml_um=1.0, pad_um=1.0, t_cap=800.0, groove_side="source"):
        if groove_side not in ("source", "back"):
            raise ValueError(f"groove_side = {groove_side!r} 는 'source' 또는 'back' 이어야 합니다.")
        self.side = groove_side
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
        self.t_cap = float(t_cap)

        # 전파 방향(우리 z = MEEP y) 셀 길이
        self.L = 2 * self.dpml + 2 * self.pad + self.t
        self.z_src = -self.L / 2 + self.dpml + 0.1 * self.pad
        self.z_refl = -self.t / 2 - 0.5 * self.pad
        self.z_tran = +self.t / 2 + 0.5 * self.pad
        self.z_probe = self.z_tran

        fmin, fmax = self.freqs.min(), self.freqs.max()
        self.fcen = 0.5 * (fmin + fmax)
        self.fwidth = max(1.5 * (fmax - fmin), 0.2 * self.fcen)

        self._norm = None

    # -------------------------------------------------------------------------
    @staticmethod
    def _pt(x, z):
        """우리 좌표 (x, z) → MEEP 2D 좌표 (x, y=z, 0)."""
        return mp.Vector3(x, z, 0)

    def geometry(self, material):
        """Si 필름(material) + 윗면 공기 홈. 나중에 쓴 객체가 앞의 것을 덮는다."""
        if material is None:
            return []
        geom = [mp.Block(size=mp.Vector3(mp.inf, self.t, mp.inf),
                         center=self._pt(0, 0), material=material)]
        if self.h > 0 and self.f < 1.0:
            w = (1.0 - self.f) * self.P
            ext = 0.2   # 홈을 필름 위로 0.2 μm 더 뻗게 해서 윗면과 면이 겹치지 않게 한다
            # 광원은 −z 쪽(z_src < 0). "source" 면 = z = −t/2, "back" 면 = z = +t/2
            sgn = -1.0 if self.side == "source" else +1.0
            geom.append(mp.Block(size=mp.Vector3(w, self.h + ext, mp.inf),
                                 center=self._pt(0, sgn * (self.t / 2 - self.h / 2 + ext / 2)),
                                 material=mp.air))
        return geom

    def _build(self, material):
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.fwidth),
                             component=mp.Ex,
                             center=self._pt(0, self.z_src),
                             size=mp.Vector3(self.P, 0, 0))]
        sim = mp.Simulation(cell_size=mp.Vector3(self.P, self.L, 0), dimensions=2,
                            resolution=self.res, k_point=mp.Vector3(0, 0, 0),
                            boundary_layers=[mp.PML(self.dpml, direction=mp.Y)],
                            sources=sources, geometry=self.geometry(material))
        refl = sim.add_flux(self.freqs, mp.FluxRegion(center=self._pt(0, self.z_refl),
                                                      size=mp.Vector3(self.P, 0, 0)))
        tran = sim.add_flux(self.freqs, mp.FluxRegion(center=self._pt(0, self.z_tran),
                                                      size=mp.Vector3(self.P, 0, 0)))
        return sim, refl, tran

    def _run(self, sim):
        # 고정 시간 방식: 소스가 꺼진 뒤 t_cap 만큼만 계산한다.
        # 이전의 stop_when_fields_decayed(1e-9)는 빈 셀에서도 기준에 도달하지 못해 계산이 끝나지 않았다
        # (2026-11-05 진단: 빈 셀은 t=200 까지 안정적, 잔류 필드 ~1e-8).
        sim.run(until_after_sources=self.t_cap)

    # -------------------------------------------------------------------------
    def normalize(self):
        if self._norm is None:
            sim, refl, tran = self._build(None)
            self._run(sim)
            inc = np.array(mp.get_fluxes(tran))
            if np.any(inc <= 0):
                raise RuntimeError(f"정규화 입사 플럭스가 0 이하입니다: {inc}. 셀 크기·소스 위치를 확인하세요.")
            self._norm = (inc, sim.get_flux_data(refl))
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
