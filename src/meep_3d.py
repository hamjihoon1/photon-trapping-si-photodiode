"""
meep_3d.py — 6주차: 얇은 Si 슬랩 위 원기둥 나노홀 단위셀의 수직입사 R/T/A (MEEP, 3D)

⚠ 초안 (미실행). Claude 작업 환경에 MEEP 이 없어 실행하지 못했다.
   사용자 WSL2 에서 평면 환원 체크(홀 반경 0 → 평면 Si 와 같아야 함)를 먼저 통과시켜야 한다.

=== 좌표 규약 ===
  MEEP 3D: 좌표 (x, y, z). z 가 전파 방향이다 (2D 와 달리 z 가 전파 방향이 맞다).
  셀: (P, P, L). x, y 는 주기 경계, z 는 PML.
  소스·모니터: 면(size = P × P × 0). 평면파 Ex 편광, 수직입사.

=== 구조 ===
  Si 필름 두께 t (z ∈ [−t/2, t/2]), 윗면에서 깊이 h 인 공기 원기둥 홀 (반경 r), 정사각 격자 주기 P.
  홀 반경 r = 0 이면 평면 필름 → 1D/2D 평면 결과와 같아야 함 (체크 1).

=== 계산 방식 (2D 그레이팅과 같은 MEEP 표준 R/T 절차) ===
  정규화 런 → 입사 플럭스·입사장 저장 → 본 런에서 입사장 빼기 → R, T, A = 1 − R − T

=== 주의 ===
  - 3D 는 격자 수가 2D 보다 훨씬 많다. 해상도는 낮게 시작해 메모리·시간을 먼저 본다.
  - 육각 격자(논문 사용)는 이 초안에 없다. 정사각 격자만 구현했다.
"""
import numpy as np
import meep as mp

from meep_1d import medium_from_N


class NanoHoleCell3D:
    """
    Parameters
    ----------
    period_um   : 정사각 격자 주기 P [μm]
    radius_um   : 원기둥 홀 반경 r [μm] (0 이면 평면 필름)
    depth_um    : 홀 깊이 h [μm] (필름 윗면에서 아래로)
    film_um     : Si 필름 두께 t [μm]
    freqs       : 플럭스 기록 주파수 배열 (= 1/λ[μm])
    resolution  : 1 μm 당 격자점 수
    dpml_um     : z 방향 PML 두께 [μm]
    pad_um      : 구조와 PML 사이 공기 두께 [μm]
    decay_tol   : stop_when_fields_decayed 기준
    """

    def __init__(self, period_um, radius_um, depth_um, film_um, freqs, resolution,
                 dpml_um=1.0, pad_um=1.0, decay_tol=1e-9):
        if depth_um < 0 or depth_um >= film_um:
            raise ValueError("홀 깊이는 0 이상이고 필름 두께보다 작아야 합니다.")
        if radius_um < 0 or 2 * radius_um > period_um:
            raise ValueError("홀 반경은 0 이상이고 지름이 주기보다 작아야 합니다.")
        self.P = float(period_um)
        self.r = float(radius_um)
        self.h = float(depth_um)
        self.t = float(film_um)
        self.freqs = np.asarray(freqs, dtype=float)
        self.res = resolution
        self.dpml = float(dpml_um)
        self.pad = float(pad_um)
        self.decay_tol = decay_tol

        self.L = 2 * self.dpml + 2 * self.pad + self.t
        self.z_src = -self.L / 2 + self.dpml + 0.1 * self.pad
        self.z_refl = -self.t / 2 - 0.5 * self.pad
        self.z_tran = +self.t / 2 + 0.5 * self.pad
        self.z_probe = self.z_tran

        fmin, fmax = self.freqs.min(), self.freqs.max()
        self.fcen = 0.5 * (fmin + fmax)
        self.fwidth = max(1.5 * (fmax - fmin), 0.2 * self.fcen)

        self._norm = None

    def geometry(self, material):
        if material is None:
            return []
        geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, self.t),
                         center=mp.Vector3(0, 0, 0), material=material)]
        if self.r > 0 and self.h > 0:
            ext = 0.2   # 홀을 필름 위로 0.2 μm 더 뻗게 해서 윗면과 면이 겹치지 않게 한다
            geom.append(mp.Cylinder(radius=self.r, height=self.h + ext,
                                    center=mp.Vector3(0, 0, self.t / 2 - self.h / 2 + ext / 2),
                                    material=mp.air))
        return geom

    def _build(self, material):
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.fwidth),
                             component=mp.Ex,
                             center=mp.Vector3(0, 0, self.z_src),
                             size=mp.Vector3(self.P, self.P, 0))]
        sim = mp.Simulation(cell_size=mp.Vector3(self.P, self.P, self.L), dimensions=3,
                            resolution=self.res,
                            boundary_layers=[mp.PML(self.dpml, direction=mp.Z)],
                            sources=sources, geometry=self.geometry(material))
        refl = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_refl),
                                                      size=mp.Vector3(self.P, self.P, 0)))
        tran = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_tran),
                                                      size=mp.Vector3(self.P, self.P, 0)))
        return sim, refl, tran

    def _run(self, sim):
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ex, mp.Vector3(0, 0, self.z_probe), self.decay_tol))

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
        inc, refl_data = self.normalize()
        sim, refl, tran = self._build(material)
        sim.load_minus_flux_data(refl, refl_data)
        self._run(sim)
        R = -np.array(mp.get_fluxes(refl)) / inc
        T = np.array(mp.get_fluxes(tran)) / inc
        sim.reset_meep()
        return R, T, 1.0 - R - T
