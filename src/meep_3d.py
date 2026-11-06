"""
meep_3d.py — 6주차: 얇은 Si 슬랩 위 원기둥 나노홀 단위셀의 수직입사 R/T/A (MEEP, 3D)

검증 상태 (2026-11-06, Claude 환경 MEEP 1.35.0, scripts/validate_nanohole_3d_flat.py):
  - 평면 환원(r = 0) vs TMM, 940 nm: 해상도 100/200/400 = −0.204/−0.057/−0.014 %p (2차 수렴) → 통과
    5파장(800–1000 nm) 최대 오차는 해상도 100/200 = 2.22/0.59 %p (1D·2D 와 같은 이산화 오차)
  - 같은 해상도의 2D 평면(meep_2d.py)과 R, T 차이 < 1e-7
  - 단위셀 크기 무관성: P = 0.2 vs 1.0 μm 차이 6e-15
  - 홀 기하(ε 분포): 홀 구간 Si 면적비 0.595 (기대 0.615, 해상도 40 계단 근사), 바닥 꽉 참
  - 수정 이력: k_point 누락 → 금속벽 경계로 계산되던 문제 수정 (아래 _build)
  ※ 홀 위치: hole_side="source"(기본값, 2026-11-06 변경) → 홀이 광원 쪽 면(z = −t/2), 빛이 홀 면으로 들어감.
    "back" 은 이전 방향(z = +t/2). 평면 환원 검증은 홀이 없어 방향과 무관.

=== 좌표 규약 ===
  MEEP 3D: 좌표 (x, y, z). z 가 전파 방향이다 (2D 와 달리 z 가 전파 방향이 맞다).
  셀: (P, P, L). x, y 는 주기 경계, z 는 PML.
  소스·모니터: 면(size = P × P × 0). 평면파 Ex 편광, 수직입사.

=== 구조 ===
  Si 필름 두께 t (z ∈ [−t/2, t/2]), 광원 쪽 면(hole_side="source")에서 깊이 h 인 공기 원기둥 홀 (반경 r), 정사각 격자 주기 P.
  홀 반경 r = 0 이면 평면 필름 → 1D/2D 평면 결과와 같아야 함 (체크 1).

=== 계산 방식 (2D 그레이팅과 같은 MEEP 표준 R/T 절차) ===
  정규화 런 → 입사 플럭스·입사장 저장 → 본 런에서 입사장 빼기 → R, T, A = 1 − R − T

=== 주의 ===
  - 3D 는 격자 수가 2D 보다 훨씬 많다. 해상도는 낮게 시작해 메모리·시간을 먼저 본다.
  - 7주차 (2026-11-06) 추가: lattice="hex" (육각 격자, Gao 2017), bottom_radius_um (깔때기·테이퍼 홀, mp.Cone).
    육각 격자는 직사각 셀 P × √3P 에 홀 2개(가운데 1 + 모서리 1/4 × 4)를 넣는다. 셀이 x·y 거울 대칭이라
    use_symmetry 도 그대로 성립한다. 기본값(lattice="square", bottom_radius_um=None)은 6주차 코드와 같은 기하.
"""
import numpy as np
import meep as mp

from meep_1d import medium_from_N


class NanoHoleCell3D:
    """
    Parameters
    ----------
    period_um   : 격자 주기 P [μm] (가장 가까운 홀 사이 거리)
    radius_um   : 홀 반경 r [μm] (깔때기면 입구 쪽 반경, 0 이면 평면 필름)
    depth_um    : 홀 깊이 h [μm] (필름 윗면에서 아래로)
    film_um     : Si 필름 두께 t [μm]
    freqs       : 플럭스 기록 주파수 배열 (= 1/λ[μm])
    resolution  : 1 μm 당 격자점 수
    dpml_um     : z 방향 PML 두께 [μm]
    pad_um      : 구조와 PML 사이 공기 두께 [μm]
    t_cap       : 소스 종료 후 계산 시간 상한 (MEEP 단위). 2D 와 같은 고정 시간 방식. 기본값 800 = 진행 노트 확정값 (2026-11-06)
    hole_side   : "source" (홀이 광원 쪽 면 z = −t/2, 기본값) 또는 "back" (광원 반대편 면 z = +t/2)
    use_symmetry: True 면 x·y 거울 대칭으로 셀 1/4 만 계산 (약 4배 빠름).
                  Ex 편광 수직입사 + 홀이 단위셀 중심 원기둥일 때만 성립:
                  Ex 는 x 거울에 대해 홀함수(phase −1), y 거울에 대해 짝함수(phase +1).
                  근거: MEEP 대칭 규칙 (doc/docs/Exploiting_Symmetry.md) — 소스 성분이 거울면에 수직이면 phase −1.
                  MEEP 튜토리얼 Mie 산란(Ez 소스)도 Mirror(Z, −1), Mirror(Y, +1) 을 쓴다. 최종 근거는 수치 비교.
                  대칭 없는 계산과 같은 값이 나오는지 확인 후 사용 (scripts/nanohole_3d.py --symcheck)
    lattice     : "square" (셀 P × P, 홀 1개) 또는 "hex" (셀 P × √3P, 홀 2개 — 가운데 + 네 모서리)
    bottom_radius_um : None 이면 원기둥. 값이 있으면 홀 바닥 반경 (입구 r → 바닥 r_bot 으로 선형 변화, mp.Cone).
                  r_bot < r 이 깔때기(Gao 의 +각), r_bot > r 이 역깔때기(Gao 의 −각). 측벽각 → r_bot 변환은 스크립트에서.
    """

    def __init__(self, period_um, radius_um, depth_um, film_um, freqs, resolution,
                 dpml_um=1.0, pad_um=1.0, t_cap=800.0, hole_side="source", use_symmetry=False,
                 lattice="square", bottom_radius_um=None):
        if hole_side not in ("source", "back"):
            raise ValueError(f"hole_side = {hole_side!r} 는 'source' 또는 'back' 이어야 합니다.")
        self.side = hole_side
        self.sym = bool(use_symmetry)
        if depth_um < 0 or depth_um >= film_um:
            raise ValueError("홀 깊이는 0 이상이고 필름 두께보다 작아야 합니다.")
        if radius_um < 0 or 2 * radius_um > period_um:
            raise ValueError("홀 반경은 0 이상이고 지름이 주기보다 작아야 합니다.")
        if lattice not in ("square", "hex"):
            raise ValueError(f"lattice = {lattice!r} 는 'square' 또는 'hex' 이어야 합니다.")
        self.lattice = lattice
        self.P = float(period_um)
        self.Py = self.P * (np.sqrt(3.0) if lattice == "hex" else 1.0)
        if lattice == "hex":
            # √3P 는 격자 칸의 정수배가 아니라 MEEP 이 셀을 몰래 반올림한다 → 직접 반올림하고 모서리 홀도 그 경계에 둔다.
            # y 칸 수는 짝수로: 홀수(해상도 40 → 69칸)면 대칭 + 분산 재료에서 t ≈ 250 에 필드가 발산했다
            # (2026-11-06, 70칸은 정상 감쇠). 왜곡이 작은 해상도: 30, 45, 60 (+0.08%). 40 은 +1.0%, 50 은 −0.7%.
            ny = 2 * int(round(self.Py * resolution / 2))
            py_exact = self.Py
            self.Py = ny / resolution
            self.hex_distortion = self.Py / py_exact - 1
            if abs(self.hex_distortion) > 2e-3:
                print(f"[meep_3d] 경고: 해상도 {resolution} 에서 육각 셀 y 길이 왜곡 {100*self.hex_distortion:+.2f}% "
                      f"(30, 45, 60 권장)", flush=True)
        self.rb = None if bottom_radius_um is None else float(bottom_radius_um)
        if self.rb is not None and (self.rb < 0 or 2 * self.rb > period_um):
            raise ValueError("홀 바닥 반경은 0 이상이고 지름이 주기보다 작아야 합니다.")
        self.r = float(radius_um)
        self.h = float(depth_um)
        self.t = float(film_um)
        self.freqs = np.asarray(freqs, dtype=float)
        self.res = resolution
        self.dpml = float(dpml_um)
        self.pad = float(pad_um)
        self.t_cap = float(t_cap)

        self.L = 2 * self.dpml + 2 * self.pad + self.t
        self.z_src = -self.L / 2 + self.dpml + 0.1 * self.pad
        self.z_refl = -self.t / 2 - 0.5 * self.pad
        self.z_tran = +self.t / 2 + 0.5 * self.pad
        self.z_probe = self.z_tran

        fmin, fmax = self.freqs.min(), self.freqs.max()
        self.fcen = 0.5 * (fmin + fmax)
        self.fwidth = max(1.5 * (fmax - fmin), 0.2 * self.fcen)

        self._norm = None

    def hole_centers(self):
        """홀 중심 (x, y). 육각: 가운데 + 네 모서리 (주기 경계로 모서리 1/4 네 개가 홀 하나가 된다)."""
        if self.lattice == "square":
            return [(0.0, 0.0)]
        a, b = self.P / 2, self.Py / 2
        return [(0.0, 0.0), (a, b), (-a, b), (a, -b), (-a, -b)]

    def geometry(self, material):
        if material is None:
            return []
        geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, self.t),
                         center=mp.Vector3(0, 0, 0), material=material)]
        if self.r > 0 and self.h > 0:
            ext = 0.2   # 홀을 필름 밖으로 0.2 μm 더 뻗게 해서 표면과 면이 겹치지 않게 한다
            sgn = -1.0 if self.side == "source" else +1.0   # 광원은 −z 쪽. 홀 입구 면 z = sgn·t/2
            z_open = sgn * self.t / 2
            inward = -sgn                                   # 입구에서 필름 안쪽 방향
            for (x, y) in self.hole_centers():
                if self.rb is None:
                    geom.append(mp.Cylinder(radius=self.r, height=self.h + ext,
                                            center=mp.Vector3(x, y, sgn * (self.t / 2 - self.h / 2 + ext / 2)),
                                            material=mp.air))
                else:
                    # 필름 밖 연장부(공기 속이라 모양 무관)는 입구 반경 원기둥, 필름 안은 원뿔대.
                    # mp.Cone: radius 는 center − height/2·axis 쪽(= 입구), radius2 는 반대쪽(= 바닥) 반경.
                    geom.append(mp.Cylinder(radius=self.r, height=ext,
                                            center=mp.Vector3(x, y, z_open - inward * ext / 2), material=mp.air))
                    geom.append(mp.Cone(radius=self.r, radius2=self.rb, height=self.h,
                                        axis=mp.Vector3(0, 0, inward),
                                        center=mp.Vector3(x, y, z_open + inward * self.h / 2), material=mp.air))
        return geom

    def _build(self, material):
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.fwidth),
                             component=mp.Ex,
                             center=mp.Vector3(0, 0, self.z_src),
                             size=mp.Vector3(self.P, self.Py, 0))]
        # k_point = 0 → x, y 를 Bloch 주기 경계(수직입사)로 만든다. 이것을 빼면 MEEP 기본값(k_point=False)이
        # 금속벽(PEC)이라 y 벽에서 Ex 가 0 으로 강제되고, 평면파 대신 도파관 모드가 생긴다 (2026-11-06 체크 1b 에서 발견).
        sim = mp.Simulation(cell_size=mp.Vector3(self.P, self.Py, self.L), dimensions=3,
                            resolution=self.res, k_point=mp.Vector3(0, 0, 0),
                            boundary_layers=[mp.PML(self.dpml, direction=mp.Z)],
                            symmetries=([mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)]
                                        if self.sym else []),
                            sources=sources, geometry=self.geometry(material))
        refl = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_refl),
                                                      size=mp.Vector3(self.P, self.Py, 0)))
        tran = sim.add_flux(self.freqs, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_tran),
                                                      size=mp.Vector3(self.P, self.Py, 0)))
        return sim, refl, tran

    def _run(self, sim):
        # 고정 시간 방식 (meep_2d.py 와 동일한 이유로 변경)
        sim.run(until_after_sources=self.t_cap)

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
