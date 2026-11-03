"""
diag_2d_vacuum.py — 진단용 (5주차 발산 원인 찾기). 재료·홈 없이 빈 2D 셀에서 필드가 안정적인지 본다.

- 셀: (P, L, 0), dimensions=2, PML 은 y 방향 (meep_2d.py 와 같은 설정)
- 두 가지 소스로 비교:
    A) 면 소스 (size = P)  — meep_2d.py 가 쓰는 방식
    B) 점 소스 (size = 0)  — 소스 크기 문제인지 확인용
- 결과: 시간별 프로브 필드 크기. 빈 공간에서는 소스가 꺼진 뒤 값이 작아지거나 일정해야 한다.
        계속 커지면(10배씩 증가) 설정 문제다.

실행 (WSL2, photontrap-meep):
  python scripts/diag_2d_vacuum.py
"""
import numpy as np
import meep as mp

P, L, RES = 0.8, 6.0, 50
Z_SRC, Y_PROBE = -1.9, 1.5


def run(size_x, label):
    src = mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ex,
                    center=mp.Vector3(0, Z_SRC, 0), size=mp.Vector3(size_x, 0, 0))
    sim = mp.Simulation(cell_size=mp.Vector3(P, L, 0), dimensions=2, resolution=RES,
                        boundary_layers=[mp.PML(1.0, direction=mp.Y)], sources=[src])
    probe = mp.Vector3(0, Y_PROBE, 0)
    print(f"\n[{label}] size_x = {size_x}")
    for _ in range(8):
        sim.run(until=5)
        v = sim.get_field_point(mp.Ex, probe)
        print(f"  t = {sim.meep_time():7.2f}   |Ex(probe)| = {abs(v):.3e}")


run(P, "A: 면 소스 (meep_2d 방식)")
run(0.0, "B: 점 소스")
