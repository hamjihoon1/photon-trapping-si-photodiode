"""
validate_incoherent.py — 3주차: 비간섭 슬랩 함수(incoherent_slab) 검증

  (1) 코팅 없는 공기/Si/공기: 교과서 닫힌식
        A = (1-R)(1-x)(1+Rx) / (1-R^2 x^2)
      과 일치 (R = Fresnel 반사율, x = exp(-alpha d))
  (2) 에너지 보존: 무손실 코팅(ARC)일 때 R + T + A = 1
  (3) 핵심 검증: coherent TMM을 두께 한 주기(lambda/(2n)) 동안 평균한 값 = incoherent 값
      (간섭무늬를 '평균'한 것이 비간섭 계산이라는 정의 그 자체를 확인)
      - 코팅 없음 / SiO2 ARC 있음 두 경우 모두
  (4) 극한: Si가 아주 두꺼우면(1 mm) 투과 0, A -> 1 - R_front
  (5) 극한: k = 0 이면 A = 0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tmm import tmm_normal_incidence, incoherent_slab, fresnel_single_interface
from si_data import SiOpticalConstants

N_SIO2 = 1.45  # SiO2 굴절률 가정값 (Malitson 1965 기준 ~1.45 @ 900nm대, 검증용으로만 사용)


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}  {detail}")
    return cond


def coherent_thickness_average(lam, N_si, d_nm, front_layers=(), front_d=(), n_samples=2000):
    """두께를 d ~ d + lambda/(2n) 한 주기 동안 균일하게 바꿔가며 coherent A, R, T 평균."""
    period = lam / (2 * N_si.real)
    ds = d_nm + np.linspace(0, period, n_samples, endpoint=False)
    out = np.array([
        tmm_normal_incidence(lam, list(front_layers) + [N_si], list(front_d) + [d], 1.0, 1.0)
        for d in ds
    ])
    R, T, A_tot = out.mean(axis=0)
    return R, T, A_tot


def main():
    si = SiOpticalConstants(verbose=False)
    ok_all = True

    print("=== 검증 1: 코팅 없는 슬랩 vs 닫힌식 ===")
    for lam in (800, 900, 940, 1000):
        for d in (1000, 2000, 10000):
            N = si.N(lam)
            n, k = N.real, -N.imag
            x = np.exp(-4 * np.pi * k / lam * d)
            Rf = ((n - 1) / (n + 1)) ** 2
            A_ref = (1 - Rf) * (1 - x) * (1 + Rf * x) / (1 - Rf ** 2 * x ** 2)
            _, _, A = incoherent_slab(lam, N, d, 1.0, 1.0)
            ok = abs(A - A_ref) < 1e-12
            ok_all &= ok
            if lam == 940 or not ok:
                check(f"  {lam}nm, d={d/1000:g}um", ok, f"(닫힌식 {A_ref:.6f} | 함수 {A:.6f})")

    print("\n=== 검증 2: 에너지 보존 R+T+A=1 (무손실 SiO2 ARC 포함) ===")
    worst = 0.0
    for lam in np.linspace(800, 1000, 41):
        for d in (1000, 5000, 10000):
            for d_arc in (0, 100, 162, 300):
                R, T, A = incoherent_slab(lam, si.N(lam), d, 1.0, 1.0, [N_SIO2], [d_arc])
                worst = max(worst, abs(R + T + A - 1))
    ok = worst < 1e-12
    ok_all &= check("  41파장 x 3두께 x 4 ARC두께", ok, f"(최대 오차 {worst:.1e})")

    print("\n=== 검증 3: coherent 두께평균 = incoherent (핵심) ===")
    cases = [(940, 2000, False), (940, 10000, False), (850, 5000, False),
             (940, 2000, True), (1000, 10000, True)]
    for lam, d, use_arc in cases:
        N = si.N(lam)
        fl, fd = ([N_SIO2], [162.0]) if use_arc else ([], [])
        Rc, Tc, Ac = coherent_thickness_average(lam, N, d, fl, fd)
        Ri, Ti, Ai = incoherent_slab(lam, N, d + lam / (4 * N.real), 1.0, 1.0, fl, fd)
        # 평균 구간 한가운데 두께로 비교. 허용오차: 상대 1%
        # (한 주기 동안 두께가 lambda/2n 만큼 변하면서 생기는 alpha*dd 효과 + 표본화 오차)
        rel = abs(Ac - Ai) / Ai
        ok = rel < 0.01
        ok_all &= ok
        tag = "ARC" if use_arc else "코팅X"
        check(f"  {lam}nm, d={d/1000:g}um, {tag}", ok,
              f"(coherent 평균 A={Ac*100:.3f}% | incoherent A={Ai*100:.3f}% | 상대차 {rel*100:.2f}%)")

    print("\n=== 검증 4: 아주 두꺼운 Si (1 mm) -> T=0, A = 1 - R_front ===")
    for use_arc in (False, True):
        fl, fd = ([N_SIO2], [162.0]) if use_arc else ([], [])
        N = si.N(940)
        R, T, A = incoherent_slab(940, N, 1e6, 1.0, 1.0, fl, fd)
        R_front, _, _ = tmm_normal_incidence(940, fl, fd, 1.0, N.real)
        ok = T < 1e-6 and abs(A - (1 - R_front)) < 1e-6
        ok_all &= ok
        check(f"  {'ARC' if use_arc else '코팅X'}", ok,
              f"(T={T:.1e}, A={A*100:.2f}%, 1-R_front={(1-R_front)*100:.2f}%)")

    print("\n=== 검증 5: k=0 -> 흡수 0 ===")
    R, T, A = incoherent_slab(940, complex(3.6, 0.0), 5000, 1.0, 1.0)
    ok_all &= check("  k=0", abs(A) < 1e-15 and abs(R + T - 1) < 1e-12, f"(A={A:.1e}, R+T={R+T:.12f})")

    print("\n" + ("모든 검증 통과" if ok_all else "실패한 검증 있음"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
