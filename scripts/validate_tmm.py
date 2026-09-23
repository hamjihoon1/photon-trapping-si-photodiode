"""
validate_tmm.py — 2주차 체크포인트: TMM 코드 검증
계획서 요구사항: "단일 계면 해석해와 에너지 보존(R+T+A=1)으로 반드시 직접 검증"

3가지 검증을 수행한다:
  (1) 내부 층이 0개인 TMM = 단일계면 Fresnel 해석해와 정확히 일치해야 함
  (2) 흡수가 없는(k=0) 다층 구조: R+T = 1 (A~0)이어야 함 (수치오차 수준)
  (3) 흡수가 있는 임의 다층 구조: R+T+A = 1이 항상 성립해야 함 (에너지 보존)
  (4) 참고문헌값 대조: HeNe 레이저 파장(632.8nm)에서 공기/Si 계면 반사율이
      문헌에 보고된 값(R≈35%)과 맞는지 확인 (출처: Filmetrics/KLA Si 굴절률
      데이터베이스, n=3.88163, k=0.01896923 @ 632.8nm)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tmm import fresnel_single_interface, tmm_normal_incidence

TOL = 1e-10


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}  {detail}")
    return cond


def main():
    all_pass = True

    # ---------------------------------------------------------------
    # (1) 내부 층 0개 TMM vs 단일계면 Fresnel 해석해
    # ---------------------------------------------------------------
    print("=== 검증 1: 단일 계면 해석해 비교 (내부 층 없음) ===")
    test_cases = [
        (1.0 + 0j, 3.6 - 0.001j, "공기 -> 손실 있는 Si (940nm 근방 값)"),
        (1.0 + 0j, 3.9 - 0.02j, "공기 -> Si (632.8nm 근방 값)"),
        (1.0 + 0j, 1.45 + 0j, "공기 -> SiO2 (무손실)"),
        (1.5 + 0j, 1.0 + 0j, "유리 -> 공기 (n1>n2)"),
    ]
    for n0, ns, desc in test_cases:
        r_ana, t_ana, R_ana, T_ana = fresnel_single_interface(n0, ns)
        R_tmm, T_tmm, A_tmm = tmm_normal_incidence(940.0, [], [], n0, ns)
        ok = check(
            f"  {desc}",
            abs(R_tmm - R_ana) < TOL and abs(T_tmm - T_ana) < TOL,
            f"(해석해 R={R_ana:.6f}, T={T_ana:.6f} | TMM R={R_tmm:.6f}, T={T_tmm:.6f})",
        )
        all_pass &= ok

    # ---------------------------------------------------------------
    # (2) 무손실 다층 구조: R+T=1, A~0
    # ---------------------------------------------------------------
    print("\n=== 검증 2: 무손실(k=0) 다층 구조 -> R+T=1 ===")
    rng = np.random.default_rng(42)
    for i in range(20):
        n_layers = [complex(rng.uniform(1.4, 4.0), -0.0) for _ in range(rng.integers(1, 5))]
        d_layers = [rng.uniform(50, 500) for _ in n_layers]
        lam = rng.uniform(800, 1000)
        R, T, A = tmm_normal_incidence(lam, n_layers, d_layers, 1.0 + 0j, 1.5 + 0j)
        ok = abs(R + T + A - 1.0) < 1e-9 and abs(A) < 1e-9
        all_pass &= ok
        if not ok or i < 3:
            check(f"  랜덤 무손실 스택 #{i}", ok, f"(R={R:.6f}, T={T:.6f}, A={A:.2e}, layers={len(n_layers)})")

    # ---------------------------------------------------------------
    # (3) 흡수 있는 임의 다층 구조: R+T+A = 1 (에너지 보존)
    # ---------------------------------------------------------------
    print("\n=== 검증 3: 흡수 있는 다층 구조 -> R+T+A=1 (에너지 보존) ===")
    n_fail = 0
    N = 500
    for i in range(N):
        # 주의: 이 프로젝트의 부호 규약은 N = n - i*k (k>=0 가 흡수).
        # complex(n, -k) 형태로 만들어야 물리적으로 "손실"이 된다.
        n_layers = [complex(rng.uniform(1.4, 4.5), -rng.uniform(0.0, 0.5)) for _ in range(rng.integers(1, 6))]
        d_layers = [rng.uniform(10, 3000) for _ in n_layers]
        lam = rng.uniform(400, 1200)
        n_inc = 1.0 + 0j
        n_sub = complex(rng.uniform(1.4, 4.5), -rng.uniform(0.0, 0.5))
        R, T, A = tmm_normal_incidence(lam, n_layers, d_layers, n_inc, n_sub)
        residual = abs(R + T + A - 1.0)
        valid_range = (-1e-9 <= R <= 1 + 1e-9) and (-1e-9 <= T <= 1 + 1e-9) and (A >= -1e-9)
        if residual > 1e-8 or not valid_range:
            n_fail += 1
    ok = n_fail == 0
    all_pass &= ok
    check(f"  랜덤 흡수 다층 스택 {N}개", ok, f"(실패 {n_fail}/{N}, |R+T+A-1|max 확인됨)")

    # ---------------------------------------------------------------
    # (4) 문헌값 대조 — 632.8nm 공기/Si 반사율
    # ---------------------------------------------------------------
    print("\n=== 검증 4: 문헌 참고값 대조 (632.8nm 공기/Si 반사율) ===")
    n_si_hene = complex(3.88163, -0.01896923)  # 출처: Filmetrics/KLA Si n,k DB
    R, T, A = tmm_normal_incidence(632.8, [], [], 1.0 + 0j, n_si_hene)
    expected_R_approx = 0.348  # 직접 계산: |(1-3.88163-i0.019)/(1+3.88163+i0.019)|^2 근사
    ok = abs(R - expected_R_approx) < 0.005
    all_pass &= check(
        "  632.8nm 공기/Si 반사율",
        ok,
        f"(TMM R={R:.4f}, 기대값~{expected_R_approx:.3f}, 실리콘 가시광 반사율 '약 35%'로 "
        f"흔히 인용되는 값과 일치)",
    )

    print("\n" + "=" * 60)
    print("전체 결과:", "모두 통과 (ALL PASS)" if all_pass else "일부 실패 (SOME FAILED)")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
