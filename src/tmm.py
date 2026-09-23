"""
tmm.py — 전송행렬법(Transfer Matrix Method) 핵심 모듈
2주차: 포톤 트래핑 Si 포토다이오드 프로젝트

=== 부호 규약 (반드시 지킬 것) ===
복소 굴절률: N = n - i*k  (k >= 0, 흡수를 나타내는 소광계수)
  - 이 규약은 Green 2008 / pveducation.org 데이터 테이블이 명시한 그대로다
    ("refractive index = n - ik").
  - 시간 의존성은 exp(+i*omega*t) (광학 박막 이론의 표준, Macleod 규약)이며,
    이 규약에서는 흡수매질에서 파동이 자연스럽게 감쇠한다. 다른 책(Born&Wolf,
    Byrnes arXiv:1603.02720 등)은 N = n + i*k, exp(-i*omega*t)를 쓰기도 하므로
    남의 코드/공식을 그대로 섞어 쓰면 부호가 뒤집혀 "이득(gain)"이 나오는 흔한
    버그가 생긴다. 이 파일 안에서는 전부 N = n - ik로 통일한다.

=== 방법론 ===
Macleod, H. A., "Thin-Film Optical Filters", 4th ed. (CRC Press, 2010)의
특성행렬(characteristic matrix) 방법을 수직입사(normal incidence)로 구현.
각 층 j에 대해:
    delta_j = (2*pi/lambda0) * N_j * d_j            (위상 두께, 복소수)
    M_j = [[ cos(delta_j),          i*sin(delta_j)/N_j ],
           [ i*N_j*sin(delta_j),    cos(delta_j)       ]]
전체 행렬 M = M_1 @ M_2 @ ... @ M_k
[B, C]^T = M @ [1, N_sub]^T   (N_sub: 마지막 매질의 굴절률)
r = (N_0*B - C) / (N_0*B + C)
t = 2*N_0 / (N_0*B + C)
R = |r|^2
T = Re(N_sub)/Re(N_0) * |t|^2   (입사매질이 무손실 실수일 때 표준식)
A = 1 - R - T
"""

import numpy as np


def fresnel_single_interface(n1: complex, n2: complex):
    """
    단일 계면(반무한 매질1 -> 반무한 매질2)의 Fresnel 반사/투과 계수.
    수직입사. TMM 결과 검증용 해석해(analytic solution)로만 사용한다.

    반환: r, t (복소 진폭 계수), R, T (에너지 반사율/투과율)
    """
    n1 = complex(n1)
    n2 = complex(n2)
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    R = np.abs(r) ** 2
    # 입사매질이 무손실 실수라는 가정 하의 표준 투과율식
    T = (n2.real / n1.real) * np.abs(t) ** 2
    return r, t, R, T


def tmm_normal_incidence(wavelength_nm, n_layers, d_layers_nm, n_incident, n_substrate):
    """
    수직입사 다층박막 TMM. 하나의 파장에 대한 R, T, A를 반환.

    Parameters
    ----------
    wavelength_nm : float
    n_layers : list[complex]      내부 층들의 복소 굴절률 (N = n - ik), 입사/기판 제외
    d_layers_nm : list[float]     내부 층들의 두께 [nm], n_layers와 길이 동일
    n_incident : complex          입사매질 (보통 공기 = 1+0j)
    n_substrate : complex         투과 측 반무한 매질

    Returns
    -------
    R, T, A : float
    """
    assert len(n_layers) == len(d_layers_nm), "층 개수와 두께 개수가 다릅니다"

    k0 = 2 * np.pi / wavelength_nm  # nm^-1

    M = np.eye(2, dtype=complex)
    for N_j, d_j in zip(n_layers, d_layers_nm):
        N_j = complex(N_j)
        delta = k0 * N_j * d_j
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        M_j = np.array([
            [cos_d,              1j * sin_d / N_j],
            [1j * N_j * sin_d,   cos_d],
        ], dtype=complex)
        M = M @ M_j

    n0 = complex(n_incident)
    ns = complex(n_substrate)
    BC = M @ np.array([1.0, ns], dtype=complex)
    B, C = BC[0], BC[1]

    r = (n0 * B - C) / (n0 * B + C)
    t = 2 * n0 / (n0 * B + C)

    R = np.abs(r) ** 2
    T = (ns.real / n0.real) * np.abs(t) ** 2
    A = 1.0 - R - T
    return R, T, A


def tmm_spectrum(wavelengths_nm, n_layers_of_lambda, d_layers_nm, n_incident_of_lambda, n_substrate_of_lambda):
    """
    여러 파장에 대해 tmm_normal_incidence를 반복 적용.

    n_layers_of_lambda(lam) -> list[complex]  각 파장에서 내부 층들의 N
    n_incident_of_lambda(lam) -> complex
    n_substrate_of_lambda(lam) -> complex
    """
    R = np.zeros(len(wavelengths_nm))
    T = np.zeros(len(wavelengths_nm))
    A = np.zeros(len(wavelengths_nm))
    for i, lam in enumerate(wavelengths_nm):
        n_layers = n_layers_of_lambda(lam)
        n_inc = n_incident_of_lambda(lam)
        n_sub = n_substrate_of_lambda(lam)
        R[i], T[i], A[i] = tmm_normal_incidence(lam, n_layers, d_layers_nm, n_inc, n_sub)
    return R, T, A
