"""
si_data.py — Si 복소 굴절률 n(lambda), k(lambda) 로더 + 보간

=== 실제 파일 형식 (2026-09-22 확인) ===
사용자의 data/Si_Green2008.csv는 refractiveindex.info에서 받은 원본 그대로이며,
아래와 같은 "2단 블록" 구조다:

    wl,n
    2.5000e-01,1.6650e+00
    ...
    1.4500e+00,3.4850e+00
    <빈 줄>
    wl,k
    2.5000e-01,3.6650e+00
    ...

- 파장 단위는 nm이 아니라 **um(마이크로미터)** (예: 2.5000e-01 = 0.25um = 250nm)
- n 블록과 k 블록은 서로 다른 파장 간격으로 샘플링되어 있을 수 있음
  (실제로 refractiveindex.info는 종종 n, k를 다른 원본 문헌에서 따로 따와서
  붙이기 때문에 grid가 다르다). 그래서 n용 보간기와 k용 보간기를 완전히
  독립적으로 만든다 — 같은 줄 번호끼리 짝지으면 안 됨.

우선순위:
  1) data/Si_Green2008.csv  (실제 refractiveindex.info 원본, 위 형식)
  2) data/Si_nk_reference_800_1000nm.csv (이 세션에서 만든 참고값 - 출처 확인 필요,
     단일 wavelength_nm,n,k 표 형식. Si_Green2008.csv가 없을 때만 폴백)
"""

import csv
import os
import numpy as np
from scipy.interpolate import interp1d

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

_PRIMARY_PATH = os.path.join(_REPO_ROOT, "data", "Si_Green2008.csv")
_FALLBACK_PATH = os.path.join(_REPO_ROOT, "data", "Si_nk_reference_800_1000nm.csv")


def _parse_rii_two_block(path):
    """
    refractiveindex.info 원본 형식 파서: 'wl,n' 블록 + 빈줄 + 'wl,k' 블록.
    파장은 um 단위로 저장되어 있으므로 nm으로 변환해서 반환한다.

    Returns
    -------
    wl_n_nm, n_vals, wl_k_nm, k_vals : np.ndarray (각각 독립적인 길이/그리드)
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f]

    blocks = []
    current = []
    for line in lines:
        if line == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    wl_n = n_vals = wl_k = k_vals = None

    for block in blocks:
        header = block[0].lower().replace(" ", "")
        if header not in ("wl,n", "wl,k"):
            continue
        wl, val = [], []
        for row in block[1:]:
            parts = row.split(",")
            if len(parts) < 2:
                continue
            try:
                wl.append(float(parts[0]))
                val.append(float(parts[1]))
            except ValueError:
                continue
        wl_nm = np.array(wl) * 1000.0  # um -> nm
        val_arr = np.array(val)
        if header == "wl,n":
            wl_n, n_vals = wl_nm, val_arr
        else:
            wl_k, k_vals = wl_nm, val_arr

    if wl_n is None or wl_k is None:
        raise ValueError(
            f"{path}: 'wl,n' / 'wl,k' 두 블록을 모두 찾지 못했습니다. "
            "refractiveindex.info 원본 형식이 맞는지 확인하세요."
        )
    return wl_n, n_vals, wl_k, k_vals


def _parse_simple_table(path):
    """폴백용: wavelength_nm,n,k 단일 표 형식."""
    wl, n, k = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wl.append(float(row["wavelength_nm"]))
            n.append(float(row["n"]))
            k.append(float(row["k"]))
    wl = np.array(wl)
    return wl, np.array(n), wl, np.array(k)


class SiOpticalConstants:
    def __init__(self, csv_path=None, verbose=True):
        is_rii_format = True
        if csv_path is None:
            if os.path.exists(_PRIMARY_PATH):
                csv_path = _PRIMARY_PATH
                is_rii_format = True
            elif os.path.exists(_FALLBACK_PATH):
                csv_path = _FALLBACK_PATH
                is_rii_format = False
            else:
                raise FileNotFoundError(
                    "Si n,k CSV를 찾을 수 없습니다. data/Si_Green2008.csv를 준비하세요."
                )

        self.source_path = csv_path

        try:
            wl_n, n_vals, wl_k, k_vals = (
                _parse_rii_two_block(csv_path) if is_rii_format else _parse_simple_table(csv_path)
            )
        except Exception as e:
            if is_rii_format:
                wl_n, n_vals, wl_k, k_vals = _parse_simple_table(csv_path)
            else:
                raise e

        order_n = np.argsort(wl_n)
        order_k = np.argsort(wl_k)
        self.wl_n_nm, self.n_data = wl_n[order_n], n_vals[order_n]
        self.wl_k_nm, self.k_data = wl_k[order_k], k_vals[order_k]

        self._n_interp = interp1d(self.wl_n_nm, self.n_data, kind="cubic", bounds_error=True)
        self._k_interp = interp1d(self.wl_k_nm, self.k_data, kind="cubic", bounds_error=True)

        if verbose:
            is_reference = os.path.basename(csv_path) == "Si_nk_reference_800_1000nm.csv"
            tag = "[참고 데이터 - 출처 확인 필요]" if is_reference else "[Si_Green2008.csv 원본 데이터]"
            print(
                f"Si n,k 데이터 로드: {csv_path} {tag}\n"
                f"  n 그리드: {self.wl_n_nm.min():.1f}-{self.wl_n_nm.max():.1f} nm ({len(self.wl_n_nm)}점)\n"
                f"  k 그리드: {self.wl_k_nm.min():.1f}-{self.wl_k_nm.max():.1f} nm ({len(self.wl_k_nm)}점)"
            )

    def N(self, wavelength_nm):
        """복소 굴절률 N = n - i*k (이 프로젝트의 통일된 부호 규약)."""
        n = float(self._n_interp(wavelength_nm))
        k = float(self._k_interp(wavelength_nm))
        return complex(n, -k)
