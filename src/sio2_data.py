"""
sio2_data.py — SiO2(용융 실리카) 굴절률 로더 (3주차 ARC 비교용)

데이터: data/SiO2_Malitson1965.csv  (refractiveindex.info, shelf=MAIN, book=SiO2, page=Malitson)
  - 형식: 'wl,n' 단일 블록, 파장 단위 um, 0.21–6.7 um
  - k 열이 없다 = 이 파장대에서 흡수를 무시(k=0). 용융 실리카는 800–1000 nm에서
    사실상 투명하므로 ARC 계산에서 무손실로 둔다.
원 출처: I. H. Malitson, "Interspecimen Comparison of the Refractive Index of Fused Silica",
         J. Opt. Soc. Am. 55, 1205-1209 (1965)
  - refractiveindex.info의 이 페이지 값은 측정표가 아니라 논문의 Sellmeier 식으로 계산된 값.
    그래서 sellmeier_n()으로 같은 식을 직접 계산해 CSV와 교차검증할 수 있다
    (scripts/validate_sio2.py).
"""
import os
import numpy as np
from scipy.interpolate import interp1d

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "data", "SiO2_Malitson1965.csv")

# Malitson 1965 Sellmeier 계수 (파장 um):  n^2 - 1 = sum B_i * L^2 / (L^2 - C_i^2)
_B = (0.6961663, 0.4079426, 0.8974794)
_C = (0.0684043, 0.1162414, 9.896161)


def sellmeier_n(wavelength_nm):
    L2 = (np.asarray(wavelength_nm, dtype=float) / 1000.0) ** 2
    n2 = 1.0 + sum(B * L2 / (L2 - C ** 2) for B, C in zip(_B, _C))
    return np.sqrt(n2)


class SiO2OpticalConstants:
    def __init__(self, csv_path=_PATH):
        data = np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8-sig")
        self.source_path = csv_path
        self.wl_nm = data["wl"] * 1000.0  # um -> nm
        self.n_data = data["n"]
        self._n = interp1d(self.wl_nm, self.n_data, kind="cubic", bounds_error=True)

    def N(self, wavelength_nm):
        """복소 굴절률 N = n - ik (k=0, 무손실)."""
        return complex(float(self._n(wavelength_nm)), 0.0)
