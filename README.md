# 포톤 트래핑 나노구조 Si 포토다이오드 시뮬레이션

포톤 트래핑 나노구조를 이용한 Si 포토다이오드의 근적외선(NIR) 흡수 향상을 
TMM + FDTD 시뮬레이션으로 분석하는 15주 학부연구 프로젝트입니다.

## 연구 질문
- Q1: 평면 Si가 놓치는 800-1000nm 광을 나노홀 배열이 얼마나 더 흡수시키는가?
- Q2: 940nm에서 최적의 지름·주기·깊이는? 그 물리적 이유는?
- Q3: 주기 배열을 무질서하게 만들면 광대역 흡수와 각도 강건성이 개선되는가?

## 진행 상황
- [x] 1주차: 환경 세팅, Si 흡수계수 베이스라인 계산
- [x] 2주차: TMM 구현 (해석해·에너지보존 검증)
- [x] 3주차: 베이스라인 정량화 (두께 스윕, SiO2 ARC, Fig.1)
- [x] 4주차: MEEP 설치, 1D Si 슬랩 MEEP–TMM 일치 검증 (`scripts/validate_meep_1d.py`)
- [ ] 5-7주차: 2D/3D FDTD 및 문헌 재현
- [ ] 8-11주차: 파라미터 스윕
- [ ] 12-13주차: 무질서 배열 비교
- [ ] 14-15주차: 정리 및 발표

## 환경
- Python 3.11 (Miniconda, 가상환경명: photontrap)
- NumPy, Matplotlib, SciPy, JupyterLab
- MEEP (pymeep): Windows 미지원 → WSL2 + conda 별도 환경 (`notes/04_meep_install_wsl2.md`)

## 지도교수
이도언 교수 (한양대학교 ERICA 전자공학부)
