# 4주차: MEEP 설치 (Windows → WSL2 + conda)

## 왜 WSL2인가
MEEP 공식 설치 문서(Installation.md)에 따르면 conda-forge의 `pymeep` 패키지는
Linux·macOS용만 있고 **native Windows는 지원하지 않는다**. 공식 권장 경로는
WSL(Windows Subsystem for Linux)로 Ubuntu를 깔고 그 안에서 conda로 설치하는 것.
계획서 4주차도 같은 경로("WSL2/Linux에 conda install -c conda-forge pymeep")를 지정한다.

→ 지금 Windows 쪽 `photontrap` 환경(TMM 코드용)은 그대로 두고, **WSL 안에 MEEP용 환경을 하나 더** 만든다.

## 1. WSL2 + Ubuntu 설치 (Windows, 관리자 권한 PowerShell)
```
wsl --install -d Ubuntu
```
- 재부팅 후 Ubuntu가 안 뜨고 `wsl -l -v`가 "설치된 배포가 없습니다"이면 같은 명령을 한 번 더
- 다운로드 중 `WININET_E_CONNECTION_RESET` → `wsl --install -d Ubuntu --web-download`
- 창 제목에 "선택"이 붙으면 진행이 멈춘 것 → Esc (실행 중 창 안 왼쪽 클릭 금지)
- 재부팅 후 Ubuntu 창이 뜨면 리눅스 사용자 이름/비밀번호 설정
- 확인: PowerShell에서 `wsl -l -v` → Ubuntu의 VERSION이 2

## 2. Ubuntu 안에 Miniconda 설치 (Ubuntu 창)
```
cd ~        # 꼭 홈으로 먼저 (처음엔 /mnt/c/WINDOWS/system32 에서 시작함)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh      # 약관 yes, 위치 Enter, 마지막 init 질문 yes (기본값이 no라 직접 입력)
exec bash                                   # 셸 다시 열기
```

## 3. MEEP 환경 만들기
```
conda create -n photontrap-meep -c conda-forge python=3.11 pymeep numpy scipy matplotlib jupyterlab
conda activate photontrap-meep
python -c "import meep as mp; print(mp.__version__)"
```
- 공식 문서 주의사항: 이 환경에 패키지를 더 넣을 때는 항상 `-c conda-forge`를 붙인다
  (defaults 채널의 MKL numpy와 섞이면 충돌 가능).
- `python=3.11`로 해결이 안 된다고 나오면, 그 부분만 빼고 다시 만들기 (MEEP 쪽 코드는 버전 무관).
- 참고로 Claude 작업 환경에서는 MEEP **1.31.0** (Python 3.11.15, 소스 빌드)로 검증했고, 사용자 PC(WSL2 + conda, MEEP **1.35.0**, Python 3.11.17)에서도 모든 수치가 같게 재현됐다 (2026-10-11).
- conda가 기본 채널 약관 동의를 요구하며 멈추면 명령에 `--override-channels`를 붙인다 (실제 설치에 쓴 명령):
  `conda create -n photontrap-meep -c conda-forge --override-channels python=3.11 pymeep numpy scipy matplotlib`

## 4. 프로젝트 폴더에서 실행
Windows 폴더는 WSL에서 `/mnt/c/...`로 보인다.
```
conda activate photontrap-meep
cd /mnt/c/projects/photon-trapping-project
python scripts/validate_meep_1d.py --quick   # 1–2분, 동작 확인용
python scripts/validate_meep_1d.py           # 전체 검증 (약 10분)
```
- WSL에서는 경로 구분자가 `\`가 아니라 `/` (python scripts/xxx.py)
- 그림은 figures/ 에 저장되므로 Windows 탐색기에서 그대로 열면 된다
- 그림의 한글: 스크립트가 `/mnt/c/Windows/Fonts/malgun.ttf`(맑은 고딕)를 자동 등록한다

## 5. git은 Windows 쪽(Anaconda Prompt)에서만
같은 폴더를 WSL의 git으로도 만지면 파일 권한·줄바꿈 차이 때문에 모든 파일이 "modified"로
보이는 일이 흔하다. **스크립트 실행은 WSL, git pull/commit/push는 지금처럼 Anaconda Prompt**로 나눈다.

## 안 될 때 (계획서 축소 방안)
"WSL2 + conda가 표준 경로. 그래도 안 되면 랩 워크스테이션에 원격 접속" — 막히면 3시간 룰대로 정리해서 공유.
