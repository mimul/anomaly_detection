# anomaly_detection

이상탐지 공부 시리즈([mimul.com](https://mimul.com))에서 쓰는 예제 코드 저장소입니다.
[ghmagazine/python_anomaly_detection_book](https://github.com/ghmagazine/python_anomaly_detection_book)의 예제를 참고해 한글로 옮기고,
시리즈 편별로 필요한 부분만 골라 Python 스크립트로 재구성했습니다.

## 구조

시리즈 편 번호를 그대로 디렉터리 이름으로 씁니다.

```
series01/
  common.py       샘플 데이터 생성 (다른 스크립트에서 공용으로 사용)
  eda.py          탐색적 데이터 분석과 시각화
  supervised.py   SVM, 로지스틱 회귀

series02/
  common.py         샘플 데이터 생성 (생산 라인 데이터, GMM/kNN용 합성 데이터)
  univariate.py     확률분포 최우추정, 1변량 홀텔링 이론, 비정규분포
  count_data.py     이항분포, 포아송분포, 과분산(음이항분포)
  multivariate.py   다변량 홀텔링 이론, 마할라노비스-다구치법
  nonparametric.py  GMM, kNN
  datasets/
    earthquake_kma.csv   국내 지진 실측 데이터 (아래 "데이터 출처" 참고)

outputs/          실행 결과 이미지 저장 위치 (실행 시 자동 생성)
```

## 실행 방법

```bash
pip install -r requirements.txt

python series01/eda.py
python series01/supervised.py

python series02/univariate.py
python series02/count_data.py
python series02/multivariate.py
python series02/nonparametric.py
```

series01, series02의 대부분 데이터는 고정된 시드로 매번 실행 시 새로 생성되므로 별도의
데이터 생성 스크립트를 먼저 실행할 필요가 없습니다. `series02/count_data.py`의 지진 데이터만
`series02/datasets/earthquake_kma.csv`에 미리 내려받아둔 파일을 사용합니다.

모든 스크립트는 저장소 루트 디렉터리에서 실행하는 것을 기준으로 상대 경로(`outputs/`)를
사용합니다. 생성된 이미지는 `outputs/` 아래에 블로그 글에서 쓰는 파일명 그대로 저장됩니다.

## 데이터 출처

- `series02/datasets/earthquake_kma.csv`: 기상청 지진 조회 서비스에서 내려받은 국내 지진
  관측 데이터 (2020.1~2026.7, 규모 2.0 이상).
- 그 밖의 실데이터(NBA 3점슛 성공률 등)는 원서 저장소
  [ghmagazine/python_anomaly_detection_book](https://github.com/ghmagazine/python_anomaly_detection_book)의
  공개 CSV를 그대로 참조한다.

## 참고

Python 3.14 환경에서는 `requirements.txt`에 적힌 최소 버전보다 낮은 numpy/pandas/matplotlib
조합에서 세그멘테이션 폴트나 렌더링 오류가 발생할 수 있다. 문제가 생기면
`pip install -U numpy pandas matplotlib`로 최신 버전을 받아 해결한다.
