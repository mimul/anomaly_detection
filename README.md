# Anomaly Detection

이상탐지 공부 시리즈([mimul.com](https://mimul.com))에서 쓰는 예제 코드 저장소입니다.

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

series03/
  common.py     중고차 데이터셋 로딩과 원-핫 인코딩
  regression.py 선형회귀, 다변량 선형회귀, 리지회귀, 감마회귀
  bayesian.py   베이지안 선형회귀, Metropolis-Hastings MCMC, 베이지안 로지스틱/포아송 회귀, 판매자 그룹차 예시
  datasets/
    usedcar_train.csv, usedcar_inference.csv           중고차 가격 합성 데이터
    earthquakes_volcanoes_train.csv, _test.csv          국가별 지진·화산 실측 데이터 (아래 "데이터 출처" 참고)

series04/
  common.py           계절성·추세가 있는 합성 시계열 생성 (점·맥락적·집단적 이상을 각각 하나씩 주입)
  distance_based.py   거리 기반 중 불일치 기반, Matrix Profile로 discord 탐지
  density_based.py    밀도 기반 중 트리 기반, 슬라이딩 윈도우 특징 + Isolation Forest
  prediction_based.py 예측 기반 중 예측 오차 기반, Holt-Winters 지수평활의 한 스텝 앞 예측 오차

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

python series03/regression.py
python series03/bayesian.py

python series04/distance_based.py
python series04/density_based.py
python series04/prediction_based.py
```

series01, series02, series04의 대부분 데이터는 고정된 시드로 매번 실행 시 새로 생성되므로 별도의 데이터 생성 스크립트를 먼저 실행할 필요가 없습니다. `series02/count_data.py`의 지진 데이터와 `series03/`의 데이터셋은 모두 `datasets/`에 미리 내려받아둔 파일을 사용합니다.

모든 스크립트는 저장소 루트 디렉터리에서 실행하는 것을 기준으로 상대 경로(`outputs/`)를 사용합니다. 생성된 이미지는 `outputs/` 아래에 블로그 글에서 쓰는 파일명 그대로 저장됩니다.

## 데이터 출처

- `series02/datasets/earthquake_kma.csv`: 기상청 지진 조회 서비스에서 내려받은 국내 지진 관측 데이터 (2020.1~2026.7, 규모 2.0 이상).
- `series02/count_data.py`, `series03/bayesian.py`의 NBA 3점슛 성공률 데이터(`player_stats_2022.csv`, `player_stats_2023.csv`)의 출처는 [NBA.com/stats](https://www.nba.com/stats/players/traditional)이다.
- `series03/datasets/usedcar_train.csv`, `usedcar_inference.csv`: 연식·주행거리·차종·판매자로 가격을 생성한 합성 데이터. price는 만원 단위, odometer는 만km 단위다.
- `series03/datasets/earthquakes_volcanoes_train.csv`, `_test.csv`: 미국 지질조사국(USGS)의 지진 데이터와 스미스소니언 박물관의 화산 데이터를 국가·연도 단위로 결합한 실측 데이터.

## 참고

Python 3.14 환경에서는 `requirements.txt`에 적힌 최소 버전보다 낮은 numpy/pandas/matplotlib 조합에서 세그멘테이션 폴트나 렌더링 오류가 발생할 수 있다. 문제가 생기면 `pip install -U numpy pandas matplotlib`로 최신 버전을 받아 해결한다.
