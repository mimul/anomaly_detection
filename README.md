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
outputs/          실행 결과 이미지 저장 위치 (실행 시 자동 생성)
```

## 실행 방법

```bash
pip install -r requirements.txt

python series01/eda.py
python series01/supervised.py
```

샘플 데이터는 고정된 시드로 매번 실행 시 새로 생성되므로 별도의 데이터 생성 스크립트를
먼저 실행할 필요가 없습니다. 모든 스크립트는 저장소 루트 디렉터리에서 실행하는 것을
기준으로 상대 경로(`outputs/`)를 사용합니다.

생성된 이미지는 `outputs/` 아래에 블로그 글에서 쓰는 파일명 그대로 저장됩니다
(`eda-histogram.png`, `eda-pca.png`, `svm-boundary.png`, `svm-overfitting.png`,
`logreg-boundary.png`, `logreg-probability.png`).
