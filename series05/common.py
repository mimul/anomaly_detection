"""
5편 - 전처리, 성능평가, 교차검증에서 공용으로 쓰는 데이터 로딩 함수

전처리와 성능평가, 교차검증은 특정 모델에 종속된 주제가 아니라 어떤
데이터셋에도 적용되는 절차이므로, 1편에서 만든 5센서 생산 라인
데이터셋을 그대로 재사용한다.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "series01", "common.py")
_spec = importlib.util.spec_from_file_location("series01_common", _path)
_series01_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_series01_common)

COLUMNS = _series01_common.COLUMNS
generate_dataset = _series01_common.generate_dataset
