## 관계사 스필오버(자본관계) feature pilot (2026-09)

### 배경
Temporal GNN(공급망/자본흐름 파급효과) 검토. quant_sector_rotation의 "섹터
스필오버" 미탐구 재료를 이어받아, GNN 전에 최대주주 관계(현대자동차->현대로템,
지분 33.8%)의 수익률 feature 2개만으로 신호가 있는지 fail-fast 검증.

### 결과
- AUC 1차 스크리닝: 5/5 통과 (평균 +0.0076, std/mean 37.2%)
- backtest 체크포인트 (threshold=0.65, 거래비용 반영):
  - BASE+RELATED > BASE: 1/5
  - BASE+RELATED > Buy&Hold: 0/5
  - 집중 연도(4/5 시드에서 2025년) 제외 후: 0/5

### 판정: [실패]
lag feature/코스피·코스닥 PER와 동일한 "AUC 개선 ≠실전 손익 개선" 패턴 재현.
2025년 국면 집중도 관계형 feature로 해소되지 않음. Temporal GNN 탐색 라인
종료 -- 관계형 feature 자체는 다른 조합(다른 관계사, 다른 라벨 프레임)에
재사용 가능하도록 코드는 보존.