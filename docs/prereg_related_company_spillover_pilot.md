# 사전등록: 관계사 스필오버(related-company spillover) feature pilot

- 레포(제안): `quant_network_spillover`
- 브랜치(제안): `experiment/related-company-spillover-pilot`
- 작성일: 2026-09-18
- 배경: Temporal GNN(시계열 그래프 신경망, 공급망/자본흐름 파급효과 모델링) 검토.
  `quant_sector_rotation`의 PROJECT_SUMMARY.md 향후 과제에 "섹터 스필오버"가 아직
  다루지 않은 재료로 이미 명시돼 있었음. GNN을 바로 구현하기 전에, `quant_sector_rotation`이
  본 모델 전에 `check_sector_reproducibility.py`로 값싸게 먼저 확인했던 것과 같은 방식으로,
  **관계형 정보 자체가 신호가 있는가**만 XGBoost feature 1~2개 추가로 먼저 검증한다.

## 1. 가설 / 질문

064350(현대로템)은 현대자동차(005380)가 지분 33.8%를 보유한 계열사다 (자본관계 실재,
상상으로 만든 관계 아님). 최대주주 현대자동차의 최근 수익률(모멘텀)을 feature로 추가하면,
064350의 triple-barrier 방향 예측력이 BASE 13개 feature만 쓸 때보다 나아지는가?

## 2. 고정할 파라미터 (사전등록 -- 결과 보고 바꾸지 않음)

| 파라미터 | 값 |
|---|---|
| 대상 종목 | 064350 (현대로템) |
| 관계사 | 005380.KS (현대자동차) -- 최대주주(33.8% 지분), 자본관계 기준 |
| feature set | FEATURE_COLS_BASE(13개) + RELATED(2개: `related_return_5d`, `related_return_20d`) |
| 라벨 | `label_tb_binary` (pt_sl=(2,1), num_days=20 -- production 설정과 동일) |
| walk-forward | train_size=300, test_size=60, step=60, embargo=20 |
| 날짜 정렬 | 두 종목 다 KRX 거래일 기준이라 shift 불필요 (같은 날 종가 기준 동시 관측) |
| seeds | 42, 1, 7, 123, 2024 |
| 1차 평가지표 | fold별 AUC (BASE vs BASE+RELATED), 5-seed 비교 |

## 3. 통과 기준 (1차 스크리닝 -- AUC만 봄)

- 5-seed 중 4개 이상에서 `related_auc_mean > base_auc_mean`
- std/mean < 50%

## 4. 이 pilot에서 하지 않는 것

- 그래프 구조/GNN 구현하지 않음 -- 관계형 정보 자체의 유효성만 확인
- 실전 backtest는 이 단계에서 하지 않음 (1차 통과 시에만 다음 단계)
- 상관관계 기반 그래프(경로 A)는 순환논리 위험이 있어 이번엔 사용하지 않음 -- 자본관계(실제
  지분 보유)라는, 가격 상관관계와 독립적인 근거로 관계를 정의함

## 5. 알려진 리스크 (미리 인지)

- feature 2개 추가가 노이즈로 작용해 과적합만 시킬 수 있음 (lag feature 실험 때 AUC는
  오르고 vs_base_rate는 떨어졌던 패턴 참고)
- 현대자동차는 초대형/고유동성 종목이라 현대로템(저유동성)과 변동성 스케일이 달라서
  단순 수익률만으로는 신호가 희석될 수 있음 -- 신호가 약하게 나오면 정규화 방식(z-score 등)
  변경도 고려할 수 있으나, 이번 1차 pilot에서는 하지 않고 실패로 기록

## 6. 판정 후 처리

- 1차 통과 (4/5 이상) -> 다음 단계: DART 관계 데이터(특수관계자/주요계약 공시)로 진짜
  그래프 구축 + Temporal GNN 검토
- 1차 실패 -> `quant_sector_rotation`과 같은 결론(관계형 정보 자체가 이 파이프라인에서
  신호 없음)으로 기록하고, Temporal GNN 탐색 라인 종료