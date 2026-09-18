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

- 1차 통과 (4/5 이상) -> 다음 단계: 실제 backtest 체크포인트(Addendum 참고) 통과 후에만
  DART 관계 데이터로 진짜 그래프 구축 + Temporal GNN 검토
- 1차 실패 -> `quant_sector_rotation`과 같은 결론(관계형 정보 자체가 이 파이프라인에서
  신호 없음)으로 기록하고, Temporal GNN 탐색 라인 종료

## 7. 결과 (2026-09-18)

5/5 시드 전부 양수 (AUC 차이 평균 +0.0076, std/mean 37.2%). 사전등록 기준(4/5, std/mean
<50%) 통과.

**판정: [1차 스크리닝 통과]**

## Addendum (2026-09-18): backtest 체크포인트 삽입

원래 6절은 "1차 통과 시 바로 GNN 검토"로 적어뒀으나, 이건 `learnings.md`의 "AUC 개선 ≠
실전 손익 개선" 원칙과 어긋난다. 이 프로젝트에서 AUC 5/5 통과 후 백테스트/PBO 단계에서
뒤집힌 전례가 다수(lag feature: AUC 5/5인데 vs_base_rate 0/5; 코스피/코스닥 PER: ablation
4/5인데 백테스트 1/5; triple-barrier 풀링: 5-seed 통과 후 PBO 95.6%로 배포 부적합) 있으므로,
GNN 투자 전에 같은 패턴이 재현되는지 먼저 확인한다.

### 추가 단계

BASE와 BASE+RELATED 각각으로 threshold=0.65(production 채택값) 진입 규칙의 실제 거래를
생성하고, 거래비용(0.2% 왕복) 반영한 순수익을 고정 Buy & Hold 구간과 비교한다. 국면 배제
재검증(집중 연도 제외)까지 포함.

### 판정 기준

- 5-seed 중 4개 이상에서 BASE+RELATED net_return > BASE net_return **그리고** > Buy & Hold
- 집중 연도 제외 후에도 우위 유지 (국면 우연 배제)

### 판정 후 처리 (갱신)

- 통과 -> DART 관계 데이터로 진짜 그래프 구축 + Temporal GNN 검토
- 실패 -> AUC 개선이 이번에도 실전 edge로 안 이어진 것으로 기록, Temporal GNN 탐색 라인
  종료 (관계형 feature 자체는 향후 다른 조합에서 재사용 가능하도록 코드만 남김)