"""
Pilot: 관계사 스필오버(related-company spillover) feature -- BASE vs BASE+RELATED.

Temporal GNN을 바로 구현하는 대신, 최대주주 관계(현대자동차 -> 현대로템, 지분 33.8%)의
최근 수익률을 feature 2개로 추가했을 때 신호가 있는지부터 XGBoost로 싸게 확인한다.
quant_sector_rotation이 본 모델 전에 check_sector_reproducibility.py로 먼저 확인했던
것과 같은 fail-fast 절차.

⚠️ 상관관계로 그래프를 추론하지 않는다 (순환논리 위험). 대신 이미 알려진 자본관계
(현대자동차가 현대로템 지분 33.8% 보유)라는, 가격 움직임과 독립적인 근거로 관계를 정의한다.

전제:
    - data/064350_features_triple_barrier_pt2sl1_nd20_hl_base.csv 가 이미 있어야 함
      (production 설정과 동일, pt_sl=(2,1), num_days=20)
    - 005380.KS(현대자동차) 가격 데이터는 이 스크립트가 yfinance로 직접 받아옴

사전등록: docs/prereg_related_company_spillover_pilot.md 참고

사용법 (레포 루트에서):
    python -m src.experiments.run_related_company_pilot
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TICKER_KRX = "064350"
CONFIG_LABEL = "pt2sl1_nd20_hl"  # production 채택 설정과 동일

RELATED_TICKER = "005380.KS"  # 현대자동차 -- 064350 최대주주(지분 33.8%)

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
RELATED_FEATURE_COLS = ["related_return_5d", "related_return_20d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + RELATED_FEATURE_COLS

TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, 20
SEEDS = [42, 1, 7, 123, 2024]

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
)


# ------------------------------------------------------------------
# 1. 데이터 로드 -- 064350 triple-barrier BASE + 현대자동차 관계형 feature 병합
# ------------------------------------------------------------------
def load_base_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_base.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 -- quant_xgboost/quant_position_sizing에서 먼저 이 config로 "
            f"triple-barrier BASE 데이터셋을 만들어야 함 (pt_sl=(2,1), num_days=20)."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def load_related_features(start: str, end: str) -> pd.DataFrame:
    related = yf.download(RELATED_TICKER, start=start, end=end, progress=False)
    if isinstance(related.columns, pd.MultiIndex):
        related.columns = related.columns.get_level_values(0)
    close = related["Close"]

    out = pd.DataFrame(index=close.index)
    out["related_return_5d"] = close.pct_change(5)
    out["related_return_20d"] = close.pct_change(20)
    return out.dropna()


def build_dataset() -> pd.DataFrame:
    base = load_base_dataset()
    related = load_related_features(
        start=(base.index.min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
        end=(base.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    merged = base.join(related, how="inner")
    dropped = len(base) - len(merged)
    print(f"관계형 feature 병합: {len(base)}행 -> {len(merged)}행 "
          f"(거래일 불일치로 {dropped}행 제외)")
    return merged


# ------------------------------------------------------------------
# 2. Walk-Forward 분할 (기존 ablation 스크립트들과 동일)
# ------------------------------------------------------------------
def walk_forward_splits(n_rows: int, train_size: int, test_size: int, step: int, embargo: int):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = list(range(start, start + train_size))
        test_start = start + train_size + embargo
        test_idx = list(range(test_start, test_start + test_size))
        splits.append((train_idx, test_idx))
        start += step
    return splits


# ------------------------------------------------------------------
# 3. fold별 학습 + 평가 (BASE vs BASE+RELATED, 같은 fold에서 나란히 비교)
# ------------------------------------------------------------------
def run_seed(df: pd.DataFrame, seed: int) -> dict:
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    base_auc, related_auc = [], []
    for train_idx, test_idx in splits:
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if len(set(y_test)) < 2:
            continue

        m_base = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        m_base.fit(df[FEATURE_COLS_BASE].iloc[train_idx], y_train)
        proba_base = m_base.predict_proba(df[FEATURE_COLS_BASE].iloc[test_idx])[:, 1]

        m_rel = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        m_rel.fit(df[FEATURE_COLS_COMBINED].iloc[train_idx], y_train)
        proba_rel = m_rel.predict_proba(df[FEATURE_COLS_COMBINED].iloc[test_idx])[:, 1]

        base_auc.append(roc_auc_score(y_test, proba_base))
        related_auc.append(roc_auc_score(y_test, proba_rel))

    return {
        "seed": seed,
        "n_folds": len(base_auc),
        "base_auc_mean": float(np.mean(base_auc)),
        "related_auc_mean": float(np.mean(related_auc)),
        "auc_diff": float(np.mean(related_auc) - np.mean(base_auc)),
    }


# ------------------------------------------------------------------
# 4. 메인
# ------------------------------------------------------------------
def main():
    print(f"=== 관계형 feature(자본관계 스필오버) pilot: {TICKER_KRX} <- {RELATED_TICKER} ===\n")
    df = build_dataset()
    print(f"최종 데이터 {len(df)}행 ({df.index.min().date()} ~ {df.index.max().date()})\n")

    results = [run_seed(df, seed) for seed in SEEDS]
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    n_pass = int((result_df["auc_diff"] > 0).sum())
    diff_mean = result_df["auc_diff"].mean()
    diff_std = result_df["auc_diff"].std()

    print(f"\n{n_pass}/{len(SEEDS)} 시드에서 관계형 feature 추가가 baseline AUC를 앞섬")
    if diff_mean != 0:
        print(f"AUC 차이 평균: {diff_mean:+.4f} (표준편차 {diff_std:.4f}, "
              f"std/mean = {abs(diff_std / diff_mean):.1%})")

    if n_pass < 4:
        print(
            "\n판정: [1차 스크리닝 실패] -- 관계형 정보 자체가 이 파이프라인에서 신호 없음.\n"
            "quant_sector_rotation과 같은 결론: Temporal GNN으로 확장할 근거 부족, "
            "탐색 라인 종료 권장."
        )
    else:
        print(
            "\n판정: [1차 스크리닝 통과] -- 다음 단계로 DART 관계 데이터(특수관계자/주요계약)로 "
            "진짜 그래프를 구축하고 Temporal GNN 검토 필요.\n"
            "⚠️ 여전히 AUC 단계일 뿐, 실전 손익 검증 전까지 본 모델을 짓지 말 것."
        )


if __name__ == "__main__":
    main()