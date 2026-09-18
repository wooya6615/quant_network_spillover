"""
Backtest 체크포인트: BASE vs BASE+RELATED (관계형 feature) 실전 거래 성과 비교.

AUC pilot(run_related_company_pilot.py)이 5/5 통과했지만, 이 프로젝트에서
AUC 통과가 실전 손익으로 안 이어진 전례가 많아(lag feature, 코스피/코스닥 PER 등)
GNN 투자 전에 반드시 거쳐야 하는 단계. threshold=0.65(production 채택값)로 실제
거래를 만들어 거래비용 반영 순수익을 고정 Buy & Hold와 비교하고, 집중 연도를
제외해도 우위가 유지되는지까지 확인한다.

사전등록: docs/prereg_related_company_spillover_pilot.md의 Addendum 절 참고

사용법 (레포 루트에서):
    python -m src.experiments.run_related_company_backtest
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

from src.experiments.run_related_company_pilot import (
    TICKER_KRX, RELATED_TICKER, FEATURE_COLS_BASE, FEATURE_COLS_COMBINED,
    TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO, SEEDS, XGB_PARAMS,
    build_dataset, walk_forward_splits,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

THRESHOLD = 0.65  # production 채택값
ROUND_TRIP_COST = 0.002


# ------------------------------------------------------------------
# 1. Buy & Hold 기준 -- 064350 실제 가격으로 고정 구간 계산 (fold별로 달라지지 않게)
# ------------------------------------------------------------------
def load_close(start: str, end: str) -> pd.Series:
    px = yf.download(f"{TICKER_KRX}.KS", start=start, end=end, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    return px["Close"]


def buy_and_hold_return(close: pd.Series, start_date, end_date) -> float:
    window = close.loc[start_date:end_date]
    if len(window) < 2:
        return float("nan")
    return float(window.iloc[-1] / window.iloc[0] - 1)


# ------------------------------------------------------------------
# 2. fold별 거래 생성 (threshold 진입, holding_rows_tb만큼 보유)
# ------------------------------------------------------------------
def generate_trades(df: pd.DataFrame, feature_cols: list, seed: int) -> pd.DataFrame:
    X, y = df[feature_cols], df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    trades = []
    for train_idx, test_idx in splits:
        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]

        i = 0
        while i < len(test_idx):
            if proba[i] >= THRESHOLD:
                row_idx = test_idx[i]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = int(df["holding_rows_tb"].iloc[row_idx]) if pd.notna(
                    df["holding_rows_tb"].iloc[row_idx]) else 1
                holding = max(holding, 1)
                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": df.index[row_idx],
                        "exit_date": df.index[exit_row],
                        "net_return": net_return,
                    })
                i += holding
            else:
                i += 1
    return pd.DataFrame(trades)


def compounded_return(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((1 + trades["net_return"]).prod() - 1)


def exclude_year(trades: pd.DataFrame, year: int) -> pd.DataFrame:
    return trades[trades["exit_date"].dt.year != year]


def dominant_year(trades: pd.DataFrame) -> int:
    if trades.empty:
        return None
    by_year = trades.groupby(trades["exit_date"].dt.year)["net_return"].apply(
        lambda s: float((1 + s).prod() - 1)
    )
    return int(by_year.idxmax())


# ------------------------------------------------------------------
# 3. 메인
# ------------------------------------------------------------------
def main():
    print(f"=== 관계형 feature backtest 체크포인트: {TICKER_KRX} <- {RELATED_TICKER} ===\n")
    df = build_dataset()

    if "ret_tb" not in df.columns or "holding_rows_tb" not in df.columns:
        raise KeyError(
            "ret_tb / holding_rows_tb 컬럼이 없음 -- triple-barrier BASE CSV에 이 두 "
            "컬럼이 포함돼 있는지 확인할 것 (quant_xgboost의 labeling_triple_barrier.py 산출물)."
        )

    close = load_close(
        start=(df.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(df.index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
    )

    rows = []
    for seed in SEEDS:
        base_trades = generate_trades(df, FEATURE_COLS_BASE, seed)
        rel_trades = generate_trades(df, FEATURE_COLS_COMBINED, seed)

        base_net = compounded_return(base_trades)
        rel_net = compounded_return(rel_trades)

        bh_start = min(base_trades["entry_date"].min(), rel_trades["entry_date"].min())
        bh_end = max(base_trades["exit_date"].max(), rel_trades["exit_date"].max())
        bh_net = buy_and_hold_return(close, bh_start, bh_end)

        # 국면 배제 -- BASE+RELATED 기준 집중 연도 제외 후 재계산
        dom_year = dominant_year(rel_trades)
        rel_trades_excl = exclude_year(rel_trades, dom_year) if dom_year else rel_trades
        rel_net_excl = compounded_return(rel_trades_excl)

        rows.append({
            "seed": seed,
            "n_trades_base": len(base_trades),
            "n_trades_related": len(rel_trades),
            "base_net": base_net,
            "related_net": rel_net,
            "buy_and_hold_net": bh_net,
            "related_beats_base": rel_net > base_net,
            "related_beats_bh": rel_net > bh_net,
            "dominant_year": dom_year,
            "related_net_excl_dominant_year": rel_net_excl,
            "related_excl_beats_base": rel_net_excl > base_net,
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    n_beats_base = int(result_df["related_beats_base"].sum())
    n_beats_bh = int(result_df["related_beats_bh"].sum())
    n_excl_beats_base = int(result_df["related_excl_beats_base"].sum())

    print(f"\nBASE+RELATED가 BASE를 이긴 시드: {n_beats_base}/{len(SEEDS)}")
    print(f"BASE+RELATED가 Buy&Hold를 이긴 시드: {n_beats_bh}/{len(SEEDS)}")
    print(f"집중 연도 제외 후에도 BASE를 이긴 시드: {n_excl_beats_base}/{len(SEEDS)}")

    if n_beats_base >= 4 and n_beats_bh >= 4 and n_excl_beats_base >= 4:
        print(
            "\n판정: [통과] -- AUC 개선이 실전 손익 개선으로도 이어지고, 국면 우연도 아님.\n"
            "다음 단계로 DART 관계 데이터 기반 그래프 구축 + Temporal GNN 검토 진행 가능."
        )
    else:
        print(
            "\n판정: [실패] -- AUC 개선이 실전 손익으로 이어지지 않음 (또는 국면 우연).\n"
            "lag feature/코스피PER와 같은 패턴 반복. Temporal GNN 탐색 라인 종료 권장."
        )


if __name__ == "__main__":
    main()