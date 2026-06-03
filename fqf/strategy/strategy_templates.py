"""
策略模板 (Strategy Templates)

实现3个预设策略：
  1. 低估值 + 高质量选股（核心策略）
  2. 业绩反转策略
  3. 质量 + 红利策略
"""

from __future__ import annotations

import pandas as pd
from typing import Optional


# ────────────────────────────────────
# 策略 1：低估值 + 高质量选股（核心策略）
# ────────────────────────────────────


def strategy_value_quality(
    date: str,
    financial_df: pd.DataFrame | None = None,
    valuation_df: pd.DataFrame | None = None,
    stock_universe: list[str] | None = None,
    top_n: int = 30,
) -> dict[str, float]:
    """
    低估值 + 高质量选股策略（核心策略）

    选股池：沪深300 + 中证500（由 stock_universe 指定）
    筛选条件：
      - ROE TTM > 15%
      - FCF/净利润 > 0.8
      - 资产负债率 < 60%
      - PE 5年分位 < 40%
      - 扣非净利润 3年CAGR > 10%
    排序：综合得分从高到低（等权）
    持仓：前 top_n 只，等权

    Returns
    -------
    dict : {stock_code: weight}，权重和为 1.0
    """
    if financial_df is None or valuation_df is None:
        return {}

    # 合并财务和估值数据
    df = financial_df.join(valuation_df, how="inner")

    # 筛选条件
    mask = pd.Series(True, index=df.index)

    if "roe_ttm" in df.columns:
        mask &= df["roe_ttm"] > 0.15
    if "fcf_to_profit" in df.columns:
        mask &= df["fcf_to_profit"] > 0.8
    if "debt_ratio" in df.columns:
        mask &= df["debt_ratio"] < 0.60
    if "pe_pct_5y" in df.columns:
        mask &= df["pe_pct_5y"] < 0.40
    if "profit_growth_3y" in df.columns:
        mask &= df["profit_growth_3y"] > 0.10

    # 如果指定了股票池，进一步过滤
    if stock_universe is not None:
        mask &= df.index.isin(stock_universe)

    filtered = df[mask].copy()
    if filtered.empty:
        return {}

    # 综合评分（等权归一化）
    score_cols = []
    if "roe_ttm" in filtered.columns:
        filtered["_s_roe"] = _normalize(filtered["roe_ttm"])
        score_cols.append("_s_roe")
    if "fcf_to_profit" in filtered.columns:
        filtered["_s_fcf"] = _normalize(filtered["fcf_to_profit"])
        score_cols.append("_s_fcf")
    if "pe_pct_5y" in filtered.columns:
        # PE分位越低越好，取反
        filtered["_s_pe"] = _normalize(1.0 - filtered["pe_pct_5y"])
        score_cols.append("_s_pe")
    if "profit_growth_3y" in filtered.columns:
        filtered["_s_growth"] = _normalize(filtered["profit_growth_3y"])
        score_cols.append("_s_growth")

    if not score_cols:
        # 没有可评分列，等权返回
        top = filtered.head(top_n)
        return {code: 1.0 / len(top) for code in top.index}

    filtered["_total_score"] = filtered[score_cols].sum(axis=1)
    top = filtered.nlargest(top_n, "_total_score")

    # 等权
    weight = 1.0 / len(top)
    return {code: weight for code in top.index}


# ────────────────────────────────────
# 策略 2：业绩反转策略
# ────────────────────────────────────


def strategy_reversal(
    date: str,
    financial_df: pd.DataFrame | None = None,
    valuation_df: pd.DataFrame | None = None,
    stock_universe: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, float]:
    """
    业绩反转策略

    选股池：全A（剔除ST、上市不满1年）
    筛选条件：
      - 最新季净利润同比 > 0（本季转正）
      - 前4季中有2季以上净利润同比 < -20%（确认之前困境）
      - 最新季毛利率 > 前4季均值（盈利质量改善）
      - 应计利润占比 < 0（非会计操作）
    排序：净利润改善幅度
    持仓：前 top_n 只，等权

    Returns
    -------
    dict : {stock_code: weight}
    """
    if financial_df is None:
        return {}

    df = financial_df.copy()
    mask = pd.Series(True, index=df.index)

    # 最新季净利润同比 > 0
    if "net_profit_yoy_latest" in df.columns:
        mask &= df["net_profit_yoy_latest"] > 0

    # 前4季净利润同比 < -20% 的个数 >= 2
    if "net_profit_yoy_history" in df.columns:

        def _count_bad(x):
            return (pd.Series(x) < -0.20).sum()

        n_bad = df["net_profit_yoy_history"].apply(_count_bad)
        mask &= n_bad >= 2

    # 最新季毛利率 > 前4季均值
    if "gross_margin_latest" in df.columns and "gross_margin_history" in df.columns:

        def _margin_ok(row):
            hist = row["gross_margin_history"]
            if not hasattr(hist, "__len__") or len(hist) < 1:
                return True
            return row["gross_margin_latest"] > pd.Series(hist).mean()

        mask &= df.apply(_margin_ok, axis=1)

    # 应计利润占比 < 0
    if "accruals_ratio" in df.columns:
        mask &= df["accruals_ratio"] < 0

    # 剔除ST
    if "is_st" in df.columns:
        mask &= ~df["is_st"]

    # 剔除上市不满1年
    if "list_days" in df.columns:
        mask &= df["list_days"] > 252

    if stock_universe is not None:
        mask &= df.index.isin(stock_universe)

    filtered = df[mask].copy()
    if filtered.empty:
        return {}

    # 按净利润改善幅度排序
    sort_col = (
        "net_profit_yoy_latest"
        if "net_profit_yoy_latest" in filtered.columns
        else filtered.columns[0]
    )
    top = filtered.nlargest(top_n, sort_col)

    weight = 1.0 / len(top)
    return {code: weight for code in top.index}


# ────────────────────────────────────
# 策略 3：质量 + 红利策略
# ────────────────────────────────────


def strategy_quality_dividend(
    date: str,
    financial_df: pd.DataFrame | None = None,
    valuation_df: pd.DataFrame | None = None,
    stock_universe: list[str] | None = None,
    top_n: int = 30,
) -> dict[str, float]:
    """
    质量 + 红利策略

    选股池：中证800
    筛选条件：
      - 近3年连续分红
      - 股息率 > 2%
      - ROE > 10%
      - FCF/净利润 > 0.9
      - 有息负债率 < 40%
    排序：股息率从高到低
    持仓：前 top_n 只，等权

    Returns
    -------
    dict : {stock_code: weight}
    """
    if financial_df is None or valuation_df is None:
        return {}

    df = financial_df.join(valuation_df, how="inner")
    mask = pd.Series(True, index=df.index)

    # 近3年连续分红
    if "dividend_years" in df.columns:
        mask &= df["dividend_years"] >= 3

    # 股息率 > 2%
    if "dividend_yield" in df.columns:
        mask &= df["dividend_yield"] > 0.02

    # ROE > 10%
    if "roe_ttm" in df.columns:
        mask &= df["roe_ttm"] > 0.10

    # FCF/净利润 > 0.9
    if "fcf_to_profit" in df.columns:
        mask &= df["fcf_to_profit"] > 0.9

    # 有息负债率 < 40%
    if "interest_bearing_debt_ratio" in df.columns:
        mask &= df["interest_bearing_debt_ratio"] < 0.40

    if stock_universe is not None:
        mask &= df.index.isin(stock_universe)

    filtered = df[mask].copy()
    if filtered.empty:
        return {}

    # 按股息率排序
    sort_col = (
        "dividend_yield"
        if "dividend_yield" in filtered.columns
        else filtered.columns[0]
    )
    top = filtered.nlargest(top_n, sort_col)

    weight = 1.0 / len(top)
    return {code: weight for code in top.index}


# ────────────────────────────────────
# 辅助函数
# ────────────────────────────────────


def _normalize(series: pd.Series) -> pd.Series:
    """Min-Max 归一化到 [0, 1]"""
    mn = series.min()
    mx = series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ────────────────────────────────────
# 策略注册表
# ────────────────────────────────────

STRATEGY_REGISTRY: dict[str, Callable] = {
    "quality_value": strategy_value_quality,
    "earnings_reversal": strategy_reversal,
    "quality_dividend": strategy_quality_dividend,
}


def get_strategy(name: str) -> Callable | None:
    """根据名称获取策略函数"""
    return STRATEGY_REGISTRY.get(name)


def list_strategies() -> list[str]:
    """列出所有已注册策略"""
    return list(STRATEGY_REGISTRY.keys())


__all__ = [
    "strategy_quality_value",
    "strategy_earnings_reversal",
    "strategy_quality_dividend",
    "get_strategy",
    "list_strategies",
    "STRATEGY_REGISTRY",
]
