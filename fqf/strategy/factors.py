"""
因子库 (Factor Library)

将投资哲学转化为可回测的因子，作为策略引擎的输入。

因子分类：
  - 基本面因子：来自 Analyzer 的财务指标
  - 估值因子：PE/PB 历史分位等
  - 行业与宏观因子：来自 Sentinel 模块
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Literal


# ──────────────────────────────────────────────
# 枚举与数据结构
# ──────────────────────────────────────────────

class Direction(IntEnum):
    """因子方向：1=越高越好（正向），-1=越低越好（负向）"""

    LONG = 1
    SHORT = -1


@dataclass
class FactorResult:
    """单因子计算结果"""

    name: str
    values: pd.Series  # index=股票代码, value=因子值
    direction: Direction
    date: str = ""
    description: str = ""


@dataclass
class FactorSpec:
    """因子规格，描述一个因子的元数据"""

    name: str
    direction: Direction
    freq: Literal["daily", "quarterly"] = "quarterly"
    description: str = ""


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────


def _winsorize(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """去极值（Winsorize）"""
    lb = series.quantile(lower)
    ub = series.quantile(upper)
    return series.clip(lower=lb, upper=ub)


def _standardize(series: pd.Series) -> pd.Series:
    """Z-Score 标准化（横截面）"""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def compute_factor(
    name: str,
    financial_df: pd.DataFrame | None = None,
    valuation_df: pd.DataFrame | None = None,
    macro_result: dict | None = None,
    industry_df: pd.DataFrame | None = None,
    date: str = "",
) -> FactorResult | None:
    """
    统一因子计算入口。
    根据因子名路由到对应的计算函数。

    Parameters
    ----------
    name : 因子名称（见 FACTOR_REGISTRY 键名）
    financial_df : 财务数据 DataFrame，index=股票代码
    valuation_df : 估值数据 DataFrame，index=股票代码
    macro_result : Sentinel 宏观分析结果 dict
    industry_df : Sentinel 行业评分 DataFrame
    date : 计算日期（可选，用于记录）
    """
    if name not in _FACTOR_REGISTRY:
        return None
    func = _FACTOR_REGISTRY[name]
    return func(
        financial_df=financial_df,
        valuation_df=valuation_df,
        macro_result=macro_result,
        industry_df=industry_df,
        date=date,
    )


# ──────────────────────────────────────────────
# 基本面因子
# ──────────────────────────────────────────────


def _factor_roe_ttm(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    ROE(TTM) — 核心盈利能力因子

    方向：正向（越高越好）
    更新：季报后
    """
    if financial_df is None or "roe_ttm" not in financial_df.columns:
        return None
    values = financial_df["roe_ttm"]
    values = _winsorize(values)
    return FactorResult(
        name="roe_ttm",
        values=values,
        direction=Direction.LONG,
        description="ROE(TTM)，正向，季报后更新",
    )


def _factor_roe_stability(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    ROE 稳定性 — 近5年ROE的1/变异系数

    变异系数 = 标准差 / 均值，越小越稳定。
    取倒数使得"越稳定得分越高"。
    方向：正向
    """
    if financial_df is None:
        return None
    # 需要近5年ROE历史数据，这里用单次截面数据模拟
    # 实际实现中 financial_df 应包含 history_roe_5y 列
    if "roe_5y_history" in financial_df.columns:
        # roe_5y_history 应为 list-like，每个元素是近5年ROE序列
        def _stability(x):
            arr = np.array(x)
            mu = arr.mean()
            if mu == 0:
                return 0.0
            cv = arr.std() / mu
            return 1.0 / cv if cv > 0 else 0.0

        values = financial_df["roe_5y_history"].apply(_stability)
    elif "roe_ttm" in financial_df.columns:
        # fallback：没有历史数据时用 ROE 绝对值代替
        values = financial_df["roe_ttm"]
    else:
        return None
    values = _winsorize(values)
    return FactorResult(
        name="roe_stability",
        values=values,
        direction=Direction.LONG,
        description="近5年ROE稳定性（1/变异系数），正向",
    )


def _factor_gross_margin_trend(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    毛利率趋势 — 近8季毛利率回归斜率

    方向：正向（斜率越大越好）
    """
    if financial_df is None:
        return None
    # gross_margin_history：每个股票的近8季毛利率序列
    if "gross_margin_history" in financial_df.columns:

        def _slope(x):
            arr = np.array(x)
            if len(arr) < 2:
                return 0.0
            x_vals = np.arange(len(arr))
            slope = np.polyfit(x_vals, arr, 1)[0]
            return slope

        values = financial_df["gross_margin_history"].apply(_slope)
    elif "gross_margin_ttm" in financial_df.columns:
        values = financial_df["gross_margin_ttm"]
    else:
        return None
    values = _winsorize(values)
    return FactorResult(
        name="gross_margin_trend",
        values=values,
        direction=Direction.LONG,
        description="近8季毛利率回归斜率，正向",
    )


def _factor_fcf_yield(
    financial_df: pd.DataFrame,
    valuation_df: pd.DataFrame | None = None,
    **kwargs,
) -> FactorResult | None:
    """
    FCF Yield = 经营现金流 / 市值

    方向：正向
    """
    if financial_df is None:
        return None
    fcf = financial_df.get("fcf_ttm")
    if fcf is None:
        return None
    mkt_cap = valuation_df.get("market_cap") if valuation_df is not None else None
    if mkt_cap is not None:
        values = fcf / mkt_cap
    else:
        # 没有市值时用 FCF 绝对值（标准化后会消除量纲）
        values = fcf
    values = _winsorize(values)
    return FactorResult(
        name="fcf_yield",
        values=values,
        direction=Direction.LONG,
        description="FCF/市值，正向",
    )


def _factor_fcf_to_profit(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    FCF / 净利润 — 利润"含金量"

    方向：正向（>1 为佳）
    """
    if financial_df is None:
        return None
    fcf = financial_df.get("fcf_ttm")
    net_profit = financial_df.get("net_profit_ttm")
    if fcf is None or net_profit is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        values = fcf / net_profit
        values = values.replace([np.inf, -np.inf], np.nan)
    values = _winsorize(values)
    return FactorResult(
        name="fcf_to_profit",
        values=values,
        direction=Direction.LONG,
        description="经营现金流/归母净利润，正向，>1为佳",
    )


def _factor_accruals(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    应计利润占比 = (净利润 - 经营现金流) / 总资产

    方向：负向（应计利润越高，盈利质量越差，取反后为正向）
    """
    if financial_df is None:
        return None
    net_profit = financial_df.get("net_profit_ttm")
    fcf = financial_df.get("fcf_ttm")
    total_assets = financial_df.get("total_assets")
    if net_profit is None or fcf is None or total_assets is None:
        return None
    accruals = (net_profit - fcf) / total_assets
    # 取反：应计利润越高越差
    values = -accruals
    values = _winsorize(values)
    return FactorResult(
        name="accruals",
        values=values,
        direction=Direction.LONG,
        description="-(应计利润占比)，正向（越小越好）",
    )


def _factor_debt_ratio(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    资产负债率 = 总负债 / 总资产

    方向：负向（越低越好）
    """
    if financial_df is None:
        return None
    debt = financial_df.get("total_liabilities")
    assets = financial_df.get("total_assets")
    if debt is None or assets is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        values = debt / assets
        values = values.replace([np.inf, -np.inf], np.nan)
    values = _winsorize(values)
    return FactorResult(
        name="debt_ratio",
        values=values,
        direction=Direction.SHORT,
        description="资产负债率，负向",
    )


def _factor_profit_growth_3y(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    归母净利润 3年CAGR

    方向：正向
    """
    if financial_df is None or "profit_growth_3y" not in financial_df.columns:
        return None
    values = financial_df["profit_growth_3y"]
    values = _winsorize(values)
    return FactorResult(
        name="profit_growth_3y",
        values=values,
        direction=Direction.LONG,
        description="归母净利润3年CAGR，正向",
    )


def _factor_revenue_growth_stability(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    营收增速稳定性 = 1 / 近12季营收同比变异系数

    方向：正向（越稳定越好）
    """
    if financial_df is None:
        return None
    if "revenue_growth_history" in financial_df.columns:

        def _stability(x):
            arr = np.array(x)
            mu = arr.mean()
            if mu == 0:
                return 0.0
            cv = arr.std() / abs(mu)
            return 1.0 / cv if cv > 0 else 0.0

        values = financial_df["revenue_growth_history"].apply(_stability)
    elif "revenue_growth_3y" in financial_df.columns:
        values = financial_df["revenue_growth_3y"]
    else:
        return None
    values = _winsorize(values)
    return FactorResult(
        name="revenue_growth_stability",
        values=values,
        direction=Direction.LONG,
        description="营收增速稳定性（1/变异系数），正向",
    )


def _factor_roic(
    financial_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    ROIC = NOPAT / (净债务 + 净资产)

    方向：正向
    """
    if financial_df is None:
        return None
    nopat = financial_df.get("nopat")
    net_debt = financial_df.get("net_debt")
    equity = financial_df.get("total_equity")
    if nopat is None or equity is None:
        return None
    denom = (net_debt if net_debt is not None else 0) + equity
    with np.errstate(divide="ignore", invalid="ignore"):
        values = nopat / denom
        values = values.replace([np.inf, -np.inf], np.nan)
    values = _winsorize(values)
    return FactorResult(
        name="roic",
        values=values,
        direction=Direction.LONG,
        description="ROIC，正向",
    )


# ──────────────────────────────────────────────
# 估值因子
# ──────────────────────────────────────────────


def _factor_pe_percentile_5y(
    valuation_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    PE-TTM 5年历史分位

    方向：负向（分位越低越有吸引力）
    """
    if valuation_df is None or "pe_ttm_pct_5y" not in valuation_df.columns:
        return None
    # 分位 0-1，越低越好，所以取反使得 "高值=好"
    values = 1.0 - valuation_df["pe_ttm_pct_5y"]
    return FactorResult(
        name="pe_percentile_5y",
        values=values,
        direction=Direction.LONG,
        description="PE-TTM 5年分位（取反，低分位=高得分），负向原值",
    )


def _factor_pb_percentile_5y(
    valuation_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    PB 5年历史分位

    方向：负向
    """
    if valuation_df is None or "pb_pct_5y" not in valuation_df.columns:
        return None
    values = 1.0 - valuation_df["pb_pct_5y"]
    return FactorResult(
        name="pb_percentile_5y",
        values=values,
        direction=Direction.LONG,
        description="PB 5年分位（取反），负向原值",
    )


def _factor_peg(
    valuation_df: pd.DataFrame,
    financial_df: pd.DataFrame | None = None,
    **kwargs,
) -> FactorResult | None:
    """
    PEG = PE-TTM / 净利润3年CAGR

    方向：负向（PEG < 1 为低估）
    """
    if valuation_df is None:
        return None
    pe = valuation_df.get("pe_ttm")
    if financial_df is not None and "profit_growth_3y" in financial_df.columns:
        growth = financial_df["profit_growth_3y"]
    else:
        growth = valuation_df.get("profit_growth_3y")
    if pe is None or growth is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        values = pe / (growth * 100)  # growth 为小数形式（0.15 = 15%）
        values = values.replace([np.inf, -np.inf], np.nan)
    # PEG 越低越好，取反
    values = -values
    values = _winsorize(values)
    return FactorResult(
        name="peg",
        values=values,
        direction=Direction.LONG,
        description="PEG = PE/净利润CAGR（取反，低PEG=高得分）",
    )


def _factor_ev_ebitda(
    valuation_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    EV / EBITDA

    方向：负向（越低越好）
    """
    if valuation_df is None or "ev_ebitda" not in valuation_df.columns:
        return None
    values = -valuation_df["ev_ebitda"]
    values = _winsorize(values)
    return FactorResult(
        name="ev_ebitda",
        values=values,
        direction=Direction.LONG,
        description="EV/EBITDA（取反），负向原值",
    )


def _factor_dividend_yield(
    valuation_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    股息率 = 近12月分红 / 市值

    方向：正向
    """
    if valuation_df is None or "dividend_yield" not in valuation_df.columns:
        return None
    values = valuation_df["dividend_yield"]
    values = _winsorize(values)
    return FactorResult(
        name="dividend_yield",
        values=values,
        direction=Direction.LONG,
        description="股息率，正向",
    )


# ──────────────────────────────────────────────
# 行业与宏观因子
# ──────────────────────────────────────────────


def _factor_industry_pe_percentile(
    industry_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    所属行业 PE 分位

    方向：负向（行业整体估值低时更好）
    """
    if industry_df is None or "valuation_pct" not in industry_df.columns:
        return None
    # valuation_pct 越低越好，取反
    values = 1.0 - industry_df["valuation_pct"]
    return FactorResult(
        name="industry_pe_percentile",
        values=values,
        direction=Direction.LONG,
        description="行业估值分位（取反），负向原值",
    )


def _factor_industry_momentum_60d(
    industry_df: pd.DataFrame,
    **kwargs,
) -> FactorResult | None:
    """
    行业 60日动量

    方向：正向（顺势）
    """
    if industry_df is None or "momentum_60d" not in industry_df.columns:
        return None
    values = industry_df["momentum_60d"]
    values = _winsorize(values)
    return FactorResult(
        name="industry_momentum_60d",
        values=values,
        direction=Direction.LONG,
        description="行业60日动量，正向（顺势）",
    )


def _factor_macro_score(
    macro_result: dict,
    **kwargs,
) -> FactorResult | None:
    """
    宏观综合得分（0-100）

    方向：正向（高分=高仓位）
    """
    if macro_result is None or "total_score" not in macro_result:
        return None
    # macro_result["total_score"] 是单个值，需要广播到所有股票
    # 实际调用时由引擎处理，这里返回 None 让引擎用广播方式处理
    return FactorResult(
        name="macro_score",
        values=pd.Series(dtype=float),  # 空 Series，引擎会填充
        direction=Direction.LONG,
        description="宏观综合得分（0-100），正向",
    )


def _factor_north_flow_20d(
    valuation_df: pd.DataFrame | None = None,
    **kwargs,
) -> FactorResult | None:
    """
    北向资金 20日累计持仓变化

    方向：正向
    """
    if valuation_df is None or "north_flow_20d" not in valuation_df.columns:
        return None
    values = valuation_df["north_flow_20d"]
    values = _winsorize(values)
    return FactorResult(
        name="north_flow_20d",
        values=values,
        direction=Direction.LONG,
        description="北向资金20日累计变化，正向",
    )


# ──────────────────────────────────────────────
# 因子注册表
# ──────────────────────────────────────────────

_FACTOR_REGISTRY: dict[str, Callable] = {
    # 基本面因子
    "roe_ttm": _factor_roe_ttm,
    "roe_stability": _factor_roe_stability,
    "gross_margin_trend": _factor_gross_margin_trend,
    "fcf_yield": _factor_fcf_yield,
    "fcf_to_profit": _factor_fcf_to_profit,
    "accruals": _factor_accruals,
    "debt_ratio": _factor_debt_ratio,
    "profit_growth_3y": _factor_profit_growth_3y,
    "revenue_growth_stability": _factor_revenue_growth_stability,
    "roic": _factor_roic,
    # 估值因子
    "pe_percentile_5y": _factor_pe_percentile_5y,
    "pb_percentile_5y": _factor_pb_percentile_5y,
    "peg": _factor_peg,
    "ev_ebitda": _factor_ev_ebitda,
    "dividend_yield": _factor_dividend_yield,
    # 行业与宏观因子
    "industry_pe_percentile": _factor_industry_pe_percentile,
    "industry_momentum_60d": _factor_industry_momentum_60d,
    "macro_score": _factor_macro_score,
    "north_flow_20d": _factor_north_flow_20d,
}


def list_factors() -> list[str]:
    """返回所有已注册因子名称"""
    return list(_FACTOR_REGISTRY.keys())


def get_factor_spec(name: str) -> FactorSpec | None:
    """获取因子规格（元数据）"""
    if name not in _FACTOR_REGISTRY:
        return None
    # 根据名称推断 freq 和 direction
    daily_factors = {
        "pe_percentile_5y",
        "pb_percentile_5y",
        "dividend_yield",
        "industry_pe_percentile",
        "industry_momentum_60d",
        "north_flow_20d",
    }
    freq: Literal["daily", "quarterly"] = (
        "daily" if name in daily_factors else "quarterly"
    )
    # direction 从函数计算结果推断（通过调用一次获取）
    # 简化：从注册表的 metadata 读取
    _spec_map = {
        "roe_ttm": (Direction.LONG, "quarterly", "ROE(TTM)，核心盈利因子"),
        "roe_stability": (Direction.LONG, "quarterly", "ROE稳定性"),
        "gross_margin_trend": (Direction.LONG, "quarterly", "毛利率趋势"),
        "fcf_yield": (Direction.LONG, "quarterly", "FCF/市值"),
        "fcf_to_profit": (Direction.LONG, "quarterly", "现金流利润比"),
        "accruals": (Direction.LONG, "quarterly", "应计利润（取反）"),
        "debt_ratio": (Direction.SHORT, "quarterly", "资产负债率"),
        "profit_growth_3y": (Direction.LONG, "quarterly", "利润3年CAGR"),
        "revenue_growth_stability": (Direction.LONG, "quarterly", "营收增速稳定性"),
        "roic": (Direction.LONG, "quarterly", "ROIC"),
        "pe_percentile_5y": (Direction.LONG, "daily", "PE 5年分位（取反）"),
        "pb_percentile_5y": (Direction.LONG, "daily", "PB 5年分位（取反）"),
        "peg": (Direction.LONG, "quarterly", "PEG（取反）"),
        "ev_ebitda": (Direction.LONG, "quarterly", "EV/EBITDA（取反）"),
        "dividend_yield": (Direction.LONG, "daily", "股息率"),
        "industry_pe_percentile": (Direction.LONG, "daily", "行业PE分位（取反）"),
        "industry_momentum_60d": (Direction.LONG, "daily", "行业60日动量"),
        "macro_score": (Direction.LONG, "daily", "宏观综合得分"),
        "north_flow_20d": (Direction.LONG, "daily", "北向资金20日"),
    }
    _direction, _freq, desc = _spec_map.get(
        name, (Direction.LONG, "quarterly", "")
    )
    return FactorSpec(
        name=name, direction=_direction, freq=_freq, description=desc
    )


# 别名，保持导入兼容
get_factor = get_factor_spec


__all__ = [
    "Direction",
    "FactorResult",
    "FactorSpec",
    "compute_factor",
    "list_factors",
    "get_factor_spec",
    "_winsorize",
    "_standardize",
]
