"""
绩效指标计算 (Performance Metrics)

计算回测结果的各种绩效指标。
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────


@dataclass
class PerformanceResult:
    """绩效指标汇总"""

    cumulative_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    win_rate: float = 0.0
    monthly_excess_return: dict[str, float] | None = None
    turnover: float = 0.0
    max_excess_drawdown: float = 0.0


# ──────────────────────────────────────────
# 核心指标计算
# ──────────────────────────────────────────


def compute_cumulative_return(nav: pd.Series) -> float:
    """
    累计收益率 = (期末净值 - 期初净值) / 期初净值

    Parameters
    ----------
    nav : 净值序列，index=日期，首值为 1.0

    Returns
    -------
    float : 累计收益率（小数形式，0.56 = 56%）
    """
    if nav.empty or len(nav) < 2:
        return 0.0
    return (nav.iloc[-1] / nav.iloc[0]) - 1.0


def compute_annualized_return(nav: pd.Series) -> float:
    """
    年化收益率 = (1 + 累计收益) ^ (1 / 年数) - 1

    Parameters
    ----------
    nav : 净值序列
    """
    if nav.empty or len(nav) < 2:
        return 0.0
    total_return = (nav.iloc[-1] / nav.iloc[0]) - 1.0
    years = len(nav) / 252.0
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def compute_annualized_volatility(daily_returns: pd.Series) -> float:
    """
    年化波动率 = 日收益率标准差 × sqrt(252)

    Parameters
    ----------
    daily_returns : 日收益率序列（小数形式）
    """
    if daily_returns.empty or len(daily_returns) < 2:
        return 0.0
    return daily_returns.std() * np.sqrt(252.0)


def compute_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.025,
) -> float:
    """
    夏普比率 = (年化收益 - 无风险利率) / 年化波动率

    Parameters
    ----------
    daily_returns : 日收益率序列
    risk_free_rate : 无风险利率（年化，小数形式，默认 2.5%）
    """
    if daily_returns.empty or len(daily_returns) < 2:
        return 0.0
    ann_return = compute_annualized_return_from_returns(daily_returns)
    ann_vol = compute_annualized_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def compute_annualized_return_from_returns(daily_returns: pd.Series) -> float:
    """从日收益率序列计算年化收益"""
    if daily_returns.empty or len(daily_returns) < 2:
        return 0.0
    total_return = (1.0 + daily_returns).prod() - 1.0
    years = len(daily_returns) / 252.0
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def compute_max_drawdown(nav: pd.Series) -> float:
    """
    最大回撤 = max((净值 - 峰值) / 峰值)

    Returns
    -------
    float : 负数，例如 -0.285 = -28.5%
    """
    if nav.empty or len(nav) < 2:
        return 0.0
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    return float(drawdown.min())


def compute_max_drawdown_duration(nav: pd.Series) -> int:
    """
    最大回撤持续天数

    Returns
    -------
    int : 从回撤开始到创新高的最长天数
    """
    if nav.empty or len(nav) < 2:
        return 0
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    # 找到最大回撤的结束点
    max_dd_idx = drawdown.idxmin()
    # 找到对应的峰值点
    peak_before = peak.loc[:max_dd_idx].idmax()
    return (max_dd_idx - peak_before).days


def compute_calmar_ratio(
    nav: pd.Series,
    risk_free_rate: float = 0.025,
) -> float:
    """
    Calmar 比率 = 年化收益 / |最大回撤|

    Parameters
    ----------
    nav : 净值序列
    risk_free_rate : 无风险利率（用于计算超额收益）
    """
    if nav.empty or len(nav) < 2:
        return 0.0
    daily_ret = nav.pct_change().dropna()
    ann_return = compute_annualized_return_from_returns(daily_ret)
    max_dd = compute_max_drawdown(nav)
    if max_dd == 0:
        return 0.0
    return ann_return / abs(max_dd)


def compute_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    信息比率 = 超额收益均值 / 跟踪误差

    跟踪误差 = 超额收益标准差

    Parameters
    ----------
    strategy_returns : 策略日收益率
    benchmark_returns : 基准日收益率
    """
    if strategy_returns.empty or benchmark_returns.empty:
        return 0.0
    # 对齐索引
    aligned = pd.concat(
        [strategy_returns, benchmark_returns],
        axis=1,
        join="inner",
    )
    if aligned.empty:
        return 0.0
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    mean_excess = excess.mean()
    std_excess = excess.std()
    if std_excess == 0:
        return 0.0
    # 年化（252个交易日）
    return (mean_excess / std_excess) * np.sqrt(252.0)


def compute_win_rate(returns: pd.Series) -> float:
    """
    胜率 = 盈利月份数 / 总月份数

    Parameters
    ----------
    returns : 日收益率序列，将聚合为月度收益
    """
    if returns.empty:
        return 0.0
    # 聚合为月度收益
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0
    wins = (monthly > 0).sum()
    return wins / len(monthly)


def compute_monthly_excess_return(
    nav: pd.Series,
    benchmark_nav: pd.Series,
) -> dict[str, float]:
    """
    月度超额收益 = 策略月收益率 - 基准月收益率

    Returns
    -------
    dict : {"2023-01": 0.025, ...}，月度超额收益（小数形式）
    """
    if nav.empty or benchmark_nav.empty:
        return {}
    strat_monthly = (1.0 + nav.pct_change()).resample("ME").prod() - 1.0
    bench_monthly = (
        (1.0 + benchmark_nav.pct_change()).resample("ME").prod() - 1.0
    )
    aligned = pd.concat(
        [strat_monthly, bench_monthly], axis=1, join="inner"
    )
    if aligned.empty:
        return {}
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return {str(k)[:7]: v for k, v in excess.items()}


def compute_turnover(
    trades: list[dict],
    nav: pd.Series,
) -> float:
    """
    年化换手率 = 单边年交易量 / 平均市值

    Parameters
    ----------
    trades : 交易记录列表，每笔含 "action" 和 "cost" 或 "proceeds"
    nav : 净值序列（用于计算平均市值）

    Returns
    -------
    float : 年化换手率（小数形式，1.5 = 150%）
    """
    if not trades or nav.empty:
        return 0.0
    total_traded = 0.0
    for t in trades:
        if t.get("action") == "buy":
            total_traded += t.get("cost", 0.0)
        elif t.get("action") == "sell":
            total_traded += t.get("proceeds", 0.0)
    avg_aum = nav.mean() * (trades[0].get("cost", 1.0) / nav.iloc[0])
    years = len(nav) / 252.0
    if avg_aum == 0 or years == 0:
        return 0.0
    return (total_traded / 2.0) / (avg_aum * years)


def compute_max_excess_drawdown(
    nav: pd.Series,
    benchmark_nav: pd.Series,
) -> float:
    """
    超额收益最大回撤

    计算策略累计超额收益的回撤。
    """
    if nav.empty or benchmark_nav.empty:
        return 0.0
    strat_cum = (1.0 + nav.pct_change()).cumprod()
    bench_cum = (1.0 + benchmark_nav.pct_change()).cumprod()
    aligned = pd.concat([strat_cum, bench_cum], axis=1, join="inner")
    if aligned.empty:
        return 0.0
    excess_cum = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    peak = excess_cum.cummax()
    dd = (excess_cum - peak) / peak
    return float(dd.min())


# ──────────────────────────────────────────
# 汇总函数
# ──────────────────────────────────────────


def compute_all_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    trades: list[dict] | None = None,
    risk_free_rate: float = 0.025,
) -> PerformanceResult:
    """
    计算全部绩效指标，返回 PerformanceResult。

    Parameters
    ----------
    nav : 策略净值序列
    benchmark_nav : 基准净值序列（可选）
    trades : 交易记录（用于计算换手率）
    risk_free_rate : 无风险利率（年化）
    """
    daily_ret = nav.pct_change().dropna()
    bench_ret = (
        benchmark_nav.pct_change().dropna()
        if benchmark_nav is not None and benchmark_nav.notna().any()
        else pd.Series(dtype=float)
    )

    cum_ret = compute_cumulative_return(nav)
    ann_ret = compute_annualized_return(nav)
    ann_vol = compute_annualized_volatility(daily_ret)
    sharpe = compute_sharpe_ratio(daily_ret, risk_free_rate)
    max_dd = compute_max_drawdown(nav)
    calmar = compute_calmar_ratio(nav, risk_free_rate)

    info_ratio = 0.0
    monthly_excess = {}
    max_excess_dd = 0.0
    if (
        benchmark_nav is not None
        and benchmark_nav.notna().any()
        and not bench_ret.empty
    ):
        info_ratio = compute_information_ratio(daily_ret, bench_ret)
        monthly_excess = compute_monthly_excess_return(nav, benchmark_nav)
        max_excess_dd = compute_max_excess_drawdown(nav, benchmark_nav)

    win = compute_win_rate(daily_ret)
    turn = compute_turnover(trades, nav) if trades else 0.0

    return PerformanceResult(
        cumulative_return=cum_ret,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        information_ratio=info_ratio,
        win_rate=win,
        monthly_excess_return=monthly_excess or None,
        turnover=turn,
        max_excess_drawdown=max_excess_dd,
    )


__all__ = [
    "PerformanceResult",
    "compute_cumulative_return",
    "compute_annualized_return",
    "compute_annualized_volatility",
    "compute_sharpe_ratio",
    "compute_max_drawdown",
    "compute_calmar_ratio",
    "compute_information_ratio",
    "compute_win_rate",
    "compute_monthly_excess_return",
    "compute_turnover",
    "compute_max_excess_drawdown",
    "compute_all_metrics",
]
