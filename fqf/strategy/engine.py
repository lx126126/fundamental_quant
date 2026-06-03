"""
回测引擎 (Backtest Engine)

对策略进行历史回放，计算净值、持仓和绩效指标。
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Literal, Optional

from fqf.strategy.metrics import (
    compute_cumulative_return,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_information_ratio,
    compute_win_rate,
    compute_monthly_excess_return,
    compute_turnover,
    compute_max_excess_drawdown,
    PerformanceResult,
)


# ──────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────


class RebalanceFreq(str, Enum):
    """调仓频率"""

    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"


@dataclass
class BacktestConfig:
    """回测配置"""

    start_date: str  # "2019-01-01"
    end_date: str  # "2023-12-31"
    benchmark: str = "000300"  # 沪深300
    initial_cash: float = 1_000_000.0
    commission: float = 0.0003  # 万三
    slippage: float = 0.001  # 千一
    max_position_pct: float = 0.10  # 单票最大仓位 10%
    rebalance_freq: RebalanceFreq = RebalanceFreq.QUARTERLY
    lookback_days: int = 20  # 停牌/涨跌停回看天数
    risk_free_rate: float = 0.025  # 无风险利率 2.5%


@dataclass
class PortfolioSnapshot:
    """单日持仓快照"""

    date: str
    cash: float
    positions: dict[str, float]  # {stock_code: market_value}
    total_value: float
    returns: float = 0.0  # 当日收益率


@dataclass
class BacktestResult:
    """回测结果"""

    config: BacktestConfig
    nav: pd.Series  # index=date, value=净值（初始=1.0）
    benchmark_nav: pd.Series  # 基准净值
    positions: list[PortfolioSnapshot]
    trades: list[dict]  # 每笔交易记录
    performance: PerformanceResult | None = None


# ──────────────────────────────────────────
# 策略函数类型
# ──────────────────────────────────────────

# 策略函数签名：接受日期和可用数据，返回 {stock_code: weight}
StrategyFunc = Callable[[str, pd.DataFrame, pd.DataFrame | None], dict[str, float]]


# ──────────────────────────────────────────
# 引擎核心
# ──────────────────────────────────────────


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        config: BacktestConfig,
        strategy_fn: StrategyFunc,
    ):
        self.config = config
        self.strategy_fn = strategy_fn
        self.nav: pd.Series = pd.Series(dtype=float)
        self.benchmark_nav: pd.Series = pd.Series(dtype=float)
        self.positions: list[PortfolioSnapshot] = []
        self.trades: list[dict] = []
        self._cash = config.initial_cash
        self._current_positions: dict[str, float] = {}  # {code: mkt_value}
        self._prev_positions: dict[str, float] = {}

    # ── 公共入口 ────────────────────────────

    def run(
        self,
        price_df: pd.DataFrame,
        benchmark_price: pd.Series | None = None,
        financial_df: pd.DataFrame | None = None,
        valuation_df: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """
        执行回测。

        Parameters
        ----------
        price_df : columns=股票代码, index=日期, values=复权收盘价
        benchmark_price : index=日期, value=基准收盘价
        financial_df : 财务数据（季报后更新）
        valuation_df : 估值数据（日频）
        """
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)

        # 截取回测区间
        prices = price_df.loc[start:end].copy()
        if benchmark_price is not None:
            benchmark_price = benchmark_price.loc[start:end].copy()

        # 生成调仓日序列
        rebalance_dates = self._generate_rebalance_dates(prices.index)

        # 初始化
        self._cash = self.config.initial_cash
        self._current_positions = {}
        self._prev_positions = {}
        self.nav = pd.Series(index=prices.index, dtype=float, name="nav")
        self.nav.iloc[0] = 1.0

        if benchmark_price is not None:
            self.benchmark_nav = benchmark_price / benchmark_price.iloc[0]
        else:
            self.benchmark_nav = pd.Series(
                index=prices.index, dtype=float, name="benchmark"
            )

        # 逐日循环
        prev_date = None
        for i, date in enumerate(prices.index):
            is_rebalance = date in rebalance_dates

            # 执行调仓（在开盘前计算目标权重，以收盘价成交）
            if is_rebalance:
                target_weights = self.strategy_fn(
                    date.strftime("%Y-%m-%d"),
                    financial_df,
                    valuation_df,
                )
                self._rebalance(
                    date, prices.loc[date], target_weights
                )

            # 计算当日收益率和净值
            daily_ret = self._compute_daily_return(date, prices.loc[:date])
            if i == 0:
                self.nav.iloc[i] = 1.0 * (1 + daily_ret)
            else:
                self.nav.iloc[i] = self.nav.iloc[i - 1] * (1 + daily_ret)

            # 记录快照
            snapshot = PortfolioSnapshot(
                date=date.strftime("%Y-%m-%d"),
                cash=self._cash,
                positions=self._current_positions.copy(),
                total_value=self._total_value(),
                returns=daily_ret,
            )
            self.positions.append(snapshot)
            prev_date = date

        # 组装结果
        result = BacktestResult(
            config=self.config,
            nav=self.nav,
            benchmark_nav=self.benchmark_nav,
            positions=self.positions,
            trades=self.trades,
        )
        result.performance = compute_performance(result)
        return result

    # ── 调仓日生成 ─────────────────────────

    def _generate_rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """根据 rebalance_freq 生成调仓日列表"""
        if self.config.rebalance_freq == RebalanceFreq.DAILY:
            return list(dates)

        freq_map = {
            RebalanceFreq.WEEKLY: "W",
            RebalanceFreq.MONTHLY: "ME",
            RebalanceFreq.QUARTERLY: "QE",
        }
        rule = freq_map.get(self.config.rebalance_freq, "Q")
        return list(dates.to_series().resample(rule).last().dropna().values)

    # ── 调仓逻辑 ─────────────────────────────

    def _rebalance(
        self,
        date: pd.Timestamp,
        prices_today: pd.Series,
        target_weights: dict[str, float],
    ) -> None:
        """
        执行调仓：卖出不在目标列表中的持仓，按目标权重买入。
        使用当日收盘价成交，扣除佣金和滑点。
        """
        # 可交易股票（有价格、非 NaN）
        tradable = set(prices_today.dropna().index)
        target_stocks = set(target_weights.keys()) & tradable

        # 当前持仓股票列表
        current_stocks = set(self._current_positions.keys())

        # 卖出（不在目标中或权重降低的）
        for stock in current_stocks - target_stocks:
            self._sell(stock, prices_today[stock], date, reason="remove")

        # 计算总市值（调仓前）
        total_value = self._total_value()

        # 按目标权重买入/调仓
        for stock in target_stocks:
            target_weight = target_weights[stock]
            max_value = total_value * min(
                target_weight, self.config.max_position_pct
            )
            cur_value = self._current_positions.get(stock, 0.0)
            if max_value > cur_value:
                buy_value = max_value - cur_value
                self._buy(stock, buy_value, prices_today[stock], date)
            elif max_value < cur_value:
                sell_value = cur_value - max_value
                self._sell(
                    stock, prices_today[stock], date, reason="rebalance"
                )

    def _sell(
        self,
        stock: str,
        price: float,
        date: pd.Timestamp,
        reason: str = "sell",
    ) -> None:
        """卖出指定股票"""
        if stock not in self._current_positions:
            return
        mkt_value = self._current_positions[stock]
        shares = mkt_value / price if price > 0 else 0
        proceeds = shares * price * (1 - self.config.commission - self.config.slippage)
        self._cash += proceeds
        del self._current_positions[stock]
        self.trades.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "stock": stock,
                "action": "sell",
                "price": price,
                "shares": shares,
                "proceeds": proceeds,
                "reason": reason,
            }
        )

    def _buy(
        self,
        stock: str,
        value: float,
        price: float,
        date: pd.Timestamp,
    ) -> None:
        """买入指定股票"""
        if price <= 0 or value <= 0:
            return
        cost = value * (1 + self.config.commission + self.config.slippage)
        if cost > self._cash:
            cost = self._cash
            value = cost / (1 + self.config.commission + self.config.slippage)
        shares = value / price
        self._cash -= cost
        self._current_positions[stock] = shares * price * (
            1 + self.config.commission + self.config.slippage
        )
        self.trades.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "stock": stock,
                "action": "buy",
                "price": price,
                "shares": shares,
                "cost": cost,
            }
        )

    # ── 收益计算 ─────────────────────────────

    def _compute_daily_return(
        self, date: pd.Timestamp, prices_hist: pd.DataFrame
    ) -> float:
        """计算当日组合收益率"""
        if not self._current_positions:
            return 0.0
        # 简化的日收益：各持仓涨跌幅的加权平均
        total_mkt = sum(self._current_positions.values())
        if total_mkt == 0:
            return 0.0
        daily_ret = 0.0
        for stock, mkt_value in self._current_positions.items():
            if stock in prices_hist.columns:
                price_today = prices_hist.iloc[-1][stock]
                price_yest = (
                    prices_hist.iloc[-2][stock]
                    if len(prices_hist) >= 2
                    else price_today
                )
                if price_yest and not pd.isna(price_yest) and price_yest > 0:
                    stock_ret = (price_today - price_yest) / price_yest
                    weight = mkt_value / total_mkt
                    daily_ret += weight * (stock_ret if not pd.isna(stock_ret) else 0.0)
        return daily_ret

    def _total_value(self) -> float:
        """计算当前总市值"""
        return self._cash + sum(self._current_positions.values())


# ──────────────────────────────────────────
# 绩效计算入口
# ──────────────────────────────────────────


def compute_performance(result: BacktestResult) -> PerformanceResult:
    """从回测结果计算全部绩效指标"""
    nav = result.nav
    benchmark_nav = result.benchmark_nav

    # 日收益率
    strategy_ret = nav.pct_change().dropna()
    bench_ret = (
        benchmark_nav.pct_change().dropna()
        if benchmark_nav.notna().any()
        else pd.Series(dtype=float)
    )

    excess_ret = strategy_ret - bench_ret

    return PerformanceResult(
        cumulative_return=compute_cumulative_return(nav),
        annualized_return=compute_annualized_return(nav),
        annualized_volatility=compute_annualized_volatility(strategy_ret),
        sharpe_ratio=compute_sharpe_ratio(
            strategy_ret, result.config.risk_free_rate
        ),
        max_drawdown=compute_max_drawdown(nav),
        calmar_ratio=compute_calmar_ratio(nav, result.config.risk_free_rate),
        information_ratio=compute_information_ratio(strategy_ret, bench_ret),
        win_rate=compute_win_rate(strategy_ret),
        monthly_excess_return=compute_monthly_excess_return(
            nav, benchmark_nav
        ),
        turnover=compute_turnover(result.trades, nav),
        max_excess_drawdown=compute_max_excess_drawdown(nav, benchmark_nav),
    )


# ──────────────────────────────────────────
# 回测报告生成
# ──────────────────────────────────────────


def generate_backtest_report(result: BacktestResult) -> str:
    """生成 Markdown 格式回测报告"""
    p = result.performance
    if p is None:
        return "无绩效数据"

    lines = [
        "# 回测绩效报告",
        "",
        f"- 回测区间：{result.config.start_date} ~ {result.config.end_date}",
        f"- 初始资金：¥{result.config.initial_cash:,.0f}",
        f"- 佣金：{result.config.commission * 10000:.1f}‱  |  滑点：{result.config.slippage * 1000:.1f}‱",
        "",
        "| 指标 | 策略 | 基准 |",
        "|------|------|------|",
        f"| 累计收益率 | {p.cumulative_return * 100:.2f}% | {_benchmark_cumret(result):.2f}% |",
        f"| 年化收益率 | {p.annualized_return * 100:.2f}% | {_benchmark_annret(result):.2f}% |",
        f"| 年化波动率 | {p.annualized_volatility * 100:.2f}% | {_benchmark_annvol(result):.2f}% |",
        f"| 夏普比率 | {p.sharpe_ratio:.2f} | — |",
        f"| 最大回撤 | {p.max_drawdown * 100:.2f}% | {_benchmark_maxdd(result):.2f}% |",
        f"| Calmar 比率 | {p.calmar_ratio:.2f} | — |",
        f"| 信息比率 | {p.information_ratio:.2f} | — |",
        f"| 胜率 | {p.win_rate * 100:.1f}% | — |",
        f"| 换手率（年化） | {p.turnover * 100:.2f}% | — |",
        f"| 超额收益最大回撤 | {p.max_excess_drawdown * 100:.2f}% | — |",
        "",
        "## 月度超额收益",
        "",
    ]

    if p.monthly_excess_return:
        for month, ret in sorted(p.monthly_excess_return.items()):
            lines.append(f"- {month}：{ret * 100:+.2f}%")

    return "\n".join(lines)


def _benchmark_cumret(result: BacktestResult) -> float:
    b = result.benchmark_nav
    if b.notna().any():
        return (b.iloc[-1] / b.iloc[0] - 1) * 100
    return 0.0


def _benchmark_annret(result: BacktestResult) -> float:
    b = result.benchmark_nav
    if b.notna().any():
        total_ret = b.iloc[-1] / b.iloc[0] - 1
        years = len(b) / 252
        return ((1 + total_ret) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    return 0.0


def _benchmark_annvol(result: BacktestResult) -> float:
    b = result.benchmark_nav
    if b.notna().any():
        ret = b.pct_change().dropna()
        return ret.std() * np.sqrt(252) * 100
    return 0.0


def _benchmark_maxdd(result: BacktestResult) -> float:
    b = result.benchmark_nav
    if b.notna().any():
        peak = b.cummax()
        dd = (b - peak) / peak
        return dd.min() * 100
    return 0.0


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
    "RebalanceFreq",
    "compute_performance",
    "generate_backtest_report",
]
