"""策略构建与回测模块 (Strategy & Backtest)

将投资哲学转化为因子，通过历史数据验证逻辑。
"""

from fqf.strategy.factors import (
    compute_factor,
    list_factors,
    get_factor_spec,
    Direction,
    FactorResult,
    FactorSpec,
)
from fqf.strategy.metrics import (
    compute_all_metrics,
    PerformanceResult,
)
from fqf.strategy.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    RebalanceFreq,
    compute_performance,
    generate_backtest_report,
)
from fqf.strategy.strategy_templates import (
    strategy_value_quality,
    strategy_reversal,
    strategy_quality_dividend,
    get_strategy,
    list_strategies,
)

__all__ = [
    # factors
    "compute_factor",
    "list_factors",
    "get_factor_spec",
    "Direction",
    "FactorResult",
    "FactorSpec",
    # metrics
    "compute_all_metrics",
    "PerformanceResult",
    # engine
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "RebalanceFreq",
    "compute_performance",
    "generate_backtest_report",
    # strategy templates
    "strategy_quality_value",
    "strategy_earnings_reversal",
    "strategy_quality_dividend",
    "get_strategy",
    "list_strategies",
]
