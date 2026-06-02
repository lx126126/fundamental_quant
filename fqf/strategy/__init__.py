"""策略构建与回测模块 (Strategy & Backtest)

将投资哲学转化为因子，通过历史数据验证逻辑。
"""

from fqf.strategy.factors import FactorLibrary
from fqf.strategy.engine import BacktestEngine
from fqf.strategy.metrics import PerformanceMetrics

__all__ = ["FactorLibrary", "BacktestEngine", "PerformanceMetrics"]
