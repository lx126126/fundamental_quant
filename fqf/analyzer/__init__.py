"""标准化投研模块 (Analyzer)

统一个股财务指标评价体系，生成 Value Line 风格的一页投研报告。
"""

from fqf.analyzer.financial import FinancialDataLoader
from fqf.analyzer.metrics import MetricsCalculator
from fqf.analyzer.report import ValueLineReport

__all__ = ["FinancialDataLoader", "MetricsCalculator", "ValueLineReport"]
