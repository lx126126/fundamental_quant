"""宏观与行业监测模块 (Sentinel)

追踪流动性、利率、行业估值水位线，生成每日宏观简报。

模块组成:
- MacroFactorAnalyzer: 宏观因子分析（货币政策/增长/通胀/外部环境）
- IndustryAnalyzer: 行业估值水位、动量与情绪分析
- BriefingGenerator: Markdown 格式每日宏观简报生成
"""

from fqf.sentinel.macro import MacroFactorAnalyzer
from fqf.sentinel.industry import IndustryAnalyzer
from fqf.sentinel.briefing import BriefingGenerator

__all__ = [
    "MacroFactorAnalyzer",
    "IndustryAnalyzer",
    "BriefingGenerator",
]
