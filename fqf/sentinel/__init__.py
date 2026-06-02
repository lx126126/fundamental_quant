"""宏观与行业监测模块 (Sentinel)

追踪流动性、利率、行业估值水位线，生成每日宏观简报。
"""

from fqf.sentinel.macro import MacroMonitor
from fqf.sentinel.industry import IndustryMonitor
from fqf.sentinel.briefing import BriefingGenerator

__all__ = ["MacroMonitor", "IndustryMonitor", "BriefingGenerator"]
