"""宏观/行业简报生成器 (BriefingGenerator)

基于规则引擎生成 Markdown 格式的每日宏观简报，
包含宏观综合评分、关键信号、行业热力图和风险提示。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BriefingGenerator:
    """简报生成器。

    使用方式：
        gen = BriefingGenerator()
        report = gen.generate(macro_result, industry_df)
        print(report)

    Parameters
    ----------
    fetcher : 可选，DataFetcher 实例
    """

    fetcher: Any = None

    # ═══════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════

    def generate(
        self,
        macro_result: Optional[dict] = None,
        industry_df: Optional[pd.DataFrame] = None,
        date: Optional[str] = None,
        title: str = "宏观日报",
    ) -> str:
        """生成完整的 Markdown 宏观简报。

        Parameters
        ----------
        macro_result : MacroFactorAnalyzer.compute_all() 的结果
        industry_df : IndustryAnalyzer.score_all() 的结果
        date : 日期字符串，默认今天
        title : 简报标题

        Returns
        -------
        str Markdown 格式简报
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        sections = [
            self._header(title, date),
            self._key_signals(macro_result) if macro_result else "",
            self._macro_scorecard(macro_result) if macro_result else "",
            self._industry_heatmap(industry_df) if industry_df is not None else "",
            self._risk_warnings(macro_result) if macro_result else "",
            self._disclaimer(),
        ]

        return "\n\n".join(s for s in sections if s)

    # ═══════════════════════════════════════════════════
    #  各章节
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _header(title: str, date: str) -> str:
        return (
            f"# {title} | {date}\n\n"
            f"> 本简报由 fqf Sentinel 规则引擎自动生成。"
        )

    def _key_signals(self, macro: dict) -> str:
        """生成「关键变化」章节。"""
        lines = ["## 关键变化"]
        ml = macro.get("monetary_liquidity", {})
        g = macro.get("growth", {})
        infl = macro.get("inflation", {})
        ext = macro.get("external", {})

        # 货币政策
        sig_texts = []
        if ml.get("monetary_signal") == "easing":
            sig_texts.append(
                f"货币政策信号：**宽松** — LPR 变动 {ml.get('lpr_1y_change', 0)}%"
            )
        elif ml.get("monetary_signal") == "tightening":
            sig_texts.append("货币政策信号：**收紧**")
        else:
            sig_texts.append("货币政策信号：**中性**")

        # 国债收益率
        bond = ml.get("bond_10y")
        if bond is not None:
            trend = ml.get("bond_10y_trend", "flat")
            trend_text = {"up": "上行", "down": "下行", "flat": "持平"}[trend]
            sig_texts.append(f"10Y 国债收益率 {bond:.2f}%，近期{trend_text}")

        # 期限利差
        spread = ml.get("term_spread")
        if spread is not None:
            sig_texts.append(f"期限利差 (10Y-1Y)：{spread:.2f}%")

        # M2-M1 剪刀差
        m2m1 = ml.get("m2_m1_spread")
        if m2m1 is not None:
            sig_texts.append(f"M2-M1 剪刀差：{m2m1:.1f}%")

        # PMI
        pmi = g.get("pmi_mfg")
        if pmi is not None:
            sig_texts.append(f"制造业 PMI：{pmi:.1f}（荣枯线 50）")

        # GDP
        gdp = g.get("gdp_yoy")
        if gdp is not None:
            sig_texts.append(f"GDP 同比：{gdp:.1f}%")

        # 通胀
        cpi = infl.get("cpi_yoy")
        ppi = infl.get("ppi_yoy")
        if cpi is not None or ppi is not None:
            parts = []
            if cpi is not None:
                parts.append(f"CPI {cpi:.1f}%")
            if ppi is not None:
                parts.append(f"PPI {ppi:.1f}%")
            sig_texts.append(" / ".join(parts))

        # 中美利差
        cn_us = ext.get("cn_us_spread")
        if cn_us is not None:
            sig_texts.append(f"中美利差：{cn_us:.2f}%")

        # 北向资金
        north = ext.get("north_flow_20d")
        if north is not None:
            direction = "净流入" if north > 0 else "净流出"
            sig_texts.append(f"北向资金 20 日{direction}：{abs(north):.0f} 亿")

        for t in sig_texts:
            lines.append(f"- {t}")

        return "\n".join(lines)

    def _macro_scorecard(self, macro: dict) -> str:
        """生成「宏观综合评分」章节。"""
        from fqf.sentinel.macro import MacroFactorAnalyzer

        analyzer = MacroFactorAnalyzer()
        score = analyzer.score(macro)

        def _arrow(prev_key: str) -> str:
            # 简化：无历史对比时显示水平箭头
            return "→"

        lines = [
            "## 宏观综合评分",
            f"\n**总分：{score['total_score']}/100**（评级：{score['grade']}）",
            "",
            f"- 流动性：{score['liquidity_score']}/100 {_arrow('liquidity')}",
            f"- 增长：{score['growth_score']}/100 {_arrow('growth')}",
            f"- 通胀：{score['inflation_score']}/100 {_arrow('inflation')}",
            f"- 外部：{score['external_score']}/100 {_arrow('external')}",
        ]
        return "\n".join(lines)

    def _industry_heatmap(self, industry_df: pd.DataFrame) -> str:
        """生成「行业热力图」章节。"""
        if industry_df.empty:
            return "## 行业热力图\n\n（暂无行业数据）"

        lines = ["## 行业热力图", ""]

        # 表头
        lines.append(
            "| 行业 | 综合 | 估值 | 基本面 | 动量 | 情绪 | 信号 |"
        )
        lines.append(
            "|------|------|------|--------|------|------|------|"
        )

        # 取 Top 10 + Bottom 5
        top_n = min(10, len(industry_df))
        bottom_n = min(5, max(0, len(industry_df) - 10))

        for _, row in industry_df.head(top_n).iterrows():
            lines.append(self._format_row(row))

        if bottom_n > 0:
            lines.append("| ... | ... | ... | ... | ... | ... | ... |")
            for _, row in industry_df.tail(bottom_n).iterrows():
                lines.append(self._format_row(row))

        return "\n".join(lines)

    @staticmethod
    def _format_row(row) -> str:
        """格式化行业热力图行。"""
        name = row.get("industry", "?")

        def _stars(score_val):
            s = score_val if not pd.isna(score_val) else 50
            if s >= 80:
                return "⭐⭐⭐⭐⭐"
            elif s >= 60:
                return "⭐⭐⭐⭐"
            elif s >= 40:
                return "⭐⭐⭐"
            elif s >= 20:
                return "⭐⭐"
            return "⭐"

        return (
            f"| {name} "
            f"| {_stars(row.get('total_score', 50))} "
            f"| {row.get('valuation_score', '-')} "
            f"| {row.get('fundamental_score', '-')} "
            f"| {row.get('momentum_score', '-')} "
            f"| {row.get('sentiment_score', '-')} "
            f"| {row.get('valuation_signal', '-')} "
            f"|"
        )

    def _risk_warnings(self, macro: dict) -> str:
        """生成「风险提示」章节。"""
        lines = ["## 风险提示"]
        warnings = []

        ml = macro.get("monetary_liquidity", {})
        g = macro.get("growth", {})
        infl = macro.get("inflation", {})
        ext = macro.get("external", {})

        # 流动性风险
        liq = ml.get("liquidity_signal", "neutral")
        if liq == "tight":
            warnings.append("短期流动性偏紧，SHIBOR/DR007 处于高位，关注资金面压力。")

        # 通胀风险
        inf_sig = infl.get("inflation_signal", "neutral")
        if inf_sig == "stagflation_risk":
            warnings.append("CPI 上行 + PPI 下行，滞胀风险上升，警惕企业盈利承压。")
        elif inf_sig == "deflation":
            warnings.append("通缩压力显现，总需求不足，关注消费和投资数据。")
        elif inf_sig == "overheat":
            warnings.append("通胀超预期上行，关注货币政策收紧风险。")

        # 中美利差风险
        cn_us = ext.get("cn_us_spread")
        if cn_us is not None and cn_us < -1.5:
            warnings.append(f"中美利差倒挂 {abs(cn_us):.2f}%，外资流出压力较大。")

        # 北向资金风险
        north = ext.get("north_flow_20d")
        if north is not None and north < -200:
            warnings.append(f"北向资金近 20 日净流出 {abs(north):.0f} 亿元，外资情绪偏谨慎。")

        # 汇率风险
        fx_trend = ext.get("usdcny_trend")
        if fx_trend == "up":
            warnings.append("人民币处于贬值通道，关注汇率对进口成本和外资流出的影响。")

        # 增长风险
        pmi = g.get("pmi_mfg", 50)
        if pmi is not None and pmi < 49:
            warnings.append(f"制造业 PMI {pmi:.1f}，连续处于荣枯线下方，经济下行压力加大。")

        if not warnings:
            lines.append("\n当前未触发重大风险信号。")
        else:
            for w in warnings:
                lines.append(f"- ⚠️ {w}")

        return "\n".join(lines)

    @staticmethod
    def _disclaimer() -> str:
        return (
            "---\n\n"
            "*免责声明：本简报由 fqf Sentinel 规则引擎自动生成，"
            "仅供研究参考，不构成任何投资建议。数据来源包括 AKShare、东方财富等公开渠道，"
            "可能存在延迟或偏差。投资有风险，入市需谨慎。*"
        )

    # ═══════════════════════════════════════════════════
    #  便捷函数
    # ═══════════════════════════════════════════════════

    def generate_brief(
        self,
        macro_result: Optional[dict] = None,
        industry_df: Optional[pd.DataFrame] = None,
        date: Optional[str] = None,
    ) -> str:
        """别名：同 generate()。"""
        return self.generate(macro_result, industry_df, date)
