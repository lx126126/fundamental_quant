"""Value Line 风格一页式投研报告生成器

基于 FinancialCleaner + FinancialMetrics 的输出，生成标准化的
Markdown 格式单页投研报告（One-Pager）。

报告结构：
1. 封面信息：公司名称、代码、行业、报告日期
2. 关键估值指标：PE/PB/PS/市值（从行情数据传入）
3. 盈利能力摘要：ROE/毛利率/净利率/ROIC（最新期）
4. 成长趋势：营收/利润 3 年 CAGR + 最近一年 YoY
5. 财务质量：FCF 含金量、现金转化率、负债率
6. 历史数据表：近 5 年关键指标时序
7. 杜邦拆解：三因子 ROE 分解
8. 综合评分：雷达图描述 + 0-100 分
9. 风险提示 & 免责声明
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd


# ── 报告模板 ────────────────────────────────────────────────

_SECTION_SEP = "\n---\n"
_DOUBLE_SEP = "\n===\n"


class ReportGenerator:
    """Value Line 风格一页式投研报告生成器。

    Parameters
    ----------
    company_name : str
        公司名称
    symbol : str
        股票代码（纯数字，如 "000001"）
    industry : str, optional
        所属行业，如 "银行"
    analyst : str, optional
        分析师名称，默认 "fqf-auto"
    """

    def __init__(
        self,
        company_name: str,
        symbol: str,
        industry: str = "未知行业",
        analyst: str = "fqf-auto",
    ):
        self.company_name = company_name
        self.symbol = symbol
        self.industry = industry
        self.analyst = analyst
        self._today = date.today().strftime("%Y-%m-%d")

    # ═══════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════

    def generate(
        self,
        metrics_df: pd.DataFrame,
        scores: pd.Series,
        valuation: Optional[dict[str, Any]] = None,
    ) -> str:
        """生成完整的 Markdown 报告。

        Parameters
        ----------
        metrics_df : pd.DataFrame
            FinancialMetrics.compute_all() 的输出
        scores : pd.Series
            FinancialMetrics.score() 的输出
        valuation : dict, optional
            估值快照，keys 可包含: 最新价, 市盈率-动态, 市净率, 市销率,
            总市值, 流通市值

        Returns
        -------
        str
            完整的 Markdown 文本
        """
        sections = [
            self._header(valuation),
            self._valuation_snapshot(valuation),
            self._profitability_summary(metrics_df),
            self._growth_summary(metrics_df),
            self._cash_quality_summary(metrics_df),
            self._historical_table(metrics_df),
            self._dupont_breakdown(metrics_df),
            self._score_section(scores),
            self._risk_disclaimer(),
        ]

        return "\n\n".join(s for s in sections if s)

    # ═══════════════════════════════════════════════════════
    #  各章节
    # ═══════════════════════════════════════════════════════

    def _header(self, valuation: Optional[dict] = None) -> str:
        """封面信息。"""
        market = "深市" if self.symbol.startswith(("0", "3")) else "沪市"
        if self.symbol.startswith(("8", "4")):
            market = "北交所"

        latest_price = ""
        if valuation:
            price = valuation.get("最新价") or valuation.get("close")
            if price:
                latest_price = f" | 最新价: **¥{price:.2f}**"

        return f"""# {self.company_name}（{self.symbol}）基本面投研报告

> **行业**: {self.industry} | **市场**: {market} | **报告日期**: {self._today}{latest_price}
>
> *本报告由 [fqf](https://github.com/lx126126/fundamental_quant) 自动生成，仅供研究参考，不构成投资建议。*"""

    def _valuation_snapshot(self, valuation: Optional[dict] = None) -> str:
        """估值快照表。"""
        if not valuation:
            return ""

        rows = []
        field_map = {
            "市盈率-动态": ("市盈率(PE-TTM)", "{:.1f}x"),
            "市净率": ("市净率(PB)", "{:.2f}x"),
            "市销率": ("市销率(PS)", "{:.2f}x"),
            "总市值": ("总市值(亿)", "{:.1f}"),
            "流通市值": ("流通市值(亿)", "{:.1f}"),
        }

        for raw_key, (display_name, fmt) in field_map.items():
            val = valuation.get(raw_key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    # 总市值/流通市值单位通常是元，转为亿
                    if "市值" in display_name and "PE" not in display_name and "PB" not in display_name:
                        val = float(val) / 1e8
                    formatted = fmt.format(float(val))
                    rows.append(f"| {display_name} | {formatted} |")
                except (ValueError, TypeError):
                    pass

        if not rows:
            return ""

        header = "## 📊 估值快照\n\n| 指标 | 数值 |\n|------|------|\n"
        return header + "\n".join(rows)

    def _profitability_summary(self, df: pd.DataFrame) -> str:
        """盈利能力摘要。"""
        if df.empty:
            return ""

        latest = df.iloc[-1]
        lines = ["## 💹 盈利能力（最新报告期）\n", "| 指标 | 数值 | 说明 |", "|------|------|------|"]

        indicators = [
            ("gross_margin",     "毛利率",     _fmt_pct, "越高说明产品定价能力越强"),
            ("net_margin",       "净利率",     _fmt_pct, "扣除全部成本后的盈利水平"),
            ("roe",              "ROE",        _fmt_pct, "股东权益回报率，核心盈利指标"),
            ("roic",             "ROIC",       _fmt_pct, "投入资本回报率，衡量资本配置效率"),
            ("roa",              "ROA",        _fmt_pct, "总资产回报率"),
            ("ebitda_margin",    "EBITDA利润率", _fmt_pct, "息税折旧前利润率"),
            ("operating_margin", "营业利润率",  _fmt_pct, "主营业务盈利能力"),
        ]

        has_data = False
        for col, name, fmt_fn, note in indicators:
            if col in latest and not pd.isna(latest[col]):
                lines.append(f"| {name} | **{fmt_fn(latest[col])}** | {note} |")
                has_data = True

        if not has_data:
            return ""

        return "\n".join(lines)

    def _growth_summary(self, df: pd.DataFrame) -> str:
        """成长趋势摘要。"""
        if df.empty:
            return ""

        latest = df.iloc[-1]
        lines = ["## 📈 成长能力\n", "| 指标 | 数值 | 解读 |", "|------|------|------|"]

        indicators = [
            ("revenue_yoy",        "营收同比增速",     _fmt_pct, _interpret_growth),
            ("net_profit_yoy",     "归母净利润同比",   _fmt_pct, _interpret_growth),
            ("operating_profit_yoy","营业利润同比",    _fmt_pct, _interpret_growth),
            ("fcf_yoy",            "FCF同比增速",      _fmt_pct, _interpret_growth),
            ("revenue_cagr3",      "营收3年CAGR",      _fmt_pct, _interpret_cagr),
            ("net_profit_cagr3",   "净利润3年CAGR",    _fmt_pct, _interpret_cagr),
        ]

        has_data = False
        for col, name, fmt_fn, interp_fn in indicators:
            if col in latest and not pd.isna(latest[col]):
                val = latest[col]
                lines.append(f"| {name} | **{fmt_fn(val)}** | {interp_fn(val)} |")
                has_data = True

        if not has_data:
            return ""

        return "\n".join(lines)

    def _cash_quality_summary(self, df: pd.DataFrame) -> str:
        """现金质量与负债稳健性摘要。"""
        if df.empty:
            return ""

        latest = df.iloc[-1]
        lines = ["## 💰 现金质量 & 财务稳健\n", "| 指标 | 数值 | 解读 |", "|------|------|------|"]

        indicators = [
            ("fcf_to_net_profit", "FCF/净利润",  _fmt_x,   _interpret_fcf_ratio),
            ("cfo_to_net_profit", "CFO/净利润",  _fmt_x,   _interpret_fcf_ratio),
            ("fcf_margin",        "FCF利润率",   _fmt_pct, _interpret_growth),
            ("debt_ratio",        "资产负债率",  _fmt_pct, _interpret_debt),
            ("current_ratio",     "流动比率",    _fmt_x,   _interpret_liquidity),
            ("quick_ratio",       "速动比率",    _fmt_x,   _interpret_liquidity),
            ("interest_coverage", "利息覆盖倍数", _fmt_x,   _interpret_coverage),
            ("net_debt_to_equity","净负债率",    _fmt_pct, _interpret_debt),
        ]

        has_data = False
        for col, name, fmt_fn, interp_fn in indicators:
            if col in latest and not pd.isna(latest[col]):
                val = latest[col]
                lines.append(f"| {name} | **{fmt_fn(val)}** | {interp_fn(val)} |")
                has_data = True

        if not has_data:
            return ""

        return "\n".join(lines)

    def _historical_table(self, df: pd.DataFrame) -> str:
        """近 5 年关键指标历史数据表。"""
        if df.empty:
            return ""

        key_cols = [
            "gross_margin", "net_margin", "roe", "revenue_yoy",
            "net_profit_yoy", "fcf_to_net_profit", "debt_ratio",
        ]
        col_names = {
            "gross_margin": "毛利率",
            "net_margin": "净利率",
            "roe": "ROE",
            "revenue_yoy": "营收YoY",
            "net_profit_yoy": "利润YoY",
            "fcf_to_net_profit": "FCF质量",
            "debt_ratio": "负债率",
        }

        available = [c for c in key_cols if c in df.columns]
        if not available:
            return ""

        # 最近 5 期
        sub = df[available].tail(5).copy()
        sub.index = sub.index.strftime("%Y-%m-%d")

        # 格式化
        def _fmt_cell(val, col):
            if pd.isna(val):
                return "—"
            if col in ("gross_margin", "net_margin", "roe", "revenue_yoy",
                       "net_profit_yoy", "fcf_to_net_profit", "debt_ratio"):
                return f"{val:.1%}"
            return f"{val:.2f}"

        lines = ["## 📋 历史关键指标（近5年）\n"]
        header_row = "| 报告期 | " + " | ".join(col_names.get(c, c) for c in available) + " |"
        sep_row = "|--------|" + "---------|" * len(available)
        lines += [header_row, sep_row]

        for idx, row in sub.iterrows():
            cells = [_fmt_cell(row[c], c) for c in available]
            lines.append(f"| {idx} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _dupont_breakdown(self, df: pd.DataFrame) -> str:
        """杜邦三因子拆解。"""
        if df.empty:
            return ""

        dp_cols = ["dp_net_margin", "dp_asset_turnover", "dp_equity_multiplier", "dp_roe_3factor"]
        available = [c for c in dp_cols if c in df.columns]
        if len(available) < 3:
            return ""

        latest = df.iloc[-1]

        lines = ["## 🔬 杜邦分析（ROE 三因子拆解）\n"]
        lines.append("$$\\text{ROE} = \\text{净利率} \\times \\text{总资产周转率} \\times \\text{权益乘数}$$\n")

        row_cells = []
        names = {
            "dp_net_margin": "净利率",
            "dp_asset_turnover": "资产周转率",
            "dp_equity_multiplier": "权益乘数",
            "dp_roe_3factor": "= ROE(验证)",
        }

        for col in available:
            val = latest.get(col, np.nan)
            name = names.get(col, col)
            if pd.isna(val):
                row_cells.append(f"**{name}**: —")
            elif col in ("dp_net_margin", "dp_roe_3factor"):
                row_cells.append(f"**{name}**: {val:.1%}")
            else:
                row_cells.append(f"**{name}**: {val:.2f}x")

        lines.append(" × ".join(row_cells[:3]))
        if len(row_cells) == 4:
            lines.append("\n" + row_cells[3])

        # 三期趋势
        if len(df) >= 2:
            lines.append("\n**趋势（近3年）**\n")
            sub = df[available].tail(3)
            sub.index = sub.index.strftime("%Y")
            header = "| 年份 | " + " | ".join(names.get(c, c) for c in available) + " |"
            sep = "|------|" + "------|" * len(available)
            lines += [header, sep]
            for idx, row in sub.iterrows():
                cells = []
                for c in available:
                    v = row[c]
                    if pd.isna(v):
                        cells.append("—")
                    elif c in ("dp_net_margin", "dp_roe_3factor"):
                        cells.append(f"{v:.1%}")
                    else:
                        cells.append(f"{v:.2f}x")
                lines.append(f"| {idx} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _score_section(self, scores: pd.Series) -> str:
        """综合评分展示。"""
        if scores.empty:
            return ""

        composite = scores.get("composite_score", 50.0)

        # 评级映射
        if composite >= 80:
            grade, emoji = "A（优质）", "⭐⭐⭐⭐⭐"
        elif composite >= 65:
            grade, emoji = "B（良好）", "⭐⭐⭐⭐"
        elif composite >= 50:
            grade, emoji = "C（一般）", "⭐⭐⭐"
        elif composite >= 35:
            grade, emoji = "D（偏弱）", "⭐⭐"
        else:
            grade, emoji = "E（弱）", "⭐"

        lines = [
            f"## 🏆 综合评分\n",
            f"**总分**: {composite:.1f} / 100 — 评级: **{grade}** {emoji}\n",
            "| 维度 | 分数 |",
            "|------|------|",
        ]

        dim_names = {
            "gross_margin": "毛利率得分",
            "net_margin": "净利率得分",
            "roe": "ROE得分",
            "roic": "ROIC得分",
            "revenue_yoy": "营收增速得分",
            "net_profit_yoy": "利润增速得分",
            "fcf_quality": "FCF质量得分",
            "debt_ratio": "负债稳健得分",
            "current_ratio": "流动性得分",
        }

        for key, name in dim_names.items():
            val = scores.get(key)
            if val is not None and not pd.isna(val):
                lines.append(f"| {name} | {val:.1f} |")

        return "\n".join(lines)

    def _risk_disclaimer(self) -> str:
        """风险提示。"""
        return (
            "## ⚠️ 风险提示 & 免责声明\n\n"
            "1. 本报告数据来源于公开财务报告（AKShare），不对数据准确性作任何保证。\n"
            "2. 历史财务指标不代表未来业绩，本报告不构成任何投资建议。\n"
            "3. 投资有风险，决策请结合完整的定性分析、行业研究及个人风险承受能力。\n"
            f"4. 报告生成时间: {self._today}，数据存在滞后，请以最新披露为准。\n\n"
            "*本报告由 [fqf 基本面量化框架](https://github.com/lx126126/fundamental_quant) 自动生成*"
        )


# ── 格式化工具 ───────────────────────────────────────────────


def _fmt_pct(val: float) -> str:
    return f"{val:.1%}"


def _fmt_x(val: float) -> str:
    return f"{val:.2f}x"


# ── 解读函数 ─────────────────────────────────────────────────


def _interpret_growth(val: float) -> str:
    if val > 0.3:
        return "🚀 高速增长"
    elif val > 0.1:
        return "✅ 稳健增长"
    elif val > 0:
        return "➡️ 温和增长"
    elif val > -0.1:
        return "⚠️ 轻微下滑"
    else:
        return "🔴 明显下滑"


def _interpret_cagr(val: float) -> str:
    if val > 0.2:
        return "🚀 3年高增长"
    elif val > 0.1:
        return "✅ 3年稳健"
    elif val > 0:
        return "➡️ 3年温和增长"
    else:
        return "⚠️ 3年增速偏弱"


def _interpret_fcf_ratio(val: float) -> str:
    if val > 1.2:
        return "💎 利润含金量极高"
    elif val > 0.8:
        return "✅ 利润含金量良好"
    elif val > 0.5:
        return "➡️ 利润含金量一般"
    elif val > 0:
        return "⚠️ 利润含金量偏低"
    else:
        return "🔴 FCF为负，注意风险"


def _interpret_debt(val: float) -> str:
    if val < 0.3:
        return "💎 低杠杆，财务稳健"
    elif val < 0.5:
        return "✅ 适中杠杆"
    elif val < 0.7:
        return "⚠️ 杠杆偏高"
    else:
        return "🔴 高杠杆，财务风险较大"


def _interpret_liquidity(val: float) -> str:
    if val > 2.0:
        return "💎 流动性充裕"
    elif val > 1.0:
        return "✅ 流动性良好"
    elif val > 0.8:
        return "⚠️ 流动性偏紧"
    else:
        return "🔴 流动性风险"


def _interpret_coverage(val: float) -> str:
    if val > 10:
        return "💎 利息压力极小"
    elif val > 5:
        return "✅ 利息保障充足"
    elif val > 2:
        return "➡️ 利息保障一般"
    elif val > 1:
        return "⚠️ 偿息压力较大"
    else:
        return "🔴 偿息风险"


# ── 便捷函数 ─────────────────────────────────────────────────


def generate_report(
    company_name: str,
    symbol: str,
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
    cash_flow: pd.DataFrame,
    valuation: Optional[dict[str, Any]] = None,
    industry: str = "未知行业",
    annual_only: bool = True,
) -> str:
    """一站式生成投研报告的便捷函数。

    Parameters
    ----------
    company_name : str
        公司名称
    symbol : str
        股票代码
    balance_sheet, income_statement, cash_flow : pd.DataFrame
        经过 FinancialCleaner 清洗的三张报表
    valuation : dict, optional
        估值快照（来自 AKShareFetcher.fetch_valuation）
    industry : str
        行业名称
    annual_only : bool
        是否只用年报数据计算指标

    Returns
    -------
    str
        Markdown 格式完整报告
    """
    from fqf.analyzer.metrics import FinancialMetrics

    calc = FinancialMetrics(annual_only=annual_only)
    metrics_df = calc.compute_all(balance_sheet, income_statement, cash_flow)
    scores = calc.score(metrics_df)

    gen = ReportGenerator(
        company_name=company_name,
        symbol=symbol,
        industry=industry,
    )
    return gen.generate(metrics_df, scores, valuation=valuation)
