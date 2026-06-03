"""财务指标计算模块

基于清洗后的三张报表（FinancialCleaner 输出），计算 40+ 核心投研指标。

模块分为六个维度：
1. 盈利能力   Profitability  — ROE/ROIC/毛利率/净利率/EBITDA
2. 成长能力   Growth         — 营收/利润/FCF 同比增速 & CAGR
3. 现金质量   Cash Quality   — FCF/净利润比、经营现金流覆盖率
4. 负债能力   Leverage       — 资产负债率、有息负债率、利息覆盖倍数
5. 运营效率   Efficiency     — 应收账款/存货/总资产周转率
6. 杜邦分析   DuPont         — ROE 三/五因子拆解

所有指标以 pd.DataFrame 形式返回，index 为 report_date（datetime）。
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


class FinancialMetrics:
    """财务指标计算器。

    Parameters
    ----------
    annual_only : bool
        是否只用年报期（12-31）计算，默认 True（避免季度数累加失真）
    min_periods : int
        计算同比增速所需最少期数，默认 2
    """

    def __init__(self, annual_only: bool = True, min_periods: int = 2):
        self.annual_only = annual_only
        self.min_periods = min_periods

    # ═══════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════

    def compute_all(
        self,
        balance_sheet: pd.DataFrame,
        income_statement: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算全部财务指标，返回宽格式 DataFrame。

        Parameters
        ----------
        balance_sheet : pd.DataFrame
            FinancialCleaner.clean_balance_sheet() 的输出
        income_statement : pd.DataFrame
            FinancialCleaner.clean_income_statement() 的输出
        cash_flow : pd.DataFrame
            FinancialCleaner.clean_cash_flow() 的输出

        Returns
        -------
        pd.DataFrame
            index = report_date，columns = 各指标名称
        """
        bs = self._filter_annual(balance_sheet) if self.annual_only else balance_sheet
        is_ = self._filter_annual(income_statement) if self.annual_only else income_statement
        cf = self._filter_annual(cash_flow) if self.annual_only else cash_flow

        parts = []
        parts.append(self.profitability(bs, is_, cf))
        parts.append(self.growth(is_, cf))
        parts.append(self.cash_quality(is_, cf))
        parts.append(self.leverage(bs, is_))
        parts.append(self.efficiency(bs, is_))
        parts.append(self.dupont(bs, is_))

        # 合并所有维度（过滤空 DataFrame，避免 concat 报错）
        valid_parts = [p for p in parts if p is not None and not p.empty]
        if not valid_parts:
            return pd.DataFrame()
        result = pd.concat(valid_parts, axis=1)
        return result.sort_index()

    # ═══════════════════════════════════════════════════════
    #  1. 盈利能力
    # ═══════════════════════════════════════════════════════

    def profitability(
        self,
        bs: pd.DataFrame,
        is_: pd.DataFrame,
        cf: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算盈利能力指标。

        Indicators
        ----------
        gross_margin      : 毛利率 = (营收 - 营业成本) / 营收
        net_margin        : 净利率 = 归母净利润 / 营收
        operating_margin  : 营业利润率 = 营业利润 / 营收
        roe               : 净资产收益率 = 归母净利润 / 平均归母净资产
        roa               : 总资产收益率 = 净利润 / 平均总资产
        roic              : 投入资本回报率 = NOPAT / 投入资本
        ebitda_margin     : EBITDA 利润率（近似）
        """
        metrics: dict[str, pd.Series] = {}

        # 毛利率
        rev = _first_valid(is_, "revenue", "total_revenue")
        cogs = _get(is_, "cogs")
        if rev is not None and cogs is not None:
            metrics["gross_margin"] = _safe_div(rev - cogs, rev)

        # 净利率
        net_profit = _first_valid(is_, "net_profit_parent", "net_profit")
        if rev is not None and net_profit is not None:
            metrics["net_margin"] = _safe_div(net_profit, rev)

        # 营业利润率
        op_profit = _get(is_, "operating_profit")
        if rev is not None and op_profit is not None:
            metrics["operating_margin"] = _safe_div(op_profit, rev)

        # ROE（用平均净资产）
        equity = _first_valid(bs, "equity_parent", "total_equity")
        if equity is not None and net_profit is not None:
            avg_equity = _rolling_mean(equity, 2)
            metrics["roe"] = _safe_div(net_profit, avg_equity)

        # ROA（用平均总资产）
        total_assets = _get(bs, "total_assets")
        net_profit_all = _get(is_, "net_profit")
        if total_assets is not None and net_profit_all is not None:
            avg_assets = _rolling_mean(total_assets, 2)
            metrics["roa"] = _safe_div(net_profit_all, avg_assets)

        # ROIC = NOPAT / (有息负债 + 权益)
        # NOPAT 近似 = 营业利润 * (1 - 税率)
        tax = _get(is_, "income_tax")
        pretax = _get(is_, "pretax_profit")
        st_loan = _get(bs, "short_term_loan")
        lt_loan = _get(bs, "long_term_loan")
        if (op_profit is not None and pretax is not None and tax is not None
                and equity is not None):
            tax_rate = _safe_div(tax.abs(), pretax.abs()).clip(0, 0.5)
            nopat = op_profit * (1 - tax_rate)
            invested_capital = equity.copy()
            if st_loan is not None:
                invested_capital = invested_capital + st_loan.fillna(0)
            if lt_loan is not None:
                invested_capital = invested_capital + lt_loan.fillna(0)
            avg_ic = _rolling_mean(invested_capital, 2)
            metrics["roic"] = _safe_div(nopat, avg_ic)

        # EBITDA margin（近似: 净利润 + 所得税 + 财务费用 + 折旧摊销近似用0）
        fin_exp = _get(is_, "finance_expense")
        if (net_profit_all is not None and tax is not None
                and fin_exp is not None and rev is not None):
            ebitda = net_profit_all + tax.fillna(0) + fin_exp.fillna(0)
            metrics["ebitda_margin"] = _safe_div(ebitda, rev)

        return _to_df(metrics)

    # ═══════════════════════════════════════════════════════
    #  2. 成长能力
    # ═══════════════════════════════════════════════════════

    def growth(self, is_: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
        """计算成长能力指标（同比增速 & CAGR）。

        Indicators
        ----------
        revenue_yoy       : 营收同比增速
        net_profit_yoy    : 归母净利润同比增速
        operating_profit_yoy : 营业利润同比增速
        fcf_yoy           : FCF 同比增速
        revenue_cagr3     : 营收 3 年 CAGR
        net_profit_cagr3  : 归母净利润 3 年 CAGR
        """
        metrics: dict[str, pd.Series] = {}

        rev = _first_valid(is_, "revenue", "total_revenue")
        net_profit = _first_valid(is_, "net_profit_parent", "net_profit")
        op_profit = _get(is_, "operating_profit")
        cfo = _get(cf, "cfo")
        capex = _get(cf, "capex")

        if rev is not None:
            metrics["revenue_yoy"] = _yoy(rev)
            cagr3 = _cagr(rev, 3)
            if cagr3 is not None:
                metrics["revenue_cagr3"] = cagr3

        if net_profit is not None:
            metrics["net_profit_yoy"] = _yoy(net_profit)
            cagr3 = _cagr(net_profit, 3)
            if cagr3 is not None:
                metrics["net_profit_cagr3"] = cagr3

        if op_profit is not None:
            metrics["operating_profit_yoy"] = _yoy(op_profit)

        # FCF = 经营现金流 - 资本支出
        if cfo is not None and capex is not None:
            fcf = cfo - capex.abs()
            metrics["fcf_yoy"] = _yoy(fcf)

        return _to_df(metrics)

    # ═══════════════════════════════════════════════════════
    #  3. 现金质量
    # ═══════════════════════════════════════════════════════

    def cash_quality(self, is_: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
        """计算现金质量指标。

        Indicators
        ----------
        fcf               : 自由现金流 = CFO - CAPEX
        fcf_margin        : FCF 利润率 = FCF / 营收
        fcf_to_net_profit : FCF / 归母净利润（>1 表示利润含金量高）
        cfo_to_net_profit : 经营现金流 / 净利润
        cash_conversion   : 净利润现金含量（同 cfo_to_net_profit）
        """
        metrics: dict[str, pd.Series] = {}

        rev = _first_valid(is_, "revenue", "total_revenue")
        net_profit = _first_valid(is_, "net_profit_parent", "net_profit")
        cfo = _get(cf, "cfo")
        capex = _get(cf, "capex")

        if cfo is not None and capex is not None:
            fcf = cfo - capex.abs()
            metrics["fcf"] = fcf
            if rev is not None:
                metrics["fcf_margin"] = _safe_div(fcf, rev)
            if net_profit is not None:
                metrics["fcf_to_net_profit"] = _safe_div(fcf, net_profit)

        if cfo is not None and net_profit is not None:
            metrics["cfo_to_net_profit"] = _safe_div(cfo, net_profit)
            metrics["cash_conversion"] = _safe_div(cfo, net_profit)

        return _to_df(metrics)

    # ═══════════════════════════════════════════════════════
    #  4. 负债能力
    # ═══════════════════════════════════════════════════════

    def leverage(self, bs: pd.DataFrame, is_: pd.DataFrame) -> pd.DataFrame:
        """计算负债与偿债能力指标。

        Indicators
        ----------
        debt_ratio         : 资产负债率 = 负债合计 / 总资产
        equity_multiplier  : 权益乘数 = 总资产 / 股东权益
        interest_bearing_debt_ratio : 有息负债率 = (短贷+长贷) / 总资产
        interest_coverage  : 利息覆盖倍数 = 营业利润 / 财务费用
        current_ratio      : 流动比率 = 流动资产 / 流动负债
        quick_ratio        : 速动比率 = (流动资产 - 存货) / 流动负债
        net_debt           : 净有息负债 = 短贷 + 长贷 - 货币资金
        net_debt_to_equity : 净负债率 = 净有息负债 / 股东权益
        """
        metrics: dict[str, pd.Series] = {}

        total_assets = _get(bs, "total_assets")
        total_liab = _get(bs, "total_liabilities")
        equity = _first_valid(bs, "equity_parent", "total_equity")
        curr_assets = _get(bs, "total_current_assets")
        curr_liab = _get(bs, "total_current_liabilities")
        inventory = _get(bs, "inventory")
        st_loan = _get(bs, "short_term_loan")
        lt_loan = _get(bs, "long_term_loan")
        cash = _get(bs, "cash")
        op_profit = _get(is_, "operating_profit")
        fin_exp = _get(is_, "finance_expense")

        if total_assets is not None and total_liab is not None:
            metrics["debt_ratio"] = _safe_div(total_liab, total_assets)

        if total_assets is not None and equity is not None:
            metrics["equity_multiplier"] = _safe_div(total_assets, equity)

        if total_assets is not None and st_loan is not None and lt_loan is not None:
            ib_debt = st_loan.fillna(0) + lt_loan.fillna(0)
            metrics["interest_bearing_debt_ratio"] = _safe_div(ib_debt, total_assets)

        if op_profit is not None and fin_exp is not None:
            # 财务费用可能为负（收益），取绝对值
            metrics["interest_coverage"] = _safe_div(op_profit, fin_exp.abs())

        if curr_assets is not None and curr_liab is not None:
            metrics["current_ratio"] = _safe_div(curr_assets, curr_liab)
            if inventory is not None:
                quick_assets = curr_assets - inventory.fillna(0)
                metrics["quick_ratio"] = _safe_div(quick_assets, curr_liab)

        if st_loan is not None and lt_loan is not None and cash is not None:
            net_debt = st_loan.fillna(0) + lt_loan.fillna(0) - cash.fillna(0)
            metrics["net_debt"] = net_debt
            if equity is not None:
                metrics["net_debt_to_equity"] = _safe_div(net_debt, equity)

        return _to_df(metrics)

    # ═══════════════════════════════════════════════════════
    #  5. 运营效率
    # ═══════════════════════════════════════════════════════

    def efficiency(self, bs: pd.DataFrame, is_: pd.DataFrame) -> pd.DataFrame:
        """计算运营效率指标（周转率/周转天数）。

        Indicators
        ----------
        asset_turnover        : 总资产周转率 = 营收 / 平均总资产
        receivable_turnover   : 应收账款周转率 = 营收 / 平均应收账款
        receivable_days       : 应收账款周转天数 = 365 / 应收账款周转率
        inventory_turnover    : 存货周转率 = 营业成本 / 平均存货
        inventory_days        : 存货周转天数 = 365 / 存货周转率
        payable_turnover      : 应付账款周转率 = 营业成本 / 平均应付账款
        payable_days          : 应付账款周转天数
        cash_conversion_cycle : 现金转化周期 = 应收天数 + 存货天数 - 应付天数
        """
        metrics: dict[str, pd.Series] = {}

        rev = _first_valid(is_, "revenue", "total_revenue")
        cogs = _get(is_, "cogs")
        total_assets = _get(bs, "total_assets")
        receivable = _get(bs, "accounts_receivable")
        inventory = _get(bs, "inventory")
        payable = _get(bs, "accounts_payable")

        if rev is not None and total_assets is not None:
            avg_assets = _rolling_mean(total_assets, 2)
            metrics["asset_turnover"] = _safe_div(rev, avg_assets)

        recv_days = None
        inv_days = None
        pay_days = None

        if rev is not None and receivable is not None:
            avg_recv = _rolling_mean(receivable, 2)
            rt = _safe_div(rev, avg_recv)
            metrics["receivable_turnover"] = rt
            recv_days = _safe_div(pd.Series(365, index=rt.index), rt)
            metrics["receivable_days"] = recv_days

        if cogs is not None and inventory is not None:
            avg_inv = _rolling_mean(inventory, 2)
            it = _safe_div(cogs, avg_inv)
            metrics["inventory_turnover"] = it
            inv_days = _safe_div(pd.Series(365, index=it.index), it)
            metrics["inventory_days"] = inv_days

        if cogs is not None and payable is not None:
            avg_pay = _rolling_mean(payable, 2)
            pt = _safe_div(cogs, avg_pay)
            metrics["payable_turnover"] = pt
            pay_days = _safe_div(pd.Series(365, index=pt.index), pt)
            metrics["payable_days"] = pay_days

        # 现金转化周期
        if recv_days is not None and inv_days is not None and pay_days is not None:
            idx = recv_days.index.intersection(
                inv_days.index
            ).intersection(pay_days.index)
            if len(idx) > 0:
                metrics["cash_conversion_cycle"] = (
                    recv_days.reindex(idx)
                    + inv_days.reindex(idx)
                    - pay_days.reindex(idx)
                )

        return _to_df(metrics)

    # ═══════════════════════════════════════════════════════
    #  6. 杜邦分析
    # ═══════════════════════════════════════════════════════

    def dupont(self, bs: pd.DataFrame, is_: pd.DataFrame) -> pd.DataFrame:
        """杜邦三/五因子拆解。

        三因子拆解:
            ROE = 净利率 × 总资产周转率 × 权益乘数

        五因子拆解（扩展版）:
            ROE = 税收效应 × 利息效应 × 营业利润率 × 总资产周转率 × 权益乘数

        Indicators
        ----------
        dp_net_margin      : 杜邦-净利率
        dp_asset_turnover  : 杜邦-总资产周转率
        dp_equity_multiplier: 杜邦-权益乘数
        dp_roe_3factor     : 杜邦三因子 ROE（乘积验证）
        dp_tax_effect      : 杜邦五因子-税收效应
        dp_interest_effect : 杜邦五因子-利息效应
        dp_op_margin       : 杜邦五因子-营业利润率
        dp_roe_5factor     : 杜邦五因子 ROE（乘积验证）
        """
        metrics: dict[str, pd.Series] = {}

        rev = _first_valid(is_, "revenue", "total_revenue")
        net_profit = _first_valid(is_, "net_profit_parent", "net_profit")
        net_profit_all = _get(is_, "net_profit")
        total_assets = _get(bs, "total_assets")
        equity = _first_valid(bs, "equity_parent", "total_equity")
        op_profit = _get(is_, "operating_profit")
        pretax = _get(is_, "pretax_profit")

        if rev is not None and net_profit is not None:
            metrics["dp_net_margin"] = _safe_div(net_profit, rev)

        if rev is not None and total_assets is not None:
            avg_assets = _rolling_mean(total_assets, 2)
            metrics["dp_asset_turnover"] = _safe_div(rev, avg_assets)

        if total_assets is not None and equity is not None:
            metrics["dp_equity_multiplier"] = _safe_div(total_assets, equity)

        # 三因子 ROE
        if all(k in metrics for k in ("dp_net_margin", "dp_asset_turnover", "dp_equity_multiplier")):
            idx = metrics["dp_net_margin"].index.intersection(
                metrics["dp_asset_turnover"].index
            ).intersection(metrics["dp_equity_multiplier"].index)
            metrics["dp_roe_3factor"] = (
                metrics["dp_net_margin"].reindex(idx)
                * metrics["dp_asset_turnover"].reindex(idx)
                * metrics["dp_equity_multiplier"].reindex(idx)
            )

        # 五因子扩展
        if net_profit_all is not None and pretax is not None:
            metrics["dp_tax_effect"] = _safe_div(net_profit_all, pretax)

        if pretax is not None and op_profit is not None:
            metrics["dp_interest_effect"] = _safe_div(pretax, op_profit)

        if rev is not None and op_profit is not None:
            metrics["dp_op_margin"] = _safe_div(op_profit, rev)

        if all(k in metrics for k in (
            "dp_tax_effect", "dp_interest_effect", "dp_op_margin",
            "dp_asset_turnover", "dp_equity_multiplier"
        )):
            idx = metrics["dp_tax_effect"].index
            for k in ("dp_interest_effect", "dp_op_margin", "dp_asset_turnover", "dp_equity_multiplier"):
                idx = idx.intersection(metrics[k].index)
            metrics["dp_roe_5factor"] = (
                metrics["dp_tax_effect"].reindex(idx)
                * metrics["dp_interest_effect"].reindex(idx)
                * metrics["dp_op_margin"].reindex(idx)
                * metrics["dp_asset_turnover"].reindex(idx)
                * metrics["dp_equity_multiplier"].reindex(idx)
            )

        return _to_df(metrics)

    # ═══════════════════════════════════════════════════════
    #  综合评分
    # ═══════════════════════════════════════════════════════

    def score(self, metrics_df: pd.DataFrame) -> pd.Series:
        """基于最新期指标生成 0-100 综合评分（越高越好）。

        评分维度（各占一定权重）：
        - 盈利质量：毛利率、净利率、ROE、ROIC
        - 成长动能：营收/利润 YoY
        - 现金含金量：FCF/净利润
        - 财务稳健：资产负债率（逆向）、流动比率
        """
        if metrics_df.empty:
            return pd.Series(dtype=float)

        latest = metrics_df.iloc[-1]

        def _score_metric(val, low, high, reverse=False):
            """将单一指标线性映射到 0-100。"""
            if pd.isna(val):
                return 50.0  # 无数据给中性分
            score = (val - low) / (high - low) * 100
            score = float(np.clip(score, 0, 100))
            return 100 - score if reverse else score

        scores = {}

        # 盈利质量（权重 40%）
        if "gross_margin" in latest:
            scores["gross_margin"] = _score_metric(latest["gross_margin"], 0, 0.6)
        if "net_margin" in latest:
            scores["net_margin"] = _score_metric(latest["net_margin"], 0, 0.3)
        if "roe" in latest:
            scores["roe"] = _score_metric(latest["roe"], 0, 0.3)
        if "roic" in latest:
            scores["roic"] = _score_metric(latest["roic"], 0, 0.25)

        # 成长动能（权重 30%）
        if "revenue_yoy" in latest:
            scores["revenue_yoy"] = _score_metric(latest["revenue_yoy"], -0.1, 0.4)
        if "net_profit_yoy" in latest:
            scores["net_profit_yoy"] = _score_metric(latest["net_profit_yoy"], -0.2, 0.5)

        # 现金质量（权重 20%）
        if "fcf_to_net_profit" in latest:
            scores["fcf_quality"] = _score_metric(latest["fcf_to_net_profit"], 0, 1.5)

        # 财务稳健（权重 10%）
        if "debt_ratio" in latest:
            scores["debt_ratio"] = _score_metric(latest["debt_ratio"], 0.2, 0.8, reverse=True)
        if "current_ratio" in latest:
            scores["current_ratio"] = _score_metric(latest["current_ratio"], 0.5, 3.0)

        if not scores:
            return pd.Series({"composite_score": 50.0})

        composite = float(np.mean(list(scores.values())))
        scores["composite_score"] = composite

        return pd.Series(scores)

    # ─────────────────────────────────────────────────────────
    #  内部工具
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _filter_annual(df: pd.DataFrame) -> pd.DataFrame:
        """只保留年报期（12月31日报告期）。"""
        if df.empty:
            return df
        return df[df.index.month == 12]


# ── 模块级私有工具函数 ────────────────────────────────────────


def _get(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """安全取列，不存在时返回 None。"""
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col]
    if s.dtype == object:
        s = pd.to_numeric(s, errors="coerce")
    return s


def _first_valid(df: pd.DataFrame, *cols: str) -> Optional[pd.Series]:
    """依次尝试列名，返回第一个存在且非空的 Series；全不存在则返回 None。

    用于替代 `_get(df, "a") or _get(df, "b")` — 后者对 pd.Series 会
    触发 ValueError: The truth value of a Series is ambiguous。
    """
    for col in cols:
        result = _get(df, col)
        if result is not None:
            return result
    return None


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全除法，分母为 0 或 NaN 时返回 NaN。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = numerator / denominator.replace(0, np.nan)
    return result


def _rolling_mean(s: pd.Series, window: int = 2) -> pd.Series:
    """滚动均值，用于计算平均值（如平均资产、平均净资产）。"""
    return s.rolling(window=window, min_periods=1).mean()


def _yoy(s: pd.Series) -> pd.Series:
    """计算同比增速（年度数据用 pct_change(1)，百分比形式）。"""
    return s.pct_change(1)


def _cagr(s: pd.Series, years: int) -> Optional[pd.Series]:
    """计算 N 年 CAGR。需要至少 years+1 期数据。"""
    if len(s.dropna()) <= years:
        return None

    result = {}
    for date in s.index:
        end_val = s.loc[date]
        # 找 years 期之前的值（按 index 位置）
        idx_pos = s.index.get_loc(date)
        if idx_pos < years:
            result[date] = np.nan
            continue
        start_val = s.iloc[idx_pos - years]
        if pd.isna(start_val) or start_val == 0:
            result[date] = np.nan
            continue
        try:
            cagr_val = (end_val / start_val) ** (1 / years) - 1
            result[date] = cagr_val
        except Exception:
            result[date] = np.nan
    return pd.Series(result)


def _to_df(metrics: dict[str, pd.Series]) -> pd.DataFrame:
    """将指标字典合并为 DataFrame。"""
    if not metrics:
        return pd.DataFrame()
    # 找公共 index
    valid = {k: v for k, v in metrics.items() if v is not None and len(v) > 0}
    if not valid:
        return pd.DataFrame()
    return pd.DataFrame(valid)
