"""财务三表标准化清洗模块

从 AKShareFetcher 获取的原始三张报表往往存在：
- 列名不统一（中文/英文混用、不同接口字段名不同）
- 数值为字符串类型（含逗号、单位"亿"等）
- 报告期格式不统一
- 缺失值和异常值

本模块负责：
1. 字段重命名 → 统一英文 snake_case
2. 类型转换 → 数值列转 float，报告期转 datetime
3. 时间轴标准化 → 以报告期为 index，按时序排列
4. 异常值/缺失值基础处理
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd


# ── 资产负债表字段映射 ────────────────────────────────────
# 东方财富 / 新浪 / AKShare 各接口字段名存在差异，尽量覆盖常见变体
_BS_COL_MAP: dict[str, str] = {
    # ── 报告期 ──
    "报告期": "report_date",
    "REPORT_DATE": "report_date",
    "report_date": "report_date",
    # ── 资产 ──
    "货币资金": "cash",
    "应收账款": "accounts_receivable",
    "应收账款净额": "accounts_receivable",
    "存货": "inventory",
    "流动资产合计": "total_current_assets",
    "固定资产": "fixed_assets",
    "无形资产": "intangible_assets",
    "商誉": "goodwill",
    "长期股权投资": "long_term_equity_investment",
    "非流动资产合计": "total_non_current_assets",
    "资产总计": "total_assets",
    "资产合计": "total_assets",
    # ── 负债 ──
    "短期借款": "short_term_loan",
    "应付账款": "accounts_payable",
    "流动负债合计": "total_current_liabilities",
    "长期借款": "long_term_loan",
    "非流动负债合计": "total_non_current_liabilities",
    "负债合计": "total_liabilities",
    "负债总计": "total_liabilities",
    # ── 权益 ──
    "股本": "share_capital",
    "资本公积": "capital_surplus",
    "盈余公积": "surplus_reserve",
    "未分配利润": "retained_earnings",
    "归属于母公司所有者权益合计": "equity_parent",
    "所有者权益合计": "total_equity",
    "股东权益合计": "total_equity",
    "负债和股东权益合计": "total_liabilities_equity",
    "负债和所有者权益总计": "total_liabilities_equity",
}

# ── 利润表字段映射 ────────────────────────────────────────
_IS_COL_MAP: dict[str, str] = {
    "报告期": "report_date",
    "REPORT_DATE": "report_date",
    "report_date": "report_date",
    "营业总收入": "total_revenue",
    "营业收入": "revenue",
    "营业成本": "cogs",
    "毛利润": "gross_profit",
    "销售费用": "selling_expense",
    "管理费用": "admin_expense",
    "财务费用": "finance_expense",
    "研发费用": "rd_expense",
    "营业利润": "operating_profit",
    "利润总额": "pretax_profit",
    "所得税费用": "income_tax",
    "净利润": "net_profit",
    "归属于母公司所有者的净利润": "net_profit_parent",
    "归母净利润": "net_profit_parent",
    "少数股东损益": "minority_interest",
    "基本每股收益": "eps_basic",
    "稀释每股收益": "eps_diluted",
}

# ── 现金流量表字段映射 ────────────────────────────────────
_CF_COL_MAP: dict[str, str] = {
    "报告期": "report_date",
    "REPORT_DATE": "report_date",
    "report_date": "report_date",
    # 经营活动
    "经营活动产生的现金流量净额": "cfo",
    "经营活动现金流入小计": "operating_inflow",
    "经营活动现金流出小计": "operating_outflow",
    # 投资活动
    "投资活动产生的现金流量净额": "cfi",
    "投资活动现金流入小计": "investing_inflow",
    "投资活动现金流出小计": "investing_outflow",
    # 购置资产
    "购建固定资产、无形资产和其他长期资产支付的现金": "capex",
    "购建固定资产无形资产及其他长期资产": "capex",
    # 筹资活动
    "筹资活动产生的现金流量净额": "cff",
    # 净增加额
    "现金及现金等价物净增加额": "net_cash_change",
    "期末现金及现金等价物余额": "ending_cash",
}


class FinancialCleaner:
    """财务三表标准化清洗器。

    Parameters
    ----------
    n_periods : int
        保留最近多少个报告期，默认 12（约 3 年季报）
    fill_method : str
        缺失值填充方式: "ffill"(向前填充) | "zero"(填0) | "none"(不填)
    annual_only : bool
        是否只保留年报（12-31 报告期），默认 False 保留全部
    """

    def __init__(
        self,
        n_periods: int = 12,
        fill_method: str = "ffill",
        annual_only: bool = False,
    ):
        self.n_periods = n_periods
        self.fill_method = fill_method
        self.annual_only = annual_only

    # ─────────────────────────────────────────────────────────
    #  公开接口
    # ─────────────────────────────────────────────────────────

    def clean_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗资产负债表。"""
        return self._clean(df, _BS_COL_MAP, sheet_type="balance_sheet")

    def clean_income_statement(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗利润表。"""
        return self._clean(df, _IS_COL_MAP, sheet_type="income_statement")

    def clean_cash_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗现金流量表。"""
        return self._clean(df, _CF_COL_MAP, sheet_type="cash_flow")

    def clean_all(
        self,
        balance_sheet: pd.DataFrame,
        income_statement: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """一次性清洗三张报表，返回 dict。

        Returns
        -------
        dict with keys: "balance_sheet", "income_statement", "cash_flow"
        """
        return {
            "balance_sheet": self.clean_balance_sheet(balance_sheet),
            "income_statement": self.clean_income_statement(income_statement),
            "cash_flow": self.clean_cash_flow(cash_flow),
        }

    # ─────────────────────────────────────────────────────────
    #  内部处理流程
    # ─────────────────────────────────────────────────────────

    def _clean(
        self,
        df: pd.DataFrame,
        col_map: dict[str, str],
        sheet_type: str,
    ) -> pd.DataFrame:
        """通用清洗流程。"""
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # 1. 字段重命名（只重命名已知字段，保留未知字段原名）
        df = self._rename_columns(df, col_map)

        # 2. 确保 report_date 列存在
        df = self._ensure_report_date(df)
        if df is None or df.empty:
            return pd.DataFrame()

        # 3. 报告期标准化为 datetime
        df["report_date"] = self._parse_report_date(df["report_date"])
        df = df.dropna(subset=["report_date"])

        # 4. 年报过滤（可选）
        if self.annual_only:
            df = df[df["report_date"].dt.month == 12]

        # 5. 去重、按报告期降序排列
        df = df.drop_duplicates(subset=["report_date"]).sort_values(
            "report_date", ascending=False
        )

        # 6. 取最近 N 期
        df = df.head(self.n_periods)

        # 7. 数值列类型转换
        df = self._convert_numeric(df)

        # 8. 缺失值处理
        df = self._fill_missing(df)

        # 9. 以报告期为 index
        df = df.set_index("report_date").sort_index()

        return df

    @staticmethod
    def _rename_columns(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
        """重命名已知列，未知列保留原名。"""
        rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
        return df.rename(columns=rename_dict)

    @staticmethod
    def _ensure_report_date(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """确保有 report_date 列；尝试常见备用列名。"""
        if "report_date" in df.columns:
            return df

        # 尝试其他常见名称
        candidates = ["REPORT_DATE", "date", "DATE", "期间", "会计期间"]
        for col in candidates:
            if col in df.columns:
                df = df.rename(columns={col: "report_date"})
                return df

        # 如果行是多索引或列索引，尝试 reset_index
        if isinstance(df.index, pd.DatetimeIndex) or df.index.name in (
            "report_date",
            "REPORT_DATE",
            "date",
        ):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "report_date"})
            return df

        warnings.warn(
            "无法找到报告期列，返回空 DataFrame", stacklevel=3
        )
        return None

    @staticmethod
    def _parse_report_date(series: pd.Series) -> pd.Series:
        """将各种格式的报告期解析为 datetime。

        支持: "2023-12-31", "20231231", "2023/12/31", "2023Q4" 等
        """
        def _parse_single(v):
            if pd.isna(v):
                return pd.NaT
            s = str(v).strip()
            # 处理 "2023Q4" 格式
            if "Q" in s.upper():
                try:
                    return pd.Period(s, freq="Q").to_timestamp(how="end")
                except Exception:
                    pass
            # 标准解析
            try:
                return pd.to_datetime(s)
            except Exception:
                return pd.NaT

        return series.map(_parse_single)

    @staticmethod
    def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """将数值列从字符串/对象类型转为 float。"""
        skip_cols = {"report_date"}
        for col in df.columns:
            if col in skip_cols:
                continue
            # pandas >= 2.0 字符串列 dtype 可能是 'object' 或 'string'，统一检查非数值
            if not pd.api.types.is_numeric_dtype(df[col]):
                # 去掉千分位逗号和特殊字符
                cleaned = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("--", "", regex=False)
                    .str.replace("－", "-", regex=False)
                    .str.strip()
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    numeric = pd.to_numeric(cleaned, errors="coerce")
                # 只在能成功转换超过半数时才替换
                non_null = numeric.notna().sum()
                if non_null > len(df) * 0.3:
                    df[col] = numeric
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """缺失值填补。"""
        if self.fill_method == "ffill":
            # 时序向前填充（用最近已知值）
            df = df.sort_values("report_date").ffill()
        elif self.fill_method == "zero":
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(0)
        # "none" 不处理
        return df

    # ─────────────────────────────────────────────────────────
    #  辅助工具
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def merge_statements(
        balance_sheet: pd.DataFrame,
        income_statement: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> pd.DataFrame:
        """将三表合并为宽格式 DataFrame（以报告期为 index）。

        合并后的 DataFrame 方便一次性计算所有指标。
        重复列会加前缀: bs_/is_/cf_

        Returns
        -------
        pd.DataFrame
        """
        dfs = {
            "bs": balance_sheet,
            "is": income_statement,
            "cf": cash_flow,
        }
        merged = None
        for prefix, df in dfs.items():
            if df is None or df.empty:
                continue
            df_prefixed = df.add_prefix(f"{prefix}_")
            if merged is None:
                merged = df_prefixed
            else:
                merged = merged.join(df_prefixed, how="outer")
        if merged is None:
            return pd.DataFrame()
        return merged.sort_index()
