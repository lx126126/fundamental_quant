"""AKShare 数据源实现

基于 AKShare (>= 1.14.0) 封装全量投研数据获取接口：
- 行情数据：个股日线、指数日线
- 财务数据：三张报表（资产负债表、利润表、现金流量表）
- 估值数据：PE/PB/PS、总市值、流通市值
- 宏观数据：PMI、M1/M2、国债收益率、CPI/PPI、LPR、社融
- 行业数据：申万行业分类、行业指数、成分股

Stock Code Convention:
    AKShare 不同接口对股票代码格式要求不同，本模块统一使用纯数字代码
    (如 "000001")，内部自动转换为各接口所需格式。
"""

import time
from typing import Optional

import akshare as ak
import pandas as pd

from fqf.data.fetcher import DataFetcher


# ── 股票代码工具 ──────────────────────────────────────


def _code_to_ak(symbol: str) -> str:
    """将纯数字代码转为 AKShare 东方财富格式。

    "000001" → "sz000001" (深市)
    "600000" → "sh600000" (沪市)
    "688001" → "sh688001" (科创板-沪市)
    "300001" → "sz300001" (创业板-深市)
    """
    if symbol.startswith(("sh", "sz", "bj")):
        return symbol
    code = symbol.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith(("8", "4")):
        return f"bj{code}"
    else:
        return f"sz{code}"


class AKShareFetcher(DataFetcher):
    """AKShare 数据获取器。

    Parameters
    ----------
    cache_dir : Path, optional
        本地缓存目录
    rate_limit : float, optional
        API 调用间隔（秒），默认 0.5
    """

    rate_limit: float = 0.5
    _last_call: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self._last_call = 0.0

    def _throttle(self):
        """API 调用节流，避免请求过快被封。"""
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call = time.time()

    # ═══════════════════════════════════════════════════
    #  行情数据
    # ═══════════════════════════════════════════════════

    def fetch_daily_hist(
        self, symbol: str, start: str, end: str, adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取个股日线行情。

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"
        start : str
            起始日期 "YYYYMMDD"
        end : str
            结束日期 "YYYYMMDD"
        adjust : str
            复权方式: "qfq"(前复权), "hfq"(后复权), ""(不复权)

        Returns
        -------
        pd.DataFrame
            columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅,
                     涨跌幅, 涨跌额, 换手率
        """
        cache_key = f"{symbol}_{adjust}_{start}_{end}"

        def _fetch():
            self._throttle()
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            return self._standardize_daily_columns(df)

        return self._cached_fetch(
            "market", cache_key, _fetch, ttl_seconds=86400
        )

    def fetch_index_daily(
        self, index_code: str, start: str, end: str
    ) -> pd.DataFrame:
        """获取指数日线行情。

        Parameters
        ----------
        index_code : str
            指数代码，如 "000001"（上证指数）、"399001"（深证成指）、
            "000300"（沪深300）
        start : str
            起始日期 "YYYYMMDD"
        end : str
            结束日期 "YYYYMMDD"

        Returns
        -------
        pd.DataFrame
        """
        cache_key = f"idx_{index_code}_{start}_{end}"

        def _fetch():
            self._throttle()
            # 指数代码需要加 sh/sz 前缀
            if index_code.startswith("000"):
                symbol = f"sh{index_code}"
            elif index_code.startswith("399"):
                symbol = f"sz{index_code}"
            else:
                symbol = index_code
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.rename(columns={"date": "日期"})
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df[(df["日期"] >= start) & (df["日期"] <= end)]
            return df

        return self._cached_fetch(
            "market", cache_key, _fetch, ttl_seconds=86400
        )

    # ═══════════════════════════════════════════════════
    #  财务数据
    # ═══════════════════════════════════════════════════

    def fetch_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """获取资产负债表。

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"

        Returns
        -------
        pd.DataFrame
            列包含: 报告期, 各项资产/负债/权益科目
        """
        cache_key = f"bs_{symbol}"

        def _fetch():
            self._throttle()
            try:
                df = ak.stock_financial_balance_sheet(symbol=symbol)
                return df
            except Exception:
                # 备用方案：东方财富接口
                code = _code_to_ak(symbol)
                df = ak.stock_financial_report_sina(stock=code, symbol="balance")
                return df

        return self._cached_fetch(
            "financial", cache_key, _fetch, ttl_seconds=86400 * 5
        )

    def fetch_income_statement(self, symbol: str) -> pd.DataFrame:
        """获取利润表。

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"

        Returns
        -------
        pd.DataFrame
        """
        cache_key = f"is_{symbol}"

        def _fetch():
            self._throttle()
            try:
                df = ak.stock_financial_profit(symbol=symbol)
                return df
            except Exception:
                code = _code_to_ak(symbol)
                df = ak.stock_financial_report_sina(stock=code, symbol="profit")
                return df

        return self._cached_fetch(
            "financial", cache_key, _fetch, ttl_seconds=86400 * 5
        )

    def fetch_cash_flow(self, symbol: str) -> pd.DataFrame:
        """获取现金流量表。

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"

        Returns
        -------
        pd.DataFrame
        """
        cache_key = f"cf_{symbol}"

        def _fetch():
            self._throttle()
            try:
                df = ak.stock_financial_cash_flow(symbol=symbol)
                return df
            except Exception:
                code = _code_to_ak(symbol)
                df = ak.stock_financial_report_sina(
                    stock=code, symbol="cash_flow"
                )
                return df

        return self._cached_fetch(
            "financial", cache_key, _fetch, ttl_seconds=86400 * 5
        )

    # ═══════════════════════════════════════════════════
    #  估值与参考数据
    # ═══════════════════════════════════════════════════

    def fetch_valuation(self, symbol: str) -> dict:
        """获取单只股票最新估值数据。

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"

        Returns
        -------
        dict
            keys: 代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额,
                  振幅, 换手率, 量比, 市盈率-动态, 市净率, 市销率,
                  总市值, 流通市值, 行业
        """
        cache_key = f"val_{symbol}"

        def _fetch():
            self._throttle()
            # 获取全市场然后筛选（AKShare 没有单只查询接口）
            df_all = ak.stock_zh_a_spot_em()
            row = df_all[df_all["代码"] == symbol]
            if row.empty:
                raise ValueError(f"未找到股票: {symbol}")
            return row.iloc[0].to_dict()

        return self._cached_fetch(
            "market", cache_key, _fetch, ttl_seconds=300
        )

    def fetch_stock_list(self) -> pd.DataFrame:
        """获取全市场 A 股列表及估值快照。

        Returns
        -------
        pd.DataFrame
            列包含: 代码, 名称, 最新价, 涨跌幅, 市盈率-动态, 市净率,
                  市销率, 总市值, 流通市值, 行业 等
        """
        cache_key = "stock_list_all"

        def _fetch():
            self._throttle()
            df = ak.stock_zh_a_spot_em()
            return df

        return self._cached_fetch(
            "market", cache_key, _fetch, ttl_seconds=300
        )

    # ═══════════════════════════════════════════════════
    #  宏观数据
    # ═══════════════════════════════════════════════════

    # 支持的宏观指标映射
    _MACRO_MAP = {
        "pmi": "_fetch_pmi",
        "money_supply": "_fetch_money_supply",
        "bond_yield": "_fetch_bond_yield",
        "cpi": "_fetch_cpi",
        "ppi": "_fetch_ppi",
        "lpr": "_fetch_lpr",
        "social_financing": "_fetch_social_financing",
    }

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame:
        """获取宏观指标时序数据（统一入口）。

        Parameters
        ----------
        indicator : str
            指标名称:
            - "pmi": 制造业/非制造业 PMI
            - "money_supply": M0/M1/M2 货币供应
            - "bond_yield": 国债收益率曲线
            - "cpi": 居民消费价格指数
            - "ppi": 工业生产者出厂价格指数
            - "lpr": 贷款市场报价利率
            - "social_financing": 社会融资规模

        Returns
        -------
        pd.DataFrame
        """
        if indicator not in self._MACRO_MAP:
            raise ValueError(
                f"未知宏观指标: {indicator}，可选: {list(self._MACRO_MAP.keys())}"
            )
        method = getattr(self, self._MACRO_MAP[indicator])
        return method()

    def _fetch_pmi(self) -> pd.DataFrame:
        """获取 PMI 数据。"""
        cache_key = "pmi"

        def _fetch():
            self._throttle()
            df = ak.macro_china_pmi()
            df = df.rename(columns={
                "日期": "date",
                "制造业": "pmi_manufacturing",
                "非制造业": "pmi_non_manufacturing",
            })
            return df

        return self._cached_fetch(
            "macro", cache_key, _fetch, ttl_seconds=86400
        )

    def _fetch_money_supply(self) -> pd.DataFrame:
        """获取货币供应量 (M0/M1/M2)。"""
        cache_key = "money_supply"

        def _fetch():
            self._throttle()
            df = ak.macro_china_money_supply()
            return df

        return self._cached_fetch(
            "macro", cache_key, _fetch, ttl_seconds=86400
        )

    def _fetch_bond_yield(self) -> pd.DataFrame:
        """获取国债收益率曲线。"""
        cache_key = "bond_yield"

        def _fetch():
            self._throttle()
            df = ak.bond_china_yield(start_date="20100101")
            return df

        return self._cached_fetch(
            "macro", cache_key, _fetch, ttl_seconds=86400
        )

    def _fetch_cpi(self) -> pd.DataFrame:
        """获取 CPI 月度数据。"""
        cache_key = "cpi"

        def _fetch():
            self._throttle()
            df = ak.macro_china_cpi_monthly()
            return df

        return self._cached_fetch(
            "macro", cache_key, _fetch, ttl_seconds=86400
        )

    def _fetch_ppi(self) -> pd.DataFrame:
        """获取 PPI 月度数据。"""
        cache_key = "ppi"

        def _fetch():
            self._throttle()
            df = ak.macro_china_ppi_yearly()
            return df

        return self._cached_fetch(
            "macro", cache_key, _fetch, ttl_seconds=86400
        )

    def _fetch_lpr(self) -> pd.DataFrame:
        """获取 LPR 数据。"""
        cache_key = "lpr"

        def _fetch():
            self._throttle()
            df = ak.macro_china_lpr()
            return df

        return self._cached_fetch(
            "macro", cache_key, _fetch, ttl_seconds=86400
        )

    def _fetch_social_financing(self) -> pd.DataFrame:
        """获取社会融资规模数据。"""
        cache_key = "social_financing"

        def _fetch():
            self._throttle()
            df = ak.macro_china_shrzgm()
            return df

        return self._cached_fetch(
            "macro", cache_key, _fetch, ttl_seconds=86400
        )

    # ═══════════════════════════════════════════════════
    #  行业数据
    # ═══════════════════════════════════════════════════

    def fetch_industry_list(self) -> pd.DataFrame:
        """获取申万行业分类列表。

        Returns
        -------
        pd.DataFrame
            列包含: 行业名称, 公司数量, 平均价格, 涨跌幅 等
        """
        cache_key = "industry_list"

        def _fetch():
            self._throttle()
            df = ak.stock_board_industry_name_em()
            return df

        return self._cached_fetch(
            "industry", cache_key, _fetch, ttl_seconds=86400
        )

    def fetch_industry_index(
        self, industry: str, start: str, end: str
    ) -> pd.DataFrame:
        """获取行业板块指数行情。

        Parameters
        ----------
        industry : str
            行业名称，如 "食品饮料"、"银行"、"医药生物"
        start : str
            起始日期 "YYYYMMDD"
        end : str
            结束日期 "YYYYMMDD"

        Returns
        -------
        pd.DataFrame
            含 日期/开盘/收盘/最高/最低/成交量/成交额 等
        """
        cache_key = f"idx_{industry}_{start}_{end}"

        def _fetch():
            self._throttle()
            df = ak.stock_board_industry_index_em(
                symbol=industry,
                start_date=start,
                end_date=end,
            )
            return df

        return self._cached_fetch(
            "industry", cache_key, _fetch, ttl_seconds=86400
        )

    def fetch_industry_constituents(self, industry: str) -> pd.DataFrame:
        """获取行业成分股列表。

        Parameters
        ----------
        industry : str
            行业名称，如 "食品饮料"

        Returns
        -------
        pd.DataFrame
            列包含: 代码, 名称, 最新价, 涨跌幅, 市盈率, 市值 等
        """
        cache_key = f"cons_{industry}"

        def _fetch():
            self._throttle()
            df = ak.stock_board_industry_cons_em(symbol=industry)
            return df

        return self._cached_fetch(
            "industry", cache_key, _fetch, ttl_seconds=86400
        )

    # ═══════════════════════════════════════════════════
    #  工具方法
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _standardize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化日线行情列名。"""
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    @staticmethod
    def to_akshare_code(symbol: str) -> str:
        """将纯数字代码转为 AKShare 东方财富格式。"""
        return _code_to_ak(symbol)

    @staticmethod
    def to_pure_code(symbol: str) -> str:
        """从任意格式提取纯数字代码。

        "sh600000" → "600000"
        "SZ000001" → "000001"
        """
        symbol = symbol.lower()
        for prefix in ("sh", "sz", "bj"):
            if symbol.startswith(prefix):
                return symbol[len(prefix):]
        return symbol.zfill(6)

    def get_macro_indicators(self) -> list[str]:
        """返回所有支持的宏观指标名称。"""
        return list(self._MACRO_MAP.keys())
