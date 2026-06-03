"""宏观因子分析器 (MacroFactorAnalyzer)

追踪中国宏观经济关键变量，覆盖四大维度：
- 货币政策与流动性（11 指标）
- 经济增长（10 指标）
- 通胀（3 指标）
- 外部环境（4 指标）

生成信号判断和 0-100 宏观综合评分。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 辅助函数 ──────────────────────────────────────────


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    """安全除法。"""
    if b == 0 or pd.isna(a) or pd.isna(b):
        return default
    return a / b


def _yoy(series: pd.Series) -> pd.Series:
    """计算同比增长率（假设按时间升序排列）。"""
    return series.pct_change(periods=1) * 100 if len(series) > 1 else pd.Series(dtype=float)


def _ma(series: pd.Series, window: int) -> pd.Series:
    """移动平均。"""
    return series.rolling(window=window, min_periods=1).mean()


def _trend(series: pd.Series, window: int = 3) -> str:
    """判断序列近期趋势方向。

    取最近 window 个值做线性回归斜率判断。
    返回 "up" / "down" / "flat"。
    """
    vals = series.dropna().values
    if len(vals) < 2:
        return "flat"
    recent = vals[-window:] if len(vals) >= window else vals
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    threshold = np.abs(recent).mean() * 0.005 if np.abs(recent).mean() > 0 else 1e-6
    if slope > threshold:
        return "up"
    elif slope < -threshold:
        return "down"
    return "flat"


def _latest(series: pd.Series) -> float:
    """获取最新非空值。"""
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) > 0 else np.nan


def _map_score(value: float, low: float, high: float, reverse: bool = False) -> float:
    """将指标值线性映射到 0-100 分。

    Parameters
    ----------
    value : 指标当前值
    low : 低水位（对应 0 或 100，取决于 reverse）
    high : 高水位（对应 100 或 0，取决于 reverse）
    reverse : True 时值越低分越高（如利率、利差倒挂）
    """
    if pd.isna(value):
        return 50.0
    if low == high:
        return 50.0
    ratio = (value - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    if reverse:
        ratio = 1.0 - ratio
    return round(ratio * 100, 1)


# ── 宏观因子分析器 ───────────────────────────────────


@dataclass
class MacroFactorAnalyzer:
    """宏观因子分析器。

    从 AKShareFetcher（或直接传入 DataFrame）获取宏观数据，
    计算多维度指标体系，生成信号和综合评分。

    使用方式：
        fetcher = AKShareFetcher()
        analyzer = MacroFactorAnalyzer(fetcher)
        result = analyzer.compute_all()
        score = analyzer.score()
    """

    fetcher: Any = None  # DataFetcher 实例（可选，用于自动拉取数据）

    # ═══════════════════════════════════════════════════
    #  货币政策与流动性
    # ═══════════════════════════════════════════════════

    def monetary_liquidity(
        self,
        bond_yield_df: Optional[pd.DataFrame] = None,
        money_supply_df: Optional[pd.DataFrame] = None,
        lpr_df: Optional[pd.DataFrame] = None,
        social_financing_df: Optional[pd.DataFrame] = None,
        shibor_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """计算货币政策与流动性指标。

        Parameters
        ----------
        bond_yield_df : 国债收益率数据。需含 "date" 列及收益率列。
        money_supply_df : 货币供应量数据。
        lpr_df : LPR 数据。
        social_financing_df : 社融数据。
        shibor_df : SHIBOR 数据。

        Returns
        -------
        dict
            keys: bond_10y, bond_1y, term_spread, term_spread_trend,
                  m2_yoy, m1_yoy, m2_m1_spread,
                  lpr_1y, lpr_5y, lpr_1y_change, lpr_5y_change,
                  social_financing_yoy, shibor_on,
                  monetary_signal, liquidity_signal
        """
        result: dict[str, Any] = {}

        # -- 国债收益率 --
        if bond_yield_df is not None and len(bond_yield_df) > 0:
            df = self._normalize_date(bond_yield_df.copy())
            # 查找 10Y / 1Y 列
            bond_10y_col = self._find_col(df, ["10年期", "10年", "y10", "bond_10y", "中国国债收益率10年"])
            bond_1y_col = self._find_col(df, ["1年期", "1年", "y1", "bond_1y", "中国国债收益率1年"])

            if bond_10y_col:
                bond_10y = df.set_index("date")[bond_10y_col].astype(float)
                result["bond_10y"] = _latest(bond_10y)
                result["bond_10y_ma20"] = _latest(_ma(bond_10y, 20))
                result["bond_10y_ma60"] = _latest(_ma(bond_10y, 60))
                result["bond_10y_trend"] = _trend(bond_10y)
            if bond_1y_col:
                bond_1y = df.set_index("date")[bond_1y_col].astype(float)
                result["bond_1y"] = _latest(bond_1y)

            if bond_10y_col and bond_1y_col:
                spread = bond_10y - bond_1y
                result["term_spread"] = _latest(spread)
                result["term_spread_trend"] = _trend(spread)

        # -- M2 / M1 --
        if money_supply_df is not None and len(money_supply_df) > 0:
            df = self._normalize_date(money_supply_df.copy())
            m2_col = self._find_col(df, ["M2", "m2", "M2同比", "货币和准货币(M2)"])
            m1_col = self._find_col(df, ["M1", "m1", "M1同比", "货币(M1)"])

            if m2_col:
                m2 = df.set_index("date")[m2_col].astype(float)
                result["m2_yoy"] = _latest(m2)
                result["m2_trend"] = _trend(m2)
            if m1_col:
                m1 = df.set_index("date")[m1_col].astype(float)
                result["m1_yoy"] = _latest(m1)

            if m2_col and m1_col:
                m2_series = df.set_index("date")[m2_col].astype(float)
                m1_series = df.set_index("date")[m1_col].astype(float)
                spread = m2_series - m1_series
                result["m2_m1_spread"] = _latest(spread)
                result["m2_m1_spread_trend"] = _trend(spread)

            # Monetary signal
            result["monetary_signal"] = self._monetary_signal(result)

        # -- LPR --
        if lpr_df is not None and len(lpr_df) > 0:
            df = self._normalize_date(lpr_df.copy())
            lpr_1y_col = self._find_col(df, ["1年期", "LPR1Y", "lpr_1y", "1年"])
            lpr_5y_col = self._find_col(df, ["5年期", "LPR5Y", "lpr_5y", "5年"])

            if lpr_1y_col:
                lpr_1y = df.set_index("date")[lpr_1y_col].astype(float)
                result["lpr_1y"] = _latest(lpr_1y)
                if len(lpr_1y) >= 2:
                    result["lpr_1y_change"] = round(
                        float(lpr_1y.iloc[-1]) - float(lpr_1y.iloc[-2]), 4
                    )
            if lpr_5y_col:
                lpr_5y = df.set_index("date")[lpr_5y_col].astype(float)
                result["lpr_5y"] = _latest(lpr_5y)
                if len(lpr_5y) >= 2:
                    result["lpr_5y_change"] = round(
                        float(lpr_5y.iloc[-1]) - float(lpr_5y.iloc[-2]), 4
                    )

        # -- 社融 --
        if social_financing_df is not None and len(social_financing_df) > 0:
            df = self._normalize_date(social_financing_df.copy())
            sf_col = self._find_col(df, ["社会融资规模增量", "社融", "social_financing"])
            if sf_col:
                sf = df.set_index("date")[sf_col].astype(float)
                result["social_financing"] = _latest(sf)
                result["social_financing_ma3"] = _latest(_ma(sf, 3))

        # -- SHIBOR / DR007 --
        if shibor_df is not None and len(shibor_df) > 0:
            df = self._normalize_date(shibor_df.copy())
            shibor_col = self._find_col(df, ["O/N", "隔夜", "ON", "shibor_on"])
            dr_col = self._find_col(df, ["DR007", "dr007"])
            if shibor_col:
                result["shibor_on"] = _latest(df.set_index("date")[shibor_col].astype(float))
            if dr_col:
                result["dr007"] = _latest(df.set_index("date")[dr_col].astype(float))

        # Liquidity signal
        result["liquidity_signal"] = self._liquidity_signal(result)

        return result

    @staticmethod
    def _monetary_signal(indicators: dict) -> str:
        """判断货币政策方向。"""
        lpr_change = indicators.get("lpr_1y_change", 0) or 0
        m2_trend = indicators.get("m2_trend", "flat")
        # 简单规则：LPR下行 or M2上升 → easing
        if lpr_change < 0 or m2_trend == "up":
            return "easing"
        elif lpr_change > 0 and m2_trend == "down":
            return "tightening"
        return "neutral"

    @staticmethod
    def _liquidity_signal(indicators: dict) -> str:
        """判断流动性松紧。"""
        shibor = indicators.get("shibor_on", 2.0)
        dr007_val = indicators.get("dr007", 2.0)
        if pd.isna(shibor):
            shibor = 2.0
        if pd.isna(dr007_val):
            dr007_val = 2.0
        avg = (shibor + dr007_val) / 2
        if avg < 1.5:
            return "loose"
        elif avg > 2.5:
            return "tight"
        return "neutral"

    # ═══════════════════════════════════════════════════
    #  经济增长
    # ═══════════════════════════════════════════════════

    def growth(
        self,
        pmi_df: Optional[pd.DataFrame] = None,
        industrial_df: Optional[pd.DataFrame] = None,
        trade_df: Optional[pd.DataFrame] = None,
        gdp_df: Optional[pd.DataFrame] = None,
        retail_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """计算经济增长维度指标。

        Returns
        -------
        dict
            keys: pmi_mfg, pmi_non_mfg, pmi_mfg_trend,
                  industrial_value_added_yoy, fixed_asset_inv_yoy,
                  retail_sales_yoy, export_yoy, import_yoy,
                  gdp_yoy, growth_signal
        """
        result: dict[str, Any] = {}

        # -- PMI --
        if pmi_df is not None and len(pmi_df) > 0:
            df = self._normalize_date(pmi_df.copy())
            mfg_col = self._find_col(df, ["制造业", "pmi_manufacturing", "PMI_manufacturing"])
            non_mfg_col = self._find_col(df, ["非制造业", "pmi_non_manufacturing", "PMI_non_manufacturing"])
            if mfg_col:
                pmi = df.set_index("date")[mfg_col].astype(float)
                result["pmi_mfg"] = _latest(pmi)
                result["pmi_mfg_trend"] = _trend(pmi)
            if non_mfg_col:
                result["pmi_non_mfg"] = _latest(df.set_index("date")[non_mfg_col].astype(float))

        # -- 工业增加值 --
        if industrial_df is not None and len(industrial_df) > 0:
            df = self._normalize_date(industrial_df.copy())
            col = self._find_col(df, ["工业增加值", "industrial_value_added", "同比增长"])
            if col:
                iva = df.set_index("date")[col].astype(float)
                result["industrial_value_added_yoy"] = _latest(iva)
                result["industrial_value_added_ma3"] = _latest(_ma(iva, 3))

        # -- 固投 --
        if industrial_df is not None and len(industrial_df) > 0:
            df = self._normalize_date(industrial_df.copy())
            col = self._find_col(df, ["固定资产投资", "fixed_asset_inv", "固定资产投资完成额"])
            if col:
                result["fixed_asset_inv_yoy"] = _latest(df.set_index("date")[col].astype(float))

        # -- 社零 --
        if retail_df is not None and len(retail_df) > 0:
            df = self._normalize_date(retail_df.copy())
            col = self._find_col(df, ["社会消费品零售总额", "retail_sales", "同比增长"])
            if col:
                result["retail_sales_yoy"] = _latest(df.set_index("date")[col].astype(float))

        # -- 进出口 --
        if trade_df is not None and len(trade_df) > 0:
            df = self._normalize_date(trade_df.copy())
            export_col = self._find_col(df, ["出口", "export", "出口同比", "出口金额"])
            import_col = self._find_col(df, ["进口", "import", "进口同比", "进口金额"])
            if export_col:
                result["export_yoy"] = _latest(df.set_index("date")[export_col].astype(float))
            if import_col:
                result["import_yoy"] = _latest(df.set_index("date")[import_col].astype(float))

        # -- GDP --
        if gdp_df is not None and len(gdp_df) > 0:
            df = self._normalize_date(gdp_df.copy())
            col = self._find_col(df, ["GDP", "gdp", "国内生产总值", "同比增长"])
            if col:
                result["gdp_yoy"] = _latest(df.set_index("date")[col].astype(float))

        # Growth signal
        result["growth_signal"] = self._growth_signal(result)
        return result

    @staticmethod
    def _growth_signal(indicators: dict) -> str:
        """判断经济增速信号。"""
        pmi = indicators.get("pmi_mfg", 50)
        gdp = indicators.get("gdp_yoy", 5)
        if pd.isna(pmi):
            pmi = 50
        if pd.isna(gdp):
            gdp = 5
        if pmi > 51 and gdp >= 5.5:
            return "expansion"
        elif pmi < 49 or gdp < 4.5:
            return "contraction"
        return "stable"

    # ═══════════════════════════════════════════════════
    #  通胀
    # ═══════════════════════════════════════════════════

    def inflation(
        self,
        cpi_df: Optional[pd.DataFrame] = None,
        ppi_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """计算通胀维度指标。

        Returns
        -------
        dict
            keys: cpi_yoy, ppi_yoy, ppi_cpi_spread, inflation_signal
        """
        result: dict[str, Any] = {}

        if cpi_df is not None and len(cpi_df) > 0:
            df = self._normalize_date(cpi_df.copy())
            col = self._find_col(df, ["CPI", "cpi", "全国", "居民消费价格指数", "同比增长", "当月同比"])
            if col:
                cpi = df.set_index("date")[col].astype(float)
                result["cpi_yoy"] = _latest(cpi)
                result["cpi_trend"] = _trend(cpi)

        if ppi_df is not None and len(ppi_df) > 0:
            df = self._normalize_date(ppi_df.copy())
            col = self._find_col(df, ["PPI", "ppi", "全部工业品", "工业生产者出厂价格指数", "当月同比"])
            if col:
                ppi = df.set_index("date")[col].astype(float)
                result["ppi_yoy"] = _latest(ppi)
                result["ppi_trend"] = _trend(ppi)

        if "cpi_yoy" in result and "ppi_yoy" in result:
            result["ppi_cpi_spread"] = round(
                result["ppi_yoy"] - result["cpi_yoy"], 2
            )

        result["inflation_signal"] = self._inflation_signal(result)
        return result

    @staticmethod
    def _inflation_signal(indicators: dict) -> str:
        """判断通胀信号。"""
        cpi = indicators.get("cpi_yoy", 1.5)
        ppi = indicators.get("ppi_yoy", 0)
        if pd.isna(cpi):
            cpi = 1.5
        if pd.isna(ppi):
            ppi = 0
        if cpi > 3 or ppi < -3:
            return "stagflation_risk" if cpi > 3 and ppi < 0 else "overheat"
        elif cpi < 0 and ppi < -1:
            return "deflation"
        elif 1 <= cpi <= 3 and ppi >= 0:
            return "moderate"
        return "neutral"

    # ═══════════════════════════════════════════════════
    #  外部环境
    # ═══════════════════════════════════════════════════

    def external(
        self,
        usdcny_df: Optional[pd.DataFrame] = None,
        fed_rate_df: Optional[pd.DataFrame] = None,
        north_flow_df: Optional[pd.DataFrame] = None,
        bond_yield_df: Optional[pd.DataFrame] = None,
        us_bond_yield_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """计算外部环境指标。

        中美利差 = 中国10Y国债 - 美国10Y国债。

        Returns
        -------
        dict
            keys: usdcny, usdcny_trend, fed_rate,
                  north_flow_5d, north_flow_20d,
                  cn_us_spread, external_signal
        """
        result: dict[str, Any] = {}

        # -- 汇率 --
        if usdcny_df is not None and len(usdcny_df) > 0:
            df = self._normalize_date(usdcny_df.copy())
            col = self._find_col(df, ["USDCNY", "usdcny", "美元人民币", "中间价", "收盘价", "close"])
            if col:
                fx = df.set_index("date")[col].astype(float)
                result["usdcny"] = _latest(fx)
                result["usdcny_ma20"] = _latest(_ma(fx, 20))
                result["usdcny_trend"] = _trend(fx)

        # -- 美联储利率 --
        if fed_rate_df is not None and len(fed_rate_df) > 0:
            df = self._normalize_date(fed_rate_df.copy())
            col = self._find_col(df, ["fed_rate", "联邦基金利率", "利率", "rate"])
            if col:
                result["fed_rate"] = _latest(df.set_index("date")[col].astype(float))

        # -- 北向资金 --
        if north_flow_df is not None and len(north_flow_df) > 0:
            df = self._normalize_date(north_flow_df.copy())
            col = self._find_col(df, ["北向资金", "净流入", "net_flow", "当日成交净买额"])
            if col:
                nf = df.set_index("date")[col].astype(float)
                result["north_flow_5d"] = _latest(nf.rolling(5, min_periods=1).sum())
                result["north_flow_20d"] = _latest(nf.rolling(20, min_periods=1).sum())

        # -- 中美利差 --
        cn_10y = None
        us_10y = None
        if bond_yield_df is not None and len(bond_yield_df) > 0:
            cn_10y_col = self._find_col(
                bond_yield_df, ["10年期", "10年", "y10", "中国国债收益率10年"]
            )
            if cn_10y_col:
                cn_yield = self._normalize_date(bond_yield_df.copy())
                cn_10y = _latest(cn_yield.set_index("date")[cn_10y_col].astype(float))
        if us_bond_yield_df is not None and len(us_bond_yield_df) > 0:
            us_10y_col = self._find_col(
                us_bond_yield_df, ["10年期", "10年", "y10", "美国国债收益率10年", "close"]
            )
            if us_10y_col:
                us_yield = self._normalize_date(us_bond_yield_df.copy())
                us_10y = _latest(us_yield.set_index("date")[us_10y_col].astype(float))
        if cn_10y is not None and us_10y is not None:
            result["cn_us_spread"] = round(cn_10y - us_10y, 4)

        result["external_signal"] = self._external_signal(result)
        return result

    @staticmethod
    def _external_signal(indicators: dict) -> str:
        """判断外部环境信号。"""
        fx_trend = indicators.get("usdcny_trend", "flat")
        spread = indicators.get("cn_us_spread", 0)
        if pd.isna(spread):
            spread = 0
        north = indicators.get("north_flow_20d", 0)
        if pd.isna(north):
            north = 0

        risks = 0
        if fx_trend == "up":  # 人民币贬值
            risks += 1
        if spread < -1.5:  # 中美利差显著倒挂
            risks += 1
        if north < -500:  # 北向大幅流出
            risks += 1

        if risks >= 2:
            return "high_pressure"
        elif risks == 1:
            return "caution"
        return "stable"

    # ═══════════════════════════════════════════════════
    #  综合输出
    # ═══════════════════════════════════════════════════

    def compute_all(self, **kwargs: pd.DataFrame) -> dict:
        """计算所有宏观因子指标并返回结构化结果。

        可通过 kwargs 直接传入各数据源的 DataFrame（用于测试），
        也可依赖 fetcher 自动拉取。

        Returns
        -------
        dict
            { "monetary_liquidity": {...}, "growth": {...},
              "inflation": {...}, "external": {...}, "date": "..." }
        """
        ml = self.monetary_liquidity(
            bond_yield_df=kwargs.get("bond_yield_df"),
            money_supply_df=kwargs.get("money_supply_df"),
            lpr_df=kwargs.get("lpr_df"),
            social_financing_df=kwargs.get("social_financing_df"),
            shibor_df=kwargs.get("shibor_df"),
        )
        g = self.growth(
            pmi_df=kwargs.get("pmi_df"),
            industrial_df=kwargs.get("industrial_df"),
            trade_df=kwargs.get("trade_df"),
            gdp_df=kwargs.get("gdp_df"),
            retail_df=kwargs.get("retail_df"),
        )
        infl = self.inflation(
            cpi_df=kwargs.get("cpi_df"),
            ppi_df=kwargs.get("ppi_df"),
        )
        ext = self.external(
            usdcny_df=kwargs.get("usdcny_df"),
            fed_rate_df=kwargs.get("fed_rate_df"),
            north_flow_df=kwargs.get("north_flow_df"),
            bond_yield_df=kwargs.get("bond_yield_df"),
            us_bond_yield_df=kwargs.get("us_bond_yield_df"),
        )
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "monetary_liquidity": ml,
            "growth": g,
            "inflation": infl,
            "external": ext,
        }

    def score(
        self,
        indicators: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """计算宏观综合评分（0-100）。

        权重：流动性 30% + 增长 30% + 通胀 20% + 外部 20%

        Parameters
        ----------
        indicators : 由 compute_all() 返回的指标字典，若为 None 则自动计算。

        Returns
        -------
        dict
            { "total_score", "liquidity_score", "growth_score",
              "inflation_score", "external_score", "grade" }
        """
        if indicators is None:
            indicators = self.compute_all(**kwargs)

        ml = indicators.get("monetary_liquidity", {})
        g = indicators.get("growth", {})
        infl = indicators.get("inflation", {})
        ext = indicators.get("external", {})

        liquidity_score = self._score_liquidity(ml)
        growth_score = self._score_growth(g)
        inflation_score = self._score_inflation(infl)
        external_score = self._score_external(ext)

        total = round(
            liquidity_score * 0.30
            + growth_score * 0.30
            + inflation_score * 0.20
            + external_score * 0.20,
            1,
        )

        grade = self._to_grade(total)

        return {
            "total_score": total,
            "liquidity_score": liquidity_score,
            "growth_score": growth_score,
            "inflation_score": inflation_score,
            "external_score": external_score,
            "grade": grade,
        }

    # ── 各维度评分逻辑 ──

    @staticmethod
    def _score_liquidity(ml: dict) -> float:
        """流动性评分：综合 SHIBOR/DR007 水位、M2 趋势、LPR 变动。"""
        scores = []
        # SHIBOR: 1.0=宽松, 3.0=紧张
        shibor = ml.get("shibor_on", 2.0)
        if not pd.isna(shibor):
            scores.append(_map_score(shibor, 1.0, 3.0, reverse=True))
        # DR007
        dr = ml.get("dr007", 2.0)
        if not pd.isna(dr):
            scores.append(_map_score(dr, 1.5, 3.0, reverse=True))
        # M2-M1 剪刀差：小=好
        spread = ml.get("m2_m1_spread", 5)
        if not pd.isna(spread):
            scores.append(_map_score(spread, 2, 10, reverse=True))
        # Monetary signal
        sig = ml.get("monetary_signal", "neutral")
        if sig == "easing":
            scores.append(80)
        elif sig == "tightening":
            scores.append(30)
        else:
            scores.append(50)
        return round(float(np.mean(scores)), 1) if scores else 50.0

    @staticmethod
    def _score_growth(g: dict) -> float:
        """增长评分。"""
        scores = []
        pmi = g.get("pmi_mfg", 50)
        if not pd.isna(pmi):
            scores.append(_map_score(pmi, 48, 52))
        gdp = g.get("gdp_yoy", 5)
        if not pd.isna(gdp):
            scores.append(_map_score(gdp, 3, 7))
        iva = g.get("industrial_value_added_yoy", 4)
        if not pd.isna(iva):
            scores.append(_map_score(iva, 0, 10))
        return round(float(np.mean(scores)), 1) if scores else 50.0

    @staticmethod
    def _score_inflation(infl: dict) -> float:
        """通胀评分：温和通胀最优。"""
        scores = []
        cpi = infl.get("cpi_yoy", 1.5)
        if not pd.isna(cpi):
            # 0-3 区间最优
            if 1 <= cpi <= 3:
                scores.append(80)
            elif 0 <= cpi < 1:
                scores.append(60)
            elif cpi < 0:
                scores.append(30)
            else:
                scores.append(40)
        ppi = infl.get("ppi_yoy", 0)
        if not pd.isna(ppi):
            if -1 <= ppi <= 3:
                scores.append(70)
            elif ppi < -3:
                scores.append(30)
            else:
                scores.append(50)
        return round(float(np.mean(scores)), 1) if scores else 50.0

    @staticmethod
    def _score_external(ext: dict) -> float:
        """外部环境评分。"""
        scores = []
        spread = ext.get("cn_us_spread", 0)
        if not pd.isna(spread):
            scores.append(_map_score(spread, -2, 2))
        fx_trend = ext.get("usdcny_trend", "flat")
        if fx_trend == "down":
            scores.append(70)
        elif fx_trend == "up":
            scores.append(30)
        else:
            scores.append(50)
        north = ext.get("north_flow_20d", 0)
        if not pd.isna(north):
            scores.append(_map_score(north, -1000, 1000))
        return round(float(np.mean(scores)), 1) if scores else 50.0

    @staticmethod
    def _to_grade(score: float) -> str:
        """评分转字母等级。"""
        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        elif score >= 20:
            return "D"
        return "E"

    # ── 工具方法 ──

    @staticmethod
    def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
        """统一日期列处理。"""
        for col in ["date", "日期", "报告期", "指标名称", "月份", "时间"]:
            if col in df.columns:
                df = df.rename(columns={col: "date"})
                break
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date")
        return df

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """从候选列名中查找第一个存在的列。"""
        for cand in candidates:
            if cand in df.columns:
                return cand
            # 模糊匹配
            for col in df.columns:
                if cand.lower() == col.lower().strip():
                    return col
                if cand in str(col):
                    return col
        return None
