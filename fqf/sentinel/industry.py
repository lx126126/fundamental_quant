"""行业分析器 (IndustryAnalyzer)

基于申万一级行业分类，计算行业估值水位、动量/情绪指标与综合评分。

核心功能：
- 行业估值水位：PE / PB 历史分位
- 行业动量：20日/60日涨跌幅、相对强度
- 行业情绪：成交额占比及其历史分位
- 行业综合评分模型
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from fqf.sentinel.macro import _map_score, _latest, _ma

logger = logging.getLogger(__name__)


def _percentile(series: pd.Series, value: float) -> float:
    """计算 value 在 series 中的历史分位（0-100）。

    分位越低 → 估值越低（越低估）。
    """
    clean = series.dropna()
    if len(clean) == 0 or pd.isna(value):
        return 50.0
    return round((clean < value).sum() / len(clean) * 100, 1)


def _return_pct(series: pd.Series, window: int) -> float:
    """计算最近 window 期的涨跌幅（%）。"""
    if len(series) < window:
        return 0.0
    start_val = float(series.iloc[-window])
    end_val = float(series.iloc[-1])
    if start_val == 0:
        return 0.0
    return round((end_val / start_val - 1) * 100, 2)


@dataclass
class IndustryAnalyzer:
    """行业分析器。

    使用方式：
        analyzer = IndustryAnalyzer()
        scores = analyzer.compute_all_scores(industry_data_dict)
    """

    fetcher: Any = None

    # ── 申万一级行业列表 ──

    SW_INDUSTRIES: tuple = (
        "农林牧渔", "基础化工", "钢铁", "有色金属", "电子",
        "汽车", "家用电器", "食品饮料", "纺织服饰", "轻工制造",
        "医药生物", "公用事业", "交通运输", "房地产", "商贸零售",
        "社会服务", "银行", "非银金融", "综合", "建筑材料",
        "建筑装饰", "电力设备", "国防军工", "计算机", "传媒",
        "通信", "煤炭", "石油石化", "环保", "美容护理",
        "机械设备",
    )

    # ═══════════════════════════════════════════════════
    #  行业估值水位
    # ═══════════════════════════════════════════════════

    def valuation(
        self,
        industry: str,
        index_df: Optional[pd.DataFrame] = None,
        pe_series: Optional[pd.Series] = None,
        pb_series: Optional[pd.Series] = None,
    ) -> dict:
        """计算单个行业的估值水位。

        Parameters
        ----------
        industry : 行业名称
        index_df : 行业指数日线数据（含 close 列）。用于 PE/PB 分位替代计算。
        pe_series : 行业历史 PE 序列（如果数据源提供）。
        pb_series : 行业历史 PB 序列（如果数据源提供）。

        Returns
        -------
        dict with keys: industry, pe_current, pe_pct_3y, pe_pct_5y,
                        pb_current, pb_pct_3y, pb_pct_5y, valuation_signal
        """
        result: dict[str, Any] = {"industry": industry}

        # PE 分位
        if pe_series is not None and len(pe_series.dropna()) > 0:
            pe = pe_series.dropna()
            result["pe_current"] = _latest(pe)
            result["pe_pct_3y"] = self._trailing_percentile(pe, 3 * 250)
            result["pe_pct_5y"] = self._trailing_percentile(pe, 5 * 250)
        elif index_df is not None and len(index_df) > 0:
            # 用指数点位替代（价格越低 → 估值越低）
            df = self._prepare_index(index_df)
            close = df["close"].dropna()
            result["pe_pct_3y"] = self._trailing_percentile(close, 3 * 250)
            result["pe_pct_5y"] = self._trailing_percentile(close, 5 * 250)

        # PB 分位
        if pb_series is not None and len(pb_series.dropna()) > 0:
            pb = pb_series.dropna()
            result["pb_current"] = _latest(pb)
            result["pb_pct_3y"] = self._trailing_percentile(pb, 3 * 250)
            result["pb_pct_5y"] = self._trailing_percentile(pb, 5 * 250)

        # Valuation signal
        pe_pct = result.get("pe_pct_3y", 50)
        if pe_pct < 20:
            result["valuation_signal"] = "undervalued"
        elif pe_pct < 40:
            result["valuation_signal"] = "fair_low"
        elif pe_pct < 60:
            result["valuation_signal"] = "fair"
        elif pe_pct < 80:
            result["valuation_signal"] = "fair_high"
        else:
            result["valuation_signal"] = "overvalued"

        return result

    # ═══════════════════════════════════════════════════
    #  行业动量
    # ═══════════════════════════════════════════════════

    def momentum(
        self,
        industry: str,
        index_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """计算行业动量指标。

        Parameters
        ----------
        industry : 行业名称
        index_df : 行业指数日线数据（含 close 列）
        benchmark_df : 基准指数日线数据（如沪深300），用于计算相对强度。

        Returns
        -------
        dict with keys: industry, ret_20d, ret_60d, relative_strength_20d,
                        relative_strength_60d, momentum_signal
        """
        df = self._prepare_index(index_df)
        close = df["close"]
        result: dict[str, Any] = {"industry": industry}
        result["ret_20d"] = _return_pct(close, 20)
        result["ret_60d"] = _return_pct(close, 60)

        if benchmark_df is not None and len(benchmark_df) > 0:
            bm = self._prepare_index(benchmark_df)
            bm_close = bm["close"]
            result["relative_strength_20d"] = round(
                result["ret_20d"] - _return_pct(bm_close, 20), 2
            )
            result["relative_strength_60d"] = round(
                result["ret_60d"] - _return_pct(bm_close, 60), 2
            )

        # Momentum signal
        ret = result.get("ret_20d", 0)
        if ret > 10:
            result["momentum_signal"] = "strong"
        elif ret > 3:
            result["momentum_signal"] = "positive"
        elif ret > -3:
            result["momentum_signal"] = "neutral"
        elif ret > -10:
            result["momentum_signal"] = "negative"
        else:
            result["momentum_signal"] = "weak"

        return result

    # ═══════════════════════════════════════════════════
    #  行业情绪
    # ═══════════════════════════════════════════════════

    def sentiment(
        self,
        industry: str,
        index_df: pd.DataFrame,
        market_volume_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """计算行业情绪指标（成交额占比）。

        Parameters
        ----------
        industry : 行业名称
        index_df : 行业指数日线数据（含 amount 列，即成交额）
        market_volume_df : 全市场日成交额数据（含 amount 列）。

        Returns
        -------
        dict with keys: industry, volume_share, volume_share_pct,
                        sentiment_signal
        """
        df = self._prepare_index(index_df)
        result: dict[str, Any] = {"industry": industry}

        # 行业近期成交额
        if "amount" in df.columns:
            recent_vol = _latest(_ma(df["amount"], 5))
        elif "volume" in df.columns:
            recent_vol = _latest(_ma(df["volume"], 5))
        else:
            return result

        result["avg_amount_5d"] = recent_vol

        # 成交额占比
        if market_volume_df is not None and len(market_volume_df) > 0:
            bm = self._prepare_index(market_volume_df)
            if "amount" in bm.columns:
                market_vol = _latest(_ma(bm["amount"], 5))
                if market_vol and market_vol > 0:
                    result["volume_share"] = round(recent_vol / market_vol * 100, 2)

        # 成交额历史分位
        if "amount" in df.columns:
            result["volume_pct"] = self._trailing_percentile(
                _ma(df["amount"], 5), 500
            )

        # Sentiment signal
        vol_pct = result.get("volume_pct", 50)
        if vol_pct > 80:
            result["sentiment_signal"] = "overheated"
        elif vol_pct > 60:
            result["sentiment_signal"] = "active"
        elif vol_pct < 20:
            result["sentiment_signal"] = "cold"
        else:
            result["sentiment_signal"] = "neutral"

        return result

    # ═══════════════════════════════════════════════════
    #  综合评分
    # ═══════════════════════════════════════════════════

    def score_industry(
        self,
        industry: str,
        val_result: dict,
        mom_result: dict,
        sent_result: dict,
    ) -> dict:
        """单个行业综合评分。

        权重：估值 30% + 基本面 35% + 动量 20% + 情绪 15%

        当前版本中，基本面得分使用动量替代（后续版本接入真实财务数据）。
        """
        val_score = self._val_score(val_result)
        # 基本面得分为 HEURISTIC：估值低+动量好的行业基本面也不差
        mom_score = self._mom_score(mom_result)
        fund_score = round((val_score * 0.4 + mom_score * 0.6), 1)
        sent_score = self._sent_score(sent_result)

        total = round(
            val_score * 0.30 + fund_score * 0.35 + mom_score * 0.20 + sent_score * 0.15, 1
        )

        return {
            "industry": industry,
            "total_score": total,
            "valuation_score": val_score,
            "fundamental_score": fund_score,
            "momentum_score": mom_score,
            "sentiment_score": sent_score,
            "valuation_signal": val_result.get("valuation_signal", "fair"),
            "momentum_signal": mom_result.get("momentum_signal", "neutral"),
            "sentiment_signal": sent_result.get("sentiment_signal", "neutral"),
        }

    def score_all(
        self,
        results: list[dict],
    ) -> pd.DataFrame:
        """对所有行业综合评分并排名。

        Parameters
        ----------
        results : list[dict]，每个元素包含 valuation/momentum/sentiment 结果。

        Returns
        -------
        pd.DataFrame 按综合得分降序排列。
        """
        scores = []
        for r in results:
            s = self.score_industry(
                r["industry"], r["valuation"], r["momentum"], r["sentiment"]
            )
            scores.append(s)
        df = pd.DataFrame(scores)
        if not df.empty:
            df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
        return df

    # ── 子评分逻辑 ──

    @staticmethod
    def _val_score(val: dict) -> float:
        """估值评分：分位越低分数越高。"""
        pp = val.get("pe_pct_3y", 50)
        if pd.isna(pp):
            pp = 50
        return _map_score(pp, 0, 100, reverse=True)

    @staticmethod
    def _mom_score(mom: dict) -> float:
        """动量评分：涨幅越大分数越高。"""
        r20 = mom.get("ret_20d", 0)
        if pd.isna(r20):
            r20 = 0
        return _map_score(r20, -15, 15)

    @staticmethod
    def _sent_score(sent: dict) -> float:
        """情绪评分：适中最好。"""
        pct = sent.get("volume_pct", 50)
        if pd.isna(pct):
            pct = 50
        # 中间最优，两端低
        if 40 <= pct <= 60:
            return 80
        elif 20 <= pct < 40 or 60 < pct <= 80:
            return 60
        elif pct < 20:
            return 40
        else:
            return 30

    # ── 全行业快照 ──

    def snapshot(
        self,
        industry_indices: dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        market_volume_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """生成全行业快照表。

        Parameters
        ----------
        industry_indices : {行业名: 指数DataFrame} 的字典
        benchmark_df : 基准指数日线
        market_volume_df : 全市场成交额

        Returns
        -------
        pd.DataFrame 全行业评分排名表
        """
        results = []
        for ind_name, ind_df in industry_indices.items():
            if ind_df is None or len(ind_df) == 0:
                continue
            val = self.valuation(ind_name, index_df=ind_df)
            mom = self.momentum(ind_name, ind_df, benchmark_df)
            sent = self.sentiment(ind_name, ind_df, market_volume_df)
            results.append(
                {"industry": ind_name, "valuation": val, "momentum": mom, "sentiment": sent}
            )
        return self.score_all(results)

    # ── 工具方法 ──

    @staticmethod
    def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
        """标准化指数数据：统一列名 + 日期处理。"""
        df = df.copy()
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "涨跌幅": "pct_change",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        return df

    @staticmethod
    def _trailing_percentile(series: pd.Series, window: int = 500) -> float:
        """最近 window 期的分位值。"""
        s = series.dropna()
        if len(s) < 2:
            return 50.0
        recent = s.tail(min(window, len(s)))
        current = float(s.iloc[-1])
        return round((recent < current).sum() / len(recent) * 100, 1)
