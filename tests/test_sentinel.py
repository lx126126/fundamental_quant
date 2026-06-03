"""Sentinel 模块单元测试

使用 mock 数据，不依赖真实网络请求。
覆盖：
- MacroFactorAnalyzer: 四大维度的指标计算、信号生成、综合评分
- IndustryAnalyzer: 估值水位、动量、情绪、综合评分
- BriefingGenerator: 简报各章节结构完整性
- 辅助函数: _map_score, _trend, _safe_div, _yoy, _percentile 等
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from fqf.sentinel.macro import (
    MacroFactorAnalyzer,
    _map_score,
    _trend,
    _safe_div,
    _yoy,
    _ma,
    _latest,
)
from fqf.sentinel.industry import IndustryAnalyzer, _percentile, _return_pct
from fqf.sentinel.briefing import BriefingGenerator


# ═══════════════════════════════════════════════════════════
#  Fixtures：模拟宏观数据
# ═══════════════════════════════════════════════════════════


def _make_dates(start_year: int, n: int, freq: str = "YS") -> list:
    """生成日期序列。"""
    start = datetime(start_year, 1, 1)
    if freq == "YS":
        return [start + timedelta(days=365 * i) for i in range(n)]
    elif freq == "MS":
        return [start + timedelta(days=30 * i) for i in range(n)]
    else:
        # daily
        return [start + timedelta(days=i) for i in range(n)]


@pytest.fixture
def macro_analyzer():
    return MacroFactorAnalyzer()


@pytest.fixture
def mock_bond_yield():
    """模拟国债收益率（200个交易日）。"""
    dates = _make_dates(2020, 200, "D")
    np.random.seed(42)
    trend = np.linspace(3.2, 2.5, 200)
    noise = np.random.normal(0, 0.05, 200)
    return pd.DataFrame({
        "date": dates,
        "中国国债收益率10年": trend + noise,
        "中国国债收益率1年": trend - 0.6 + noise * 0.5,
    })


@pytest.fixture
def mock_money_supply():
    """模拟货币供应量（24个月）。"""
    dates = _make_dates(2022, 24, "MS")
    return pd.DataFrame({
        "date": dates,
        "货币和准货币(M2)": np.linspace(10, 8, 24) + np.random.uniform(-0.3, 0.3, 24),
        "货币(M1)": np.linspace(5, 3, 24) + np.random.uniform(-0.3, 0.3, 24),
    })


@pytest.fixture
def mock_lpr():
    """模拟 LPR（12个月）。"""
    dates = _make_dates(2023, 12, "MS")
    # LPR 1Y 从 3.65 降到 3.45
    return pd.DataFrame({
        "date": dates,
    "1年期": [3.65]*3 + [3.55]*2 + [3.50]*1 + [3.45]*5 + [3.40],
    "5年期": [4.30]*3 + [4.20]*2 + [4.00]*1 + [3.95]*5 + [3.85],
    })


@pytest.fixture
def mock_pmi():
    """模拟 PMI（24个月）。"""
    dates = _make_dates(2022, 24, "MS")
    return pd.DataFrame({
        "date": dates,
        "制造业": [49.0, 49.5, 50.1, 50.2, 49.8, 50.5,
                   50.8, 51.0, 50.3, 49.5, 49.2, 50.1,
                   50.5, 51.2, 51.5, 50.8, 50.0, 49.5,
                   49.8, 50.3, 50.8, 51.0, 51.2, 50.5],
        "非制造业": [52.0, 52.5, 53.0, 53.5, 52.8, 53.2,
                     54.0, 54.5, 53.8, 53.0, 52.5, 53.0,
                     53.5, 54.0, 54.5, 53.8, 53.0, 52.5,
                     52.8, 53.3, 53.8, 54.2, 54.5, 53.8],
    })


@pytest.fixture
def mock_cpi():
    """模拟 CPI（24个月）。"""
    dates = _make_dates(2022, 24, "MS")
    return pd.DataFrame({
        "date": dates,
        "当月同比": [0.9, 0.8, 0.7, 0.5, 0.3, 0.0,
                     -0.2, -0.3, 0.0, 0.2, 0.5, 0.8,
                     1.0, 1.2, 1.5, 1.8, 2.0, 2.2,
                     2.0, 1.8, 1.5, 1.3, 1.0, 0.8],
    })


@pytest.fixture
def mock_ppi():
    """模拟 PPI（24个月）。"""
    dates = _make_dates(2022, 24, "MS")
    return pd.DataFrame({
        "date": dates,
        "当月同比": [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
                     -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
                     0.5, 1.0, 1.5, 1.0, 0.5, 0.0,
                     -0.5, -1.0, -0.5, 0.0, 0.5, 0.3],
    })


@pytest.fixture
def mock_gdp():
    """模拟 GDP（8个季度）。"""
    dates = _make_dates(2022, 8, "MS")
    return pd.DataFrame({
        "date": dates,
        "国内生产总值": [4.0, 3.8, 3.5, 3.0, 3.2, 3.8, 4.5, 5.0],
    })


@pytest.fixture
def mock_usdcny():
    """模拟汇率（200个交易日）。"""
    dates = _make_dates(2023, 200, "D")
    np.random.seed(24)
    base = np.linspace(6.90, 7.20, 200)
    noise = np.random.normal(0, 0.02, 200)
    return pd.DataFrame({
        "date": dates,
        "close": base + noise,
    })


# ═══════════════════════════════════════════════════════════
#  Fixtures：模拟行业数据
# ═══════════════════════════════════════════════════════════


def _make_index_data(name: str, days: int, start_price: float,
                     trend: float = 0.0, volatility: float = 0.02) -> pd.DataFrame:
    """生成模拟行业指数日线数据。"""
    np.random.seed(hash(name) % 2**32)
    dates = _make_dates(2020, days, "D")
    returns = np.random.normal(trend / 252, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))
    volume = np.random.lognormal(15, 0.5, days)
    amount = prices * volume
    return pd.DataFrame({
        "日期": dates,
        "开盘": prices * (1 - np.random.uniform(0, 0.01, days)),
        "收盘": prices,
        "最高": prices * (1 + np.random.uniform(0, 0.02, days)),
        "最低": prices * (1 - np.random.uniform(0, 0.02, days)),
        "成交量": volume,
        "成交额": amount,
        "涨跌幅": np.concatenate([[0], np.diff(prices) / prices[:-1] * 100]),
    })


@pytest.fixture
def industry_analyzer():
    return IndustryAnalyzer()


@pytest.fixture
def mock_food_beverage():
    """模拟食品饮料行业指数（偏强行情）。"""
    return _make_index_data("食品饮料", 600, 5000, trend=0.10, volatility=0.018)


@pytest.fixture
def mock_bank():
    """模拟银行行业指数（稳健行情）。"""
    return _make_index_data("银行", 600, 3000, trend=0.05, volatility=0.012)


@pytest.fixture
def mock_real_estate():
    """模拟房地产行业指数（偏弱行情）。"""
    return _make_index_data("房地产", 600, 2000, trend=-0.08, volatility=0.025)


@pytest.fixture
def mock_hs300():
    """模拟沪深300基准指数。"""
    return _make_index_data("沪深300", 600, 4000, trend=0.06, volatility=0.015)


@pytest.fixture
def mock_industry_dict(mock_food_beverage, mock_bank):
    """多行业模拟数据字典。"""
    return {"食品饮料": mock_food_beverage, "银行": mock_bank}


@pytest.fixture
def mock_market_volume():
    """模拟全市场成交额。"""
    dates = _make_dates(2020, 600, "D")
    return pd.DataFrame({
        "日期": dates,
        "成交额": np.random.lognormal(20, 0.3, 600) * 1e8,
    })


# ═══════════════════════════════════════════════════════════
#  辅助函数测试
# ═══════════════════════════════════════════════════════════


class TestHelperFunctions:
    """测试 sentinel 辅助函数。"""

    def test_safe_div_normal(self):
        assert _safe_div(10, 2) == 5.0

    def test_safe_div_zero(self):
        assert np.isnan(_safe_div(10, 0))

    def test_safe_div_nan(self):
        assert np.isnan(_safe_div(np.nan, 2))
        assert np.isnan(_safe_div(10, np.nan))

    def test_map_score_normal(self):
        assert _map_score(50, 0, 100) == 50.0
        assert _map_score(0, 0, 100) == 0.0
        assert _map_score(100, 0, 100) == 100.0

    def test_map_score_reverse(self):
        assert _map_score(0, 0, 100, reverse=True) == 100.0
        assert _map_score(100, 0, 100, reverse=True) == 0.0

    def test_map_score_nan(self):
        assert _map_score(np.nan, 0, 100) == 50.0

    def test_map_score_clip(self):
        assert _map_score(-10, 0, 100) == 0.0
        assert _map_score(200, 0, 100) == 100.0

    def test_trend_up(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        assert _trend(s) == "up"

    def test_trend_down(self):
        s = pd.Series([5, 4, 3, 2, 1], dtype=float)
        assert _trend(s) == "down"

    def test_trend_flat(self):
        s = pd.Series([2.0, 2.01, 1.99, 2.0, 2.0], dtype=float)
        assert _trend(s) == "flat"

    def test_trend_short(self):
        s = pd.Series([5.0], dtype=float)
        assert _trend(s) == "flat"

    def test_yoy(self):
        s = pd.Series([100, 110, 121], dtype=float)
        result = _yoy(s)
        assert round(result.iloc[2], 1) == 10.0

    def test_ma(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = _ma(s, 3)
        assert result.iloc[4] == 4.0  # (3+4+5)/3

    def test_latest(self):
        s = pd.Series([1.0, 2.0, np.nan, 3.0])
        assert _latest(s) == 3.0

    def test_percentile_high(self):
        s = pd.Series(range(100), dtype=float)
        assert _percentile(s, 90) == 90.0

    def test_percentile_low(self):
        s = pd.Series(range(100), dtype=float)
        assert _percentile(s, 10) == 10.0

    def test_return_pct(self):
        s = pd.Series([100, 105, 110, 115, 120], dtype=float)
        assert _return_pct(s, 5) == 20.0

    def test_return_pct_insufficient(self):
        s = pd.Series([100, 110], dtype=float)
        assert _return_pct(s, 20) == 0.0


# ═══════════════════════════════════════════════════════════
#  MacroFactorAnalyzer 测试
# ═══════════════════════════════════════════════════════════


class TestMacroMonetary:
    """货币政策与流动性维度。"""

    def test_bond_yield_indicators(self, macro_analyzer, mock_bond_yield):
        result = macro_analyzer.monetary_liquidity(bond_yield_df=mock_bond_yield)
        assert "bond_10y" in result
        assert "bond_1y" in result
        assert "term_spread" in result
        assert "bond_10y_trend" in result
        assert isinstance(result["bond_10y"], float)
        assert result["term_spread"] > 0  # 10Y > 1Y normally

    def test_money_supply_indicators(self, macro_analyzer, mock_money_supply):
        result = macro_analyzer.monetary_liquidity(money_supply_df=mock_money_supply)
        assert "m2_yoy" in result
        assert "m1_yoy" in result
        assert "m2_m1_spread" in result
        assert result["monetary_signal"] in ("easing", "neutral", "tightening")

    def test_lpr_indicators(self, macro_analyzer, mock_lpr):
        result = macro_analyzer.monetary_liquidity(lpr_df=mock_lpr)
        assert "lpr_1y" in result
        assert "lpr_5y" in result
        assert result["lpr_1y"] == 3.40  # 最新值
        assert result["lpr_1y_change"] < 0  # LPR 下调

    def test_liquidity_signal(self, macro_analyzer, mock_bond_yield, mock_money_supply, mock_lpr):
        result = macro_analyzer.monetary_liquidity(
            bond_yield_df=mock_bond_yield,
            money_supply_df=mock_money_supply,
            lpr_df=mock_lpr,
        )
        assert "liquidity_signal" in result
        assert result["liquidity_signal"] in ("loose", "neutral", "tight")

    def test_empty_data(self, macro_analyzer):
        result = macro_analyzer.monetary_liquidity()
        assert isinstance(result, dict)
        # 空数据时不应崩溃
        assert result.get("liquidity_signal") == "neutral"


class TestMacroGrowth:
    """经济增长维度。"""

    def test_pmi_indicators(self, macro_analyzer, mock_pmi):
        result = macro_analyzer.growth(pmi_df=mock_pmi)
        assert "pmi_mfg" in result
        assert "pmi_non_mfg" in result
        assert "pmi_mfg_trend" in result
        assert 45 <= result["pmi_mfg"] <= 55

    def test_gdp_indicators(self, macro_analyzer, mock_gdp):
        result = macro_analyzer.growth(gdp_df=mock_gdp)
        assert "gdp_yoy" in result
        assert result["gdp_yoy"] == 5.0

    def test_growth_signal(self, macro_analyzer, mock_pmi, mock_gdp):
        result = macro_analyzer.growth(pmi_df=mock_pmi, gdp_df=mock_gdp)
        assert "growth_signal" in result
        assert result["growth_signal"] in ("expansion", "stable", "contraction")

    def test_empty_growth(self, macro_analyzer):
        result = macro_analyzer.growth()
        assert isinstance(result, dict)
        # Default PMI=50, GDP=5 → not expansion (PMI>51+GDP≥5.5), not contraction → "stable"
        assert result["growth_signal"] == "stable"


class TestMacroInflation:
    """通胀维度。"""

    def test_cpi_indicators(self, macro_analyzer, mock_cpi):
        result = macro_analyzer.inflation(cpi_df=mock_cpi)
        assert "cpi_yoy" in result
        assert "cpi_trend" in result

    def test_ppi_indicators(self, macro_analyzer, mock_ppi):
        result = macro_analyzer.inflation(ppi_df=mock_ppi)
        assert "ppi_yoy" in result
        assert "ppi_trend" in result

    def test_ppi_cpi_spread(self, macro_analyzer, mock_cpi, mock_ppi):
        result = macro_analyzer.inflation(cpi_df=mock_cpi, ppi_df=mock_ppi)
        assert "ppi_cpi_spread" in result
        assert isinstance(result["ppi_cpi_spread"], float)

    def test_inflation_signal(self, macro_analyzer, mock_cpi, mock_ppi):
        result = macro_analyzer.inflation(cpi_df=mock_cpi, ppi_df=mock_ppi)
        assert "inflation_signal" in result
        assert result["inflation_signal"] in (
            "moderate", "neutral", "deflation", "overheat", "stagflation_risk"
        )


class TestMacroExternal:
    """外部环境维度。"""

    def test_usdcny_indicators(self, macro_analyzer, mock_usdcny):
        result = macro_analyzer.external(usdcny_df=mock_usdcny)
        assert "usdcny" in result
        assert "usdcny_trend" in result
        assert 6.5 <= result["usdcny"] <= 8.0

    def test_external_signal(self, macro_analyzer, mock_usdcny):
        result = macro_analyzer.external(usdcny_df=mock_usdcny)
        assert "external_signal" in result
        assert result["external_signal"] in ("stable", "caution", "high_pressure")


class TestMacroComputeAll:
    """compute_all 与 score 端到端测试。"""

    def test_compute_all(
        self, macro_analyzer,
        mock_bond_yield, mock_money_supply, mock_lpr,
        mock_pmi, mock_cpi, mock_ppi, mock_gdp, mock_usdcny,
    ):
        result = macro_analyzer.compute_all(
            bond_yield_df=mock_bond_yield,
            money_supply_df=mock_money_supply,
            lpr_df=mock_lpr,
            pmi_df=mock_pmi,
            cpi_df=mock_cpi,
            ppi_df=mock_ppi,
            gdp_df=mock_gdp,
            usdcny_df=mock_usdcny,
        )
        assert "date" in result
        assert "monetary_liquidity" in result
        assert "growth" in result
        assert "inflation" in result
        assert "external" in result

    def test_score(
        self, macro_analyzer,
        mock_bond_yield, mock_money_supply, mock_lpr,
        mock_pmi, mock_cpi, mock_ppi, mock_gdp, mock_usdcny,
    ):
        result = macro_analyzer.compute_all(
            bond_yield_df=mock_bond_yield,
            money_supply_df=mock_money_supply,
            lpr_df=mock_lpr,
            pmi_df=mock_pmi,
            cpi_df=mock_cpi,
            ppi_df=mock_ppi,
            gdp_df=mock_gdp,
            usdcny_df=mock_usdcny,
        )
        score = macro_analyzer.score(result)
        assert 0 <= score["total_score"] <= 100
        assert 0 <= score["liquidity_score"] <= 100
        assert 0 <= score["growth_score"] <= 100
        assert 0 <= score["inflation_score"] <= 100
        assert 0 <= score["external_score"] <= 100
        assert score["grade"] in ("A", "B", "C", "D", "E")

    def test_score_empty(self, macro_analyzer):
        score = macro_analyzer.score({})
        assert 0 <= score["total_score"] <= 100

    def test_closed_economy(self, macro_analyzer, mock_pmi, mock_cpi, mock_ppi, mock_gdp):
        """仅国内指标也能正常评分。"""
        result = macro_analyzer.compute_all(
            pmi_df=mock_pmi, cpi_df=mock_cpi, ppi_df=mock_ppi, gdp_df=mock_gdp,
        )
        score = macro_analyzer.score(result)
        assert 0 <= score["total_score"] <= 100


# ═══════════════════════════════════════════════════════════
#  IndustryAnalyzer 测试
# ═══════════════════════════════════════════════════════════


class TestIndustryValuation:
    """行业估值水位。"""

    def test_valuation_from_index(self, industry_analyzer, mock_food_beverage):
        result = industry_analyzer.valuation("食品饮料", index_df=mock_food_beverage)
        assert result["industry"] == "食品饮料"
        assert "pe_pct_3y" in result
        assert 0 <= result["pe_pct_3y"] <= 100
        assert result["valuation_signal"] in (
            "undervalued", "fair_low", "fair", "fair_high", "overvalued"
        )

    def test_valuation_with_pe(self, industry_analyzer):
        pe_series = pd.Series(
            [15, 14, 16, 18, 20, 22, 25, 30, 28, 25, 22, 20], dtype=float
        )
        result = industry_analyzer.valuation("银行", pe_series=pe_series)
        assert "pe_current" in result
        assert "pe_pct_3y" in result

    def test_valuation_empty(self, industry_analyzer):
        result = industry_analyzer.valuation("测试")
        assert result["industry"] == "测试"


class TestIndustryMomentum:
    """行业动量。"""

    def test_momentum_basic(self, industry_analyzer, mock_food_beverage, mock_hs300):
        result = industry_analyzer.momentum(
            "食品饮料", mock_food_beverage, mock_hs300
        )
        assert "ret_20d" in result
        assert "ret_60d" in result
        assert "relative_strength_20d" in result
        assert result["momentum_signal"] in (
            "strong", "positive", "neutral", "negative", "weak"
        )

    def test_momentum_no_benchmark(self, industry_analyzer, mock_bank):
        result = industry_analyzer.momentum("银行", mock_bank)
        assert "ret_20d" in result
        assert "relative_strength_20d" not in result


class TestIndustrySentiment:
    """行业情绪。"""

    def test_sentiment_basic(self, industry_analyzer, mock_food_beverage):
        result = industry_analyzer.sentiment("食品饮料", mock_food_beverage)
        assert "avg_amount_5d" in result
        assert "sentiment_signal" in result

    def test_sentiment_with_market(self, industry_analyzer, mock_food_beverage, mock_market_volume):
        result = industry_analyzer.sentiment(
            "食品饮料", mock_food_beverage, mock_market_volume
        )
        assert "volume_share" in result
        assert isinstance(result["volume_share"], float)

    def test_sentiment_empty(self, industry_analyzer):
        df = pd.DataFrame({"date": [], "收盘": []})
        result = industry_analyzer.sentiment("空行业", df)
        assert result["industry"] == "空行业"


class TestIndustryScore:
    """行业综合评分。"""

    def test_score_single(self, industry_analyzer, mock_food_beverage, mock_hs300, mock_market_volume):
        val = industry_analyzer.valuation("食品饮料", index_df=mock_food_beverage)
        mom = industry_analyzer.momentum("食品饮料", mock_food_beverage, mock_hs300)
        sent = industry_analyzer.sentiment("食品饮料", mock_food_beverage, mock_market_volume)
        score = industry_analyzer.score_industry("食品饮料", val, mom, sent)
        assert 0 <= score["total_score"] <= 100
        assert "valuation_score" in score
        assert "momentum_score" in score
        assert "sentiment_score" in score
        assert "fundamental_score" in score

    def test_score_all(self, industry_analyzer, mock_food_beverage, mock_bank, mock_hs300):
        results = []
        for name, df in [("食品饮料", mock_food_beverage), ("银行", mock_bank)]:
            val = industry_analyzer.valuation(name, index_df=df)
            mom = industry_analyzer.momentum(name, df, mock_hs300)
            sent = industry_analyzer.sentiment(name, df)
            results.append({"industry": name, "valuation": val, "momentum": mom, "sentiment": sent})
        df = industry_analyzer.score_all(results)
        assert len(df) == 2
        assert "total_score" in df.columns
        assert df["total_score"].iloc[0] >= df["total_score"].iloc[1]  # 降序排列

    def test_score_all_empty(self, industry_analyzer):
        df = industry_analyzer.score_all([])
        assert df.empty

    def test_snapshot(self, industry_analyzer, mock_industry_dict, mock_hs300, mock_market_volume):
        df = industry_analyzer.snapshot(
            mock_industry_dict, benchmark_df=mock_hs300, market_volume_df=mock_market_volume
        )
        assert len(df) == 2
        assert "total_score" in df.columns


# ═══════════════════════════════════════════════════════════
#  BriefingGenerator 测试
# ═══════════════════════════════════════════════════════════


class TestBriefingGenerator:
    """简报生成器。"""

    @pytest.fixture
    def gen(self):
        return BriefingGenerator()

    @pytest.fixture
    def sample_macro(self, macro_analyzer, mock_bond_yield, mock_money_supply, mock_lpr,
                     mock_pmi, mock_cpi, mock_ppi, mock_gdp, mock_usdcny):
        return macro_analyzer.compute_all(
            bond_yield_df=mock_bond_yield,
            money_supply_df=mock_money_supply,
            lpr_df=mock_lpr,
            pmi_df=mock_pmi,
            cpi_df=mock_cpi,
            ppi_df=mock_ppi,
            gdp_df=mock_gdp,
            usdcny_df=mock_usdcny,
        )

    @pytest.fixture
    def sample_industry_df(self, industry_analyzer, mock_industry_dict, mock_hs300, mock_market_volume):
        return industry_analyzer.snapshot(
            mock_industry_dict, benchmark_df=mock_hs300, market_volume_df=mock_market_volume
        )

    def test_generate_full(self, gen, sample_macro, sample_industry_df):
        report = gen.generate(sample_macro, sample_industry_df, date="2024-06-03")
        assert isinstance(report, str)
        assert "宏观日报" in report
        assert "2024-06-03" in report
        assert "关键变化" in report
        assert "宏观综合评分" in report
        assert "行业热力图" in report
        assert "风险提示" in report
        assert "免责声明" in report
        # 总分应该是合理的
        assert "总分" in report

    def test_generate_macro_only(self, gen, sample_macro):
        report = gen.generate(sample_macro)
        assert "关键变化" in report
        assert "行业热力图" not in report

    def test_generate_industry_only(self, gen, sample_industry_df):
        report = gen.generate(industry_df=sample_industry_df)
        assert "行业热力图" in report
        assert "食品饮料" in report or "银行" in report

    def test_generate_empty(self, gen):
        report = gen.generate()
        assert "宏观日报" in report
        assert "免责声明" in report

    def test_header(self, gen):
        header = gen._header("测试", "2024-01-01")
        assert "测试" in header
        assert "2024-01-01" in header

    def test_disclaimer(self, gen):
        d = gen._disclaimer()
        assert "免责声明" in d
        assert "投资建议" in d
