"""Analyzer 模块单元测试

使用 mock 数据，不依赖真实网络请求。
覆盖：
- FinancialCleaner: 字段重命名、类型转换、报告期解析、缺失值填补
- FinancialMetrics: 各类指标计算的数学逻辑
- ReportGenerator: 报告各章节结构完整性
- generate_report: 便捷函数端到端流程
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from fqf.analyzer.financial import FinancialCleaner
from fqf.analyzer.metrics import FinancialMetrics, _safe_div, _yoy
from fqf.analyzer.report import ReportGenerator, generate_report


# ═══════════════════════════════════════════════════════════
#  Fixtures：生成模拟三表数据
# ═══════════════════════════════════════════════════════════


def _make_dates(n=5):
    """生成 n 个年报期（倒序，最新的在前）。"""
    return [f"{2023 - i}-12-31" for i in range(n)]


@pytest.fixture
def mock_balance_sheet():
    """模拟资产负债表（5期年报）。"""
    dates = _make_dates(5)
    data = {
        "报告期": dates,
        "货币资金":         [500e8, 450e8, 400e8, 350e8, 300e8],
        "应收账款":          [200e8, 180e8, 160e8, 140e8, 120e8],
        "存货":              [150e8, 140e8, 130e8, 120e8, 110e8],
        "流动资产合计":      [1000e8, 900e8, 800e8, 700e8, 600e8],
        "固定资产":          [400e8, 380e8, 360e8, 340e8, 320e8],
        "资产总计":          [2000e8, 1800e8, 1600e8, 1400e8, 1200e8],
        "短期借款":          [100e8, 90e8, 80e8, 70e8, 60e8],
        "应付账款":          [120e8, 110e8, 100e8, 90e8, 80e8],
        "流动负债合计":      [400e8, 360e8, 320e8, 280e8, 240e8],
        "长期借款":          [200e8, 180e8, 160e8, 140e8, 120e8],
        "负债合计":          [700e8, 630e8, 560e8, 490e8, 420e8],
        "归属于母公司所有者权益合计": [1200e8, 1080e8, 960e8, 840e8, 720e8],
        "所有者权益合计":    [1300e8, 1170e8, 1040e8, 910e8, 780e8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_income_statement():
    """模拟利润表（5期年报）。"""
    dates = _make_dates(5)
    data = {
        "报告期":  dates,
        "营业收入":  [1000e8, 900e8, 800e8, 700e8, 600e8],
        "营业成本":  [600e8, 540e8, 480e8, 420e8, 360e8],
        "销售费用":  [50e8, 45e8, 40e8, 35e8, 30e8],
        "管理费用":  [30e8, 27e8, 24e8, 21e8, 18e8],
        "财务费用":  [10e8, 9e8, 8e8, 7e8, 6e8],
        "研发费用":  [20e8, 18e8, 16e8, 14e8, 12e8],
        "营业利润":  [180e8, 162e8, 144e8, 126e8, 108e8],
        "利润总额":  [175e8, 157e8, 140e8, 122e8, 105e8],
        "所得税费用": [44e8, 39e8, 35e8, 30e8, 26e8],
        "净利润":    [131e8, 118e8, 105e8, 92e8, 79e8],
        "归属于母公司所有者的净利润": [125e8, 112e8, 100e8, 88e8, 75e8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_cash_flow():
    """模拟现金流量表（5期年报）。"""
    dates = _make_dates(5)
    data = {
        "报告期": dates,
        "经营活动产生的现金流量净额": [150e8, 135e8, 120e8, 105e8, 90e8],
        "购建固定资产、无形资产和其他长期资产支付的现金": [30e8, 27e8, 24e8, 21e8, 18e8],
        "投资活动产生的现金流量净额": [-40e8, -36e8, -32e8, -28e8, -24e8],
        "筹资活动产生的现金流量净额": [-20e8, -18e8, -16e8, -14e8, -12e8],
        "现金及现金等价物净增加额": [90e8, 81e8, 72e8, 63e8, 54e8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cleaner():
    return FinancialCleaner(n_periods=12, fill_method="ffill", annual_only=False)


@pytest.fixture
def cleaner_annual():
    return FinancialCleaner(n_periods=12, fill_method="ffill", annual_only=True)


@pytest.fixture
def cleaned_tables(cleaner, mock_balance_sheet, mock_income_statement, mock_cash_flow):
    return cleaner.clean_all(mock_balance_sheet, mock_income_statement, mock_cash_flow)


# ═══════════════════════════════════════════════════════════
#  FinancialCleaner 测试
# ═══════════════════════════════════════════════════════════


class TestFinancialCleaner:

    def test_clean_balance_sheet_columns(self, cleaner, mock_balance_sheet):
        """清洗后列名应为英文 snake_case。"""
        result = cleaner.clean_balance_sheet(mock_balance_sheet)
        assert "total_assets" in result.columns
        assert "total_liabilities" in result.columns
        assert "cash" in result.columns
        assert "equity_parent" in result.columns

    def test_clean_report_date_index(self, cleaner, mock_balance_sheet):
        """report_date 应成为 DatetimeIndex。"""
        result = cleaner.clean_balance_sheet(mock_balance_sheet)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_sorted_ascending(self, cleaner, mock_balance_sheet):
        """结果按时间升序排列。"""
        result = cleaner.clean_balance_sheet(mock_balance_sheet)
        assert result.index.is_monotonic_increasing

    def test_numeric_conversion(self, cleaner, mock_balance_sheet):
        """数值列应为 float 类型。"""
        result = cleaner.clean_balance_sheet(mock_balance_sheet)
        assert result["total_assets"].dtype in (np.float64, np.float32, float)

    def test_n_periods_limit(self, mock_balance_sheet):
        """n_periods 参数应限制返回期数。"""
        cleaner = FinancialCleaner(n_periods=3)
        result = cleaner.clean_balance_sheet(mock_balance_sheet)
        assert len(result) <= 3

    def test_clean_all_returns_dict(self, cleaner, mock_balance_sheet, mock_income_statement, mock_cash_flow):
        """clean_all 应返回含三个键的 dict。"""
        result = cleaner.clean_all(mock_balance_sheet, mock_income_statement, mock_cash_flow)
        assert set(result.keys()) == {"balance_sheet", "income_statement", "cash_flow"}

    def test_empty_dataframe_returns_empty(self, cleaner):
        """空 DataFrame 输入应返回空 DataFrame。"""
        result = cleaner.clean_balance_sheet(pd.DataFrame())
        assert result.empty

    def test_income_statement_columns(self, cleaner, mock_income_statement):
        """利润表清洗后应有 revenue, cogs, net_profit 等列。"""
        result = cleaner.clean_income_statement(mock_income_statement)
        assert "revenue" in result.columns
        assert "cogs" in result.columns
        assert "net_profit" in result.columns
        assert "net_profit_parent" in result.columns

    def test_cash_flow_columns(self, cleaner, mock_cash_flow):
        """现金流量表清洗后应有 cfo, capex 等列。"""
        result = cleaner.clean_cash_flow(mock_cash_flow)
        assert "cfo" in result.columns
        assert "capex" in result.columns

    def test_string_numbers_converted(self, cleaner):
        """含逗号的字符串数值应能正确转换。"""
        df = pd.DataFrame({
            "报告期": ["2023-12-31"],
            "资产总计": ["1,234,567.89"],
        })
        result = cleaner.clean_balance_sheet(df)
        assert "total_assets" in result.columns
        assert abs(result["total_assets"].iloc[0] - 1234567.89) < 1

    def test_merge_statements(self, cleaned_tables):
        """merge_statements 应生成宽格式 DataFrame。"""
        merged = FinancialCleaner.merge_statements(
            cleaned_tables["balance_sheet"],
            cleaned_tables["income_statement"],
            cleaned_tables["cash_flow"],
        )
        assert not merged.empty
        # 应包含各表前缀的列
        bs_cols = [c for c in merged.columns if c.startswith("bs_")]
        is_cols = [c for c in merged.columns if c.startswith("is_")]
        cf_cols = [c for c in merged.columns if c.startswith("cf_")]
        assert len(bs_cols) > 0
        assert len(is_cols) > 0
        assert len(cf_cols) > 0


# ═══════════════════════════════════════════════════════════
#  FinancialMetrics 测试
# ═══════════════════════════════════════════════════════════


class TestFinancialMetrics:

    @pytest.fixture(autouse=True)
    def setup(self, cleaned_tables):
        self.bs = cleaned_tables["balance_sheet"]
        self.is_ = cleaned_tables["income_statement"]
        self.cf = cleaned_tables["cash_flow"]
        self.calc = FinancialMetrics(annual_only=False)

    def test_compute_all_returns_dataframe(self):
        """compute_all 应返回非空 DataFrame。"""
        result = self.calc.compute_all(self.bs, self.is_, self.cf)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_gross_margin_range(self):
        """毛利率应在 [0, 1] 之间。"""
        result = self.calc.profitability(self.bs, self.is_, self.cf)
        if "gross_margin" in result.columns:
            valid = result["gross_margin"].dropna()
            assert (valid >= 0).all()
            assert (valid <= 1).all()

    def test_gross_margin_value(self):
        """毛利率 = (营收 - 成本) / 营收，应为 0.4。"""
        result = self.calc.profitability(self.bs, self.is_, self.cf)
        if "gross_margin" in result.columns:
            # 营收1000e8, 成本600e8 → 毛利率 40%
            assert abs(result["gross_margin"].iloc[-1] - 0.4) < 0.01

    def test_roe_positive(self):
        """ROE 在正向盈利时应为正。"""
        result = self.calc.profitability(self.bs, self.is_, self.cf)
        if "roe" in result.columns:
            valid = result["roe"].dropna()
            assert (valid > 0).all()

    def test_revenue_yoy(self):
        """营收同比增速应接近 11.1%（900→1000）。"""
        result = self.calc.growth(self.is_, self.cf)
        if "revenue_yoy" in result.columns:
            latest = result["revenue_yoy"].dropna().iloc[-1]
            assert abs(latest - (1000 - 900) / 900) < 0.01

    def test_fcf_positive(self):
        """FCF = CFO - CAPEX 应为正。"""
        result = self.calc.cash_quality(self.is_, self.cf)
        if "fcf" in result.columns:
            valid = result["fcf"].dropna()
            assert (valid > 0).all()

    def test_debt_ratio_range(self):
        """资产负债率应在 [0, 1] 之间。"""
        result = self.calc.leverage(self.bs, self.is_)
        if "debt_ratio" in result.columns:
            valid = result["debt_ratio"].dropna()
            assert (valid >= 0).all()
            assert (valid <= 1).all()

    def test_debt_ratio_value(self):
        """负债率 = 700/2000 = 35%。"""
        result = self.calc.leverage(self.bs, self.is_)
        if "debt_ratio" in result.columns:
            latest = result["debt_ratio"].dropna().iloc[-1]
            assert abs(latest - 700 / 2000) < 0.01

    def test_current_ratio_value(self):
        """流动比率 = 流动资产 / 流动负债 = 1000/400 = 2.5。"""
        result = self.calc.leverage(self.bs, self.is_)
        if "current_ratio" in result.columns:
            latest = result["current_ratio"].dropna().iloc[-1]
            assert abs(latest - 1000 / 400) < 0.1

    def test_asset_turnover_positive(self):
        """总资产周转率应为正。"""
        result = self.calc.efficiency(self.bs, self.is_)
        if "asset_turnover" in result.columns:
            valid = result["asset_turnover"].dropna()
            assert (valid > 0).all()

    def test_dupont_three_factor_consistency(self):
        """杜邦三因子乘积应接近直接计算的 ROE。"""
        result_dup = self.calc.dupont(self.bs, self.is_)
        result_prof = self.calc.profitability(self.bs, self.is_, self.cf)
        if "dp_roe_3factor" in result_dup.columns and "roe" in result_prof.columns:
            idx = result_dup["dp_roe_3factor"].dropna().index.intersection(
                result_prof["roe"].dropna().index
            )
            if len(idx) > 0:
                diff = abs(
                    result_dup.loc[idx, "dp_roe_3factor"]
                    - result_prof.loc[idx, "roe"]
                )
                assert (diff < 0.05).all(), "杜邦三因子 ROE 与直接 ROE 差异过大"

    def test_score_returns_composite(self):
        """score 函数应返回 composite_score。"""
        metrics_df = self.calc.compute_all(self.bs, self.is_, self.cf)
        scores = self.calc.score(metrics_df)
        assert "composite_score" in scores.index

    def test_score_range(self):
        """综合评分应在 [0, 100]。"""
        metrics_df = self.calc.compute_all(self.bs, self.is_, self.cf)
        scores = self.calc.score(metrics_df)
        composite = scores.get("composite_score", 50)
        assert 0 <= composite <= 100

    def test_empty_input(self):
        """空 DataFrame 输入应返回空 DataFrame 而非报错。"""
        result = self.calc.compute_all(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert isinstance(result, pd.DataFrame)

    def test_annual_filter(self, cleaned_tables):
        """annual_only=True 时应过滤非年报期数据。"""
        calc = FinancialMetrics(annual_only=True)
        # 插入一个非年报期行
        bs = cleaned_tables["balance_sheet"].copy()
        bs.loc[pd.Timestamp("2023-06-30")] = bs.iloc[-1]
        result = calc.profitability(calc._filter_annual(bs), cleaned_tables["income_statement"], cleaned_tables["cash_flow"])
        # 确保 6月30日 数据不在结果里
        if not result.empty:
            assert pd.Timestamp("2023-06-30") not in result.index


# ═══════════════════════════════════════════════════════════
#  ReportGenerator 测试
# ═══════════════════════════════════════════════════════════


class TestReportGenerator:

    @pytest.fixture(autouse=True)
    def setup(self, cleaned_tables):
        self.bs = cleaned_tables["balance_sheet"]
        self.is_ = cleaned_tables["income_statement"]
        self.cf = cleaned_tables["cash_flow"]
        calc = FinancialMetrics(annual_only=False)
        self.metrics_df = calc.compute_all(self.bs, self.is_, self.cf)
        self.scores = calc.score(self.metrics_df)
        self.gen = ReportGenerator(
            company_name="测试科技",
            symbol="000001",
            industry="银行",
        )

    def test_report_is_string(self):
        """生成报告应为非空字符串。"""
        report = self.gen.generate(self.metrics_df, self.scores)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_header(self):
        """报告应包含公司名和代码。"""
        report = self.gen.generate(self.metrics_df, self.scores)
        assert "测试科技" in report
        assert "000001" in report

    def test_report_contains_sections(self):
        """报告应包含各主要章节标题。"""
        report = self.gen.generate(self.metrics_df, self.scores)
        assert "盈利能力" in report
        assert "成长能力" in report
        assert "现金质量" in report
        assert "综合评分" in report
        assert "风险提示" in report

    def test_report_with_valuation(self):
        """传入估值数据时报告应包含估值快照。"""
        valuation = {
            "最新价": 10.5,
            "市盈率-动态": 15.2,
            "市净率": 1.8,
            "总市值": 1e12,
        }
        report = self.gen.generate(self.metrics_df, self.scores, valuation=valuation)
        assert "估值快照" in report
        assert "15.2x" in report

    def test_report_disclaimer(self):
        """报告应包含免责声明。"""
        report = self.gen.generate(self.metrics_df, self.scores)
        assert "不构成投资建议" in report or "免责声明" in report

    def test_generate_report_function(
        self,
        mock_balance_sheet,
        mock_income_statement,
        mock_cash_flow
    ):
        """便捷函数 generate_report 端到端流程。"""
        cleaner = FinancialCleaner(annual_only=False)
        bs = cleaner.clean_balance_sheet(mock_balance_sheet)
        is_ = cleaner.clean_income_statement(mock_income_statement)
        cf = cleaner.clean_cash_flow(mock_cash_flow)

        report = generate_report(
            company_name="测试公司",
            symbol="600000",
            balance_sheet=bs,
            income_statement=is_,
            cash_flow=cf,
            industry="金融",
            annual_only=False,
        )
        assert isinstance(report, str)
        assert "测试公司" in report
        assert "600000" in report


# ═══════════════════════════════════════════════════════════
#  工具函数测试
# ═══════════════════════════════════════════════════════════


class TestHelperFunctions:

    def test_safe_div_normal(self):
        """正常除法。"""
        a = pd.Series([10.0, 20.0, 30.0])
        b = pd.Series([2.0, 4.0, 5.0])
        result = _safe_div(a, b)
        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_div_zero_denominator(self):
        """分母为 0 时返回 NaN。"""
        a = pd.Series([10.0])
        b = pd.Series([0.0])
        result = _safe_div(a, b)
        assert pd.isna(result.iloc[0])

    def test_yoy_calculation(self):
        """同比增速计算（1000/900 - 1 ≈ 11.1%）。"""
        s = pd.Series(
            [600.0, 700.0, 800.0, 900.0, 1000.0],
            index=pd.date_range("2019", periods=5, freq="YE"),
        )
        result = _yoy(s)
        latest_yoy = result.dropna().iloc[-1]
        assert abs(latest_yoy - (1000 - 900) / 900) < 0.001
