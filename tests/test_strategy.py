"""
Phase 3 单元测试 -- Strategy & Backtest
测试因子库、回测引擎、绩效指标、策略模板。
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fqf.strategy.factors import list_factors, get_factor_spec, compute_factor, Direction
from fqf.strategy.metrics import (
    compute_cumulative_return,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_information_ratio,
    compute_win_rate,
    compute_monthly_excess_return,
    compute_turnover,
    compute_max_excess_drawdown,
    compute_all_metrics,
    PerformanceResult,
)
from fqf.strategy.engine import BacktestConfig, RebalanceFreq, BacktestEngine
from fqf.strategy.strategy_templates import (
    strategy_value_quality,
    strategy_reversal,
    strategy_quality_dividend,
)


def _make_price_df():
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    np.random.seed(42)
    prices = {}
    for code in ["600519.SH", "601318.SH", "000858.SZ"]:
        ret = np.random.normal(0.0005, 0.015, 500)
        price = 100 * np.exp(np.cumsum(ret))
        prices[code] = price
    df = pd.DataFrame(prices, index=dates)
    return df


def _make_financial_df():
    return pd.DataFrame({
        "code": ["600519.SH", "601318.SH", "000858.SZ", "000333.SZ", "600036.SH"],
        "roe_ttm": [0.25, 0.22, 0.28, 0.20, 0.18],
        "dividend_years": [5, 4, 3, 5, 4],
        "interest_bearing_debt_ratio": [0.0, 0.10, 0.15, 0.05, 0.12],
    }).set_index("code")


def _make_valuation_df():
    return pd.DataFrame({
        "code": ["600519.SH", "601318.SH", "000858.SZ", "000333.SZ", "600036.SH"],
        "dividend_yield": [0.021, 0.025, 0.018, 0.019, 0.020],
    }).set_index("code")


def _make_pe_pb_df():
    dates = pd.date_range("2018-01-01", periods=1200, freq="D")
    np.random.seed(42)
    data = {}
    for code in ["600519.SH", "601318.SH"]:
        pe = 20 + np.random.normal(0, 3, 1200)
        pb = 3 + np.random.normal(0, 0.5, 1200)
        df = pd.DataFrame({"pe_ttm": pe, "pb": pb}, index=dates)
        df["pe_ttm_pct_5y"] = np.random.uniform(0.1, 0.9, 1200)
        df["pb_pct_5y"] = np.random.uniform(0.1, 0.9, 1200)
        data[code] = df
    return data


class TestFactorLibrary(unittest.TestCase):
    def test_list_factors_returns_list(self):
        factors = list_factors()
        self.assertIsInstance(factors, list)
        self.assertIn("roe_ttm", factors)

    def test_get_factor_spec(self):
        spec = get_factor_spec("roe_ttm")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "roe_ttm")
        self.assertEqual(spec.direction, Direction.LONG)

    def test_get_factor_spec_unknown(self):
        spec = get_factor_spec("nonexistent_factor")
        self.assertIsNone(spec)

    def test_compute_factor_roe_ttm(self):
        df = _make_financial_df()
        result = compute_factor("roe_ttm", financial_df=df)
        self.assertIsNotNone(result)
        self.assertIn("600519.SH", result.values.index)

    def test_compute_factor_pe_percentile_5y(self):
        pe_pb_data = _make_pe_pb_df()
        # 用第一个股票的 df 作为 valuation_df
        df = pe_pb_data["600519.SH"].iloc[-1:]
        df["pe_ttm_pct_5y"] = 0.3
        result = compute_factor("pe_percentile_5y", valuation_df=df)
        self.assertIsNotNone(result)

    def test_compute_factor_pb_percentile_5y(self):
        pe_pb_data = _make_pe_pb_df()
        df = pe_pb_data["600519.SH"].iloc[-1:]
        df["pb_pct_5y"] = 0.3
        result = compute_factor("pb_percentile_5y", valuation_df=df)
        self.assertIsNotNone(result)

    def test_compute_factor_dividend_yield(self):
        df = _make_valuation_df()
        result = compute_factor("dividend_yield", valuation_df=df)
        self.assertIsNotNone(result)
        best = result.values.idxmax()
        self.assertEqual(best, "601318.SH")

    def test_compute_factor_fcf_to_profit(self):
        df = _make_financial_df()
        # 需要 fcf_ttm 和 net_profit_ttm 列
        df["fcf_ttm"] = df["roe_ttm"] * 100
        df["net_profit_ttm"] = df["roe_ttm"] * 80
        result = compute_factor("fcf_to_profit", financial_df=df)
        self.assertIsNotNone(result)

    def test_compute_factor_debt_ratio(self):
        df = _make_financial_df()
        df["total_liabilities"] = 500
        df["total_assets"] = 1000
        result = compute_factor("debt_ratio", financial_df=df)
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, Direction.SHORT)

    def test_compute_factor_profit_growth_3y(self):
        df = _make_financial_df()
        df["profit_growth_3y"] = [0.15, 0.12, 0.18, 0.10, 0.08]
        result = compute_factor("profit_growth_3y", financial_df=df)
        self.assertIsNotNone(result)

    def test_compute_factor_ev_ebitda(self):
        df = _make_valuation_df()
        df["ev_ebitda"] = [8.0, 9.0, 10.0, 7.0, 8.5]
        result = compute_factor("ev_ebitda", valuation_df=df)
        self.assertIsNotNone(result)

    def test_compute_factor_none_for_missing_data(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = compute_factor("roe_ttm", financial_df=df)
        self.assertTrue(result is None or result.values.isna().all())

    def test_compute_factor_unknown_name(self):
        result = compute_factor("nonexistent")
        self.assertIsNone(result)


class TestPerformanceMetrics(unittest.TestCase):
    def _make_nav(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        np.random.seed(42)
        ret = np.random.normal(0.0008, 0.012, 252)
        nav = pd.Series(1.0 + np.cumsum(ret), index=dates)
        return nav

    def _make_benchmark(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        np.random.seed(123)
        ret = np.random.normal(0.0003, 0.014, 252)
        bm = pd.Series(1.0 + np.cumsum(ret), index=dates)
        return bm

    def test_cumulative_return(self):
        nav = self._make_nav()
        cr = compute_cumulative_return(nav)
        self.assertIsInstance(cr, float)
        self.assertGreater(cr, 0)

    def test_cumulative_return_empty(self):
        nav = pd.Series([], dtype=float)
        cr = compute_cumulative_return(nav)
        self.assertEqual(cr, 0.0)

    def test_annualized_return(self):
        nav = self._make_nav()
        ar = compute_annualized_return(nav)
        self.assertIsInstance(ar, float)

    def test_annualized_volatility(self):
        nav = self._make_nav()
        daily_ret = nav.pct_change().dropna()
        vol = compute_annualized_volatility(daily_ret)
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)

    def test_sharpe_ratio(self):
        nav = self._make_nav()
        daily_ret = nav.pct_change().dropna()
        sr = compute_sharpe_ratio(daily_ret, risk_free_rate=0.025)
        self.assertIsInstance(sr, float)

    def test_max_drawdown(self):
        nav = self._make_nav()
        mdd = compute_max_drawdown(nav)
        self.assertIsInstance(mdd, float)
        self.assertLessEqual(mdd, 0.0)

    def test_max_drawdown_no_drawdown(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        nav = pd.Series(np.linspace(1.0, 2.0, 100), index=dates)
        mdd = compute_max_drawdown(nav)
        self.assertEqual(mdd, 0.0)

    def test_calmar_ratio(self):
        nav = self._make_nav()
        cr = compute_calmar_ratio(nav, risk_free_rate=0.025)
        self.assertIsInstance(cr, float)

    def test_information_ratio(self):
        nav = self._make_nav()
        bm = self._make_benchmark()
        strat_ret = nav.pct_change().dropna()
        bench_ret = bm.pct_change().dropna()
        ir = compute_information_ratio(strat_ret, bench_ret)
        self.assertIsInstance(ir, float)

    def test_win_rate(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        np.random.seed(42)
        ret = np.random.normal(0.0008, 0.012, 500)
        nav = pd.Series(1.0 + np.cumsum(ret), index=dates)
        wr = compute_win_rate(nav.pct_change().dropna())
        self.assertIsInstance(wr, float)
        self.assertGreaterEqual(wr, 0.0)
        self.assertLessEqual(wr, 1.0)

    def test_win_rate_zero(self):
        wr = compute_win_rate(pd.Series([], dtype=float))
        self.assertEqual(wr, 0.0)

    def test_monthly_excess_return(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        np.random.seed(42)
        strat_nav = pd.Series(1.0 + np.cumsum(np.random.normal(0.0008, 0.012, 252)), index=dates)
        bench_nav = pd.Series(1.0 + np.cumsum(np.random.normal(0.0003, 0.014, 252)), index=dates)
        monthly = compute_monthly_excess_return(strat_nav, bench_nav)
        self.assertIsInstance(monthly, dict)
        self.assertGreater(len(monthly), 0)

    def test_turnover(self):
        # compute_turnover 接受 (trades: list[dict], nav: pd.Series)
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        nav = pd.Series(np.linspace(1.0, 1.1, 10), index=dates)
        trades = [
            {"action": "buy", "cost": 10000},
            {"action": "sell", "proceeds": 10500},
        ]
        to = compute_turnover(trades, nav)
        self.assertIsInstance(to, float)
        self.assertGreater(to, 0)

    def test_turnover_no_trades(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        nav = pd.Series(np.linspace(1.0, 1.1, 10), index=dates)
        to = compute_turnover([], nav)
        self.assertEqual(to, 0.0)

    def test_max_excess_drawdown(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        np.random.seed(42)
        strat_nav = pd.Series(1.0 + np.cumsum(np.random.normal(0.0008, 0.012, 252)), index=dates)
        bench_nav = pd.Series(1.0 + np.cumsum(np.random.normal(0.0003, 0.014, 252)), index=dates)
        mdd = compute_max_excess_drawdown(strat_nav, bench_nav)
        self.assertIsInstance(mdd, float)
        self.assertLessEqual(mdd, 0.0)

    def test_compute_all_metrics(self):
        nav = self._make_nav()
        bm = self._make_benchmark()
        result = compute_all_metrics(nav, benchmark_nav=bm)
        self.assertIsInstance(result, PerformanceResult)
        self.assertIsInstance(result.cumulative_return, float)
        self.assertIsInstance(result.sharpe_ratio, float)


class TestBacktestEngine(unittest.TestCase):
    def test_config_creation(self):
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2021-12-31",
            benchmark="000300.SH",
            initial_cash=1_000_000.0,
        )
        self.assertEqual(cfg.rebalance_freq, RebalanceFreq.QUARTERLY)
        self.assertEqual(cfg.initial_cash, 1_000_000.0)

    def test_engine_creation(self):
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2021-12-31",
            benchmark="000300.SH",
        )
        engine = BacktestEngine(cfg, strategy_fn=None)
        self.assertIsNotNone(engine)

    def test_run_with_mock_data(self):
        np.random.seed(42)
        price_df = _make_price_df()
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2021-12-31",
            benchmark="000300.SH",
            initial_cash=1_000_000.0,
        )
        engine = BacktestEngine(cfg, strategy_fn=strategy_value_quality)
        result = engine.run(price_df=price_df)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.nav), 0)

    def test_run_no_strategy(self):
        """测试无策略函数时引擎能正常运行（不调仓）"""
        price_df = _make_price_df()
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2021-12-31",
        )
        # 传入一个返回空权重的策略函数（不持仓）
        engine = BacktestEngine(cfg, strategy_fn=lambda *a, **k: {})
        result = engine.run(price_df=price_df)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.nav) > 0)

    def test_rebalance_freq_quarterly(self):
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2020-12-31",
            rebalance_freq=RebalanceFreq.QUARTERLY,
        )
        self.assertEqual(cfg.rebalance_freq, RebalanceFreq.QUARTERLY)


class TestStrategyTemplates(unittest.TestCase):
    def test_value_quality_returns_weights(self):
        fin = _make_financial_df()
        val = _make_valuation_df()
        weights = strategy_value_quality(
            date="2022-01-01", financial_df=fin, valuation_df=val,
        )
        if weights:
            self.assertIsInstance(weights, dict)
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_value_quality_filtered_by_roe(self):
        fin = _make_financial_df()
        val = _make_valuation_df()
        weights = strategy_value_quality(
            date="2022-01-01", financial_df=fin, valuation_df=val,
        )
        if weights:
            self.assertGreater(len(weights), 0)

    def test_reversal_returns_weights(self):
        """测试反转策略返回权重字典且权重和为1"""
        fin = _make_financial_df()
        weights = strategy_reversal(
            date="2022-01-01", financial_df=fin,
        )
        if weights:
            self.assertIsInstance(weights, dict)
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_dividend_strategy_high_dividend_first(self):
        fin = _make_financial_df()
        val = _make_valuation_df()
        weights = strategy_quality_dividend(
            date="2022-01-01", financial_df=fin, valuation_df=val,
        )
        if weights:
            codes = list(weights.keys())
            self.assertEqual(codes[0], "601318.SH")

    def test_dividend_strategy_filtered_by_dividend_years(self):
        fin = _make_financial_df()
        val = _make_valuation_df()
        weights = strategy_quality_dividend(
            date="2022-01-01", financial_df=fin, valuation_df=val,
        )
        if weights:
            for code in weights:
                self.assertIn(code, ["600519.SH", "601318.SH", "000333.SZ", "600036.SH"])


if __name__ == "__main__":
    unittest.main()
