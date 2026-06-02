"""数据获取模块

封装 AKShare 等多源数据接口，提供统一的数据拉取入口。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DataFetcher:
    """统一数据获取接口。

    Parameters
    ----------
    cache_dir : Path, optional
        本地缓存目录，默认 ~/.fqf_cache
    """

    cache_dir: Optional[Path] = None

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".fqf_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 行情数据 ──────────────────────────────────────

    def fetch_daily_hist(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """获取个股日线行情。"""
        raise NotImplementedError

    def fetch_index_daily(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """获取指数日线行情。"""
        raise NotImplementedError

    # ── 财务数据 ──────────────────────────────────────

    def fetch_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """获取资产负债表。"""
        raise NotImplementedError

    def fetch_income_statement(self, symbol: str) -> pd.DataFrame:
        """获取利润表。"""
        raise NotImplementedError

    def fetch_cash_flow(self, symbol: str) -> pd.DataFrame:
        """获取现金流量表。"""
        raise NotImplementedError

    # ── 估值数据 ──────────────────────────────────────

    def fetch_valuation(self, symbol: str) -> dict:
        """获取最新估值数据 (PE/PB/PS/市值)。"""
        raise NotImplementedError

    # ── 宏观数据 ──────────────────────────────────────

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame:
        """获取宏观指标时序。"""
        raise NotImplementedError

    # ── 行业数据 ──────────────────────────────────────

    def fetch_industry_valuation(self, industry: str) -> pd.DataFrame:
        """获取行业估值数据。"""
        raise NotImplementedError

    def fetch_industry_index(self, industry: str, start: str, end: str) -> pd.DataFrame:
        """获取行业指数行情。"""
        raise NotImplementedError

    # ── 缓存管理 ──────────────────────────────────────

    def _cache_path(self, category: str, key: str) -> Path:
        """构建缓存文件路径。"""
        return self.cache_dir / category / f"{key}.parquet"

    def _read_cache(self, category: str, key: str) -> Optional[pd.DataFrame]:
        """读取缓存。"""
        path = self._cache_path(category, key)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _write_cache(self, category: str, key: str, df: pd.DataFrame):
        """写入缓存。"""
        path = self._cache_path(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
