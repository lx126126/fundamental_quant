"""数据获取模块

封装 AKShare 等多源数据接口，提供统一的数据拉取入口。
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataFetcher(ABC):
    """统一数据获取抽象基类。

    Parameters
    ----------
    cache_dir : Path, optional
        本地缓存目录，默认 ~/.fqf_cache
    """

    cache_dir: Optional[Path] = None

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path.cwd() / ".fqf_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 行情数据 ──────────────────────────────────────

    @abstractmethod
    def fetch_daily_hist(
        self, symbol: str, start: str, end: str, adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取个股日线行情。"""
        ...

    @abstractmethod
    def fetch_index_daily(
        self, index_code: str, start: str, end: str
    ) -> pd.DataFrame:
        """获取指数日线行情。"""
        ...

    # ── 财务数据 ──────────────────────────────────────

    @abstractmethod
    def fetch_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """获取资产负债表。"""
        ...

    @abstractmethod
    def fetch_income_statement(self, symbol: str) -> pd.DataFrame:
        """获取利润表。"""
        ...

    @abstractmethod
    def fetch_cash_flow(self, symbol: str) -> pd.DataFrame:
        """获取现金流量表。"""
        ...

    # ── 估值与参考数据 ─────────────────────────────────

    @abstractmethod
    def fetch_valuation(self, symbol: str) -> dict:
        """获取最新估值数据 (PE/PB/PS/市值/行业)。"""
        ...

    @abstractmethod
    def fetch_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表及基本信息。"""
        ...

    # ── 宏观数据 ──────────────────────────────────────

    @abstractmethod
    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame:
        """获取宏观指标时序。"""
        ...

    # ── 行业数据 ──────────────────────────────────────

    @abstractmethod
    def fetch_industry_list(self) -> pd.DataFrame:
        """获取申万行业分类列表。"""
        ...

    @abstractmethod
    def fetch_industry_index(
        self, industry: str, start: str, end: str
    ) -> pd.DataFrame:
        """获取行业指数行情。"""
        ...

    @abstractmethod
    def fetch_industry_constituents(self, industry: str) -> pd.DataFrame:
        """获取行业成分股。"""
        ...

    # ── 缓存管理 ──────────────────────────────────────

    def _cache_path(self, category: str, key: str) -> Path:
        """构建缓存文件路径。"""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / category / f"{safe_key}.parquet"

    def _read_cache(self, category: str, key: str) -> Optional[pd.DataFrame]:
        """读取缓存。"""
        path = self._cache_path(category, key)
        if path.exists():
            logger.debug(f"Cache hit: {path}")
            return pd.read_parquet(path)
        return None

    def _write_cache(self, category: str, key: str, df: pd.DataFrame):
        """写入缓存。"""
        path = self._cache_path(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.debug(f"Cache write: {path}")

    def _cached_fetch(
        self,
        category: str,
        key: str,
        fetch_fn,
        ttl_seconds: int = 3600,
        **kwargs,
    ) -> pd.DataFrame:
        """缓存装饰器：命中缓存直接返回，否则调用 fetch_fn 并缓存。

        Parameters
        ----------
        category : str
            缓存分类 (market/financial/macro/industry)
        key : str
            缓存键
        fetch_fn : callable
            数据获取函数，返回 DataFrame
        ttl_seconds : int
            缓存有效期（秒），默认 3600（1小时）
        """
        path = self._cache_path(category, key)
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < ttl_seconds:
                logger.debug(f"Cache hit (age={age:.0f}s): {path}")
                return pd.read_parquet(path)
            logger.debug(f"Cache expired (age={age:.0f}s): {path}")

        logger.info(f"Fetching: {category}/{key}")
        df = fetch_fn(**kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df
