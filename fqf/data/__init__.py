"""数据层 (Data Layer)

多源数据获取、清洗、缓存与导出。
"""

from fqf.data.fetcher import DataFetcher
from fqf.data.akshare_fetcher import AKShareFetcher
from fqf.data.cleaner import DataCleaner

__all__ = ["DataFetcher", "AKShareFetcher", "DataCleaner"]
