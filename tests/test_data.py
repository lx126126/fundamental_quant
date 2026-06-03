"""数据层单元测试"""

import pytest
from pathlib import Path
from fqf.data import AKShareFetcher, DataFetcher, DataCleaner
from fqf.data.akshare_fetcher import _code_to_ak


class TestCodeNormalization:
    """股票代码转换测试。"""

    def test_to_akshare_shanghai(self):
        assert _code_to_ak("600000") == "sh600000"

    def test_to_akshare_shenzhen(self):
        assert _code_to_ak("000001") == "sz000001"

    def test_to_akshare_gem(self):
        assert _code_to_ak("300750") == "sz300750"

    def test_to_akshare_star(self):
        assert _code_to_ak("688001") == "sh688001"

    def test_already_prefixed(self):
        assert _code_to_ak("sh600000") == "sh600000"
        assert _code_to_ak("sz000001") == "sz000001"

    def test_short_code(self):
        assert _code_to_ak("1") == "sz000001"


class TestAKShareFetcherInit:
    """AKShareFetcher 初始化测试。"""

    def test_init_default_cache(self):
        fetcher = AKShareFetcher()
        assert fetcher.cache_dir == Path.cwd() / ".fqf_cache"

    def test_init_custom_cache(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path / "my_cache")
        assert fetcher.cache_dir == tmp_path / "my_cache"
        assert fetcher.cache_dir.exists()

    def test_rate_limit_default(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        assert fetcher.rate_limit == 0.5


class TestAKShareFetcherCache:
    """缓存读写测试。"""

    def test_cache_write_read(self, tmp_path):
        import pandas as pd
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        fetcher._write_cache("test", "key1", df)
        cached = fetcher._read_cache("test", "key1")
        assert cached is not None
        pd.testing.assert_frame_equal(cached, df)

    def test_cache_miss(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        result = fetcher._read_cache("test", "nonexistent")
        assert result is None

    def test_cache_path_sanitize(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        path = fetcher._cache_path("category", "a/b\\c")
        assert "/" not in path.name
        assert "\\" not in path.name


class TestAKShareFetcherMacro:
    """宏观指标枚举测试。"""

    def test_get_macro_indicators(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        indicators = fetcher.get_macro_indicators()
        assert "pmi" in indicators
        assert "money_supply" in indicators
        assert "bond_yield" in indicators
        assert "cpi" in indicators

    def test_invalid_indicator(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="未知宏观指标"):
            fetcher.fetch_macro_indicator("invalid_indicator")


class TestAKShareFetcherTools:
    """工具方法测试。"""

    def test_to_pure_code(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        assert fetcher.to_pure_code("sh600000") == "600000"
        assert fetcher.to_pure_code("SZ000001") == "000001"

    def test_to_akshare_code(self, tmp_path):
        fetcher = AKShareFetcher(cache_dir=tmp_path)
        assert fetcher.to_akshare_code("600000") == "sh600000"
