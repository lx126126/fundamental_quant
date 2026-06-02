"""数据清洗模块

财务报表标准化、异常值检测、缺失值填补。
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """数据清洗与标准化工具。"""

    @staticmethod
    def standardize_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化财务数据列名，统一为 snake_case 英文名。"""
        raise NotImplementedError

    @staticmethod
    def validate_balance_sheet(df: pd.DataFrame) -> bool:
        """校验资产负债表勾稽关系：资产 = 负债 + 所有者权益。"""
        raise NotImplementedError

    @staticmethod
    def detect_outliers(
        series: pd.Series, method: str = "zscore", threshold: float = 3.0
    ) -> pd.Series:
        """检测异常值。

        Parameters
        ----------
        series : pd.Series
            待检测序列
        method : str
            'zscore' 或 'iqr'
        threshold : float
            阈值（zscore 法为 sigma 倍数，iqr 法为 IQR 倍数）
        """
        if method == "zscore":
            z = (series - series.mean()) / series.std()
            return z.abs() > threshold
        elif method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            return (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def fill_missing(
        df: pd.DataFrame, method: str = "ffill", max_gap: int = 4
    ) -> pd.DataFrame:
        """填补缺失值。

        Parameters
        ----------
        df : pd.DataFrame
        method : str
            'ffill'（前值填充）、'linear'（线性插值）
        max_gap : int
            最大缺失间隔，超过此值不填充
        """
        raise NotImplementedError
