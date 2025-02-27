# src/utils/market_conditions.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np

class MarketCondition(ABC):
    """市场状态基类，用于动态调整权重"""
    @abstractmethod
    def evaluate(self, datasets: List[pd.DataFrame]) -> bool:
        """判断是否满足市场条件
        Args:
            datasets (List[pd.DataFrame]): 数据集列表
        Returns:
            bool: 是否满足条件
        """
        pass

    @abstractmethod
    def apply_adjustments(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重调整
        Args:
            weights (Dict[str, float]): 当前权重字典
        Returns:
            Dict[str, float]: 调整后的权重
        """
        pass

class HighVolatilityCondition(MarketCondition):
    """高波动状态：短期ATR高于长期均值+标准差"""
    def __init__(self, adjustments: Dict[str, float]):
        self.adjustments = adjustments

    def evaluate(self, datasets: List[pd.DataFrame]) -> bool:
        market_atr = np.mean([df['high'].tail(20).mean() - df['low'].tail(20).mean() for df in datasets])
        atr_mean = np.mean([df['high'].mean() - df['low'].mean() for df in datasets])
        atr_std = np.std([df['high'].mean() - df['low'].mean() for df in datasets])
        return market_atr > atr_mean + atr_std

    def apply_adjustments(self, weights: Dict[str, float]) -> Dict[str, float]:
        for key, multiplier in self.adjustments.items():
            if key in weights:
                weights[key] = min(weights[key] * multiplier, 2.0)
        return weights

class TrendMarketCondition(MarketCondition):
    """趋势市场状态：ADX > 25"""
    def __init__(self, adjustments: Dict[str, float]):
        self.adjustments = adjustments

    def evaluate(self, datasets: List[pd.DataFrame]) -> bool:
        high = datasets[0]['high'].tail(14)
        low = datasets[0]['low'].tail(14)
        close = datasets[0]['close'].tail(14)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.mean()
        dm_plus = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        dm_minus = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        di_plus = 100 * dm_plus.rolling(14).mean() / atr
        di_minus = 100 * dm_minus.rolling(14).mean() / atr
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean().iloc[-1]
        return adx > 25 if not np.isnan(adx) else False

    def apply_adjustments(self, weights: Dict[str, float]) -> Dict[str, float]:
        for key, multiplier in self.adjustments.items():
            if key in weights:
                weights[key] = min(weights[key] * multiplier, 2.0)
        return weights

class RangeMarketCondition(MarketCondition):
    """震荡市场状态：ADX < 20"""
    def __init__(self, adjustments: Dict[str, float]):
        self.adjustments = adjustments

    def evaluate(self, datasets: List[pd.DataFrame]) -> bool:
        high = datasets[0]['high'].tail(14)
        low = datasets[0]['low'].tail(14)
        close = datasets[0]['close'].tail(14)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.mean()
        dm_plus = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        dm_minus = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        di_plus = 100 * dm_plus.rolling(14).mean() / atr
        di_minus = 100 * dm_minus.rolling(14).mean() / atr
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean().iloc[-1]
        return adx < 20 if not np.isnan(adx) else False

    def apply_adjustments(self, weights: Dict[str, float]) -> Dict[str, float]:
        for key, multiplier in self.adjustments.items():
            if key in weights:
                weights[key] = min(weights[key] * multiplier, 1.8)
        return weights

class LowVolatilityCondition(MarketCondition):
    """低波动状态：短期ATR低于长期均值"""
    def __init__(self, adjustments: Dict[str, float]):
        self.adjustments = adjustments

    def evaluate(self, datasets: List[pd.DataFrame]) -> bool:
        market_atr = np.mean([df['high'].tail(20).mean() - df['low'].tail(20).mean() for df in datasets])
        atr_mean = np.mean([df['high'].mean() - df['low'].mean() for df in datasets])
        return market_atr < atr_mean

    def apply_adjustments(self, weights: Dict[str, float]) -> Dict[str, float]:
        for key, multiplier in self.adjustments.items():
            if key in weights:
                weights[key] = min(weights[key] * multiplier, 2.0)
        return weights