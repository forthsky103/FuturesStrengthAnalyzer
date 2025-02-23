# src/stats/evaluator.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np

class Stat(ABC):
    """统计方法基类"""
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> float:
        """计算单个合约的统计得分"""
        pass

class ReturnStat(Stat):
    """收益率统计"""
    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, data: pd.DataFrame) -> float:
        recent = data.tail(self.window)
        return (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]

class VolatilityStat(Stat):
    """波动率统计（基于高低价差的标准差）"""
    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, data: pd.DataFrame) -> float:
        recent = data.tail(self.window)
        return (recent['high'] - recent['low']).std()

class SharpeStat(Stat):
    """夏普比率统计（收益率/波动率）"""
    def __init__(self, window: int = 20, risk_free_rate: float = 0.0):
        self.window = window
        self.risk_free_rate = risk_free_rate

    def compute(self, data: pd.DataFrame) -> float:
        recent = data.tail(self.window)
        returns = recent['close'].pct_change().dropna()
        excess_return = returns.mean() - self.risk_free_rate / (self.window * 5)  # 每日调整
        volatility = returns.std()
        return excess_return / volatility if volatility != 0 else 0.0

class StatsEvaluator:
    """统计评估器"""
    def __init__(self, stats: List[Stat], weights: Dict[str, float] = None):
        self.stats = stats
        self.weights = weights or {stat.__class__.__name__: 1.0 for stat in stats}

    def evaluate(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        """综合多个统计方法评估强弱"""
        scores = {}
        for i, data in enumerate(datasets):
            total_score = 0.0
            for stat in self.stats:
                stat_name = stat.__class__.__name__
                score = stat.compute(data)
                # 标准化处理：收益率正向，波动率反向，夏普正向
                if stat_name == "VolatilityStat":
                    score = -score / 1000  # 波动率反向，缩放
                total_score += score * self.weights.get(stat_name, 1.0)
            scores[f"contract{i+1}"] = total_score / sum(self.weights.values()) if self.weights else 0.0
        return scores