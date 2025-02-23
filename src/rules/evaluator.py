# src/rules/evaluator.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd

class Rule(ABC):
    """规则基类"""
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> float:
        """应用规则，返回得分（0-1）"""
        pass

class BreakoutMARule(Rule):
    """规则：价格突破均线"""
    def __init__(self, window: int = 20):
        self.window = window

    def apply(self, data: pd.DataFrame) -> float:
        recent = data.tail(self.window)
        ma = recent['close'].mean()
        latest_close = recent['close'].iloc[-1]
        return 1.0 if latest_close > ma else 0.0

class VolumeIncreaseRule(Rule):
    """规则：成交量增加"""
    def __init__(self, window: int = 20):
        self.window = window

    def apply(self, data: pd.DataFrame) -> float:
        recent = data.tail(self.window)
        vol_change = (recent['volume'].iloc[-1] - recent['volume'].iloc[0]) / recent['volume'].iloc[0]
        return 1.0 if vol_change > 0 else 0.0

class PositionTrendRule(Rule):
    """规则：持仓量趋势上升"""
    def __init__(self, window: int = 20):
        self.window = window

    def apply(self, data: pd.DataFrame) -> float:
        recent = data.tail(self.window)
        pos_change = (recent['position'].iloc[-1] - recent['position'].iloc[0]) / recent['position'].iloc[0]
        return 1.0 if pos_change > 0 else 0.0

class RulesEvaluator:
    """规则评估器"""
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def evaluate(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        """综合多个规则评估强弱"""
        scores = {}
        for i, data in enumerate(datasets):
            total_score = 0.0
            for rule in self.rules:
                total_score += rule.apply(data)
            scores[f"contract{i+1}"] = total_score / len(self.rules) if self.rules else 0.0  # 平均得分
        return scores