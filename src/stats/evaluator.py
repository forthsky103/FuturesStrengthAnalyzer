# src/stats/evaluator.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import os
import pandas as pd
import numpy as np
import yaml
import logging
from ..utils.market_conditions import MarketCondition

class StatMetric(ABC):
    """统计指标基类"""
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        """计算统计指标
        Args:
            data (pd.DataFrame): 合约数据，至少包含基础列（如 close）
        Returns:
            Tuple[float, str]: (指标值, 解释)
        """
        pass

class MeanReturn(StatMetric):
    """平均收益率"""
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        returns = data['close'].pct_change().dropna()
        mean_return = returns.mean()
        return mean_return, f"平均收益率: {mean_return:.4f}"

class Volatility(StatMetric):
    """波动率"""
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        returns = data['close'].pct_change().dropna()
        vol = returns.std()
        return vol, f"波动率: {vol:.4f}"

class SharpeRatio(StatMetric):
    """夏普比率（假设无风险利率为0）"""
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        returns = data['close'].pct_change().dropna()
        mean_return = returns.mean()
        vol = returns.std()
        sharpe = mean_return / vol if vol != 0 else 0
        return sharpe, f"夏普比率: {sharpe:.4f}"

class MaxDrawdown(StatMetric):
    """最大回撤"""
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        returns = data['close'].pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        return max_dd, f"最大回撤: {max_dd:.4f}"

class Skewness(StatMetric):
    """偏度"""
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        returns = data['close'].pct_change().dropna()
        skew = returns.skew()
        return skew, f"偏度: {skew:.4f}"

class Kurtosis(StatMetric):
    """峰度"""
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        returns = data['close'].pct_change().dropna()
        kurt = returns.kurtosis()
        return kurt, f"峰度: {kurt:.4f}"

class QuantileRange(StatMetric):
    """分位数范围（75%-25%）"""
    def compute(self, data: pd.DataFrame) -> Tuple[float, str]:
        returns = data['close'].pct_change().dropna()
        q75 = returns.quantile(0.75)
        q25 = returns.quantile(0.25)
        qr = q75 - q25
        return qr, f"分位数范围 (75%-25%): {qr:.4f}"

class StatsEvaluator:
    """统计评估器，用于判断合约强弱"""
    def __init__(self, metrics: List[StatMetric], weights: Dict[str, float] = None):
        """初始化统计评估器
        Args:
            metrics (List[StatMetric]): 统计指标列表
            weights (Dict[str, float]): 指标权重，默认为 1.0
        """
        self.metrics = metrics
        self.base_weights = weights or {m.__class__.__name__: 1.0 for m in metrics}
        self.weights = self.base_weights.copy()

    def compute_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算单个合约的统计指标
        Args:
            data (pd.DataFrame): 合约数据
        Returns:
            Dict[str, float]: 指标名到值的映射
        """
        stats = {}
        for metric in self.metrics:
            value, _ = metric.compute(data)
            stats[metric.__class__.__name__] = value
        return stats

    def compute_score(self, stats: Dict[str, float]) -> Tuple[float, str]:
        """计算综合得分
        Args:
            stats (Dict[str, float]): 统计指标字典
        Returns:
            Tuple[float, str]: (得分, 解释)
        """
        total_score = 0.0
        total_weight = sum(self.weights.values())
        explanation = "统计指标分析：\n"
        for metric_name, value in stats.items():
            weight = self.weights.get(metric_name, 1.0)
            # 标准化处理：正向指标加分（如收益率），负向指标减分（如回撤）
            if metric_name in ["MaxDrawdown", "Volatility", "Kurtosis"]:  # 负向指标
                normalized_value = -value  # 负值转为正贡献
            else:  # 正向指标
                normalized_value = value
            weighted_value = normalized_value * weight
            total_score += weighted_value
            explanation += f"- {metric_name}: {value:.4f}, 权重: {weight}, 加权值: {weighted_value:.4f}\n"
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        explanation += f"综合得分: {final_score:.4f}"
        return final_score, explanation

    def adjust_weights(self, datasets: List[pd.DataFrame], conditions: List[MarketCondition]) -> Dict[str, float]:
        """动态调整统计指标权重
        Args:
            datasets (List[pd.DataFrame]): 数据集列表
            conditions (List[MarketCondition]): 市场状态类实例列表
        Returns:
            Dict[str, float]: 调整后的权重
        """
        self.weights = self.base_weights.copy()
        for condition in conditions:
            if condition.evaluate(datasets):
                condition.apply_adjustments(self.weights)
                logging.info(f"应用市场状态: {condition.__class__.__name__}")
        logging.info(f"动态调整权重: {self.weights}")
        return self.weights

    def evaluate(self, datasets: List[pd.DataFrame], condition_map: Dict[str, type], config_path: str = "stats_config.yaml") -> Tuple[Dict, str, str]:
        """评估所有合约的强弱
        Args:
            datasets (List[pd.DataFrame]): 数据集列表
            condition_map (Dict[str, type]): 市场状态类映射
            config_path (str): 配置文件路径，默认为 stats_config.yaml
        Returns:
            Tuple[Dict, str, str]: (结果字典, 最强合约, 最弱合约)
        """
        config_full_path = os.path.join(os.path.dirname(__file__), config_path)
        with open(config_full_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        conditions = [condition_map[cond["type"]](cond["adjustments"]) for cond in config.get("market_conditions", [])]

        if config.get("auto_weights", False):
            from ..weight_generator.generate_weights import WeightGenerator
            generator = WeightGenerator()
            self.base_weights = generator.generate("stats", datasets, self.metrics)
            self.weights = self.base_weights.copy()
            if config.get("update_config", True):
                generator.update_config("stats", self.base_weights, config_full_path)
        else:
            self.base_weights = config.get("weights", {m.__class__.__name__: 1.0 for m in self.metrics})
            self.weights = self.base_weights.copy()

        self.adjust_weights(datasets, conditions)

        results = {}
        for i, data in enumerate(datasets):
            contract = f"contract{i+1}"
            stats = self.compute_stats(data)
            score, explanation = self.compute_score(stats)
            results[contract] = (score, explanation)
            logging.info(f"{contract} 评估完成，得分: {score:.4f}\n{explanation}")

        strongest = max(results, key=lambda k: results[k][0])
        weakest = min(results, key=lambda k: results[k][0])
        return results, strongest, weakest