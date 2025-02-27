# src/weight_generator/generate_weights.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier

class WeightAdjuster(ABC):
    """权重调整基类"""
    @abstractmethod
    def adjust_weights(self, datasets: List[pd.DataFrame], components: List, target: str) -> Dict[str, float]:
        """根据数据和组件调整权重
        Args:
            datasets: 历史数据列表
            components: 规则、分析类或特征列表
            target: 强弱标签列名
        Returns:
            Dict[str, float]: 组件名到权重的映射
        """
        pass

class RulesWeightAdjuster(WeightAdjuster):
    """规则模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], rules: List, target: str) -> Dict[str, float]:
        """基于规则预测准确率调整权重"""
        weights = {}
        for rule in rules:
            facts = [rule.evaluate(df) for df in datasets]
            preds = [f[0] for f in facts]
            actual = [df[target].iloc[-1] > 0 for df in datasets]
            accuracy = sum(p == a for p, a in zip(preds, actual)) / len(preds)
            weights[rule.__class__.__name__] = accuracy * 2  # 映射到 0-2
        return weights

class ScoringWeightAdjuster(WeightAdjuster):
    """打分模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], analyses: List, target: str) -> Dict[str, float]:
        """基于分析类得分与强弱的相关性调整权重"""
        weights = {}
        for analysis in analyses:
            scores = [analysis.evaluate(df) for df in datasets]
            corr = np.corrcoef(scores, [df[target].iloc[-1] for df in datasets])[0, 1]
            weights[analysis.__class__.__name__] = max(corr, 0) * 2
        return weights

class MLWeightAdjuster(WeightAdjuster):
    """机器学习模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], features: List[str], target: str) -> Dict[str, float]:
        """基于随机森林特征重要性调整权重"""
        X = pd.concat([df[features] for df in datasets], axis=0)
        y = [1 if df[target].iloc[-1] > 0 else 0 for df in datasets]
        model = RandomForestClassifier(random_state=42).fit(X, y)
        return dict(zip(features, model.feature_importances_ * 2))

class StatsWeightAdjuster(WeightAdjuster):
    """统计模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], metrics: List, target: str) -> Dict[str, float]:
        """基于统计指标与强弱的相关性调整权重"""
        weights = {}
        for metric in metrics:
            values = [metric.compute(df)[0] for df in datasets]
            corr = np.corrcoef(values, [df[target].iloc[-1] for df in datasets])[0, 1]
            weights[metric.__class__.__name__] = max(corr, 0) * 2
        return weights

class WeightGenerator:
    """权重生成工具"""
    def __init__(self):
        """初始化调整器映射"""
        self.adjusters = {
            "rules": RulesWeightAdjuster(),
            "scoring": ScoringWeightAdjuster(),
            "ml": MLWeightAdjuster(),
            "stats": StatsWeightAdjuster()
        }

    def generate(self, module_type: str, datasets: List[pd.DataFrame], components: List, target: str = "label") -> Dict[str, float]:
        """生成指定模块的权重
        Args:
            module_type: 模块类型 ('rules', 'scoring', 'ml', 'stats')
            datasets: 历史数据列表
            components: 规则、分析类或特征列表
            target: 强弱标签列名
        Returns:
            Dict[str, float]: 权重字典
        """
        adjuster = self.adjusters.get(module_type)
        if not adjuster:
            raise ValueError(f"未知模块类型: {module_type}")
        return adjuster.adjust_weights(datasets, components, target)

    def update_config(self, module_type: str, weights: Dict[str, float], config_path: str):
        """更新配置文件中的权重
        Args:
            module_type: 模块类型
            weights: 生成的权重字典
            config_path: 配置文件路径
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        config["weights"] = weights
        config["auto_weights"] = True
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            print(f"已更新 {config_path} 中的权重")

if __name__ == "__main__":
    from ..stats.evaluator import MeanReturn, Volatility  # 调整为完整路径
    datasets = [pd.read_csv("../../data/rb2510.csv")]
    metrics = [MeanReturn(), Volatility()]
    generator = WeightGenerator()
    weights = generator.generate("stats", datasets, metrics, target="close")
    generator.update_config("stats", weights, "stats_config.json")