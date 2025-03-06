# src/weight_generator/generate_weights.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

class WeightAdjuster(ABC):
    """权重调整基类"""
    @abstractmethod
    def adjust_weights(self, datasets: List[pd.DataFrame], components: List, target: str) -> Dict[str, float]:
        """根据数据和组件调整权重"""
        pass

class RulesWeightAdjuster(WeightAdjuster):
    """规则模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], rules: List, target: str) -> Dict[str, float]:
        weights = {}
        for rule in rules:
            facts = [rule.evaluate(df) for df in datasets]
            preds = [f[0] for f in facts]
            actual = [df[target].iloc[-1] == 'strong' for df in datasets]
            accuracy = sum(p == a for p, a in zip(preds, actual)) / len(preds)
            weights[rule.__class__.__name__] = float(accuracy * 2)
        return weights

class ScoringWeightAdjuster(WeightAdjuster):
    """打分模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], analyses: List, target: str) -> Dict[str, float]:
        weights = {}
        for analysis in analyses:
            scores = [analysis.evaluate(df) for df in datasets]
            corr = np.corrcoef(scores, [df[target].iloc[-1] for df in datasets])[0, 1]
            weights[analysis.__class__.__name__] = float(max(corr, 0) * 2)
        return weights

class MLWeightAdjuster(WeightAdjuster):
    """机器学习模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], features: List[str], target: str) -> Dict[str, float]:
        X = pd.concat([df[features] for df in datasets], axis=0)
        y = [1 if df[target].iloc[-1] == 'strong' else 0 for df in datasets]
        model = RandomForestClassifier(random_state=42).fit(X, y)
        return dict(zip(features, model.feature_importances_ * 2))

class StatsWeightAdjuster(WeightAdjuster):
    """统计模块权重调整器"""
    def adjust_weights(self, datasets: List[pd.DataFrame], metrics: List, target: str) -> Dict[str, float]:
        """基于统计指标与强弱标签的相关性调整权重"""
        weights = {}
        for metric in metrics:
            values = [metric.compute(df)[0] for df in datasets]
            labels = [1 if df[target].iloc[-1] == 'strong' else 0 for df in datasets]  # 转换为布尔值
            corr = np.corrcoef(values, labels)[0, 1] if np.std(values) != 0 else 0
            weights[metric.__class__.__name__] = float(max(corr, 0) * 2)
        return weights

class WeightGenerator:
    """权重生成工具"""
    def __init__(self):
        self.adjusters = {
            "rules": RulesWeightAdjuster(),
            "scoring": ScoringWeightAdjuster(),
            "ml": MLWeightAdjuster(),
            "stats": StatsWeightAdjuster()
        }

    def generate(self, module_type: str, datasets: List[pd.DataFrame], components: List, target: str = "label") -> Dict[str, float]:
        adjuster = self.adjusters.get(module_type)
        if not adjuster:
            raise ValueError(f"未知模块类型: {module_type}")
        return adjuster.adjust_weights(datasets, components, target)

    def update_config(self, module_type: str, weights: Dict[str, float], config_path: str):
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config.get("update_config", True):
            print(f"动态权重已生成但未更新到 {config_path}（update_config: false）：{weights}")
            return
        if config.get("update_weights_only", False):
            config["weights"] = weights
        else:
            config = {"weights": weights, "auto_weights": True}
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"已更新 {config_path} 中的权重")




# # src/weight_generator/generate_weights.py
# from abc import ABC, abstractmethod
# from typing import List, Dict
# import pandas as pd
# import numpy as np
# import yaml
# from sklearn.ensemble import RandomForestClassifier

# class WeightAdjuster(ABC):
#     """权重调整基类"""
#     @abstractmethod
#     def adjust_weights(self, datasets: List[pd.DataFrame], components: List, target: str) -> Dict[str, float]:
#         """根据数据和组件调整权重"""
#         pass

# class RulesWeightAdjuster(WeightAdjuster):
#     """规则模块权重调整器"""
#     def adjust_weights(self, datasets: List[pd.DataFrame], rules: List, target: str) -> Dict[str, float]:
#         """基于规则预测准确率调整权重"""
#         weights = {}
#         for rule in rules:
#             facts = [rule.evaluate(df) for df in datasets]
#             preds = [f[0] for f in facts]
#             actual = [df[target].iloc[-1] == 'strong' for df in datasets]
#             accuracy = sum(p == a for p, a in zip(preds, actual)) / len(preds)
#             weights[rule.__class__.__name__] = float(accuracy * 2)  # 强制转换为标准浮点数
#         return weights

# class ScoringWeightAdjuster(WeightAdjuster):
#     """打分模块权重调整器"""
#     def adjust_weights(self, datasets: List[pd.DataFrame], analyses: List, target: str) -> Dict[str, float]:
#         weights = {}
#         for analysis in analyses:
#             scores = [analysis.evaluate(df) for df in datasets]
#             corr = np.corrcoef(scores, [df[target].iloc[-1] for df in datasets])[0, 1]
#             weights[analysis.__class__.__name__] = float(max(corr, 0) * 2)
#         return weights

# class MLWeightAdjuster(WeightAdjuster):
#     """机器学习模块权重调整器"""
#     def adjust_weights(self, datasets: List[pd.DataFrame], features: List[str], target: str) -> Dict[str, float]:
#         X = pd.concat([df[features] for df in datasets], axis=0)
#         y = [1 if df[target].iloc[-1] > 0 else 0 for df in datasets]
#         model = RandomForestClassifier(random_state=42).fit(X, y)
#         return dict(zip(features, model.feature_importances_ * 2))

# class StatsWeightAdjuster(WeightAdjuster):
#     """统计模块权重调整器"""
#     def adjust_weights(self, datasets: List[pd.DataFrame], metrics: List, target: str) -> Dict[str, float]:
#         weights = {}
#         for metric in metrics:
#             values = [metric.compute(df)[0] for df in datasets]
#             corr = np.corrcoef(values, [df[target].iloc[-1] for df in datasets])[0, 1]
#             weights[metric.__class__.__name__] = float(max(corr, 0) * 2)
#         return weights

# class WeightGenerator:
#     """权重生成工具"""
#     def __init__(self):
#         """初始化调整器映射"""
#         self.adjusters = {
#             "rules": RulesWeightAdjuster(),
#             "scoring": ScoringWeightAdjuster(),
#             "ml": MLWeightAdjuster(),
#             "stats": StatsWeightAdjuster()
#         }

#     def generate(self, module_type: str, datasets: List[pd.DataFrame], components: List, target: str = "label") -> Dict[str, float]:
#         """生成指定模块的权重"""
#         adjuster = self.adjusters.get(module_type)
#         if not adjuster:
#             raise ValueError(f"未知模块类型: {module_type}")
#         return adjuster.adjust_weights(datasets, components, target)

#     def update_config(self, module_type: str, weights: Dict[str, float], config_path: str):
#         """更新配置文件中的权重"""
#         with open(config_path, "r", encoding='utf-8') as f:
#             config = yaml.safe_load(f)
        
#         # 检查是否更新配置文件
#         if not config.get("update_config", True):
#             print(f"动态权重已生成但未更新到 {config_path}（update_config: false）：{weights}")
#             return
        
#         # 根据 update_weights_only 决定更新范围
#         if config.get("update_weights_only", False):
#             config["weights"] = weights  # 仅更新 weights
#         else:
#             config = {"weights": weights, "auto_weights": True}  # 覆盖整个配置
        
#         with open(config_path, "w", encoding='utf-8') as f:
#             yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
#             print(f"已更新 {config_path} 中的权重")