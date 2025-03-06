# src/utils/feature_selector.py
from typing import List, Dict
import pandas as pd
import inspect
from abc import ABC, abstractmethod
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
import src.features.labelers as labelers
from src.features.labelers import ReturnBasedLabeler
from ..scoring import analyses
from ..features import features
from ..rules import evaluator
from ..stats import evaluator as stats_evaluator
import logging
import numpy as np

class BaseFeatureSelector(ABC):
    def __init__(self, feature_classes: List, feature_names: List[str]):
        self.feature_classes = feature_classes
        self.feature_names = feature_names
        # 移除 self.modules，避免预实例化

    @abstractmethod
    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        pass

    def _get_labeler(self, config: Dict):
        labeler_type = config.get("labeler_type", "ReturnBasedLabeler")
        window = config.get("labeler_window", 20)
        try:
            labeler_class = getattr(labelers, labeler_type)
            return labeler_class(window)
        except AttributeError:
            logging.error(f"指定的标签生成器 {labeler_type} 不存在，使用默认 ReturnBasedLabeler")
            return ReturnBasedLabeler(window)

class ScoringFeatureSelector(BaseFeatureSelector):
    def __init__(self):
        classes = [
            cls for name, cls in inspect.getmembers(analyses, inspect.isclass)
            if cls.__module__ == 'src.scoring.analyses' and not inspect.isabstract(cls)
        ]
        names = [cls.__name__ for cls in classes]
        super().__init__(classes, names)

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        enable_selection = config.get("enable_feature_selection", False)
        feature_count = config.get("feature_count", 20)
        manual_features = config.get("manual_features", [])

        if not enable_selection:
            if manual_features:
                valid_features = [f for f in manual_features if f in self.feature_names]
                invalid_features = set(manual_features) - set(valid_features)
                if invalid_features:
                    logging.warning(f"以下手动指定的打分特征不存在，将被忽略: {invalid_features}")
                return valid_features if valid_features else self.feature_names
            return self.feature_names

        if feature_count <= 0 or feature_count >= len(self.feature_names):
            return self.feature_names

        labeler = self._get_labeler(config)
        labels_dict = labeler.generate_labels(datasets)
        labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]
        logging.info(f"标签数量: {len(labels)}")

        feature_scores = pd.DataFrame(
            {cls.__name__: cls().analyze(datasets) for cls in self.feature_classes}
        ).T
        feature_scores = feature_scores.T
        logging.info(f"feature_scores 形状: {feature_scores.shape}")

        n_per_class = feature_count // 2
        strong_labels = [1 if label == 'strong' else 0 for label in labels]
        weak_labels = [1 if label == 'weak' else 0 for label in labels]

        selector_strong = SelectKBest(f_regression, k=n_per_class)
        selector_strong.fit(feature_scores, strong_labels)
        strong_indices = selector_strong.get_support(indices=True)

        selector_weak = SelectKBest(f_regression, k=n_per_class)
        selector_weak.fit(feature_scores, weak_labels)
        weak_indices = selector_weak.get_support(indices=True)

        selected_indices = list(set(strong_indices).union(weak_indices))
        selected_features = [self.feature_names[i] for i in selected_indices]

        remaining = feature_count - len(selected_features)
        if remaining > 0:
            other_indices = [i for i in range(len(self.feature_names)) if i not in selected_indices][:remaining]
            selected_features.extend([self.feature_names[i] for i in other_indices])

        return selected_features

class MLFeatureSelector(BaseFeatureSelector):
    """ML 特征选择器，基于显著性筛选"""
    def __init__(self):
        classes = [
            cls for name, cls in inspect.getmembers(features, inspect.isclass)
            if cls.__module__ == 'src.features.features' and not inspect.isabstract(cls)
        ]
        names = [cls.__name__ for cls in classes]
        super().__init__(classes, names)

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        enable_selection = config.get("enable_feature_selection", False)
        manual_features = config.get("manual_features", [])
        feature_params = config.get("feature_params", {})

        if not enable_selection and manual_features:
            valid_features = [f for f in manual_features if f in self.feature_names]
            invalid_features = set(manual_features) - set(valid_features)
            if invalid_features:
                logging.warning(f"以下手动指定的特征不存在，将被忽略: {invalid_features}")
            return valid_features if valid_features else self.feature_names
        return self.feature_names  # 默认返回所有特征，后续可扩展自动选择逻辑

class RulesFeatureSelector(BaseFeatureSelector):
    """规则特征选择器，基于触发频率筛选"""
    def __init__(self):
        classes = [
            cls for name, cls in inspect.getmembers(evaluator, inspect.isclass)
            if cls.__module__ == 'src.rules.evaluator' and 
               name != 'ExpertSystem' and 
               not inspect.isabstract(cls)
        ]
        names = [cls.__name__ for cls in classes]
        super().__init__(classes, names)

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        enable_selection = config.get("enable_feature_selection", False)
        feature_count = config.get("feature_count", 13)
        manual_features = config.get("manual_features", [])

        if not enable_selection:
            if manual_features:
                valid_features = [f for f in manual_features if f in self.feature_names]
                invalid_features = set(manual_features) - set(valid_features)
                if invalid_features:
                    logging.warning(f"以下手动指定的规则特征不存在，将被忽略: {invalid_features}")
                return valid_features if valid_features else self.feature_names
            return self.feature_names

        if feature_count <= 0 or feature_count >= len(self.feature_names):
            return self.feature_names

        labeler = self._get_labeler(config)
        labels_dict = labeler.generate_labels(datasets)
        labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]

        trigger_counts = {name: 0 for name in self.feature_names}
        window = config.get("labeler_window", 20)
        for i, data in enumerate(datasets):
            for name, cls in zip(self.feature_names, self.feature_classes):
                is_met, _, _ = cls().evaluate(data.tail(window))
                if (labels[i] == 'strong' and is_met) or (labels[i] == 'weak' and not is_met):
                    trigger_counts[name] += 1

        sorted_rules = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        selected_features = [rule[0] for rule in sorted_rules[:feature_count]]

        return selected_features

class StatsFeatureSelector(BaseFeatureSelector):
    """统计特征选择器，基于显著性筛选"""
    def __init__(self):
        classes = [
            cls for name, cls in inspect.getmembers(stats_evaluator, inspect.isclass)
            if cls.__module__ == 'src.stats.evaluator' and 
               name != 'StatsEvaluator' and 
               not inspect.isabstract(cls)
        ]
        names = [cls.__name__ for cls in classes]
        super().__init__(classes, names)

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        enable_selection = config.get("enable_feature_selection", False)
        feature_count = config.get("feature_count", 7)
        manual_features = config.get("manual_features", [])

        if not enable_selection:
            if manual_features:
                valid_features = [f for f in manual_features if f in self.feature_names]
                invalid_features = set(manual_features) - set(valid_features)
                if invalid_features:
                    logging.warning(f"以下手动指定的统计特征不存在，将被忽略: {invalid_features}")
                return valid_features if valid_features else self.feature_names
            return self.feature_names

        if feature_count <= 0 or feature_count >= len(self.feature_names):
            return self.feature_names

        labeler = self._get_labeler(config)
        labels_dict = labeler.generate_labels(datasets)
        labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]

        feature_scores = pd.DataFrame()
        for i, data in enumerate(datasets):
            scores = {name: cls().compute(data)[0] for name, cls in zip(self.feature_names, self.feature_classes)}
            feature_scores[f"contract{i+1}"] = pd.Series(scores)

        n_per_class = feature_count // 2
        strong_labels = [1 if label == 'strong' else 0 for label in labels]
        weak_labels = [1 if label == 'weak' else 0 for label in labels]

        selector_strong = SelectKBest(f_regression, k=n_per_class)
        selector_strong.fit(feature_scores.T, strong_labels)
        strong_indices = selector_strong.get_support(indices=True)

        selector_weak = SelectKBest(f_regression, k=n_per_class)
        selector_weak.fit(feature_scores.T, weak_labels)
        weak_indices = selector_weak.get_support(indices=True)

        selected_indices = list(set(strong_indices).union(weak_indices))
        selected_features = [self.feature_names[i] for i in selected_indices]

        remaining = feature_count - len(selected_features)
        if remaining > 0:
            other_indices = [i for i in range(len(self.feature_names)) if i not in selected_indices][:remaining]
            selected_features.extend([self.feature_names[i] for i in other_indices])

        return selected_features

def get_feature_selector(module_type: str):
    selector_map = {
        "rules": RulesFeatureSelector,
        "stats": StatsFeatureSelector,
        "ml": MLFeatureSelector,
        "scoring": ScoringFeatureSelector
    }
    selector_class = selector_map.get(module_type)
    if not selector_class:
        raise ValueError(f"不支持的模块类型: {module_type}")
    return selector_class()


# # src/utils/feature_selector.py
# from typing import List, Dict
# import pandas as pd
# import inspect
# from abc import ABC, abstractmethod
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.ensemble import RandomForestClassifier
# import src.features.labelers as labelers
# from src.features.labelers import ReturnBasedLabeler
# from ..scoring import analyses
# from ..features import features
# from ..rules import evaluator
# from ..stats import evaluator as stats_evaluator
# import logging
# import numpy as np

# class BaseFeatureSelector(ABC):
#     def __init__(self, feature_classes: List, feature_names: List[str]):
#         self.feature_classes = feature_classes
#         self.feature_names = feature_names
#         self.modules = [cls() for cls in feature_classes]  # 预实例化模块

#     @abstractmethod
#     def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
#         pass

#     def _get_labeler(self, config: Dict):
#         labeler_type = config.get("labeler_type", "ReturnBasedLabeler")
#         window = config.get("labeler_window", 20)
#         try:
#             labeler_class = getattr(labelers, labeler_type)
#             return labeler_class(window)
#         except AttributeError:
#             logging.error(f"指定的标签生成器 {labeler_type} 不存在，使用默认 ReturnBasedLabeler")
#             return ReturnBasedLabeler(window)

# class ScoringFeatureSelector(BaseFeatureSelector):
#     def __init__(self):
#         classes = [
#             cls for name, cls in inspect.getmembers(analyses, inspect.isclass)
#             if cls.__module__ == 'src.scoring.analyses' and not inspect.isabstract(cls)
#         ]
#         names = [cls.__name__ for cls in classes]
#         super().__init__(classes, names)

#     def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
#         enable_selection = config.get("enable_feature_selection", False)
#         feature_count = config.get("feature_count", 20)
#         manual_features = config.get("manual_features", [])

#         if not enable_selection:
#             if manual_features:
#                 valid_features = [f for f in manual_features if f in self.feature_names]
#                 invalid_features = set(manual_features) - set(valid_features)
#                 if invalid_features:
#                     logging.warning(f"以下手动指定的打分特征不存在，将被忽略: {invalid_features}")
#                 return valid_features if valid_features else self.feature_names
#             return self.feature_names

#         if feature_count <= 0 or feature_count >= len(self.feature_names):
#             return self.feature_names

#         labeler = self._get_labeler(config)
#         labels_dict = labeler.generate_labels(datasets)
#         labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]
#         logging.info(f"标签数量: {len(labels)}")  # 添加日志确认

#         # 构造 feature_scores，形状为 (n_samples, n_features)
#         feature_scores = pd.DataFrame(
#             {module.__class__.__name__: module.analyze(datasets) for module in self.modules}
#         ).T  # 初始为 (n_features, n_samples)
#         feature_scores = feature_scores.T  # 转为 (n_samples, n_features)，即 (3, 35)
#         logging.info(f"feature_scores 形状: {feature_scores.shape}")  # 添加日志确认

#         n_per_class = feature_count // 2
#         strong_labels = [1 if label == 'strong' else 0 for label in labels]
#         weak_labels = [1 if label == 'weak' else 0 for label in labels]

#         selector_strong = SelectKBest(f_regression, k=n_per_class)
#         selector_strong.fit(feature_scores, strong_labels)
#         strong_indices = selector_strong.get_support(indices=True)

#         selector_weak = SelectKBest(f_regression, k=n_per_class)
#         selector_weak.fit(feature_scores, weak_labels)
#         weak_indices = selector_weak.get_support(indices=True)

#         selected_indices = list(set(strong_indices).union(weak_indices))
#         selected_features = [self.feature_names[i] for i in selected_indices]

#         remaining = feature_count - len(selected_features)
#         if remaining > 0:
#             other_indices = [i for i in range(len(self.feature_names)) if i not in selected_indices][:remaining]
#             selected_features.extend([self.feature_names[i] for i in other_indices])

#         return selected_features

# class MLFeatureSelector(BaseFeatureSelector):
#     """ML 特征选择器，基于显著性筛选"""
#     def __init__(self):
#         classes = [
#             cls for name, cls in inspect.getmembers(features, inspect.isclass)
#             if cls.__module__ == 'src.features.features' and not inspect.isabstract(cls)
#         ]
#         names = [cls.__name__ for cls in classes]
#         super().__init__(classes, names)

#     def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
#         enable_selection = config.get("enable_feature_selection", False)
#         manual_features = config.get("manual_features", [])
#         feature_params = config.get("feature_params", {})

#         if not enable_selection and manual_features:
#             valid_features = [f for f in manual_features if f in self.feature_names]
#             invalid_features = set(manual_features) - set(valid_features)
#             if invalid_features:
#                 logging.warning(f"以下手动指定的特征不存在，将被忽略: {invalid_features}")
#             return valid_features if valid_features else self.feature_names
#         return self.feature_names  # 默认返回所有特征，后续可扩展自动选择逻辑

# # class MLFeatureSelector(BaseFeatureSelector):
# #     """ML 特征选择器，使用随机森林特征重要性筛选"""
# #     # def __init__(self):
# #     #     # 动态加载 ML 特征类
# #     #     classes = [
# #     #         cls for name, cls in inspect.getmembers(features, inspect.isclass)
# #     #         if cls.__module__ == 'src.features.features'
# #     #     ]
# #     #     names = [cls.__name__ for cls in classes]
# #     #     super().__init__(classes, names)
# #     def __init__(self):
# #         # 动态加载特征类，排除抽象类
# #         classes = [
# #             cls for name, cls in inspect.getmembers(features, inspect.isclass)
# #             if cls.__module__ == 'src.features.features' and not inspect.isabstract(cls)
# #         ]
# #         names = [cls.__name__ for cls in classes]
# #         super().__init__(classes, names)

# #     def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
# #         """筛选 ML 特征，使用树模型重要性
# #         Args:
# #             datasets: 合约数据列表
# #             config: 配置字典，含 enable_feature_selection, feature_count, manual_features, labeler_type, labeler_window
# #         Returns:
# #             List[str]: 筛选后的特征名列表
# #         """
# #         enable_selection = config.get("enable_feature_selection", False)
# #         feature_count = config.get("feature_count", 20)
# #         manual_features = config.get("manual_features", [])

# #         if not enable_selection:
# #             if manual_features:
# #                 valid_features = [f for f in manual_features if f in self.feature_names]
# #                 invalid_features = set(manual_features) - set(valid_features)
# #                 if invalid_features:
# #                     logging.warning(f"以下手动指定的 ML 特征不存在，将被忽略: {invalid_features}")
# #                 return valid_features if valid_features else self.feature_names
# #             return self.feature_names

# #         if feature_count <= 0 or feature_count >= len(self.feature_names):
# #             return self.feature_names

# #         labeler = self._get_labeler(config)
# #         labels_dict = labeler.generate_labels(datasets)
# #         labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]
# #         binary_labels = [1 if label == 'strong' else 0 for label in labels]  # 二值化：强=1，其他=0

# #         # 计算特征矩阵
# #         feature_matrix = pd.DataFrame()
# #         for i, data in enumerate(datasets):
# #             scores = {name: cls().compute(data).iloc[-1] for name, cls in zip(self.feature_names, self.feature_classes)}
# #             feature_matrix[f"contract{i+1}"] = pd.Series(scores)

# #         # 使用随机森林计算特征重要性
# #         rf = RandomForestClassifier(random_state=42)
# #         rf.fit(feature_matrix.T, binary_labels)
# #         importances = rf.feature_importances_
# #         indices = np.argsort(importances)[::-1][:feature_count]  # 按重要性排序，取前 feature_count 个
# #         selected_features = [self.feature_names[i] for i in indices]

# #         return selected_features

# class RulesFeatureSelector(BaseFeatureSelector):
#     """规则特征选择器，基于触发频率筛选"""
#     # def __init__(self):
#     #     # 动态加载规则类，排除 ExpertSystem
#     #     classes = [
#     #         cls for name, cls in inspect.getmembers(evaluator, inspect.isclass)
#     #         if cls.__module__ == 'src.rules.evaluator' and name != 'ExpertSystem'
#     #     ]
#     #     names = [cls.__name__ for cls in classes]
#     #     super().__init__(classes, names)
    
#     def __init__(self):
#         # 动态加载规则类，排除 ExpertSystem 和抽象类（如 Rule）
#         classes = [
#             cls for name, cls in inspect.getmembers(evaluator, inspect.isclass)
#             if cls.__module__ == 'src.rules.evaluator' and 
#                name != 'ExpertSystem' and 
#                not inspect.isabstract(cls)
#         ]
#         names = [cls.__name__ for cls in classes]
#         super().__init__(classes, names)

#     def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
#         """筛选规则特征，基于规则触发频率
#         Args:
#             datasets: 合约数据列表
#             config: 配置字典，含 enable_feature_selection, feature_count, manual_features, labeler_type, labeler_window
#         Returns:
#             List[str]: 筛选后的规则名列表
#         """
#         enable_selection = config.get("enable_feature_selection", False)
#         feature_count = config.get("feature_count", 13)  # 默认 13 个规则
#         manual_features = config.get("manual_features", [])

#         if not enable_selection:
#             if manual_features:
#                 valid_features = [f for f in manual_features if f in self.feature_names]
#                 invalid_features = set(manual_features) - set(valid_features)
#                 if invalid_features:
#                     logging.warning(f"以下手动指定的规则特征不存在，将被忽略: {invalid_features}")
#                 return valid_features if valid_features else self.feature_names
#             return self.feature_names

#         if feature_count <= 0 or feature_count >= len(self.feature_names):
#             return self.feature_names

#         labeler = self._get_labeler(config)
#         labels_dict = labeler.generate_labels(datasets)
#         labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]

#         # 计算规则触发频率
#         trigger_counts = {name: 0 for name in self.feature_names}
#         window = config.get("labeler_window", 20)
#         for i, data in enumerate(datasets):
#             for name, cls in zip(self.feature_names, self.feature_classes):
#                 is_met, _, _ = cls().evaluate(data.tail(window))  # 检查规则是否触发
#                 if (labels[i] == 'strong' and is_met) or (labels[i] == 'weak' and not is_met):
#                     trigger_counts[name] += 1  # 匹配强或弱时计数加一

#         # 按触发频率排序，选择前 feature_count 个
#         sorted_rules = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
#         selected_features = [rule[0] for rule in sorted_rules[:feature_count]]

#         return selected_features

# class StatsFeatureSelector(BaseFeatureSelector):
#     """统计特征选择器，基于显著性筛选"""
#     # def __init__(self):
#     #     # 动态加载统计指标类，排除 StatsEvaluator
#     #     classes = [
#     #         cls for name, cls in inspect.getmembers(stats_evaluator, inspect.isclass)
#     #         if cls.__module__ == 'src.stats.evaluator' and name != 'StatsEvaluator'
#     #     ]
#     #     names = [cls.__name__ for cls in classes]
#     #     super().__init__(classes, names)
#     def __init__(self):
#         # 动态加载统计指标类，排除 StatsEvaluator 和抽象类
#         classes = [
#             cls for name, cls in inspect.getmembers(stats_evaluator, inspect.isclass)
#             if cls.__module__ == 'src.stats.evaluator' and 
#                name != 'StatsEvaluator' and 
#                not inspect.isabstract(cls)
#         ]
#         names = [cls.__name__ for cls in classes]
#         super().__init__(classes, names)

#     def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
#         """筛选统计指标，基于显著性
#         Args:
#             datasets: 合约数据列表
#             config: 配置字典，含 enable_feature_selection, feature_count, manual_features, labeler_type, labeler_window
#         Returns:
#             List[str]: 筛选后的指标名列表
#         """
#         enable_selection = config.get("enable_feature_selection", False)
#         feature_count = config.get("feature_count", 7)  # 默认 7 个指标
#         manual_features = config.get("manual_features", [])

#         if not enable_selection:
#             if manual_features:
#                 valid_features = [f for f in manual_features if f in self.feature_names]
#                 invalid_features = set(manual_features) - set(valid_features)
#                 if invalid_features:
#                     logging.warning(f"以下手动指定的统计特征不存在，将被忽略: {invalid_features}")
#                 return valid_features if valid_features else self.feature_names
#             return self.feature_names

#         if feature_count <= 0 or feature_count >= len(self.feature_names):
#             return self.feature_names

#         labeler = self._get_labeler(config)
#         labels_dict = labeler.generate_labels(datasets)
#         labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]

#         # 计算统计指标得分矩阵
#         feature_scores = pd.DataFrame()
#         for i, data in enumerate(datasets):
#             scores = {name: cls().compute(data)[0] for name, cls in zip(self.feature_names, self.feature_classes)}
#             feature_scores[f"contract{i+1}"] = pd.Series(scores)

#         # 双向筛选
#         n_per_class = feature_count // 2
#         strong_labels = [1 if label == 'strong' else 0 for label in labels]
#         weak_labels = [1 if label == 'weak' else 0 for label in labels]

#         selector_strong = SelectKBest(f_regression, k=n_per_class)
#         selector_strong.fit(feature_scores.T, strong_labels)
#         strong_indices = selector_strong.get_support(indices=True)

#         selector_weak = SelectKBest(f_regression, k=n_per_class)
#         selector_weak.fit(feature_scores.T, weak_labels)
#         weak_indices = selector_weak.get_support(indices=True)

#         selected_indices = list(set(strong_indices).union(weak_indices))
#         selected_features = [self.feature_names[i] for i in selected_indices]

#         remaining = feature_count - len(selected_features)
#         if remaining > 0:
#             other_indices = [i for i in range(len(self.feature_names)) if i not in selected_indices][:remaining]
#             selected_features.extend([self.feature_names[i] for i in other_indices])

#         return selected_features

# # def get_feature_selector(method: str) -> BaseFeatureSelector:
# #     """特征选择器工厂函数，根据方法类型返回对应的选择器
# #     Args:
# #         method: 方法类型（如 'scoring', 'ml', 'rules', 'stats'）
# #     Returns:
# #         BaseFeatureSelector: 对应的特征选择器实例
# #     """
# #     selectors = {
# #         "scoring": ScoringFeatureSelector,
# #         "ml": MLFeatureSelector,
# #         "rules": RulesFeatureSelector,
# #         "stats": StatsFeatureSelector
# #     }
# #     selector_class = selectors.get(method.lower(), ScoringFeatureSelector)  # 默认返回 Scoring 选择器
# #     return selector_class()

# def get_feature_selector(module_type: str):
#     selector_map = {
#         "rules": RulesFeatureSelector,
#         "stats": StatsFeatureSelector,
#         "ml": MLFeatureSelector,
#         "scoring": ScoringFeatureSelector
#     }
#     selector_class = selector_map.get(module_type)
#     if not selector_class:
#         raise ValueError(f"不支持的模块类型: {module_type}")
#     return selector_class()