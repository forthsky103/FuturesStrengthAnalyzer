# src/utils/feature_selector.py
from typing import List, Dict
import pandas as pd
import inspect
from abc import ABC, abstractmethod
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
import src.features.labelers as labelers  # 绝对导入 labelers 模块，用于动态加载 Labeler 类
from src.features.labelers import ReturnBasedLabeler  # 默认 Labeler 类，作为回退选项
from ..scoring import analyses  # 打分法特征模块
from ..features import features  # ML 特征模块
from ..rules import evaluator  # 规则模块
from ..stats import evaluator as stats_evaluator  # 统计模块
import logging
import numpy as np

class BaseFeatureSelector(ABC):
    """特征选择器基类，提供通用接口和共享逻辑"""
    def __init__(self, feature_classes: List, feature_names: List[str]):
        """
        初始化基类，加载特征类和名称
        Args:
            feature_classes (List): 特征或规则的类列表，用于实例化
            feature_names (List[str]): 特征或规则的名称列表，用于标识
        """
        self.feature_classes = feature_classes  # 存储特征类，用于动态实例化
        self.feature_names = feature_names  # 存储特征名称，用于筛选和返回

    @abstractmethod
    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        """抽象方法，子类必须实现，筛选特征
        Args:
            datasets: 合约数据列表
            config: 配置字典，包含筛选参数
        Returns:
            List[str]: 筛选后的特征名称列表
        """
        pass

    def _get_labeler(self, config: Dict):
        """动态加载 Labeler 类，根据配置选择标签生成器
        Args:
            config: 配置字典，含 labeler_type 和 labeler_window
        Returns:
            Labeler: 实例化的标签生成器对象
        """
        labeler_type = config.get("labeler_type", "ReturnBasedLabeler")  # 默认使用 ReturnBasedLabeler
        window = config.get("labeler_window", 20)  # 默认窗口 20 天
        try:
            labeler_class = getattr(labelers, labeler_type)  # 从 labelers 模块动态获取类
            return labeler_class(window)  # 实例化并返回
        except AttributeError:
            logging.error(f"指定的标签生成器 {labeler_type} 不存在，使用默认 ReturnBasedLabeler")
            return ReturnBasedLabeler(window)  # 回退到默认 Labeler

class ScoringFeatureSelector(BaseFeatureSelector):
    """打分法特征选择器，使用 f_regression 双向筛选"""
    def __init__(self):
        # 动态加载打分法特征类
        classes = [
            cls for name, cls in inspect.getmembers(analyses, inspect.isclass)
            if cls.__module__ == 'src.scoring.analyses'  # 只加载 analyses.py 中的类
        ]
        names = [cls.__name__ for cls in classes]  # 获取类名
        super().__init__(classes, names)  # 初始化基类

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        """筛选打分法特征，优化最强和最弱合约识别
        Args:
            datasets: 合约数据列表
            config: 配置字典，含 enable_feature_selection, feature_count, manual_features, labeler_type, labeler_window
        Returns:
            List[str]: 筛选后的特征名列表
        """
        enable_selection = config.get("enable_feature_selection", False)  # 是否启用自动筛选
        feature_count = config.get("feature_count", 20)  # 默认筛选 20 个
        manual_features = config.get("manual_features", [])  # 人工选择特征

        # 不启用自动筛选
        if not enable_selection:
            if manual_features:  # 如果指定了人工选择特征
                valid_features = [f for f in manual_features if f in self.feature_names]
                invalid_features = set(manual_features) - set(valid_features)
                if invalid_features:
                    logging.warning(f"以下手动指定的打分特征不存在，将被忽略: {invalid_features}")
                return valid_features if valid_features else self.feature_names  # 返回有效特征或全部
            return self.feature_names  # 默认返回全部特征

        # 检查 feature_count 是否有效
        if feature_count <= 0 or feature_count >= len(self.feature_names):
            return self.feature_names

        # 获取标签生成器并生成标签
        labeler = self._get_labeler(config)
        labels_dict = labeler.generate_labels(datasets)
        labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]  # 取最新标签

        # 计算特征得分矩阵
        feature_scores = pd.DataFrame()
        for i, data in enumerate(datasets):
            scores = {name: cls().evaluate(data) for name, cls in zip(self.feature_names, self.feature_classes)}
            feature_scores[f"contract{i+1}"] = pd.Series(scores)

        # 双向筛选：分别筛选强和弱相关特征
        n_per_class = feature_count // 2  # 每类选一半（如 10 强 + 10 弱）
        strong_labels = [1 if label == 'strong' else 0 for label in labels]  # 二值化：强=1，其他=0
        weak_labels = [1 if label == 'weak' else 0 for label in labels]  # 二值化：弱=1，其他=0

        # 筛选与“强”相关的特征
        selector_strong = SelectKBest(f_regression, k=n_per_class)
        selector_strong.fit(feature_scores.T, strong_labels)
        strong_indices = selector_strong.get_support(indices=True)

        # 筛选与“弱”相关的特征
        selector_weak = SelectKBest(f_regression, k=n_per_class)
        selector_weak.fit(feature_scores.T, weak_labels)
        weak_indices = selector_weak.get_support(indices=True)

        # 合并并去重
        selected_indices = list(set(strong_indices).union(weak_indices))
        selected_features = [self.feature_names[i] for i in selected_indices]

        # 若少于 feature_count，补齐
        remaining = feature_count - len(selected_features)
        if remaining > 0:
            other_indices = [i for i in range(len(self.feature_names)) if i not in selected_indices][:remaining]
            selected_features.extend([self.feature_names[i] for i in other_indices])

        return selected_features

class MLFeatureSelector(BaseFeatureSelector):
    """ML 特征选择器，使用随机森林特征重要性筛选"""
    def __init__(self):
        # 动态加载 ML 特征类
        classes = [
            cls for name, cls in inspect.getmembers(features, inspect.isclass)
            if cls.__module__ == 'src.features.features'
        ]
        names = [cls.__name__ for cls in classes]
        super().__init__(classes, names)

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        """筛选 ML 特征，使用树模型重要性
        Args:
            datasets: 合约数据列表
            config: 配置字典，含 enable_feature_selection, feature_count, manual_features, labeler_type, labeler_window
        Returns:
            List[str]: 筛选后的特征名列表
        """
        enable_selection = config.get("enable_feature_selection", False)
        feature_count = config.get("feature_count", 20)
        manual_features = config.get("manual_features", [])

        if not enable_selection:
            if manual_features:
                valid_features = [f for f in manual_features if f in self.feature_names]
                invalid_features = set(manual_features) - set(valid_features)
                if invalid_features:
                    logging.warning(f"以下手动指定的 ML 特征不存在，将被忽略: {invalid_features}")
                return valid_features if valid_features else self.feature_names
            return self.feature_names

        if feature_count <= 0 or feature_count >= len(self.feature_names):
            return self.feature_names

        labeler = self._get_labeler(config)
        labels_dict = labeler.generate_labels(datasets)
        labels = [labels_dict[f'label{i+1}'].iloc[-1] for i in range(len(datasets))]
        binary_labels = [1 if label == 'strong' else 0 for label in labels]  # 二值化：强=1，其他=0

        # 计算特征矩阵
        feature_matrix = pd.DataFrame()
        for i, data in enumerate(datasets):
            scores = {name: cls().compute(data).iloc[-1] for name, cls in zip(self.feature_names, self.feature_classes)}
            feature_matrix[f"contract{i+1}"] = pd.Series(scores)

        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(random_state=42)
        rf.fit(feature_matrix.T, binary_labels)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:feature_count]  # 按重要性排序，取前 feature_count 个
        selected_features = [self.feature_names[i] for i in indices]

        return selected_features

class RulesFeatureSelector(BaseFeatureSelector):
    """规则特征选择器，基于触发频率筛选"""
    def __init__(self):
        # 动态加载规则类，排除 ExpertSystem
        classes = [
            cls for name, cls in inspect.getmembers(evaluator, inspect.isclass)
            if cls.__module__ == 'src.rules.evaluator' and name != 'ExpertSystem'
        ]
        names = [cls.__name__ for cls in classes]
        super().__init__(classes, names)

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        """筛选规则特征，基于规则触发频率
        Args:
            datasets: 合约数据列表
            config: 配置字典，含 enable_feature_selection, feature_count, manual_features, labeler_type, labeler_window
        Returns:
            List[str]: 筛选后的规则名列表
        """
        enable_selection = config.get("enable_feature_selection", False)
        feature_count = config.get("feature_count", 13)  # 默认 13 个规则
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

        # 计算规则触发频率
        trigger_counts = {name: 0 for name in self.feature_names}
        window = config.get("labeler_window", 20)
        for i, data in enumerate(datasets):
            for name, cls in zip(self.feature_names, self.feature_classes):
                is_met, _, _ = cls().evaluate(data.tail(window))  # 检查规则是否触发
                if (labels[i] == 'strong' and is_met) or (labels[i] == 'weak' and not is_met):
                    trigger_counts[name] += 1  # 匹配强或弱时计数加一

        # 按触发频率排序，选择前 feature_count 个
        sorted_rules = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        selected_features = [rule[0] for rule in sorted_rules[:feature_count]]

        return selected_features

class StatsFeatureSelector(BaseFeatureSelector):
    """统计特征选择器，基于显著性筛选"""
    def __init__(self):
        # 动态加载统计指标类，排除 StatsEvaluator
        classes = [
            cls for name, cls in inspect.getmembers(stats_evaluator, inspect.isclass)
            if cls.__module__ == 'src.stats.evaluator' and name != 'StatsEvaluator'
        ]
        names = [cls.__name__ for cls in classes]
        super().__init__(classes, names)

    def select_features(self, datasets: List[pd.DataFrame], config: Dict) -> List[str]:
        """筛选统计指标，基于显著性
        Args:
            datasets: 合约数据列表
            config: 配置字典，含 enable_feature_selection, feature_count, manual_features, labeler_type, labeler_window
        Returns:
            List[str]: 筛选后的指标名列表
        """
        enable_selection = config.get("enable_feature_selection", False)
        feature_count = config.get("feature_count", 7)  # 默认 7 个指标
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

        # 计算统计指标得分矩阵
        feature_scores = pd.DataFrame()
        for i, data in enumerate(datasets):
            scores = {name: cls().compute(data)[0] for name, cls in zip(self.feature_names, self.feature_classes)}
            feature_scores[f"contract{i+1}"] = pd.Series(scores)

        # 双向筛选
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

def get_feature_selector(method: str) -> BaseFeatureSelector:
    """特征选择器工厂函数，根据方法类型返回对应的选择器
    Args:
        method: 方法类型（如 'scoring', 'ml', 'rules', 'stats'）
    Returns:
        BaseFeatureSelector: 对应的特征选择器实例
    """
    selectors = {
        "scoring": ScoringFeatureSelector,
        "ml": MLFeatureSelector,
        "rules": RulesFeatureSelector,
        "stats": StatsFeatureSelector
    }
    selector_class = selectors.get(method.lower(), ScoringFeatureSelector)  # 默认返回 Scoring 选择器
    return selector_class()