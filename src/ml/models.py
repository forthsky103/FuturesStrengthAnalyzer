# src/ml/models.py
"""
机器学习模型模块，定义多种分类模型及其堆叠实现。
每个模型继承自 MLModel 基类，提供 fit 和 predict 方法。
支持 scikit-learn 和其他流行的机器学习库。
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

class MLModel(ABC):
    """机器学习模型基类，定义模型训练和预测接口"""
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        训练模型
        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量（分类标签）
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率（正类概率）
        Args:
            X (pd.DataFrame): 特征矩阵
        Returns:
            np.ndarray: 预测的正类概率
        """
        pass

class RandomForestModel(MLModel):
    """随机森林分类器：基于多棵决策树的集成模型"""
    def __init__(self, **params):
        """
        初始化随机森林模型
        Args:
            **params: 可选参数，如 n_estimators（树数量，默认 100）, max_depth（最大深度）等
        """
        self.model = RandomForestClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练随机森林模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class XGBoostModel(MLModel):
    """XGBoost 分类器：基于梯度提升的高效实现"""
    def __init__(self, **params):
        """
        初始化 XGBoost 模型
        Args:
            **params: 可选参数，如 n_estimators（树数量，默认 100）, learning_rate（学习率，默认 0.3）等
        Note:
            eval_metric='logloss' 用于二分类任务，移除 use_label_encoder（1.7+ 默认禁用）
        """
        self.model = xgb.XGBClassifier(eval_metric='logloss', **params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练 XGBoost 模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class GradientBoostingModel(MLModel):
    """梯度提升决策树：scikit-learn 的经典梯度提升实现"""
    def __init__(self, **params):
        """
        初始化梯度提升模型
        Args:
            **params: 可选参数，如 n_estimators（树数量，默认 100）, learning_rate（学习率，默认 0.1）等
        """
        self.model = GradientBoostingClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练梯度提升模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class LightGBMModel(MLModel):
    """LightGBM 分类器：微软开发的轻量级梯度提升框架，优化速度和内存"""
    def __init__(self, **params):
        """
        初始化 LightGBM 模型
        Args:
            **params: 可选参数，如 n_estimators（树数量，默认 100）, learning_rate（学习率，默认 0.1）等
        """
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练 LightGBM 模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class CatBoostModel(MLModel):
    """CatBoost 分类器：专为类别特征优化的梯度提升模型"""
    def __init__(self, **params):
        """
        初始化 CatBoost 模型
        Args:
            **params: 可选参数，如 iterations（树数量，默认 1000）, learning_rate（学习率，默认自动）等
        Note:
            verbose=0 关闭训练日志输出
        """
        self.model = cat.CatBoostClassifier(verbose=0, **params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练 CatBoost 模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class LogisticRegressionModel(MLModel):
    """逻辑回归分类器：线性模型，适用于二分类任务"""
    def __init__(self, **params):
        """
        初始化逻辑回归模型
        Args:
            **params: 可选参数，如 max_iter（最大迭代次数，默认 100）, C（正则化强度，默认 1.0）等
        """
        self.model = LogisticRegression(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练逻辑回归模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class SVMModel(MLModel):
    """支持向量机分类器：基于最大间隔的非线性分类模型"""
    def __init__(self, **params):
        """
        初始化 SVM 模型
        Args:
            **params: 可选参数，如 C（正则化参数，默认 1.0）, kernel（核函数，默认 'rbf'）等
        Note:
            probability=True 启用概率估计
        """
        self.model = SVC(probability=True, **params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练 SVM 模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class AdaBoostModel(MLModel):
    """AdaBoost 分类器：自适应提升法，基于弱分类器加权"""
    def __init__(self, **params):
        """
        初始化 AdaBoost 模型
        Args:
            **params: 可选参数，如 n_estimators（弱分类器数量，默认 50）, learning_rate（学习率，默认 1.0）等
        """
        self.model = AdaBoostClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练 AdaBoost 模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class ExtraTreesModel(MLModel):
    """极端随机树分类器：随机森林的变种，决策更随机"""
    def __init__(self, **params):
        """
        初始化 ExtraTrees 模型
        Args:
            **params: 可选参数，如 n_estimators（树数量，默认 100）, max_depth（最大深度）等
        """
        self.model = ExtraTreesClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练 ExtraTrees 模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class KNeighborsModel(MLModel):
    """K 近邻分类器：基于距离的非参数模型"""
    def __init__(self, **params):
        """
        初始化 KNN 模型
        Args:
            **params: 可选参数，如 n_neighbors（邻居数，默认 5）, weights（权重，默认 'uniform'）等
        """
        self.model = KNeighborsClassifier(n_neighbors=5, **params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练 KNN 模型"""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率"""
        return self.model.predict_proba(X)[:, 1]

class StackingModel(MLModel):
    """堆叠模型：集成多个基模型，使用元模型整合预测"""
    def __init__(self, base_model_type: str = None, base_model_params: dict = {}, 
                 other_model_params: dict = {}, meta_model_params: dict = {}):
        """
        初始化堆叠模型
        Args:
            base_model_type (str): 基准模型类型（默认 'random_forest'）
            base_model_params (dict): 基准模型参数
            other_model_params (dict): 次级模型参数
            meta_model_params (dict): 元模型参数（默认 LogisticRegression）
        """
        self.all_models = {
            "random_forest": RandomForestModel(**other_model_params),
            "xgboost": XGBoostModel(**other_model_params),
            "gradient_boosting": GradientBoostingModel(**other_model_params),
            "lightgbm": LightGBMModel(**other_model_params),
            "catboost": CatBoostModel(**other_model_params),
            "svm": SVMModel(**other_model_params),
            "adaboost": AdaBoostModel(**other_model_params),
            "extra_trees": ExtraTreesModel(**other_model_params),
            "knn": KNeighborsModel(**other_model_params)
        }
        self.base_model_type = base_model_type or "random_forest"
        if self.base_model_type in self.all_models:
            self.base_model = self.all_models[self.base_model_type]
            if base_model_params:
                # 使用特定参数覆盖默认参数
                self.base_model = type(self.base_model.model)(**base_model_params)
            self.other_models = [model for key, model in self.all_models.items() if key != self.base_model_type]
        else:
            self.base_model = None
            self.other_models = list(self.all_models.values())
        self.meta_model = LogisticRegression(**meta_model_params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练堆叠模型：先训练基模型，再用基模型预测训练元模型"""
        if self.base_model:
            base_pred = self.base_model.predict(X)
            other_preds = np.column_stack([model.predict(X) for model in self.other_models])
            meta_features = np.column_stack([base_pred, other_preds])
            self.base_model.fit(X, y)
        else:
            meta_features = np.column_stack([model.predict(X) for model in self.other_models])
        self.meta_model.fit(meta_features, y)
        for model in self.other_models:
            model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类概率：基模型生成特征，元模型整合"""
        if self.base_model:
            base_pred = self.base_model.predict(X)
            other_preds = np.column_stack([model.predict(X) for model in self.other_models])
            meta_features = np.column_stack([base_pred, other_preds])
        else:
            meta_features = np.column_stack([model.predict(X) for model in self.other_models])
        return self.meta_model.predict_proba(meta_features)[:, 1]


# # src/ml/models.py
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as cat

# class MLModel:
#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         pass

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         pass

# class RandomForestModel(MLModel):
#     def __init__(self, **params):
#         self.model = RandomForestClassifier(**params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class XGBoostModel(MLModel):
#     def __init__(self, **params):
#         self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class GradientBoostingModel(MLModel):
#     def __init__(self, **params):
#         self.model = GradientBoostingClassifier(**params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class LightGBMModel(MLModel):
#     def __init__(self, **params):
#         self.model = lgb.LGBMClassifier(**params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class CatBoostModel(MLModel):
#     def __init__(self, **params):
#         self.model = cat.CatBoostClassifier(verbose=0, **params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class LogisticRegressionModel(MLModel):
#     def __init__(self, **params):
#         self.model = LogisticRegression(**params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class SVMModel(MLModel):
#     def __init__(self, **params):
#         self.model = SVC(probability=True, **params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class AdaBoostModel(MLModel):
#     def __init__(self, **params):
#         self.model = AdaBoostClassifier(**params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class ExtraTreesModel(MLModel):
#     def __init__(self, **params):
#         self.model = ExtraTreesClassifier(**params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class KNeighborsModel(MLModel):
#     def __init__(self, **params):
#         self.model = KNeighborsClassifier(n_neighbors=5, **params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         self.model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         return self.model.predict_proba(X)[:, 1]

# class StackingModel(MLModel):
#     def __init__(self, base_model_type: str = None, base_model_params: dict = {}, 
#                  other_model_params: dict = {}, meta_model_params: dict = {}):
#         self.all_models = {
#             "random_forest": RandomForestModel(**other_model_params),
#             "xgboost": XGBoostModel(**other_model_params),
#             "gradient_boosting": GradientBoostingModel(**other_model_params),
#             "lightgbm": LightGBMModel(**other_model_params),
#             "catboost": CatBoostModel(**other_model_params),
#             "svm": SVMModel(**other_model_params),
#             "adaboost": AdaBoostModel(**other_model_params),
#             "extra_trees": ExtraTreesModel(**other_model_params),
#             "knn": KNeighborsModel(**other_model_params)
#         }
#         self.base_model_type = base_model_type or "random_forest"  # 默认 random_forest
#         if self.base_model_type in self.all_models:
#             self.base_model = self.all_models[self.base_model_type]
#             if base_model_params:
#                 self.base_model = type(self.base_model.model)(**base_model_params)
#             self.other_models = [model for key, model in self.all_models.items() if key != self.base_model_type]
#         else:
#             self.base_model = None
#             self.other_models = list(self.all_models.values())
#         self.meta_model = LogisticRegression(**meta_model_params)

#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         if self.base_model:
#             base_pred = self.base_model.predict(X)
#             other_preds = np.column_stack([model.predict(X) for model in self.other_models])
#             meta_features = np.column_stack([base_pred, other_preds])
#             self.base_model.fit(X, y)
#         else:
#             meta_features = np.column_stack([model.predict(X) for model in self.other_models])
#         self.meta_model.fit(meta_features, y)
#         for model in self.other_models:
#             model.fit(X, y)

#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         if self.base_model:
#             base_pred = self.base_model.predict(X)
#             other_preds = np.column_stack([model.predict(X) for model in self.other_models])
#             meta_features = np.column_stack([base_pred, other_preds])
#         else:
#             meta_features = np.column_stack([model.predict(X) for model in self.other_models])
#         return self.meta_model.predict_proba(meta_features)[:, 1]