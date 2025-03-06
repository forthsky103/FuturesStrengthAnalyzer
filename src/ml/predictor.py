# src/ml/predictor.py
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .models import MLModel
import pandas as pd
import numpy as np
import logging

class MLPredictor:
    """ML 预测器，支持单标签列和多合约预测"""
    def __init__(self, models: List[MLModel], feature_cols: List[str], symbols: List[str]):
        self.models = models
        self.feature_cols = feature_cols
        self.symbols = symbols
        self.test_preds = {}  # 保存测试集预测

    def train(self, features: pd.DataFrame):
        """训练模型，使用每个合约的独立标签列"""
        X = features[self.feature_cols]
        for i, (model, symbol) in enumerate(zip(self.models, self.symbols)):
            y = features[f"label_{symbol}"].map({'strong': 1, 'weak': 0, 'neutral': 0})
            unique_labels = y.unique()
            logging.debug(f"Contract {symbol} 标签分布: {y.value_counts().to_dict()}")
            if len(unique_labels) < 2:
                logging.warning(f"Contract {symbol} 的标签只有单一类别 {unique_labels}, 跳过训练")
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Contract {symbol} 准确率: {accuracy:.2f}")
            self.test_preds[symbol] = {'y_test': y_test, 'y_pred_proba': y_pred_proba}

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """预测每个合约的概率，返回最后一行预测"""
        X = features[self.feature_cols]
        preds = {f"contract_{i+1}": model.predict(X) for i, model in enumerate(self.models)}
        return {key: pred[-1] for key, pred in preds.items()}

    def predict_proba_full(self, features: pd.DataFrame) -> np.ndarray:
        """预测全数据的概率，用于评估"""
        X = features[self.feature_cols]
        return np.array([model.predict(X) for model in self.models]).T  # [n_samples, n_contracts]

    def get_test_predictions(self, symbol: str) -> Dict[str, np.ndarray]:
        """获取指定合约的测试集预测"""
        return self.test_preds.get(symbol, {})



# # src/ml/predictor.py
# from typing import List, Dict
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from .models import MLModel
# import pandas as pd
# import numpy as np
# import logging

# class MLPredictor:
#     """ML 预测器，支持单标签列和多合约预测"""
#     def __init__(self, models: List[MLModel], feature_cols: List[str], symbols: List[str]):
#         self.models = models
#         self.feature_cols = feature_cols
#         self.symbols = symbols

#     def train(self, features: pd.DataFrame):
#         """训练模型，使用每个合约的独立标签列"""
#         X = features[self.feature_cols]
#         for i, (model, symbol) in enumerate(zip(self.models, self.symbols)):
#             y = features[f"label_{symbol}"].map({'strong': 1, 'weak': 0, 'neutral': 0})
#             unique_labels = y.unique()
#             if len(unique_labels) < 2:
#                 logging.warning(f"Contract {symbol} 的标签只有单一类别 {unique_labels}, 跳过训练")
#                 continue
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred_proba = model.predict(X_test)
#             y_pred = (y_pred_proba > 0.5).astype(int)
#             accuracy = accuracy_score(y_test, y_pred)
#             print(f"Contract {symbol} 准确率: {accuracy:.2f}")

#     def predict(self, features: pd.DataFrame) -> Dict[str, float]:
#         """预测每个合约的概率，返回最后一行预测"""
#         X = features[self.feature_cols]
#         # 全数据预测
#         preds = {f"contract_{i+1}": model.predict(X) for i, model in enumerate(self.models)}
#         # 返回最后一行预测用于建议
#         return {key: pred[-1] for key, pred in preds.items()}

#     def predict_proba_full(self, features: pd.DataFrame) -> np.ndarray:
#         """预测全数据的概率，用于评估"""
#         X = features[self.feature_cols]
#         return np.array([model.predict(X) for model in self.models]).T  # [n_samples, n_contracts]