# src/deeplearning/evaluator.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Flatten
import os
from ..features.extractor import FeatureExtractor

class DLModel(ABC):
    """深度学习模型基类"""
    @abstractmethod
    def build_model(self, window: int, features: int) -> Sequential:
        """构建模型"""
        pass

    @abstractmethod
    def predict(self, model: Sequential, X: np.ndarray) -> float:
        """进行预测，返回强弱概率"""
        pass

class LSTMModel(DLModel):
    def build_model(self, window: int, features: int) -> Sequential:
        model = Sequential([
            LSTM(20, input_shape=(window, features), return_sequences=True),
            LSTM(10),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def predict(self, model: Sequential, X: np.ndarray) -> float:
        return model.predict(X, verbose=0)[0][0]

class GRUModel(DLModel):
    def build_model(self, window: int, features: int) -> Sequential:
        model = Sequential([
            GRU(20, input_shape=(window, features), return_sequences=True),
            GRU(10),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def predict(self, model: Sequential, X: np.ndarray) -> float:
        return model.predict(X, verbose=0)[0][0]

class CNNModel(DLModel):
    def build_model(self, window: int, features: int) -> Sequential:
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(window, features)),
            Conv1D(16, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def predict(self, model: Sequential, X: np.ndarray) -> float:
        return model.predict(X, verbose=0)[0][0]

class DLEvaluator:
    """深度学习评估器"""
    def __init__(self, model: DLModel, feature_extractor: FeatureExtractor, window: int = 50, 
                 model_path: str = None, epochs: int = 5):
        self.model = model
        self.feature_extractor = feature_extractor
        self.window = window
        self.model_path = model_path or f"../models/dl_model_{id(self)}"
        self.epochs = epochs
        self.trained_models = {}

    def evaluate(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        """评估每个合约的强弱"""
        scores = {}
        features_df = self.feature_extractor.extract_features(datasets, include_labels=False)
        feature_cols = [col for col in features_df.columns if not col.startswith('label')]

        for i, data in enumerate(datasets):
            # 提取该合约的特征列
            contract_features = [col for col in feature_cols if col.endswith(str(i+1))]
            series = features_df[contract_features].tail(self.window + 1)
            X = series[:-1].values.reshape(1, self.window, len(contract_features))
            y = np.array([1 if data['close'].iloc[-1] > data['close'].iloc[-2] else 0])  # 简单标签

            # 检查是否已有预训练模型
            contract_key = f"contract{i+1}"
            model_file = f"{self.model_path}_{contract_key}.h5"
            if os.path.exists(model_file):
                dl_model = tf.keras.models.load_model(model_file)
            else:
                dl_model = self.model.build_model(self.window, len(contract_features))
                dl_model.fit(X, y, epochs=self.epochs, verbose=0)
                dl_model.save(model_file)
                print(f"模型已保存至: {model_file}")

            # 预测
            score = self.model.predict(dl_model, X)
            scores[contract_key] = score
            self.trained_models[contract_key] = dl_model

        return scores

    def save_model(self, contract: str):
        """保存特定合约的模型"""
        if contract in self.trained_models:
            self.trained_models[contract].save(f"{self.model_path}_{contract}.h5")
            print(f"模型已保存至: {self.model_path}_{contract}.h5")

    def load_model(self, contract: str) -> Sequential:
        """加载特定合约的模型"""
        model_file = f"{self.model_path}_{contract}.h5"
        if os.path.exists(model_file):
            return tf.keras.models.load_model(model_file)
        raise FileNotFoundError(f"模型文件未找到: {model_file}")