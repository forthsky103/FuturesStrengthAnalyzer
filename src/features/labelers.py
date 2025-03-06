# src/features/labelers.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np

class Labeler(ABC):
    @abstractmethod
    def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        """生成每个合约的强弱标签，返回字典"""
        pass

class ReturnBasedLabeler(Labeler):
    def __init__(self, window: int = 20):
        self.window = window

    def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        returns = [data['close'].pct_change(periods=self.window).dropna() for data in datasets]
        returns_df = pd.concat(returns, axis=1, keys=[f'return{i+1}' for i in range(len(returns))])
        # 如果 returns_df 为空，返回默认标签
        if returns_df.empty:
            labels = {f'label{i+1}': pd.Series(['neutral'] * len(datasets[0]), index=datasets[0].index) for i in range(len(datasets))}
            return labels
        strongest = returns_df.idxmax(axis=1, skipna=True)
        weakest = returns_df.idxmin(axis=1, skipna=True)
        labels = {}
        for i in range(len(datasets)):
            labels[f'label{i+1}'] = pd.Series(
                np.where(
                    strongest == f'return{i+1}', 'strong',
                    np.where(weakest == f'return{i+1}', 'weak', 'neutral')
                ),
                index=returns_df.index
            )
        return labels

# class ReturnBasedLabeler(Labeler):
#     def __init__(self, window: int = 20):
#         self.window = window

#     def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
#         returns = [data['close'].pct_change(periods=self.window) for data in datasets]
#         returns_df = pd.concat(returns, axis=1, keys=[f'return{i+1}' for i in range(len(returns))])
#         strongest = returns_df.idxmax(axis=1)
#         weakest = returns_df.idxmin(axis=1)
#         labels = {}
#         for i in range(len(datasets)):
#             labels[f'label{i+1}'] = pd.Series(
#                 np.where(
#                     strongest == f'return{i+1}', 'strong',
#                     np.where(weakest == f'return{i+1}', 'weak', 'neutral')
#                 ),
#                 index=returns_df.index
#             )
#         return labels

# VolumeBasedLabeler 和 VolatilityBasedLabeler 类似调整
class VolumeBasedLabeler(Labeler):
    def __init__(self, window: int = 20):
        self.window = window

    def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        volumes = [data['volume'].rolling(window=self.window).mean() for data in datasets]
        volumes_df = pd.concat(volumes, axis=1, keys=[f'vol{i+1}' for i in range(len(volumes))])
        strongest = volumes_df.idxmax(axis=1)
        weakest = volumes_df.idxmin(axis=1)
        labels = {}
        for i in range(len(datasets)):
            labels[f'label{i+1}'] = pd.Series(
                np.where(
                    strongest == f'vol{i+1}', 'strong',
                    np.where(weakest == f'vol{i+1}', 'weak', 'neutral')
                ),
                index=volumes_df.index
            )
        return labels

class VolatilityBasedLabeler(Labeler):
    def __init__(self, window: int = 20):
        self.window = window

    def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        volatilities = [(data['high'] - data['low']).rolling(window=self.window).mean() for data in datasets]
        vol_df = pd.concat(volatilities, axis=1, keys=[f'volatility{i+1}' for i in range(len(volatilities))])
        strongest = vol_df.idxmax(axis=1)
        weakest = vol_df.idxmin(axis=1)
        labels = {}
        for i in range(len(datasets)):
            labels[f'label{i+1}'] = pd.Series(
                np.where(
                    strongest == f'volatility{i+1}', 'strong',
                    np.where(weakest == f'volatility{i+1}', 'weak', 'neutral')
                ),
                index=vol_df.index
            )
        return labels

# # src/features/labelers.py
# from abc import ABC, abstractmethod
# from typing import List, Dict
# import pandas as pd
# import numpy as np

# class Labeler(ABC):
#     @abstractmethod
#     def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
#         """生成每个合约的强弱标签，返回字典"""
#         pass

# # 基于收益率的标签生成器（默认策略）
# class ReturnBasedLabeler(Labeler):
#     def __init__(self, window: int = 20):
#         self.window = window

#     def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
#         returns = [data['close'].pct_change(periods=self.window) for data in datasets]
#         returns_df = pd.concat(returns, axis=1, keys=[f'return{i+1}' for i in range(len(returns))])
#         strongest = returns_df.idxmax(axis=1)
#         weakest = returns_df.idxmin(axis=1)
#         labels = {}
#         for i in range(len(datasets)):
#             labels[f'label{i+1}'] = np.where(
#                 strongest == f'return{i+1}', 'strong',
#                 np.where(weakest == f'return{i+1}', 'weak', 'neutral')
#             )
#         return labels

# # 基于成交量的标签生成器
# class VolumeBasedLabeler(Labeler):
#     def __init__(self, window: int = 20):
#         self.window = window

#     def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
#         volumes = [data['volume'].rolling(window=self.window).mean() for data in datasets]
#         volumes_df = pd.concat(volumes, axis=1, keys=[f'vol{i+1}' for i in range(len(volumes))])
#         strongest = volumes_df.idxmax(axis=1)
#         weakest = volumes_df.idxmin(axis=1)
#         labels = {}
#         for i in range(len(datasets)):
#             labels[f'label{i+1}'] = np.where(
#                 strongest == f'vol{i+1}', 'strong',
#                 np.where(weakest == f'vol{i+1}', 'weak', 'neutral')
#             )
#         return labels

# # 基于波动率的标签生成器
# class VolatilityBasedLabeler(Labeler):
#     def __init__(self, window: int = 20):
#         self.window = window

#     def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
#         volatilities = [(data['high'] - data['low']).rolling(window=self.window).mean() for data in datasets]
#         vol_df = pd.concat(volatilities, axis=1, keys=[f'volatility{i+1}' for i in range(len(volatilities))])
#         strongest = vol_df.idxmax(axis=1)
#         weakest = vol_df.idxmin(axis=1)
#         labels = {}
#         for i in range(len(datasets)):
#             labels[f'label{i+1}'] = np.where(
#                 strongest == f'volatility{i+1}', 'strong',
#                 np.where(weakest == f'volatility{i+1}', 'weak', 'neutral')
#             )
#         return labels