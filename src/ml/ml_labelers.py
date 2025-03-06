# src/ml/ml_labelers.py
from typing import List, Dict
import pandas as pd
import numpy as np

class MLReturnBasedLabeler:
    def __init__(self, window: int = 20):
        self.window = window

    def generate_labels(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        """生成每个合约的强弱标签，专为 ML 模块优化"""
        labels = {}
        if len(datasets) == 1:
            # 单数据集，使用收益率阈值
            data = datasets[0]
            returns = data['close'].pct_change(periods=self.window).dropna()
            labels['label1'] = pd.Series(
                np.where(returns > 0.01, 'strong',
                         np.where(returns < -0.01, 'weak', 'neutral')),
                index=returns.index
            )
        else:
            # 多数据集，比较收益率
            returns = [data['close'].pct_change(periods=self.window).dropna() for data in datasets]
            returns_df = pd.concat(returns, axis=1, keys=[f'return{i+1}' for i in range(len(returns))])
            if returns_df.empty:
                labels = {f'label{i+1}': pd.Series(['neutral'] * len(datasets[0]), index=datasets[0].index) 
                          for i in range(len(datasets))}
                return labels
            strongest = returns_df.idxmax(axis=1, skipna=True)
            weakest = returns_df.idxmin(axis=1, skipna=True)
            for i in range(len(datasets)):
                labels[f'label{i+1}'] = pd.Series(
                    np.where(strongest == f'return{i+1}', 'strong',
                             np.where(weakest == f'return{i+1}', 'weak', 'neutral')),
                    index=returns_df.index
                )
        return labels