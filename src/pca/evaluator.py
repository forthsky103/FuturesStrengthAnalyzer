# src/pca/evaluator.py
from typing import List, Dict
import pandas as pd
from sklearn.decomposition import PCA

def evaluate(datasets: List[pd.DataFrame], window: int = 20) -> Dict[str, float]:
    """PCA方法：基于多维特征的主成分贡献判断强弱"""
    features = pd.concat([data.tail(window)[['close', 'volume', 'position']] for data in datasets], 
                        axis=1, keys=[f'contract{i+1}' for i in range(len(datasets))])
    pca = PCA(n_components=1)
    pca.fit(features.dropna())
    scores = pca.transform(features.dropna())[:, 0]  # 第一主成分得分
    return {f"contract{i+1}": scores[-1] for i in range(len(datasets))}  # 取最新得分