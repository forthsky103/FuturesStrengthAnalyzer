import pandas as pd
import numpy as np
from typing import List
from .features import Feature

class FeatureExtractor:
    def __init__(self, features: List[Feature]):
        self.features = features

    def extract_features(self, datasets: List[pd.DataFrame], include_labels: bool = True) -> pd.DataFrame:
        # 对齐所有合约的时间索引
        common_index = datasets[0].index
        for data in datasets[1:]:
            common_index = common_index.intersection(data.index)
        aligned_datasets = [data.loc[common_index].reset_index() for data in datasets]
        
        # 提取特征
        feature_dict = {}
        for feature in self.features:
            feature_dict.update(feature.compute(aligned_datasets))
        
        features_df = pd.DataFrame(feature_dict, index=common_index)
        
        # 生成标签：基于收益率排序
        if include_labels:
            returns = [data['close'].pct_change(periods=20) for data in aligned_datasets]
            returns_df = pd.concat(returns, axis=1, keys=[f'return{i+1}' for i in range(len(returns))])
            strongest = returns_df.idxmax(axis=1)
            weakest = returns_df.idxmin(axis=1)
            for i in range(len(datasets)):
                features_df[f'label{i+1}'] = np.where(
                    strongest == f'return{i+1}', 'strong',
                    np.where(weakest == f'return{i+1}', 'weak', 'neutral')
                )
        
        return features_df.dropna()