# src/features/extractor.py
import pandas as pd
import numpy as np
from typing import List, Optional
from .features import Feature

class FeatureExtractor:
    def __init__(self, features: List[Feature]):
        self.features = features

    def extract_features(self, datasets: List[pd.DataFrame], labeler: Optional['Labeler'] = None) -> pd.DataFrame:
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
        
        # 如果提供了标签生成器，则生成标签
        if labeler:
            labels = labeler.generate_labels(aligned_datasets)
            features_df.update(labels)
        
        return features_df.dropna()