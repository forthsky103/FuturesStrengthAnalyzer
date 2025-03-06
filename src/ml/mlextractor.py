# src/ml/mlextractor.py
import pandas as pd
from typing import List, Optional
from ..features.features import Feature
import logging

class MLFeatureExtractor:
    """ML 模块专用的单合约特征提取器"""
    def __init__(self, features: List[Feature], symbol: str):
        """初始化提取器
        Args:
            features (List[Feature]): 特征对象列表
            symbol (str): 合约标识符（如 'rb2510'）
        """
        self.features = features
        self.symbol = symbol

    def extract_features(self, data: pd.DataFrame, labeler=None) -> pd.DataFrame:
        """提取单合约特征
        Args:
            data (pd.DataFrame): 单合约数据
            labeler (Optional[Labeler]): 标签生成器，默认 None
        Returns:
            pd.DataFrame: 特征和标签的 DataFrame
        """
        # 提取特征并确保返回 Series
        feature_outputs = []
        for feature in self.features:
            output = feature.compute(data)
            if isinstance(output, pd.DataFrame):
                logging.warning(f"特征 {feature.name} 返回 DataFrame，取第一列")
                feature_outputs.append(output.iloc[:, 0])
            elif isinstance(output, pd.Series):
                feature_outputs.append(output)
            else:
                raise ValueError(f"特征 {feature.name} 返回非 Series/DataFrame 类型: {type(output)}")

        features_df = pd.concat(feature_outputs, axis=1)
        # 为特征列名添加 symbol 后缀
        features_df.columns = [f"{feature.name}_{self.symbol}" for feature in self.features]

        # 添加标签
        if labeler:
            labels = labeler.generate_labels([data])
            label_series = next(iter(labels.values()))  # 取第一个标签值（label1）
            features_df[f'label_{self.symbol}'] = label_series.reindex(features_df.index)

        # 验证列类型
        for col in features_df.columns:
            if not isinstance(features_df[col], pd.Series):
                logging.error(f"列 {col} 不是 Series 类型: {type(features_df[col])}")
                raise ValueError(f"特征提取结果包含非 Series 列: {col}")

        logging.debug(f"features_df 列类型: {features_df.dtypes}")
        return features_df.dropna()