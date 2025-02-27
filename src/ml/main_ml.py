# src/ml/main_ml.py
import json
import os
import pandas as pd
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from ..utils.logging_utils import setup_logging  # 日志工具
from ..utils.data_processor import DataProcessor  # 数据处理工具
from ..features.extractor import FeatureExtractor  # 特征提取器
from ..features import features  # ML 特征模块
from ..features.labelers import ReturnBasedLabeler  # 标签生成器
from .predictor import MLPredictor  # ML 预测器
from .models import (
    RandomForestModel, XGBoostModel, GradientBoostingModel, LightGBMModel, CatBoostModel,
    LogisticRegressionModel, SVMModel, AdaBoostModel, ExtraTreesModel, KNeighborsModel,
    StackingModel
)  # ML 模型类
from ..utils.recommender import MLTradeRecommender  # 推荐器
from ..utils.feature_selector import get_feature_selector  # 特征选择器工厂

def generate_ml_results(raw_results: Dict[str, float], auc: float, recommend_mode: str, 
                        auc_threshold: float = 0.7, prob_weight: float = 0.8, auc_weight: float = 0.2) -> Dict[str, str]:
    """根据原始预测结果生成强弱标签
    Args:
        raw_results (Dict[str, float]): 原始预测概率
        auc (float): AUC 分数
        recommend_mode (str): 推荐模式
        auc_threshold (float): AUC 阈值
        prob_weight (float): 概率权重
        auc_weight (float): AUC 权重
    Returns:
        Dict[str, str]: 合约到强弱标签的映射
    """
    if recommend_mode == "probability":
        max_contract = max(raw_results, key=raw_results.get)
        min_contract = min(raw_results, key=raw_results.get)
        results = {}
        for contract in raw_results:
            if contract == max_contract:
                results[contract] = 'strong'
            elif contract == min_contract:
                results[contract] = 'weak'
            else:
                results[contract] = 'neutral'
        return results
    else:
        raise ValueError(f"当前仅支持 'probability' 模式进行三组以上数据比对")

def main_ml(global_config_path: str = "../config.json", ml_config_path: str = "ml_config.json"):
    """运行 ML 模块，判断合约强弱
    Args:
        global_config_path (str): 全局配置文件路径，包含数据组和市场方向
        ml_config_path (str): ML 模块配置文件路径，包含特征、模型和筛选参数
    """
    # 设置日志
    setup_logging(log_file_path="../../results/ml_log.log", level=logging.INFO)
    logging.info("开始运行 ML 方法")

    # 加载全局配置
    with open(global_config_path, "r") as f:
        global_config = json.load(f)
    # 加载 ML 配置
    with open(ml_config_path, "r") as f:
        ml_config = json.load(f)

    # 检查必要配置项
    if "data_groups" not in global_config:
        raise ValueError("全局配置缺少 'data_groups'")
    if "features" not in ml_config:
        raise ValueError("ML 配置缺少 'features'")

    # 创建结果和模型目录
    os.makedirs("../../results", exist_ok=True)
    os.makedirs("../../models", exist_ok=True)

    # 模型映射，用于实例化
    model_map = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "gradient_boosting": GradientBoostingModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "logistic_regression": LogisticRegressionModel,
        "svm": SVMModel,
        "adaboost": AdaBoostModel,
        "extra_trees": ExtraTreesModel,
        "knn": KNeighborsModel,
        "stacking": lambda: StackingModel(ml_config.get("base_model", None))
    }

    # 遍历数据组
    for group_idx, data_files in enumerate(global_config["data_groups"]):
        logging.info(f"处理第 {group_idx + 1} 组数据: {data_files}")
        # 数据预处理
        processors = [DataProcessor(f"../../data/{path}") for path in data_files]
        datasets = [p.clean_data() for p in processors]
        symbols = [path.split('.')[0] for path in data_files]
        # 时间对齐
        for i, data in enumerate(datasets):
            data.set_index('date', inplace=True)
        common_index = datasets[0].index
        for data in datasets[1:]:
            common_index = common_index.intersection(data.index)
        datasets = [data.loc[common_index].reset_index() for data in datasets]
        logging.info("数据加载完成，时间对齐至重叠期")

        # 获取 ML 特征选择器并筛选特征
        selector = get_feature_selector("ml")
        selected_feature_names = selector.select_features(datasets, ml_config)
        logging.info(f"特征筛选完成，选出 {len(selected_feature_names)} 个特征: {selected_feature_names}")

        # 动态实例化特征对象
        selected_features = [getattr(features, name)(ml_config.get("window", 20)) for name in selected_feature_names]
        extractor = FeatureExtractor(selected_features)
        labeler = ReturnBasedLabeler(20)
        features_df = extractor.extract_features(datasets, labeler)
        feature_cols = [col for col in features_df.columns if not col.startswith('label')]
        logging.info(f"提取特征完成，共 {len(feature_cols)} 个特征")

        # 获取模型类型并初始化
        model_type = ml_config.get("model_type", "random_forest")
        base_model_type = ml_config.get("base_model", None) if model_type == "stacking" else None
        if model_type not in model_map:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        models = [model_map[model_type]() for _ in datasets]
        logging.info(f"使用 {model_type.replace('_', ' ').title()} 模型" + 
                     (f"（Base Model: {base_model_type}）" if base_model_type else ""))

        # 训练和预测
        predictor = MLPredictor(models, feature_cols)
        predictor.train(features_df)
        logging.info("模型训练完成")
        
        # 计算评估指标（以第一个合约为例）
        y_true = features_df[f"label_{symbols[0]}"].map({'strong': 1, 'weak': 0, 'neutral': 0}).values
        raw_results = predictor.predict(features_df)
        y_pred_proba = np.array([raw_results[f"contract_{i+1}"] for i in range(len(datasets))]).T[0]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(y_true, y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logging.info("预测完成")

        # 输出预测结果和评估指标
        print(f"\n=== {model_type.replace('_', ' ').title()} 模型预测结果 ===" + 
              (f"（Base Model: {base_model_type}）" if base_model_type else ""))
        print("合约\t\t概率")
        print("-" * 30)
        for contract, value in raw_results.items():
            symbol = symbols[int(contract[-1]) - 1]
            print(f"{symbol:<15} {value:.4f}")
        print("-" * 30)
        print("评估指标（以第一个合约为例）:")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 30)

        # 生成强弱标签和交易建议
        recommend_mode = ml_config.get("recommend_mode", "probability")
        results = generate_ml_results(
            raw_results=raw_results,
            auc=auc,
            recommend_mode=recommend_mode,
            auc_threshold=ml_config.get("auc_threshold", 0.7),
            prob_weight=ml_config.get("prob_weight", 0.8),
            auc_weight=ml_config.get("auc_weight", 0.2)
        )
        
        recommender = MLTradeRecommender(global_config["market_direction"])
        advice = recommender.recommend(results, "ml", symbols, group_idx, datasets)

        print(f"交易建议模式: {recommend_mode.replace('_', ' ').title()}")
        print(f"交易建议: {advice}")
        logging.info(f"交易建议生成（模式: {recommend_mode}）: {advice}")
        logging.info(f"评估指标 - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                     f"Recall: {recall:.4f}, F1: {f1:.4f}")

    logging.info("ML 方法运行完成")

if __name__ == "__main__":
    main_ml()