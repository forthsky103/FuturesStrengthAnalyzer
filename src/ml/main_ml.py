# src/ml/main_ml.py
import json
import os
import pandas as pd
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from ..logging_utils import setup_logging
from ..data_processor import DataProcessor
from ..features.extractor import FeatureExtractor
from ..features.features import (
    PriceFeature, VolumeFeature, SpreadFeature, PositionFeature, AmountFeature,
    PriceAccelerationFeature, IntradayVolatilityRatioFeature, PriceSkewnessFeature,
    PriceKurtosisFeature, PriceEntropyFeature, VolumePressureFeature,
    PositionVolatilityFeature, PriceMomentumFeature, IntradayPriceRangeFeature,
    PriceCycleAmplitudeFeature, PriceShadowAsymmetryFeature, TurnoverEfficiencyFeature,
    PriceVelocityFeature, IntradayPivotStrengthFeature, PriceFractalDimensionFeature,
    AmountVelocityFeature, PriceElasticityFeature, VolatilityCycleFeature,
    PriceMeanDistanceFeature, IntradayPriceSymmetryFeature, VolumeMomentumFeature,
    PriceWaveletEnergyFeature, PositionAccelerationFeature, PriceDensityFeature,
    IntradayPriceVelocityFeature, PriceRotationFrequencyFeature, VolumePriceCorrelationFeature,
    PriceBreakoutStrengthFeature, IntradayPriceCenterFeature, PriceHarmonicAmplitudeFeature,
    AmountPressureFeature, PositionMomentumFeature, PriceSpikeFrequencyFeature,
    IntradayPriceElasticityFeature, PriceTrendPersistenceFeature
)
from ..features.labelers import ReturnBasedLabeler
from .predictor import MLPredictor
from .models import (
    RandomForestModel, XGBoostModel, GradientBoostingModel, LightGBMModel, CatBoostModel,
    LogisticRegressionModel, SVMModel, AdaBoostModel, ExtraTreesModel, KNeighborsModel,
    StackingModel
)
from ..recommender import MLTradeRecommender  # 只导入 ML 专用推荐器

def get_feature_objects(feature_names: list, window: int = 20) -> list:
    feature_map = {
        "open": PriceFeature("open"), "high": PriceFeature("high"), "low": PriceFeature("low"),
        "close": PriceFeature("close"), "volume": VolumeFeature(window), "spread": SpreadFeature(),
        "position": PositionFeature(), "amount": AmountFeature(),
        "price_acceleration_20": PriceAccelerationFeature(window),
        "intraday_volatility_ratio_10": IntradayVolatilityRatioFeature(10),
        "price_skewness_20": PriceSkewnessFeature(window),
        "price_kurtosis_20": PriceKurtosisFeature(window),
        "price_entropy_20": PriceEntropyFeature(window),
        "volume_pressure_20": VolumePressureFeature(window),
        "position_volatility_20": PositionVolatilityFeature(window),
        "price_momentum_20": PriceMomentumFeature(window),
        "intraday_price_range_10": IntradayPriceRangeFeature(10),
        "price_cycle_amplitude_20": PriceCycleAmplitudeFeature(window),
        "price_shadow_asymmetry_10": PriceShadowAsymmetryFeature(10),
        "turnover_efficiency_20": TurnoverEfficiencyFeature(window),
        "price_velocity_10": PriceVelocityFeature(10),
        "intraday_pivot_strength_10": IntradayPivotStrengthFeature(10),
        "price_fractal_dimension_20": PriceFractalDimensionFeature(window),
        "amount_velocity_20": AmountVelocityFeature(window),
        "price_elasticity_20": PriceElasticityFeature(window),
        "volatility_cycle_20": VolatilityCycleFeature(window),
        "price_mean_distance_20": PriceMeanDistanceFeature(window),
        "intraday_price_symmetry_10": IntradayPriceSymmetryFeature(10),
        "volume_momentum_20": VolumeMomentumFeature(window),
        "price_wavelet_energy_20": PriceWaveletEnergyFeature(window),
        "position_acceleration_20": PositionAccelerationFeature(window),
        "price_density_20": PriceDensityFeature(window),
        "intraday_price_velocity_10": IntradayPriceVelocityFeature(10),
        "price_rotation_frequency_20": PriceRotationFrequencyFeature(window),
        "volume_price_correlation_20": VolumePriceCorrelationFeature(window),
        "price_breakout_strength_20": PriceBreakoutStrengthFeature(window),
        "intraday_price_center_10": IntradayPriceCenterFeature(10),
        "price_harmonic_amplitude_20": PriceHarmonicAmplitudeFeature(window),
        "amount_pressure_20": AmountPressureFeature(window),
        "position_momentum_20": PositionMomentumFeature(window),
        "price_spike_frequency_20": PriceSpikeFrequencyFeature(window),
        "intraday_price_elasticity_10": IntradayPriceElasticityFeature(10),
        "price_trend_persistence_20": PriceTrendPersistenceFeature(window)
    }
    return [feature_map[name] for name in feature_names if name in feature_map]

def generate_ml_results(raw_results: dict, auc: float, recommend_mode: str, 
                       auc_threshold: float = 0.7, prob_weight: float = 0.8, auc_weight: float = 0.2) -> dict:
    """生成 ML 推荐结果，转换为 'strong'/'weak' 标签"""
    if recommend_mode == "probability":
        max_contract = max(raw_results, key=raw_results.get)
        results = {contract: 'strong' if contract == max_contract else 'weak' 
                  for contract in raw_results}
    elif recommend_mode == "auc_filtered":
        max_contract = max(raw_results, key=raw_results.get)
        results = {contract: 'strong' if contract == max_contract else 'weak' 
                  for contract in raw_results}
    elif recommend_mode == "combined_score":
        scores = {contract: prob_weight * prob + auc_weight * auc 
                  for contract, prob in raw_results.items()}
        max_contract = max(scores, key=scores.get)
        results = {contract: 'strong' if contract == max_contract else 'weak' 
                  for contract in raw_results}
    else:
        raise ValueError(f"不支持的推荐模式: {recommend_mode}")
    return results

def main_ml(global_config_path="../../config.json", ml_config_path="ml_config.json"):
    setup_logging(log_file_path="../../results/ml_log.log", level=logging.INFO)
    logging.info("开始运行 ML 方法")

    with open(global_config_path, "r") as f:
        global_config = json.load(f)
    with open(ml_config_path, "r") as f:
        ml_config = json.load(f)

    if "data_groups" not in global_config:
        raise ValueError("全局配置缺少 'data_groups'")
    if "features" not in ml_config:
        raise ValueError("ML 配置缺少 'features'")

    os.makedirs("../../results", exist_ok=True)
    os.makedirs("../../models", exist_ok=True)

    for group_idx, data_files in enumerate(global_config["data_groups"]):
        logging.info(f"处理第 {group_idx + 1} 组数据: {data_files}")
        processors = [DataProcessor(f"../../data/{path}") for path in data_files]
        datasets = [p.clean_data() for p in processors]
        symbols = [path.split('.')[0] for path in data_files]
        for i, data in enumerate(datasets):
            data.set_index('date', inplace=True)
        common_index = datasets[0].index.intersection(datasets[1].index)
        datasets = [data.loc[common_index].reset_index() for data in datasets]
        logging.info("数据加载完成，时间对齐至重叠期")

        features = ml_config.get("features", ["close", "volume"])
        selected_features = get_feature_objects(features, ml_config.get("window", 20))
        extractor = FeatureExtractor(selected_features)
        labeler = ReturnBasedLabeler(20)
        features_df = extractor.extract_features(datasets, labeler)
        feature_cols = [col for col in features_df.columns if not col.startswith('label')]
        logging.info(f"提取特征完成，共 {len(feature_cols)} 个特征")

        model_type = ml_config.get("model_type", "random_forest")
        base_model_type = ml_config.get("base_model", None) if model_type == "stacking" else None
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
            "stacking": lambda: StackingModel(base_model_type)
        }
        if model_type not in model_map:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        models = [model_map[model_type]() for _ in datasets]
        logging.info(f"使用 {model_type.replace('_', ' ').title()} 模型" + 
                     (f"（Base Model: {base_model_type}）" if base_model_type else ""))

        predictor = MLPredictor(models, feature_cols)
        predictor.train(features_df)
        logging.info("模型训练完成")
        
        y_true = features_df[f"label_{symbols[0]}"].values
        raw_results = predictor.predict(features_df)
        y_pred_proba = np.array([raw_results[f"contract_{i+1}"] for i in range(len(datasets))]).T[0]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(y_true, y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        logging.info("预测完成")

        print(f"\n=== {model_type.replace('_', ' ').title()} 模型预测结果 ===" + 
              (f"（Base Model: {base_model_type}）" if base_model_type else ""))
        print("合约\t\t概率")
        print("-" * 30)
        for contract, value in raw_results.items():
            symbol = symbols[int(contract[-1]) - 1]
            print(f"{symbol:<15} {value:.4f}")
        print("-" * 30)
        print("评估指标:")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 30)

        # 生成 ML 推荐结果
        recommend_mode = ml_config.get("recommend_mode", "probability")
        results = generate_ml_results(
            raw_results=raw_results,
            auc=auc,
            recommend_mode=recommend_mode,
            auc_threshold=ml_config.get("auc_threshold", 0.7),
            prob_weight=ml_config.get("prob_weight", 0.8),
            auc_weight=ml_config.get("auc_weight", 0.2)
        )
        
        # 使用 ML 专用推荐器
        recommender = MLTradeRecommender(global_config["market_direction"])
        advice = recommender.recommend(results, "ml", symbols, group_idx, datasets)
        
        # 添加模式特定信息
        if recommend_mode == "auc_filtered" and auc < ml_config.get("auc_threshold", 0.7):
            advice += f"\n警告: AUC ({auc:.4f}) 低于 {ml_config.get('auc_threshold', 0.7)}，模型区分能力可能不足，建议谨慎"
        elif recommend_mode == "combined_score":
            scores = {contract: ml_config.get("prob_weight", 0.8) * prob + ml_config.get("auc_weight", 0.2) * auc 
                      for contract, prob in raw_results.items()}
            advice += "\n综合得分计算:"
            for contract, score in scores.items():
                symbol = symbols[int(contract[-1]) - 1]
                prob = raw_results[contract]
                advice += f"\n{symbol}: {score:.4f} = {ml_config.get('prob_weight', 0.8)} * {prob:.4f} + {ml_config.get('auc_weight', 0.2)} * {auc:.4f}"

        print(f"交易建议模式: {recommend_mode.replace('_', ' ').title()}")
        print(f"交易建议: {advice}")
        logging.info(f"交易建议生成（模式: {recommend_mode}）: {advice}")
        logging.info(f"评估指标 - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                     f"Recall: {recall:.4f}, F1: {f1:.4f}")

    logging.info("ML 方法运行完成")

if __name__ == "__main__":
    main_ml()