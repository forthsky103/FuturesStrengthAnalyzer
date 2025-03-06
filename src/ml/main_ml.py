# src/ml/main_ml.py
import os
import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from src.utils.logging_utils import setup_logging
from src.utils.data_processor import DataProcessor
from src.utils.feature_selector import get_feature_selector
from src.utils.recommender import MLTradeRecommender
from src.features import features
from src.ml.ml_labelers import MLReturnBasedLabeler
from src.ml.mlextractor import MLFeatureExtractor
from src.ml.predictor import MLPredictor
from src.ml.models import (
    RandomForestModel, XGBoostModel, GradientBoostingModel, LightGBMModel, CatBoostModel,
    LogisticRegressionModel, SVMModel, AdaBoostModel, ExtraTreesModel, KNeighborsModel,
    StackingModel
)

def generate_ml_results(raw_results: Dict[str, float], auc: float, recommend_mode: str, 
                        auc_threshold: float = 0.7, prob_weight: float = 0.8, auc_weight: float = 0.2) -> Dict[str, str]:
    """根据原始预测结果生成强弱标签"""
    if recommend_mode == "probability":
        scores = {contract: prob_weight * prob + auc_weight * auc for contract, prob in raw_results.items()}
        max_contract = max(scores, key=scores.get)
        min_contract = min(scores, key=scores.get)
        results = {}
        for contract in raw_results:
            if contract == max_contract and auc >= auc_threshold:
                results[contract] = 'strong'
            elif contract == min_contract and auc >= auc_threshold:
                results[contract] = 'weak'
            else:
                results[contract] = 'neutral'
        return results
    else:
        raise ValueError(f"当前仅支持 'probability' 模式")

def main_ml(config_path: str = "ml_config.yaml"):
    """运行 ML 模块，判断合约强弱"""
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    config_full_path = os.path.join(ROOT_DIR, "src", "ml", config_path)

    if not os.path.exists(config_full_path):
        print(f"配置文件 {config_full_path} 不存在")
        return
    with open(config_full_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if "data_groups" not in config:
        raise ValueError("配置缺少 'data_groups'")
    if "manual_features" not in config:
        raise ValueError("配置缺少 'manual_features'")

    log_dir = os.path.join(ROOT_DIR, config.get("log_dir", "results"))
    log_path = os.path.join(log_dir, "ml_log.log")
    setup_logging(log_file_path=log_path, level=logging.DEBUG)
    logging.info("开始运行 ML 方法")

    data_dir = os.path.join(ROOT_DIR, config.get("data_dir", "data"))
    results_dir = os.path.join(ROOT_DIR, config.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)

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
        "stacking": lambda: StackingModel(
            base_model_type=config.get("base_model", "random_forest"),
            base_model_params=config.get("stacking_params", {}).get("base_model_params", config.get("model_params", {})),
            other_model_params=config.get("stacking_params", {}).get("other_model_params", config.get("model_params", {})),
            meta_model_params=config.get("stacking_params", {}).get("meta_model_params", {})
        )
    }

    for group_idx, data_group in enumerate(config["data_groups"]):
        data_files = data_group["files"]
        market_direction = data_group["market_direction"]
        logging.info(f"处理第 {group_idx + 1} 组数据: {data_files}, 市场方向: {market_direction}")

        processors = [DataProcessor(os.path.join(data_dir, path)) for path in data_files]
        datasets = [p.clean_data() for p in processors]
        symbols = [path.split('.')[0] for path in data_files]
        for i, data in enumerate(datasets):
            data.set_index('date', inplace=True)
        common_index = datasets[0].index
        for data in datasets[1:]:
            common_index = common_index.intersection(data.index)
        datasets = [data.loc[common_index].reset_index() for data in datasets]
        logging.info("数据加载完成，时间对齐至重叠期")

        selector = get_feature_selector("ml")
        selected_feature_names = selector.select_features(datasets, config)
        logging.info(f"特征筛选完成，选出 {len(selected_feature_names)} 个特征: {selected_feature_names}")

        extractors = []
        for i, symbol in enumerate(symbols):
            feature_params = config.get("feature_params", {})
            selected_features = []
            for name in selected_feature_names:
                if name == "PriceFeature":
                    column = feature_params.get(name, {}).get("column", "close")
                    selected_features.append(getattr(features, name)(column))
                else:
                    window = feature_params.get(name, {}).get("window", 20)
                    selected_features.append(getattr(features, name)(window))
            extractors.append(MLFeatureExtractor(selected_features, symbol))
        
        labeler = MLReturnBasedLabeler(config.get("labeler_window", 20))
        features_dfs = [extractor.extract_features(datasets[i], labeler) for i, extractor in enumerate(extractors)]
        features_df = pd.concat(features_dfs, axis=1)
        feature_cols = [col for col in features_df.columns if not col.startswith('label')]
        logging.info(f"提取特征完成，共 {len(feature_cols)} 个特征")
        logging.debug(f"features_df 列类型: {features_df.dtypes}")

        model_type = config.get("model_type", "random_forest")
        if model_type not in model_map:
            raise ValueError(f"不支持的模型类型: {model_type}")
        base_model_type = config.get("base_model", "random_forest") if model_type == "stacking" else None
        models = [model_map[model_type]() for _ in datasets]
        logging.info(f"使用 {model_type.replace('_', ' ').title()} 模型" + 
                     (f"（基准模型: {base_model_type}）" if base_model_type else ""))

        predictor = MLPredictor(models, feature_cols, symbols)
        predictor.train(features_df)
        logging.info("模型训练完成")
        
        # 使用测试集评估第一个合约
        test_preds = predictor.get_test_predictions(symbols[0])
        y_true = test_preds.get('y_test', np.array([]))
        y_pred_proba = test_preds.get('y_pred_proba', np.array([]))
        y_pred = (y_pred_proba > 0.5).astype(int)

        if len(y_true) > 0:
            auc = roc_auc_score(y_true, y_pred_proba)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            logging.warning("测试集数据为空，使用默认值")
            auc = accuracy = precision = recall = f1 = 0.0

        # 交易建议基于最后一行
        raw_results = predictor.predict(features_df)
        print(f"\n=== {model_type.replace('_', ' ').title()} 模型预测结果 ===")
        print("合约\t\t概率")
        print("-" * 30)
        for contract, value in raw_results.items():
            symbol = symbols[int(contract.split('_')[1]) - 1]
            print(f"{symbol:<15} {value:.4f}")
        print("-" * 30)
        print("评估指标（以第一个合约测试集为例）:")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 30)

        recommend_mode = config.get("recommend_mode", "probability")
        results = generate_ml_results(
            raw_results=raw_results,
            auc=auc,
            recommend_mode=recommend_mode,
            auc_threshold=config.get("auc_threshold", 0.7),
            prob_weight=config.get("prob_weight", 0.8),
            auc_weight=config.get("auc_weight", 0.2)
        )
        
        recommender = MLTradeRecommender(market_direction, config)
        advice = recommender.recommend(results, "ml", symbols, group_idx, datasets)
        print(f"交易建议模式: {recommend_mode.replace('_', ' ').title()}")
        print(f"交易建议: {advice}")
        logging.info(f"交易建议生成（模式: {recommend_mode}）: {advice}")

        results_df = pd.DataFrame({
            'Contract': symbols,
            'Probability': list(raw_results.values()),
            'Label': list(results.values()),
            'AUC': auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        results_df.to_csv(os.path.join(results_dir, f"ml_results_group_{group_idx}.csv"), index=False)
        logging.info(f"结果已保存至 {results_dir}/ml_results_group_{group_idx}.csv")

    logging.info("ML 方法运行完成")

if __name__ == "__main__":
    main_ml()