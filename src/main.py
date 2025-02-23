# src/main.py
import json
from .data_processor import DataProcessor
from .features.extractor import FeatureExtractor
from .features.features import TrendFeature, VolumeFeature, SpreadFeature
from .features.labelers import ReturnBasedLabeler, VolumeBasedLabeler, VolatilityBasedLabeler
from .scoring.evaluator import StrengthEvaluator
from .scoring.analyses import TrendAnalysis, VolumeAnalysis
from .ml.predictor import MLPredictor
from .ml.models import RandomForestModel
from .recommender import TradeRecommender

def main():
    # 加载配置
    with open("../config.json", "r") as f:
        config = json.load(f)

    # 数据加载与对齐
    processors = [DataProcessor(f"../data/{path}") for path in config["data_files"]]
    datasets = [p.clean_data() for p in processors]
    symbols = [path.split('.')[0] for path in config["data_files"]]
    for i, data in enumerate(datasets):
        data.set_index('date', inplace=True)
    common_index = datasets[0].index
    for data in datasets[1:]:
        common_index = common_index.intersection(data.index)
    datasets = [data.loc[common_index].reset_index() for data in datasets]

    # 评估
    if config["method"] == "scoring":
        modules = [TrendAnalysis(), VolumeAnalysis()]
        evaluator = StrengthEvaluator(modules, config["weights"])
        results = evaluator.evaluate(datasets)
        print("得分结果:")
        for contract, score in results.items():
            print(f"{symbols[int(contract[-1])-1]}: {score:.2f}")
    else:
        selected_features = [
            TrendFeature(window=20),
            VolumeFeature(window=20),
            SpreadFeature()
        ]
        # 动态选择标签生成器
        labeler_type = config.get("labeler", "return")  # 默认使用收益率
        if labeler_type == "return":
            labeler = ReturnBasedLabeler(window=20)
        elif labeler_type == "volume":
            labeler = VolumeBasedLabeler(window=20)
        elif labeler_type == "volatility":
            labeler = VolatilityBasedLabeler(window=20)
        else:
            raise ValueError(f"未知的标签生成器: {labeler_type}")
        
        extractor = FeatureExtractor(selected_features)
        features = extractor.extract_features(datasets, labeler=labeler)
        feature_cols = [col for col in features.columns if not col.startswith('label')]
        
        models = [RandomForestModel() for _ in datasets]
        predictor = MLPredictor(models, feature_cols)
        predictor.train(features)
        results = predictor.predict(features)
        print("预测结果:")
        for contract, pred in results.items():
            print(f"{symbols[int(contract[-1])-1]}: {pred}")

    # 交易建议
    recommender = TradeRecommender(config["market_direction"])
    advice = recommender.recommend(results, config["method"], symbols)
    print(f"交易建议: {advice}")

if __name__ == "__main__":
    main()