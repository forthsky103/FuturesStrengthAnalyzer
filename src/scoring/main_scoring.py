# src/scoring/main_scoring.py
import json
import os
import pandas as pd
import logging
from ..logging_utils import setup_logging
from ..data_processor import DataProcessor
from .evaluator import StrengthEvaluator
from .analyses import (
    PriceTrendAccelerationAnalysis, IntradayVolatilityAnalysis, PriceSkewnessAnalysis,
    PriceKurtosisAnalysis, PriceCompressionAnalysis, IntradayReversalAnalysis,
    PriceMeanReversionAnalysis, TrendDirectionConsistencyAnalysis, PriceBounceStrengthAnalysis,
    PricePullbackStrengthAnalysis, IntradayPriceEfficiencyAnalysis, PricePressureAnalysis,
    VolumeWeightedVolatilityAnalysis, PriceMomentumDivergenceAnalysis, PriceClusterAnalysis,
    IntradayTrendStrengthAnalysis, PriceExpansionAnalysis, MarketTensionAnalysis,
    PricePathEfficiencyAnalysis, VolatilityCompressionAnalysis,
    HighLowTimingAnalysis, PriceShadowRatioAnalysis, PricePivotStrengthAnalysis,
    IntradayMomentumShiftAnalysis, PriceGapPersistenceAnalysis, MarketDepthPressureAnalysis,
    PriceRotationAnalysis, IntradayPriceSpreadAnalysis, PriceBreakoutFailureAnalysis,
    MarketSentimentShiftAnalysis, PriceVelocityAsymmetryAnalysis, IntradayPriceCenterAnalysis,
    PriceStructureComplexityAnalysis, VolumePressureDistributionAnalysis, PriceTrendRotationAnalysis
)
from ..recommender import ScoringTradeRecommender

def main_scoring(global_config_path="../../config.json", scoring_config_path="scoring_config.json"):
    setup_logging(log_file_path="../../results/scoring_log.log", level=logging.INFO)
    logging.info("开始运行 Scoring 方法")

    with open(global_config_path, "r") as f:
        global_config = json.load(f)
    with open(scoring_config_path, "r") as f:
        scoring_config = json.load(f)

    if "data_groups" not in global_config:
        raise ValueError("全局配置缺少 'data_groups'")
    if "weights" not in scoring_config:
        raise ValueError("打分配置缺少 'weights'")

    os.makedirs("../../results", exist_ok=True)

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

        modules = [
            PriceTrendAccelerationAnalysis(), IntradayVolatilityAnalysis(), PriceSkewnessAnalysis(),
            PriceKurtosisAnalysis(), PriceCompressionAnalysis(), IntradayReversalAnalysis(),
            PriceMeanReversionAnalysis(), TrendDirectionConsistencyAnalysis(), PriceBounceStrengthAnalysis(),
            PricePullbackStrengthAnalysis(), IntradayPriceEfficiencyAnalysis(), PricePressureAnalysis(),
            VolumeWeightedVolatilityAnalysis(), PriceMomentumDivergenceAnalysis(), PriceClusterAnalysis(),
            IntradayTrendStrengthAnalysis(), PriceExpansionAnalysis(), MarketTensionAnalysis(),
            PricePathEfficiencyAnalysis(), VolatilityCompressionAnalysis(),
            HighLowTimingAnalysis(), PriceShadowRatioAnalysis(), PricePivotStrengthAnalysis(),
            IntradayMomentumShiftAnalysis(), PriceGapPersistenceAnalysis(), MarketDepthPressureAnalysis(),
            PriceRotationAnalysis(), IntradayPriceSpreadAnalysis(), PriceBreakoutFailureAnalysis(),
            MarketSentimentShiftAnalysis(), PriceVelocityAsymmetryAnalysis(), IntradayPriceCenterAnalysis(),
            PriceStructureComplexityAnalysis(), VolumePressureDistributionAnalysis(), PriceTrendRotationAnalysis()
        ]
        weights = scoring_config.get("weights", {m.__class__.__name__: 1.0 for m in modules})
        evaluator = StrengthEvaluator(modules, weights)
        results = evaluator.evaluate(datasets)
        logging.info("打分计算完成")

        print("得分结果:")
        for contract, score in results.items():
            symbol = symbols[int(contract[-1]) - 1]
            print(f"{symbol:<15} {score:.2f}")

        recommender = ScoringTradeRecommender(global_config["market_direction"])
        advice = recommender.recommend(results, "scoring", symbols, group_idx, datasets)
        print(f"交易建议: {advice}")
        logging.info(f"交易建议生成: {advice}")

    logging.info("Scoring 方法运行完成")

if __name__ == "__main__":
    main_scoring()