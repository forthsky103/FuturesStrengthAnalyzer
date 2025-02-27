# src/scoring/main_scoring.py
import json
import os
import pandas as pd
import logging
from ..utils.logging_utils import setup_logging
from ..utils.data_processor import DataProcessor
from .evaluator import StrengthEvaluator
from ..utils.feature_selector import get_feature_selector
from ..utils.recommender import ScoringTradeRecommender

def main_scoring(global_config_path="../config.json", scoring_config_path="scoring_config.json"):
    """运行打分法模块，判断合约强弱
    Args:
        global_config_path: 全局配置文件路径
        scoring_config_path: 打分法配置文件路径
    """
    setup_logging(log_file_path="../../results/scoring_log.log", level=logging.INFO)
    logging.info("开始运行 Scoring 方法")

    # 加载全局配置
    with open(global_config_path, "r") as f:
        global_config = json.load(f)
    # 加载打分法配置
    with open(scoring_config_path, "r") as f:
        scoring_config = json.load(f)

    # 检查必要配置项
    if "data_groups" not in global_config:
        raise ValueError("全局配置缺少 'data_groups'")
    if "weights" not in scoring_config:
        raise ValueError("打分配置缺少 'weights'")

    # 创建结果目录
    os.makedirs("../../results", exist_ok=True)

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

        # 获取打分法特征选择器并筛选特征
        selector = get_feature_selector("scoring")
        selected_feature_names = selector.select_features(datasets, scoring_config)
        logging.info(f"特征筛选完成，选出 {len(selected_feature_names)} 个特征: {selected_feature_names}")

        # 初始化特征模块和权重
        modules = [globals()[name]() for name in selected_feature_names]
        weights = {name: scoring_config["weights"].get(name, 1.0) for name in selected_feature_names}
        evaluator = StrengthEvaluator(modules, weights)
        results = evaluator.evaluate(datasets)
        logging.info("打分计算完成")

        # 输出得分结果
        print("得分结果:")
        for contract, score in results.items():
            symbol = symbols[int(contract[-1]) - 1]
            print(f"{symbol:<15} {score:.2f}")

        # 生成交易建议
        recommender = ScoringTradeRecommender(global_config["market_direction"])
        advice = recommender.recommend(results, "scoring", symbols, group_idx, datasets)
        print(f"交易建议: {advice}")
        logging.info(f"交易建议生成: {advice}")

    logging.info("Scoring 方法运行完成")

if __name__ == "__main__":
    main_scoring()