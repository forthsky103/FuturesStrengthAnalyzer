# src/stats/main_stats.py
import os
import yaml
import pandas as pd
import logging
from typing import List, Tuple, Dict
from src.utils.logging_utils import setup_logging
from src.utils.data_processor import DataProcessor
from src.utils.feature_selector import get_feature_selector
from src.utils.recommender import ScoringTradeRecommender
from src.utils.market_conditions import (
    HighVolatilityCondition,
    TrendMarketCondition,
    RangeMarketCondition,
    LowVolatilityCondition
)
from src.features.labelers import ReturnBasedLabeler
from src.stats import evaluator

def main_stats(config_path: str = "stats_config.yaml") -> Tuple[Dict, str, str]:
    """运行统计模块，判断合约强弱
    Args:
        config_path (str): 统计模块配置文件路径，包含数据组、路径和参数
    Returns:
        Tuple[Dict, str, str]: (结果字典, 最强合约, 最弱合约)
    """
    # 获取项目根目录
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    config_full_path = os.path.join(ROOT_DIR, "src", "stats", config_path)

    # 加载配置
    if not os.path.exists(config_full_path):
        print(f"配置文件 {config_full_path} 不存在")
        return
    with open(config_full_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 检查必要配置项
    if "data_groups" not in config:
        raise ValueError("配置缺少 'data_groups'")
    if "weights" not in config:
        raise ValueError("配置缺少 'weights'")

    # 设置日志路径
    log_dir = os.path.join(ROOT_DIR, config.get("log_dir", "results"))
    log_path = os.path.join(log_dir, "stats_log.log")
    setup_logging(log_file_path=log_path, level=logging.INFO)
    logging.info("开始运行 Stats 方法")

    # 数据目录
    data_dir = os.path.join(ROOT_DIR, config.get("data_dir", "data"))

    # 定义市场状态映射
    condition_map = {
        "HighVolatilityCondition": HighVolatilityCondition,
        "TrendMarketCondition": TrendMarketCondition,
        "RangeMarketCondition": RangeMarketCondition,
        "LowVolatilityCondition": LowVolatilityCondition
    }

    # 遍历数据组
    for group_idx, data_group in enumerate(config["data_groups"]):
        logging.info(f"处理第 {group_idx + 1} 组数据: {data_group['files']}")
        # 数据预处理
        processors = [DataProcessor(os.path.join(data_dir, path)) for path in data_group["files"]]
        datasets = [p.clean_data() for p in processors]
        symbols = [path.split('.')[0] for path in data_group["files"]]
        # 时间对齐
        for i, data in enumerate(datasets):
            data.set_index('date', inplace=True)
        common_index = datasets[0].index
        for data in datasets[1:]:
            common_index = common_index.intersection(data.index)
        datasets = [data.loc[common_index].reset_index() for data in datasets]
        logging.info("数据加载完成，时间对齐至重叠期")

        # 添加标签（用于自动权重生成和特征选择）
        labeler = ReturnBasedLabeler(window=config.get("labeler_window", 20))
        labels_dict = labeler.generate_labels(datasets)
        for i, df in enumerate(datasets):
            df['label'] = labels_dict[f'label{i+1}']

        # 获取统计特征选择器并筛选指标
        selector = get_feature_selector("stats")
        selected_stats_names = selector.select_features(datasets, config)
        logging.info(f"统计指标筛选完成，选出 {len(selected_stats_names)} 个指标: {selected_stats_names}")

        # 动态实例化统计指标对象
        selected_metrics = [getattr(evaluator, name)() for name in selected_stats_names]
        weights = {name: config.get("weights", {}).get(name, 1.0) for name in selected_stats_names}

        # 创建评估器并计算结果
        evaluator_instance = evaluator.StatsEvaluator(selected_metrics, weights)
        results, strongest, weakest = evaluator_instance.evaluate(datasets, condition_map, config_path=config_full_path)
        logging.info(f"统计评估完成，最强合约: {strongest}, 最弱合约: {weakest}")

        # 输出得分结果
        print(f"Stats 评估结果 (Group {group_idx + 1}):")
        for contract, (score, explanation) in results.items():
            symbol = symbols[int(contract[-1]) - 1]
            print(f"{symbol:<15} 得分: {score:.4f}")
            print(explanation)
            print("-" * 50)

        # 生成交易建议
        recommender = ScoringTradeRecommender(data_group["market_direction"], config)
        advice = recommender.recommend({k: v[0] for k, v in results.items()}, "stats", symbols, group_idx, datasets)
        print(f"交易建议: {advice}")
        logging.info(f"交易建议生成: {advice}")

    logging.info("Stats 方法运行完成")
    return results, strongest, weakest

if __name__ == "__main__":
    main_stats()