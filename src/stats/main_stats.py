# src/stats/main_stats.py
import json
import os
import pandas as pd
import logging
from typing import List, Tuple, Dict
from ..utils.logging_utils import setup_logging  # 日志工具
from ..utils.data_processor import DataProcessor  # 数据处理工具
from ..utils.feature_selector import get_feature_selector  # 特征选择器工厂
from . import evaluator  # 统计模块

def main_stats(global_config_path: str = "../config.json", stats_config_path: str = "stats_config.json") -> Tuple[Dict, str, str]:
    """运行统计模块，判断合约强弱
    Args:
        global_config_path (str): 全局配置文件路径，包含数据组和市场方向
        stats_config_path (str): 统计模块配置文件路径，包含权重和筛选参数
    Returns:
        Tuple[Dict, str, str]: (结果字典, 最强合约, 最弱合约)
    """
    # 设置日志
    setup_logging(log_file_path="../../results/stats_log.log", level=logging.INFO)
    logging.info("开始运行 Stats 方法")

    # 加载全局配置
    with open(global_config_path, "r") as f:
        global_config = json.load(f)
    # 加载统计模块配置
    with open(stats_config_path, "r") as f:
        stats_config = json.load(f)

    # 检查必要配置项
    if "data_groups" not in global_config:
        raise ValueError("全局配置缺少 'data_groups'")
    if "weights" not in stats_config:
        raise ValueError("统计配置缺少 'weights'")

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

        # 获取统计特征选择器并筛选指标
        selector = get_feature_selector("stats")
        selected_stats_names = selector.select_features(datasets, stats_config)
        logging.info(f"统计指标筛选完成，选出 {len(selected_stats_names)} 个指标: {selected_stats_names}")

        # 动态实例化统计指标对象
        selected_metrics = [getattr(evaluator, name)() for name in selected_stats_names]
        weights = {name: stats_config.get("weights", {}).get(name, 1.0) for name in selected_stats_names}

        # 创建评估器并计算结果
        evaluator_instance = evaluator.StatsEvaluator(selected_metrics, weights)
        results, strongest, weakest = evaluator_instance.evaluate(datasets, {}, stats_config_path)
        logging.info(f"统计评估完成，最强合约: {strongest}, 最弱合约: {weakest}")

        # 输出得分结果和其他信息
        print(f"Stats 评估结果 (Group {group_idx + 1}):")
        for contract, (score, explanation) in results.items():
            symbol = symbols[int(contract[-1]) - 1]
            print(f"{symbol:<15} 得分: {score:.4f}")
            print(explanation)
            print("-" * 50)
        print(f"最强合约: {strongest}")
        print(f"最弱合约: {weakest}")

    logging.info("Stats 方法运行完成")
    return results, strongest, weakest

if __name__ == "__main__":
    main_stats()