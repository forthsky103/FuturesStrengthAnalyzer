# src/rules/main_rules.py
import json
import os
import pandas as pd
import logging
from typing import List, Dict, Tuple
from ..utils.logging_utils import setup_logging  # 日志工具
from ..utils.data_processor import DataProcessor  # 数据处理工具
from ..utils.feature_selector import get_feature_selector  # 特征选择器工厂
from ..utils.recommender import ScoringTradeRecommender  # 推荐器
from . import evaluator  # 规则模块

def main_rules(global_config_path: str = "../config.json", rules_config_path: str = "rules_config.json"):
    """运行规则模块，判断合约强弱
    Args:
        global_config_path (str): 全局配置文件路径，包含数据组和市场方向
        rules_config_path (str): 规则模块配置文件路径，包含权重和筛选参数
    """
    # 设置日志
    setup_logging(log_file_path="../../results/rules_log.log", level=logging.INFO)
    logging.info("开始运行 Rules 方法")

    # 加载全局配置
    with open(global_config_path, "r") as f:
        global_config = json.load(f)
    # 加载规则模块配置
    with open(rules_config_path, "r") as f:
        rules_config = json.load(f)

    # 检查必要配置项
    if "data_groups" not in global_config:
        raise ValueError("全局配置缺少 'data_groups'")
    if "weights" not in rules_config:
        raise ValueError("规则配置缺少 'weights'")

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

        # 获取规则特征选择器并筛选规则
        selector = get_feature_selector("rules")
        selected_rule_names = selector.select_features(datasets, rules_config)
        logging.info(f"规则筛选完成，选出 {len(selected_rule_names)} 个规则: {selected_rule_names}")

        # 动态实例化规则对象
        selected_rules = [getattr(evaluator, name)() for name in selected_rule_names]
        rule_weights = {name: rules_config.get("weights", {}).get(name, 1.0) for name in selected_rule_names}

        # 创建专家系统并评估
        expert_system = evaluator.ExpertSystem(selected_rules, rule_weights)
        results = expert_system.evaluate(datasets, {}, rules_config_path)
        logging.info("规则评估完成")

        # 输出得分结果
        print(f"Rules 评估结果 (Group {group_idx + 1}):")
        for contract, (score, explanation) in results.items():
            symbol = symbols[int(contract[-1]) - 1]
            print(f"{symbol:<15} 得分: {score:.2f}")
            print(explanation)
            print("-" * 50)

        # 生成交易建议
        recommender = ScoringTradeRecommender(global_config["market_direction"])
        advice = recommender.recommend({k: v[0] for k, v in results.items()}, "rules", symbols, group_idx, datasets)
        print(f"交易建议: {advice}")
        logging.info(f"交易建议生成: {advice}")

    logging.info("Rules 方法运行完成")

if __name__ == "__main__":
    main_rules()