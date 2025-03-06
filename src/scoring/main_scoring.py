# src/scoring/main_scoring.py
import os
import yaml
from src.utils.data_processor import DataProcessor
from src.utils.feature_selector import ScoringFeatureSelector
from src.scoring.evaluator import StrengthEvaluator
from src.utils.recommender import ScoringTradeRecommender
from src.utils.logging_utils import setup_logging
import logging

# 获取项目根目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def main():
    # 配置文件路径
    config_path = os.path.join(ROOT_DIR, "src", "scoring", "scoring_config.yaml")
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在")
        return
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 从配置文件读取路径
    data_dir = os.path.join(ROOT_DIR, config.get("data_dir", "data"))
    log_dir = os.path.join(ROOT_DIR, config.get("log_dir", "results"))
    log_path = os.path.join(log_dir, "scoring_log.log")

    # 配置日志
    setup_logging(log_path)

    # 循环处理每组数据
    for group_idx, data_group in enumerate(config["data_groups"]):
        logging.info(f"处理数据组 {group_idx + 1}，市场方向: {data_group['market_direction']}")
        datasets = []
        for file in data_group["files"]:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                logging.error(f"数据文件 {file_path} 不存在")
                return
            processor = DataProcessor(file_path)
            datasets.append(processor.clean_data())
        logging.info(f"数据组 {group_idx + 1} 加载的数据集数量: {len(datasets)}")

        # 特征选择
        selector = ScoringFeatureSelector()
        selected_features = selector.select_features(datasets, config)
        logging.info(f"数据组 {group_idx + 1} 选择的特征: {selected_features}")

        # 加载特征模块
        from src.scoring import analyses
        modules = [getattr(analyses, feat)() for feat in selected_features]

        # 评分
        evaluator = StrengthEvaluator(modules, config["weights"])
        scores = evaluator.evaluate(datasets)
        logging.info(f"数据组 {group_idx + 1} 评分结果: {scores}")

        # 生成交易建议
        recommender = ScoringTradeRecommender(data_group["market_direction"], config)
        advice = recommender.recommend(scores, "scoring", data_group["files"], group_idx)
        logging.info(f"数据组 {group_idx + 1} 交易建议: {advice}")

if __name__ == "__main__":
    main()



# # src/scoring/main_scoring.py
# import sys
# import os
# import yaml
# from src.utils.data_processor import DataProcessor
# from src.utils.feature_selector import ScoringFeatureSelector
# from src.scoring.evaluator import StrengthEvaluator
# from src.utils.recommender import ScoringTradeRecommender
# from src.utils.logging_utils import setup_logging
# import logging

# # 获取项目根目录
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# sys.path.append(ROOT_DIR)  # 添加根目录到 sys.path，确保导入一致

# def main():
#     # 配置日志，基于根目录的相对路径
#     log_path = os.path.join(ROOT_DIR, "results", "scoring_log.log")
#     setup_logging(log_path)

#     # 配置文件路径
#     config_path = os.path.join(ROOT_DIR, "src", "scoring", "scoring_config.yaml")
#     if not os.path.exists(config_path):
#         logging.error(f"配置文件 {config_path} 不存在")
#         return
#     with open(config_path, "r", encoding='utf-8') as f:
#         config = yaml.safe_load(f)

#     # 循环处理每组数据
#     for group_idx, data_group in enumerate(config["data_groups"]):
#         logging.info(f"处理数据组 {group_idx + 1}，市场方向: {data_group['market_direction']}")
#         datasets = []
#         for file in data_group["files"]:
#             # 数据文件路径，基于根目录
#             file_path = os.path.join(ROOT_DIR, "data", file)
#             if not os.path.exists(file_path):
#                 logging.error(f"数据文件 {file_path} 不存在")
#                 return
#             processor = DataProcessor(file_path)
#             datasets.append(processor.clean_data())
#         logging.info(f"数据组 {group_idx + 1} 加载的数据集数量: {len(datasets)}")

#         # 特征选择
#         selector = ScoringFeatureSelector()
#         selected_features = selector.select_features(datasets, config)
#         logging.info(f"数据组 {group_idx + 1} 选择的特征: {selected_features}")

#         # 加载特征模块
#         from src.scoring import analyses
#         modules = [getattr(analyses, feat)() for feat in selected_features]

#         # 评分
#         evaluator = StrengthEvaluator(modules, config["weights"])
#         scores = evaluator.evaluate(datasets)
#         logging.info(f"数据组 {group_idx + 1} 评分结果: {scores}")

#         # 生成交易建议
#         recommender = ScoringTradeRecommender(data_group["market_direction"])
#         advice = recommender.recommend(scores, "scoring", data_group["files"], group_idx)
#         logging.info(f"数据组 {group_idx + 1} 交易建议: {advice}")

# if __name__ == "__main__":
#     main()
