# src/main.py
import json
import logging
from .utils.logging_utils import setup_logging  # 调整为完整路径
from scoring.main_scoring import main_scoring
from ml.main_ml import main_ml
from rules.main_rules import main_rules
from stats.main_stats import main_stats
from .utils.recommender import ScoringTradeRecommender, MLTradeRecommender  # 调整为完整路径

def main(config_path="config.json"):
    setup_logging(log_file_path="../results/combined_log.log", level=logging.INFO)
    logging.info("开始运行综合 Main 方法")

    with open(config_path, "r") as f:
        config = json.load(f)

    methods = config.get("methods", ["scoring"])
    for method in methods:
        if method == "scoring":
            logging.info("运行 Scoring 方法")
            main_scoring(config_path, "src/scoring/scoring_config.json")
        elif method == "ml":
            logging.info("运行 ML 方法")
            main_ml(config_path, "src/ml/ml_config.json")
        elif method == "rules":
            logging.info("运行 Rules-Based Expert System 方法")
            main_rules(config_path, "src/rules/rules_config.json")
        elif method == "stats":
            logging.info("运行 Stats 方法")
            main_stats(config_path, "src/stats/stats_config.json")
        else:
            logging.warning(f"未知方法: {method}")

    logging.info("综合 Main 方法运行完成")

if __name__ == "__main__":
    main()