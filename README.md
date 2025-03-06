# Futures Strength Analyzer

一个用于分析期货合约强弱的项目，支持打分法和机器学习方法。

## 项目目录结构
FuturesStrengthAnalyzer/
├── data/                          # 数据目录，存放合约 CSV 文件
│   ├── SHFE.rb2510.csv           # 螺纹钢 2510 合约数据
│   ├── SHFE.rb2505.csv           # 螺纹钢 2505 合约数据
│   ├── SHFE.hc2505.csv           # 热卷 2505 合约数据
│   ├── SHFE.hc2510.csv           # 热卷 2510 合约数据
│
├── results/                      # 结果目录，存放日志文件
│   ├── ml_log.log                # ML 模块日志
│   ├── scoring_log.log           # Scoring 模块日志
│   ├── stats_log.log             # Stats 模块日志
│   ├── rules_log.log             # Rules 模块日志（未替换时可能仍为 JSON 配置）
│
├── models/                       # 模型目录，存放训练后的模型文件（若有）
│
├── src/                          # 源代码目录
│   ├── config.yaml               # 全局配置文件（YAML，已替换）
│   │                             # 定义多组数据、每组的市场方向和方法
│   │
│   ├── ml/                       # ML 模块
│   │   ├── main_ml.py            # ML 主文件（已替换为 YAML）
│   │   ├── ml_config.yaml        # ML 配置文件（YAML，已替换）
│   │   ├── predictor.py          # ML 预测器逻辑
│   │   ├── models.py             # ML 模型定义（RandomForestModel 等）
│   │
│   ├── scoring/                  # Scoring 模块
│   │   ├── main_scoring.py       # Scoring 主文件（已替换为 YAML）
│   │   ├── scoring_config.yaml   # Scoring 配置文件（YAML，已替换）
│   │   ├── evaluator.py          # Scoring 评估器逻辑
│   │   ├── analyses.py           # Scoring 特征分析类
│   │
│   ├── stats/                    # Stats 模块
│   │   ├── main_stats.py         # Stats 主文件（已替换为 YAML，测试版）
│   │   ├── stats_config.yaml     # Stats 配置文件（YAML，已替换，支持动态权重）
│   │   ├── evaluator.py          # Stats 评估器逻辑（包含动态权重调整）
│   │
│   ├── rules/                    # Rules 模块（未替换，仍基于 JSON）
│   │   ├── main_rules.py         # Rules 主文件（仍使用 JSON）
│   │   ├── rules_config.json     # Rules 配置文件（JSON，未替换）
│   │   ├── evaluator.py          # Rules 专家系统逻辑
│   │
│   ├── utils/                    # 工具模块
│   │   ├── __init__.py           # 空初始化文件
│   │   ├── logging_utils.py      # 日志工具
│   │   ├── data_processor.py     # 数据处理工具
│   │   ├── feature_selector.py   # 特征选择器（支持动态加载）
│   │   ├── market_conditions.py  # 市场条件类（动态权重调整）
│   │   ├── recommender.py        # 推荐器（交易建议生成）
│   │   ├── weight_generator/     # 自动权重生成器目录（假设存在）
│   │       ├── generate_weights.py  # 权重生成逻辑
│   │
│   ├── features/                 # 特征模块（ML 和通用）
│   │   ├── __init__.py           # 空初始化文件
│   │   ├── features.py           # ML 特征定义类
│   │   ├── extractor.py          # 特征提取器
│   │   ├── labelers.py           # 标签生成器（ReturnBasedLabeler 等）

## 项目概述

一个用于分析期货合约强弱的项目，支持打分法和机器学习方法。主要功能包括：
- **数据处理**：清洗和对齐期货数据。
- **特征提取**：提供35个创新特征，用于机器学习和深度学习。
- **强弱分析**：支持打分法、机器学习、统计方法、时间序列分析等多种方式。
- **交易建议**：根据分析结果生成具体的交易推荐。

## 使用方法

1. **准备数据**：
   - 将期货数据（如 `rb2510.csv`、`rb2505.csv`、`hc2505.csv`）放入 `data/` 目录。
   - 数据格式需包含：`date`, `symbol`, `frequency`, `open`, `high`, `low`, `close`, `volume`, `amount`, `position`。

2. **配置参数**：
   - 修改 `src/config.json`，指定数据组和分析方法：
     ```json
     {
         "data_groups": [["rb2510.csv", "rb2505.csv", "hc2505.csv"]],
         "market_direction": "up",
         "methods": ["ml"]
     }