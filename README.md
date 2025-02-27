# Futures Strength Analyzer

一个用于分析期货合约强弱的项目，支持打分法和机器学习方法。

## 项目目录结构
FuturesStrengthAnalyzer/
├── data/                        # 数据文件存放目录
│   ├── rb2505.csv              # 螺纹钢2505合约数据
│   ├── rb2510.csv              # 螺纹钢2510合约数据
│   ├── hc2505.csv              # 热卷2505合约数据
├── results/                     # 结果输出目录（运行后生成）
│   ├── combined_log.log        # 主程序综合日志（示例）
│   ├── ml_log.log             # ML方法日志（示例）
│   ├── scoring_log.log        # 打分法日志（示例）
│   └── result_group_1_*.csv    # 交易建议结果文件（示例）
├── src/                         # 源代码目录
│   ├── init.py             # 模块初始化文件
│   ├── config.json             # 全局配置文件（数据组、市场方向等）
│   ├── data_processor.py       # 数据预处理模块（清洗、时间转换等）
│   ├── logging_utils.py        # 日志配置工具
│   ├── main.py                 # 主程序（综合运行打分法和ML）
│   ├── recommender.py          # 交易建议生成模块（打分法、ML、DL推荐器）
│   ├── data_acquisition/       # 数据获取模块
│   │   ├── init.py
│   │   └── juejin_data_downloader.py  # 从掘金量化平台下载K线和Tick数据
│   ├── deeplearning/           # 深度学习模块
│   │   ├── init.py
│   │   └── evaluator.py        # DL模型（LSTM、GRU、CNN等）
│   ├── features/               # 特征提取模块
│   │   ├── init.py
│   │   ├── description.md      # 35个特征的含义与作用说明
│   │   ├── extractor.py        # 特征提取主逻辑
│   │   ├── features.py         # 定义35个创新特征（如价格动量、波动周期等）
│   │   └── labelers.py         # 标签生成器（基于收益率、成交量等）
│   ├── ml/                     # 机器学习模块
│   │   ├── init.py
│   │   ├── main_ml.py          # ML主程序
│   │   ├── ml_config.json      # ML配置（特征选择、模型类型等）
│   │   ├── models.py           # 定义多种ML模型（RF、XGBoost、Stacking等）
│   │   └── predictor.py        # ML预测器（训练与预测逻辑）
│   ├── pca/                    # 主成分分析模块
│   │   ├── init.py
│   │   └── evaluator.py        # PCA评估器（基于多维特征）
│   ├── rsi/                    # RSI对比模块
│   │   ├── init.py
│   │   └── evaluator.py        # （未完整实现）
│   ├── rules/                  # 基于规则的专家系统模块
│   │   ├── init.py
│   │   └── evaluator.py        # 规则评估器（均线突破、成交量增加等）
│   ├── scoring/                # 打分法模块
│   │   ├── init.py
│   │   ├── analyses.py         # 定义35个创新分析类（如趋势加速度、日内波动等）
│   │   ├── evaluator.py        # 打分评估器（综合多个分析模块）
│   │   ├── main_scoring.py     # 打分法主程序
│   │   ├── scoring_config.json # 打分法权重配置
│   │   └── Scoring_Analyses.md # 35个分析类的含义与作用说明
│   ├── stats/                  # 统计方法模块
│   │   ├── init.py
│   │   └── evaluator.py        # 统计评估器（收益率、波动率、夏普比率等）
│   └── timeseries/             # 时间序列分析模块
│       ├── init.py
│       └── evaluator.py        # 时间序列模型（ARIMA、GARCH、Holt-Winters等）
└── README.md                    # 项目说明文档（当前文件）

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