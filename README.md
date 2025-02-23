FuturesStrengthAnalyzer/
├── data/
│   ├── rb2505.csv
│   ├── rb2510.csv
│   ├── hc2505.csv
├── results/
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   ├── features.py
│   │   └── labelers.py
│   ├── scoring/          # 打分法
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── analyses.py
│   ├── ml/              # 机器学习
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── models.py
│   ├── stats/           # 新增：统计方法
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── timeseries/      # 新增：时间序列分析
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── rules/           # 新增：基于规则的专家系统
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── rsi/             # 新增：RSI对比
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── pca/             # 新增：主成分分析
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── deeplearning/    # 新增：深度学习（基础实现）
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── recommender.py
│   └── main.py
├── config.json
└── README.md


# Futures Strength Analyzer

一个用于分析期货合约强弱的项目，支持打分法和机器学习方法。

## 目录结构
- `data/`: 数据文件存放目录
- `src/`: 源代码目录
  - `data_processor.py`: 数据预处理
  - `features/`: 特征提取模块
  - `scoring/`: 打分法模块
  - `ml/`: 机器学习模块
  - `recommender.py`: 交易建议
  - `main.py`: 主程序

## 使用方法
1. 将数据放入 `data/` 目录。
2. 修改 `config.json` 配置方法和参数。
3. 运行 `python src/main.py`。

## 依赖
- pandas
- numpy
- scikit-learn
- xgboost