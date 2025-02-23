FuturesStrengthAnalyzer/
├── data/                   # 存放数据文件
│   ├── rb2505.csv         # 示例数据：近月合约
│   └── rb2510.csv         # 示例数据：远月合约
├── src/                   # 源代码目录
│   ├── __init__.py        # 标记为Python包
│   ├── data_processor.py  # 数据预处理模块
│   ├── features/          # 特征提取模块
│   │   ├── __init__.py
│   │   ├── extractor.py   # 特征提取器
│   │   └── features.py    # 具体特征实现
│   ├── scoring/           # 打分法模块
│   │   ├── __init__.py
│   │   ├── evaluator.py   # 打分评估器
│   │   └── analyses.py    # 具体分析方法
│   ├── ml/                # 机器学习模块
│   │   ├── __init__.py
│   │   ├── predictor.py   # ML预测器
│   │   └── models.py      # 具体ML模型
│   ├── recommender.py     # 交易建议模块
│   └── main.py            # 主程序
├── config.json            # 配置文件
└── README.md              # 项目说明


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