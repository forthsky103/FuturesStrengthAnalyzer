# src/utils/recommender.py
from typing import Union, Dict, List
import pandas as pd
from datetime import datetime
import abc
import os

class TradeRecommender(abc.ABC):
    """交易推荐基类"""
    def __init__(self, market_direction: str, config: Dict):
        self.direction = market_direction.lower()
        self.config = config
        # 获取项目根目录
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.results_dir = os.path.join(self.root_dir, config.get("results_dir", "results"))

    @abc.abstractmethod
    def recommend(self, results: Dict[str, Union[float, str]], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
        """生成交易建议"""
        pass

    def _save_results(self, result_data: pd.DataFrame, group_idx: int) -> str:
        """保存结果到 CSV 文件，从配置文件读取路径"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"result_group_{group_idx + 1}_{timestamp.replace(':', '').replace(' ', '_')}.csv"
        result_path = os.path.join(self.results_dir, filename)
        result_data.to_csv(result_path, index=False)
        print(f"结果已保存至: {result_path}")
        return result_path

class ScoringTradeRecommender(TradeRecommender):
    """打分法推荐器"""
    def recommend(self, results: Dict[str, float], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scores = {symbols[int(k[-1]) - 1]: v for k, v in results.items()}
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)
        
        if self.direction == 'up':
            advice = f"做多 {strongest}（得分: {scores[strongest]:.4f}），做空 {weakest}（得分: {scores[weakest]:.4f}）"
        elif self.direction == 'down':
            advice = f"做空 {weakest}（得分: {scores[weakest]:.4f}），做多 {strongest}（得分: {scores[strongest]:.4f}）"
        else:
            advice = "市场震荡，建议观望"

        result_data = pd.DataFrame({
            "timestamp": [timestamp] * len(symbols),
            "group_id": [group_idx + 1] * len(symbols),
            "symbol": symbols,
            "score": [scores[symbol] for symbol in symbols],
            "is_strongest": [symbol == strongest for symbol in symbols],
            "is_weakest": [symbol == weakest for symbol in symbols],
            "market_direction": [self.direction] * len(symbols)
        })
        self._save_results(result_data, group_idx)
        return advice

class MLTradeRecommender(TradeRecommender):
    """机器学习推荐器"""
    def recommend(self, results: Dict[str, Union[float, str]], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        preds = {symbols[int(k[-1]) - 1]: v for k, v in results.items()}
        
        if isinstance(list(preds.values())[0], float):
            raise ValueError("MLTradeRecommender 期望 'strong' 或 'weak' 标签，请在 main_ml.py 中预处理概率")
        strongest = next(symbol for symbol, pred in preds.items() if pred == 'strong')
        weakest = next(symbol for symbol, pred in preds.items() if pred == 'weak')
        
        if self.direction == 'up':
            advice = f"做多 {strongest}（预测: {preds[strongest]}），做空 {weakest}（预测: {preds[weakest]}）"
        elif self.direction == 'down':
            advice = f"做空 {weakest}（预测: {preds[weakest]}），做多 {strongest}（预测: {preds[strongest]}）"
        else:
            advice = "市场震荡，建议观望"

        result_data = pd.DataFrame({
            "timestamp": [timestamp] * len(symbols),
            "group_id": [group_idx + 1] * len(symbols),
            "symbol": symbols,
            "prediction": [preds[symbol] for symbol in symbols],
            "is_strongest": [symbol == strongest for symbol in symbols],
            "is_weakest": [symbol == weakest for symbol in symbols],
            "market_direction": [self.direction] * len(symbols)
        })
        self._save_results(result_data, group_idx)
        return advice

class DeepLearningTradeRecommender(TradeRecommender):
    """深度学习推荐器"""
    def recommend(self, results: Dict[str, float], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scores = {symbols[int(k[-1]) - 1]: v for k, v in results.items()}
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)
        
        if self.direction == 'up':
            advice = f"做多 {strongest}（DL概率: {scores[strongest]:.4f}），做空 {weakest}（DL概率: {scores[weakest]:.4f}）"
        elif self.direction == 'down':
            advice = f"做空 {weakest}（DL概率: {scores[weakest]:.4f}），做多 {strongest}（DL概率: {scores[strongest]:.4f}）"
        else:
            advice = "市场震荡，建议观望"

        result_data = pd.DataFrame({
            "timestamp": [timestamp] * len(symbols),
            "group_id": [group_idx + 1] * len(symbols),
            "symbol": symbols,
            "dl_probability": [scores[symbol] for symbol in symbols],
            "is_strongest": [symbol == strongest for symbol in symbols],
            "is_weakest": [symbol == weakest for symbol in symbols],
            "market_direction": [self.direction] * len(symbols)
        })
        self._save_results(result_data, group_idx)
        return advice







# # src/recommender.py
# from typing import Union, Dict, List
# import pandas as pd
# from datetime import datetime
# import abc

# class TradeRecommender(abc.ABC):
#     """交易推荐基类"""
#     def __init__(self, market_direction: str):
#         self.direction = market_direction.lower()

#     @abc.abstractmethod
#     def recommend(self, results: Dict[str, Union[float, str]], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
#         """生成交易建议"""
#         pass

#     def _save_results(self, result_data: pd.DataFrame, group_idx: int) -> str:
#         """保存结果到 CSV 文件"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         filename = f"../results/result_group_{group_idx + 1}_{timestamp.replace(':', '').replace(' ', '_')}.csv"
#         result_data.to_csv(filename, index=False)
#         print(f"结果已保存至: {filename}")
#         return filename

# class ScoringTradeRecommender(TradeRecommender):
#     """打分法推荐器"""
#     def recommend(self, results: Dict[str, float], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         scores = {symbols[int(k[-1]) - 1]: v for k, v in results.items()}
#         strongest = max(scores, key=scores.get)
#         weakest = min(scores, key=scores.get)
        
#         if self.direction == 'up':
#             advice = f"做多 {strongest}（得分: {scores[strongest]:.4f}），做空 {weakest}（得分: {scores[weakest]:.4f}）"
#         elif self.direction == 'down':
#             advice = f"做空 {weakest}（得分: {scores[weakest]:.4f}），做多 {strongest}（得分: {scores[strongest]:.4f}）"
#         else:
#             advice = "市场震荡，建议观望"

#         result_data = pd.DataFrame({
#             "timestamp": [timestamp] * len(symbols),
#             "group_id": [group_idx + 1] * len(symbols),
#             "symbol": symbols,
#             "score": [scores[symbol] for symbol in symbols],
#             "is_strongest": [symbol == strongest for symbol in symbols],
#             "is_weakest": [symbol == weakest for symbol in symbols],
#             "market_direction": [self.direction] * len(symbols)
#         })
#         self._save_results(result_data, group_idx)
#         return advice

# class MLTradeRecommender(TradeRecommender):
#     """机器学习推荐器"""
#     def recommend(self, results: Dict[str, Union[float, str]], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         preds = {symbols[int(k[-1]) - 1]: v for k, v in results.items()}
        
#         # 转换为 strong/weak 标签（若传入概率则需在外部处理）
#         if isinstance(list(preds.values())[0], float):
#             raise ValueError("MLTradeRecommender 期望 'strong' 或 'weak' 标签，请在 main_ml.py 中预处理概率")
#         strongest = next(symbol for symbol, pred in preds.items() if pred == 'strong')
#         weakest = next(symbol for symbol, pred in preds.items() if pred == 'weak')
        
#         if self.direction == 'up':
#             advice = f"做多 {strongest}（预测: {preds[strongest]}），做空 {weakest}（预测: {preds[weakest]}）"
#         elif self.direction == 'down':
#             advice = f"做空 {weakest}（预测: {preds[weakest]}），做多 {strongest}（预测: {preds[strongest]}）"
#         else:
#             advice = "市场震荡，建议观望"

#         result_data = pd.DataFrame({
#             "timestamp": [timestamp] * len(symbols),
#             "group_id": [group_idx + 1] * len(symbols),
#             "symbol": symbols,
#             "prediction": [preds[symbol] for symbol in symbols],
#             "is_strongest": [symbol == strongest for symbol in symbols],
#             "is_weakest": [symbol == weakest for symbol in symbols],
#             "market_direction": [self.direction] * len(symbols)
#         })
#         self._save_results(result_data, group_idx)
#         return advice

# class DeepLearningTradeRecommender(TradeRecommender):
#     """深度学习推荐器"""
#     def recommend(self, results: Dict[str, float], method: str, symbols: List[str], group_idx: int, datasets: List[pd.DataFrame] = None) -> str:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         scores = {symbols[int(k[-1]) - 1]: v for k, v in results.items()}
#         strongest = max(scores, key=scores.get)
#         weakest = min(scores, key=scores.get)
        
#         if self.direction == 'up':
#             advice = f"做多 {strongest}（DL概率: {scores[strongest]:.4f}），做空 {weakest}（DL概率: {scores[weakest]:.4f}）"
#         elif self.direction == 'down':
#             advice = f"做空 {weakest}（DL概率: {scores[weakest]:.4f}），做多 {strongest}（DL概率: {scores[strongest]:.4f}）"
#         else:
#             advice = "市场震荡，建议观望"

#         result_data = pd.DataFrame({
#             "timestamp": [timestamp] * len(symbols),
#             "group_id": [group_idx + 1] * len(symbols),
#             "symbol": symbols,
#             "dl_probability": [scores[symbol] for symbol in symbols],
#             "is_strongest": [symbol == strongest for symbol in symbols],
#             "is_weakest": [symbol == weakest for symbol in symbols],
#             "market_direction": [self.direction] * len(symbols)
#         })
#         self._save_results(result_data, group_idx)
#         return advice