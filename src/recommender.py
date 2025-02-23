# src/recommender.py
from typing import Union, Dict, List
import pandas as pd
from datetime import datetime

class TradeRecommender:
    def __init__(self, market_direction: str):
        self.direction = market_direction.lower()

    def recommend(self, results: Dict[str, Union[float, str]], method: str, symbols: List[str], group_idx: int) -> str:
        # 生成建议
        if method == 'scoring':
            scores = {symbols[i]: score for i, score in enumerate(results.values())}
            strongest = max(scores, key=scores.get)
            weakest = min(scores, key=scores.get)
            if self.direction == 'up':
                advice = f"做多 {strongest}（得分: {scores[strongest]:.2f}），做空 {weakest}（得分: {scores[weakest]:.2f}）"
            elif self.direction == 'down':
                advice = f"做空 {weakest}（跌势更快，得分: {scores[weakest]:.2f}），做多 {strongest}（跌势较慢，得分: {scores[strongest]:.2f}）"
            else:
                advice = "市场震荡，建议观望"
            result_data = pd.DataFrame({
                "symbol": symbols,
                "score": [scores[symbol] for symbol in symbols],
                "is_strongest": [symbol == strongest for symbol in symbols],
                "is_weakest": [symbol == weakest for symbol in symbols]
            })
        else:  # ML
            preds = {symbols[i]: pred for i, pred in enumerate(results.values())}
            strongest = next(symbol for symbol, pred in preds.items() if pred == 'strong')
            weakest = next(symbol for symbol, pred in preds.items() if pred == 'weak')
            if self.direction == 'up':
                advice = f"做多 {strongest}（强势），做空 {weakest}（弱势）"
            elif self.direction == 'down':
                advice = f"做空 {weakest}（跌势更快），做多 {strongest}（跌势较慢）"
            else:
                advice = "市场震荡，建议观望"
            result_data = pd.DataFrame({
                "symbol": symbols,
                "prediction": [preds[symbol] for symbol in symbols],
                "is_strongest": [symbol == strongest for symbol in symbols],
                "is_weakest": [symbol == weakest for symbol in symbols]
            })

        # 保存结果到本地
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../results/result_group_{group_idx + 1}_{timestamp}.csv"
        result_data.to_csv(filename, index=False)
        print(f"结果已保存至: {filename}")

        return advice