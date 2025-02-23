# src/timeseries/evaluator.py
from typing import List, Dict
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def evaluate(datasets: List[pd.DataFrame], window: int = 50) -> Dict[str, float]:
    """时间序列方法：基于ARIMA预测趋势判断强弱"""
    trends = {}
    for i, data in enumerate(datasets):
        series = data['close'].tail(window)
        try:
            model = ARIMA(series, order=(1, 1, 1)).fit()
            forecast = model.forecast(steps=1)[0]
            trend = (forecast - series.iloc[-1]) / series.iloc[-1]  # 预测变化率
            trends[f"contract{i+1}"] = trend
        except:
            trends[f"contract{i+1}"] = 0.0  # 如果模型失败，返回中性
    return trends