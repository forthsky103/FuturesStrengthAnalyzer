# src/timeseries/evaluator.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class TimeSeriesModel(ABC):
    """时间序列模型基类"""
    @abstractmethod
    def fit_and_forecast(self, data: pd.DataFrame) -> float:
        """拟合模型并预测趋势得分"""
        pass

class ARIMAModel(TimeSeriesModel):
    """ARIMA模型"""
    def __init__(self, window: int = 50, order: tuple = (1, 1, 1), field: str = "close"):
        self.window = window
        self.order = order
        self.field = field

    def fit_and_forecast(self, data: pd.DataFrame) -> float:
        series = data[self.field].tail(self.window)
        try:
            model = ARIMA(series, order=self.order).fit()
            forecast = model.forecast(steps=1)[0]
            return (forecast - series.iloc[-1]) / series.iloc[-1]  # 预测变化率
        except:
            return 0.0  # 如果失败，返回中性

class GARCHModel(TimeSeriesModel):
    """GARCH模型（预测波动率）"""
    def __init__(self, window: int = 50, p: int = 1, q: int = 1, field: str = "close"):
        self.window = window
        self.p = p
        self.q = q
        self.field = field

    def fit_and_forecast(self, data: pd.DataFrame) -> float:
        series = data[self.field].pct_change().dropna().tail(self.window)
        try:
            model = arch_model(series, vol='Garch', p=self.p, q=self.q).fit(disp='off')
            forecast = model.forecast(horizon=1).variance.iloc[-1, 0]
            return -forecast / 1000  # 波动率反向，缩放
        except:
            return 0.0

class HoltWintersModel(TimeSeriesModel):
    """Holt-Winters模型（指数平滑）"""
    def __init__(self, window: int = 50, seasonal_periods: int = 12, field: str = "close"):
        self.window = window
        self.seasonal_periods = seasonal_periods
        self.field = field

    def fit_and_forecast(self, data: pd.DataFrame) -> float:
        series = data[self.field].tail(self.window)
        try:
            model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=self.seasonal_periods).fit()
            forecast = model.forecast(1).iloc[0]
            return (forecast - series.iloc[-1]) / series.iloc[-1]
        except:
            return 0.0

class TimeSeriesEvaluator:
    """时间序列评估器"""
    def __init__(self, models: List[TimeSeriesModel], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {model.__class__.__name__: 1.0 for model in models}

    def evaluate(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        """综合多个时间序列模型评估强弱"""
        scores = {}
        for i, data in enumerate(datasets):
            total_score = 0.0
            for model in self.models:
                model_name = model.__class__.__name__
                score = model.fit_and_forecast(data)
                total_score += score * self.weights.get(model_name, 1.0)
            scores[f"contract{i+1}"] = total_score / sum(self.weights.values()) if self.weights else 0.0
        return scores