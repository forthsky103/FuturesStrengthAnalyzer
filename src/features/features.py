import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List

class Feature(ABC):
    @abstractmethod
    def compute(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        pass

class TrendFeature(Feature):
    def __init__(self, window: int = 20, name: str = "trend"):
        self.window = window
        self.name = name

    def compute(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        return {f"{self.name}{i+1}": data['close'].pct_change(periods=self.window) 
                for i, data in enumerate(datasets)}

class VolumeFeature(Feature):
    def __init__(self, window: int = 20, name: str = "vol_mean"):
        self.window = window
        self.name = name

    def compute(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        return {f"{self.name}{i+1}": data['volume'].rolling(window=self.window).mean() 
                for i, data in enumerate(datasets)}

class SpreadFeature(Feature):
    def __init__(self, name: str = "spread"):
        self.name = name

    def compute(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        avg_close = pd.concat([data['close'] for data in datasets], axis=1).mean(axis=1)
        return {f"{self.name}{i+1}": data['close'] - avg_close 
                for i, data in enumerate(datasets)}

class VolatilityFeature(Feature):
    def __init__(self, window: int = 20, name: str = "volatility"):
        self.window = window
        self.name = name

    def compute(self, datasets: List[pd.DataFrame]) -> Dict[str, pd.Series]:
        return {f"{self.name}{i+1}": (data['high'] - data['low']).rolling(window=self.window).mean() 
                for i, data in enumerate(datasets)}