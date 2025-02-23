from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

class AnalysisModule(ABC):
    @abstractmethod
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        pass

class TrendAnalysis(AnalysisModule):
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            trend = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
            scores[f"contract{i+1}"] = min(max(trend + 5, 0), 10)
        return scores

class VolumeAnalysis(AnalysisModule):
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        vols = [data['volume'].mean() for data in datasets]
        max_vol = max(vols)
        return {f"contract{i+1}": (vol / max_vol) * 10 if max_vol > 0 else 5 
                for i, vol in enumerate(vols)}