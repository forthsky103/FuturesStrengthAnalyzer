# src/scoring/evaluator.py
from typing import List, Dict
from .analyses import AnalysisModule
import pandas as pd

class StrengthEvaluator:
    def __init__(self, analysis_modules: List[AnalysisModule], weights: Dict[str, float]):
        self.modules = analysis_modules
        self.weights = weights

    def evaluate(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {f"contract{i+1}": 0 for i in range(len(datasets))}
        for module in self.modules:
            name = module.__class__.__name__
            module_scores = module.analyze(datasets)  # 返回 {"contract1": score1, ...}
            for contract, score in module_scores.items():
                scores[contract] += score * self.weights.get(name, 1.0)
        total_weight = sum(self.weights.values()) or 1  # 防止除零
        return {contract: score / total_weight for contract, score in scores.items()}