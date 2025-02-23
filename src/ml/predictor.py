from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .models import MLModel
import pandas as pd

class MLPredictor:
    def __init__(self, models: List[MLModel], feature_cols: List[str]):
        self.models = models
        self.feature_cols = feature_cols

    def train(self, features: pd.DataFrame):
        X = features[self.feature_cols]
        for i, model in enumerate(self.models):
            y = features[f'label{i+1}']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.train(X_train, y_train)
            print(f"Contract{i+1} 准确率: {accuracy_score(y_test, model.predict(X_test)):.2f}")

    def predict(self, features: pd.DataFrame) -> Dict[str, str]:
        X = features[self.feature_cols].iloc[-1:]
        return {f"contract{i+1}": model.predict(X)[0] for i, model in enumerate(self.models)}