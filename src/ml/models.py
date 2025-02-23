# src/ml/models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

class MLModel:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

class RandomForestModel(MLModel):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class XGBoostModel(MLModel):
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class GradientBoostingModel(MLModel):
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class LightGBMModel(MLModel):
    def __init__(self):
        self.model = lgb.LGBMClassifier(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class CatBoostModel(MLModel):
    def __init__(self):
        self.model = cat.CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class LogisticRegressionModel(MLModel):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class SVMModel(MLModel):
    def __init__(self):
        self.model = SVC(probability=True, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class AdaBoostModel(MLModel):
    def __init__(self):
        self.model = AdaBoostClassifier(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class ExtraTreesModel(MLModel):
    def __init__(self):
        self.model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class KNeighborsModel(MLModel):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    
class StackingModel(MLModel):
    def __init__(self, base_model_type: str = None):
        self.all_models = {
            "random_forest": RandomForestModel(),
            "xgboost": XGBoostModel(),
            "gradient_boosting": GradientBoostingModel(),
            "lightgbm": LightGBMModel(),
            "catboost": CatBoostModel(),
            "svm": SVMModel(),
            "adaboost": AdaBoostModel(),
            "extra_trees": ExtraTreesModel(),
            "knn": KNeighborsModel()
        }
        self.base_model_type = base_model_type
        if base_model_type and base_model_type in self.all_models:
            self.base_model = self.all_models[base_model_type]
            self.other_models = [model for key, model in self.all_models.items() if key != base_model_type]
        else:
            self.base_model = None
            self.other_models = list(self.all_models.values())
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 训练基模型并生成元特征
        if self.base_model:
            base_pred = self.base_model.predict(X)
            other_preds = np.column_stack([model.predict(X) for model in self.other_models])
            meta_features = np.column_stack([base_pred, other_preds])
            self.base_model.fit(X, y)
        else:
            meta_features = np.column_stack([model.predict(X) for model in self.other_models])
        self.meta_model.fit(meta_features, y)
        for model in self.other_models:
            model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.base_model:
            base_pred = self.base_model.predict(X)
            other_preds = np.column_stack([model.predict(X) for model in self.other_models])
            meta_features = np.column_stack([base_pred, other_preds])
        else:
            meta_features = np.column_stack([model.predict(X) for model in self.other_models])
        return self.meta_model.predict_proba(meta_features)[:, 1]