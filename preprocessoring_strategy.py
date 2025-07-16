from abc import ABC, abstractmethod
import pandas as pd
from errors import TargetColumnNotFoundError
from typing import Optional
from sklearn.preprocessing import PolynomialFeatures

class PreprocessingStrategy(ABC):
    @abstractmethod
    def process(self, df:pd.DataFrame, target_col:str) -> tuple[pd.DataFrame, pd.Series]:
        pass

class LinearRegressionPreprocessor(PreprocessingStrategy):
    def process(self, df:pd.DataFrame, target_col:str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')
        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)
        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']

        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X,y


class DecisionTreePreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y

class LogisticRegressionPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y

class ClusteringPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col:Optional[str] = None ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')


        str_cols = [col for col in df.columns if df[col].dtype == 'object']

        if str_cols:
            df = pd.get_dummies(df, columns=str_cols)

        return df, None

class KNNPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y

class ANNCNNPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y

class NLPPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y





class PolynomialRegressionPreprocessor(PreprocessingStrategy):
    def __init__(self, degree: int = 2):
        self.degree = degree

    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, PolynomialFeatures]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        poly = PolynomialFeatures(degree=self.degree, include_bias=False)

        return X, y, poly
