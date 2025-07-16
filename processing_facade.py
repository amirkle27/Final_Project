from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from errors import InvalidPenaltySolverCombination
from preprocessoring_strategy import *
from train_and_save_model import y_test


class LinearRegressionFacade:
    def __init__(self, test_size:float = 0.2, random_state:int = 27):
        self.preprocessor = LinearRegressionPreprocessor()
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df:pd.DataFrame, target_col:str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train,y_train)

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)

        return {
            "model": self.model,
            "scaler": self.scaler,
            "mse": mse,
            "r2": r2,
            "y_test": y_test,
            "y_pred": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        new_data_scaled = pd.DataFrame(self.scaler.transform(new_data), columns=new_data.columns)
        return self.model.predict(new_data_scaled)


class DecisionTreeClassifierFacade:
    def __init__(self, test_size:float = 0.2, random_state:int = 27):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = DecisionTreeClassifier()
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)

        return {
            "model": self.model,
            "accuracy": accuracy,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class DecisionTreeRegressorFacade:
    def __init__(self, test_size:float = 0.2, random_state:int = 27):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = DecisionTreeRegressor()
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "model": self.model,
            "mse": mse,
            "r2": r2,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class RandomForestRegressorFacade:
    def __init__(self, test_size:float = 0.2, random_state:int = 27, criterion="squared_error"):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = RandomForestRegressor(random_state=random_state, criterion=criterion)
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "model": self.model,
            "mse": mse,
            "r2": r2,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)

class RandomForestClassifierFacade:
    def __init__(self, test_size:float = 0.2, random_state:int = 27, criterion="gini"):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = RandomForestClassifier(random_state=random_state, criterion=criterion)
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)

        return {
            "model": self.model,
            "accuracy": accuracy,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class LogisticRegressionFacade:
    def __init__(self, test_size: float = 0.2, random_state: int = 27, solver='lbfgs', penalty='l2', C=1.0, max_iter: int = 1000):
        self.preprocessor = LogisticRegressionPreprocessor()
        default_penalty = 'l2'
        default_solver = 'lbfgs'

        self.penalty = penalty or default_penalty
        self.solver = solver or default_solver

        try:
            self.validate_penalty_solver(self.penalty, self.solver)
        except InvalidPenaltySolverCombination as e:
            print(f"[User Error]: {e}")
            print(f"[Default Parameters Selected: penalty='{default_penalty}', solver='{default_solver}']")

            self.penalty = default_penalty
            self.solver = default_solver

        self.model = LogisticRegression(solver=self.solver,penalty=self.penalty,C=C, max_iter=max_iter)
        self.test_size = test_size
        self.random_state = random_state

    def validate_penalty_solver(self, penalty: str, solver: str):
        valid_combinations = {
            'l1': ['liblinear', 'saga'],
            'l2': ['lbfgs', 'liblinear', 'sag', 'saga'],
            'elasticnet': ['saga'],
            None: ['lbfgs', 'saga']
        }

        if solver not in valid_combinations.get(penalty, []):
            raise InvalidPenaltySolverCombination(penalty, solver)


    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)

        return {
            "model": self.model,
            "accuracy": accuracy,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)

class LogisticRegressionCVFacade:
    def __init__(self, test_size: float = 0.2, random_state: int = 27, solver='lbfgs', penalty='l2',
                 max_iter: int = 1000, cv: int = 5, scoring: str = 'accuracy'):
        self.preprocessor = LogisticRegressionPreprocessor()
        default_penalty = 'l2'
        default_solver = 'lbfgs'

        self.penalty = penalty or default_penalty
        self.solver = solver or default_solver

        try:
            self.validate_penalty_solver(self.penalty, self.solver)
        except InvalidPenaltySolverCombination as e:
            print(f"[User Error]: {e}")
            print(f"[Default Parameters Selected: penalty='{default_penalty}', solver='{default_solver}']")

            self.penalty = default_penalty
            self.solver = default_solver

        self.model = LogisticRegressionCV(solver=self.solver,penalty=self.penalty,max_iter=max_iter,
            cv=cv, scoring=scoring, random_state=random_state, Cs=10 )
        self.test_size = test_size
        self.random_state = random_state

    def validate_penalty_solver(self, penalty: str, solver: str):
        valid_combinations = {
            'l1': ['liblinear', 'saga'],
            'l2': ['lbfgs', 'liblinear', 'sag', 'saga'],
            'elasticnet': ['saga'],
            None: ['lbfgs', 'saga']
        }

        if solver not in valid_combinations.get(penalty, []):
            raise InvalidPenaltySolverCombination(penalty, solver)


    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)

        return {
            "model": self.model,
            "best_C": self.model.C_[0],
            "accuracy": accuracy,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)

class KMeansClusteringFacade:
    def __init__(self, n_clusters:int = 3, random_state:int = 27, max_iter:int = 300):
        self.preprocessor = ClusteringPreprocessor()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
        self.scaler = StandardScaler()
        self.fitted = False

    def train_and_cluster(self, df: pd.DataFrame) -> dict:
        X, _ = self.preprocessor.process(df)
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.fitted = True

        labels = self.model.labels_
        centroids = self.model.cluster_centers_
        score = silhouette_score(X_scaled, labels)

        # 🪪 הוספת עמודת אשכול ל-DF המקורי (אופציונלי)
        clustered_df = X.copy()
        clustered_df["cluster_label"] = labels

        return {
            "clustered_data": clustered_df,
            "centroids": centroids,
            "silhouette_score": score
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise NotFittedError("Model must be fit before predicting.")

        X, _ = self.preprocessor.process(new_data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)