from matplotlib import pyplot as plt
from pandas.core.common import random_state
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from errors import InvalidPenaltySolverCombination, NotFittedError, MultipleFeaturesPolyError
from preprocessoring_strategy import *

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from scipy.optimize import minimize


from typing import Optional
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


class LinearRegressionFacade:
    def __init__(self, test_size:float = 0.2, random_state:int = 27):
        self.preprocessor = LinearRegressionPreprocessor()
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state

        self.model = LinearRegression()
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

        clustered_df = X.copy()
        clustered_df["cluster_label"] = labels

        return {
            "clustered_data": clustered_df,
            "centroids": centroids,
            "silhouette_score": score
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise NotFittedError

        X, _ = self.preprocessor.process(new_data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class DBScanClusteringFacade:
    def __init__(self, min_samples:int = 5, eps:float = 0.5):
        self.preprocessor = ClusteringPreprocessor()
        self.min_samples = min_samples
        self.eps = eps
        self.model = DBSCAN(min_samples=min_samples, eps=eps)
        self.scaler = StandardScaler()
        self.fitted = False

    def train_and_cluster(self, df: pd.DataFrame) -> dict:
        X, _ = self.preprocessor.process(df)
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.fitted = True

        labels = self.model.labels_
        score = silhouette_score(X_scaled, labels)

        clustered_df = X.copy()
        clustered_df["cluster_label"] = labels

        return {
            "clustered_data": clustered_df,
            "silhouette_score": score,
            "labels": labels
        }

class KNNFacade:
    def __init__(self, test_size: float = 0.2, random_state: int = 27, n_neighbors = 3):
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.preprocessor = KNNPreprocessor()
        self.model = KNeighborsClassifier(n_neighbors)
        self.scaler = MinMaxScaler()
        self.fitted = False

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        self.fitted = True
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "model": self.model,
            "n_neighbors": self.n_neighbors,
            "accuracy": accuracy,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise NotFittedError

        X, _ = self.preprocessor.process(new_data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)




class ANNFacade:
    def __init__(self, test_size: float = 0.2, random_state: int = 27, hidden_layers: list = [64, 32], epochs: int = 20, batch_size: int = 32):
        self.test_size = test_size
        self.random_state = random_state
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocessor = ANNCNNPreprocessor()
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def build_model(self, input_dim: int, output_dim: int):
        model = Sequential()
        model.add(Dense(self.hidden_layers[0], input_dim=input_dim, activation='relu'))
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))

        if output_dim == 1:
            model.add(Dense(1, activation='sigmoid'))
            loss = BinaryCrossentropy()
        else:
            model.add(Dense(output_dim, activation='softmax'))
            loss = SparseCategoricalCrossentropy()

        model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
        return model

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        output_dim = 1 if y.nunique() == 2 else y.nunique()
        self.model = self.build_model(input_dim=X.shape[1], output_dim=output_dim)

        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.fitted = True

        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=1)
        y_pred = self.model.predict(X_test)

        return {
            "model": self.model,
            "accuracy": test_acc,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise NotFittedError

        X, _ = self.preprocessor.process(new_data, target_col=target_col)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled)

        if probabilities.shape[1] == 1:
            predicted_class = (probabilities > 0.5).astype(int).flatten()
            confidence = probabilities.flatten()
        else:
            predicted_class = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)

        results_df = pd.DataFrame({
            "prediction": predicted_class,
            "confidence": confidence
        })
        prob_df = pd.DataFrame(probabilities, columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])])
        results_df = pd.concat([results_df, prob_df], axis=1)

        return results_df


class CNNFacade:
    def __init__(self, test_size: float = 0.2, random_state: int = 27, filters: int = 64, kernel_size: int = 3,
                 epochs: int = 20, batch_size: int = 32):
        self.test_size = test_size
        self.random_state = random_state
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocessor = ANNCNNPreprocessor()
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def build_model(self, input_shape: tuple, output_dim: int):
        model = Sequential()
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                         activation='relu', input_shape=input_shape))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))

        if output_dim == 1:
            model.add(Dense(1, activation='sigmoid'))
            loss = BinaryCrossentropy()
        else:
            model.add(Dense(output_dim, activation='softmax'))
            loss = SparseCategoricalCrossentropy()

        model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
        return model

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        output_dim = 1 if y.nunique() == 2 else y.nunique()
        self.model = self.build_model(input_shape=X_train.shape[1:], output_dim=output_dim)

        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        self.fitted = True

        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=1)
        y_pred = self.model.predict(X_test)

        return {
            "model": self.model,
            "accuracy": test_acc,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise NotFittedError

        X, _ = self.preprocessor.process(new_data, target_col=target_col)
        X_scaled = self.scaler.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        probabilities = self.model.predict(X_scaled)

        if probabilities.shape[1] == 1:
            predicted_class = (probabilities > 0.5).astype(int).flatten()
            confidence = probabilities.flatten()
        else:
            predicted_class = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)

        results_df = pd.DataFrame({
            "prediction": predicted_class,
            "confidence": confidence
        })
        prob_df = pd.DataFrame(probabilities, columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])])
        results_df = pd.concat([results_df, prob_df], axis=1)

        return results_df


class NLPFacade:
    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 27,
                 max_words: int = 10000,
                 max_len: int = 100,
                 embedding_dim: int = 64,
                 epochs: int = 10,
                 batch_size: int = 32):

        self.test_size = test_size
        self.random_state = random_state
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocessor = NLPPreprocessor(num_words=max_words, max_len=max_len)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.fitted = False

    def build_model(self, output_dim: int):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, input_length=self.max_len))
        model.add(LSTM(64))
        model.add(Dense(32, activation='relu'))

        if output_dim == 1:
            model.add(Dense(1, activation='sigmoid'))
            loss = BinaryCrossentropy()
        else:
            model.add(Dense(output_dim, activation='softmax'))
            loss = SparseCategoricalCrossentropy()

        model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
        return model

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str):
        X, y = self.preprocessor.process(df, target_col)
        output_dim = len(np.unique(y))
        if output_dim > 2:
            y = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        self.model = self.build_model(output_dim)
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        self.fitted = True

        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=1)
        y_pred = self.model.predict(X_test)

        return {
            "model": self.model,
            "accuracy": test_acc,
            "y_test": y_test,
            "prediction": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call train_and_evaluate first.")

        texts = new_data.select_dtypes(include='object').iloc[:, 0]
        sequences = self.preprocessor.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        probabilities = self.model.predict(padded)

        if probabilities.shape[1] == 1:
            predicted_class = (probabilities > 0.5).astype(int).flatten()
            confidence = probabilities.flatten()
        else:
            predicted_class = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)

        results_df = pd.DataFrame({
            "prediction": predicted_class,
            "confidence": confidence
        })

        prob_df = pd.DataFrame(probabilities, columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])])
        results_df = pd.concat([results_df, prob_df], axis=1)

        return results_df



class PolynomialFacade:
    def __init__(self, preprocessor, degree: int = 2):
        self.preprocessor = preprocessor
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.X = None
        self.y = None
        self.X_poly = None
        self.y_pred = None

    def train_and_evaluate(self, df, target_col: str):
        X_raw, y, poly = self.preprocessor.process(df, target_col)
        self.poly = poly
        self.X = X_raw
        self.y = y
        self.X_poly = self.poly.fit_transform(X_raw)
        self.model.fit(self.X_poly, y)
        self.y_pred = self.model.predict(self.X_poly)
        r2 = r2_score(self.y, self.y_pred)
        mse = mean_squared_error(self.y, self.y_pred)

        return {
            "R²": f"{r2:.4f}",
            "mse": f"{mse:.2f}",
        }


    def plot(self):
        try:
            if self.X.shape[1] != 1:
                raise MultipleFeaturesPolyError()


            x_line = np.linspace(self.X.min().values[0], self.X.max().values[0], 100).reshape(-1, 1)
            x_line_poly = self.poly.transform(x_line)
            y_line = self.model.predict(x_line_poly)

            plt.scatter(self.X, self.y, color='red', label='Data')
            plt.plot(x_line, y_line, color='blue', label='Prediction')
            plt.xlabel(self.X.columns[0])
            plt.ylabel("Target")
            plt.title("Polynomial Regression")
            plt.legend()
            plt.grid(True)
            plt.show()
        except MultipleFeaturesPolyError as e:
            print(e)

    def get_optimal_x(self):
        if self.X.shape[1] == 1 and self.degree == 2:
            a = self.model.intercept_
            b = self.model.coef_[1]
            c = self.model.coef_[2]
            x_opt = -b / (2 * c)
            y_opt = self.model.predict(self.poly.transform([[x_opt]]))[0]
            print(f"Optimal x = {x_opt:.2f}, y = {y_opt:.2f}")
            return x_opt, y_opt
        else:
            def negative_prediction(x):
                x = np.array(x).reshape(1, -1)
                x_poly = self.poly.transform(x)
                return -self.model.predict(x_poly)[0]

            bounds = []
            for col in self.X.columns:
                col_data = self.X[col]
                bounds.append((col_data.min(), col_data.max()))

            initial_guess = self.X.mean().values.tolist()
            result = minimize(negative_prediction, x0=initial_guess, bounds=bounds)
            x_opt = result.x
            y_opt = -result.fun
            print("Optimal feature values for highest prediction:")
            for name, value in zip(self.X.columns, x_opt):
                print(f"{name}: {value:.2f}")
            print(f"Predicted y = {y_opt:.2f}")
            return x_opt, y_opt



