"""
Model 'facades' that wrap scikit-learn / Keras / XGBoost estimators behind a
uniform interface.

Each facade typically provides:
- `train_and_evaluate(df, target_col)` → dict of metrics and artifacts
- `predict(new_data)` or clustering-specific methods
and delegates preprocessing to a strategy from `preprocessoring_strategy`.

All facades raise `NotFittedError` (or RuntimeError) from `errors` when
`predict` is called before training.
"""

from matplotlib import pyplot as plt
from typing import Optional, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from scipy.optimize import minimize
from errors import (
    InvalidPenaltySolverCombination, NotFittedError, MultipleFeaturesPolyError,
    PolinomialMaxMinError, PolinomialNotDFError, PolinomialForClassificationError,
    LinearForClassificationError
)
from preprocessoring_strategy import *

# -------------------------------
# Regressors
# -------------------------------

class LinearRegressionFacade:
    """Facade for linear regression on tabular data.

    Pipeline:
      1) `LinearRegressionPreprocessor` (drop id/index cols, one-hot encode).
      2) Standardize features (fit ONLY on train to avoid leakage).
      3) Fit `sklearn.linear_model.LinearRegression`.

    Raises:
        LinearForClassificationError: if the target column is non-numeric.
    """
    def __init__(self, test_size: float = 0.2, random_state: int = 27):
        self.preprocessor = LinearRegressionPreprocessor()
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state
        self.feature_cols: Optional[pd.Index] = None

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        self.feature_cols = X.columns

        if not pd.api.types.is_numeric_dtype(y):
            raise LinearForClassificationError()

        # split BEFORE scaling to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test  = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        return {
            "model": self.model,
            "scaler": self.scaler,
            "mse": float(mse),
            "rmse": rmse,
            "r2": r2,
            "y_test": y_test,
            "y_pred": y_pred
        }

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        X_new = new_data.reindex(columns=self.feature_cols, fill_value=0)
        X_new_scaled = pd.DataFrame(self.scaler.transform(X_new), columns=self.feature_cols)
        return self.model.predict(X_new_scaled)


class DecisionTreeRegressorFacade:
    """Decision tree regressor with simple preprocessing."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = DecisionTreeRegressor()
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"model": self.model, "mse": mse, "r2": r2, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class RandomForestRegressorFacade:
    """Random forest regressor with decision-tree preprocessing."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27, criterion="squared_error"):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = RandomForestRegressor(random_state=random_state, criterion=criterion)
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"model": self.model, "mse": mse, "r2": r2, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class PolynomialFacade:
    """Polynomial regression wrapper that also supports plotting and simple optimization."""
    def __init__(self, preprocessor, degree: int = 2):
        self.preprocessor = preprocessor
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.feature_cols: Optional[pd.Index] = None

    def train_and_evaluate(self, df, target_col: str):
        X_raw, y, poly = self.preprocessor.process(df, target_col)
        self.poly = poly
        self.X = X_raw
        self.y = y
        self.feature_cols = X_raw.columns

        if not pd.api.types.is_numeric_dtype(y):
            raise PolinomialForClassificationError()

        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=42
        )
        X_train_poly = self.poly.fit_transform(X_train)
        X_test_poly  = self.poly.transform(X_test)

        self.model.fit(X_train_poly, y_train)
        y_pred = self.model.predict(X_test_poly)

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        return {"mse": float(mse), "rmse": rmse, "r2": r2}

    def plot(self):
        try:
            if self.X.shape[1] != 1:
                raise MultipleFeaturesPolyError()
            x_line = np.linspace(self.X.min().values[0], self.X.max().values[0], 100).reshape(-1, 1)
            x_line_poly = self.poly.transform(x_line)
            y_line = self.model.predict(x_line_poly)
            plt.scatter(self.X, self.y, color='red', label='Data')
            plt.plot(x_line, y_line, color='blue', label='Prediction')
            plt.xlabel(self.X.columns[0]); plt.ylabel("Target")
            plt.title("Polynomial Regression"); plt.legend(); plt.grid(True); plt.show()
        except MultipleFeaturesPolyError as e:
            print(e)

    def predict_row(self, new_row: pd.DataFrame):
        new_row_aligned = new_row.reindex(columns=self.feature_cols, fill_value=0)
        Xp = self.poly.transform(new_row_aligned)
        return self.model.predict(Xp)

    def predict_opt(self, min_or_max: str = 'max'):
        if self.X.shape[1] == 1 and self.degree == 2:
            b = float(self.model.coef_[0]); c = float(self.model.coef_[1])
            x_opt = -b / (2 * c)
            y_opt = self.model.predict(self.poly.transform([[x_opt]]))[0]
            print(f"Optimal x = {x_opt:.2f}, y = {y_opt:.2f}")
            return x_opt, y_opt
        else:
            def negative_prediction(x):
                x = np.array(x).reshape(1, -1)
                x_df = pd.DataFrame(x, columns=self.feature_cols)
                x_poly = self.poly.transform(x_df)
                pred = self.model.predict(x_poly)[0]
                return -pred if min_or_max == 'max' else pred

            bounds = []
            for col in self.X.columns:
                col_data = self.X[col]
                bounds.append((col_data.min(), col_data.max()))

            initial_guess = self.X.mean().values.tolist()
            result = minimize(negative_prediction, x0=initial_guess, bounds=bounds)
            x_opt = result.x
            y_opt = -result.fun if min_or_max == 'max' else result.fun
            print("Optimal feature values for highest prediction:")
            for name, value in zip(self.X.columns, x_opt):
                print(f"{name}: {value:.2f}")
            print(f"Predicted y = {y_opt:.2f}")
            return x_opt, y_opt

    def predict(self, arg='row'):
        if isinstance(arg, pd.DataFrame):
            return self.predict_row(arg)
        if isinstance(arg, str):
            if arg not in ('max', 'min'):
                raise PolinomialMaxMinError
            return self.predict_opt(arg)
        raise PolinomialNotDFError()

# -------------------------------
# Classifiers
# -------------------------------

class DecisionTreeClassifierFacade:
    """Decision tree classifier with simple preprocessing."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = DecisionTreeClassifier()
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {"model": self.model, "accuracy": accuracy, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class RandomForestClassifierFacade:
    """Random forest classifier with decision-tree preprocessing."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27, criterion="gini"):
        self.preprocessor = DecisionTreePreprocessor()
        self.model = RandomForestClassifier(random_state=random_state, criterion=criterion)
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {"model": self.model, "accuracy": accuracy, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class LogisticRegressionFacade:
    """Logistic regression with scaling & validation of (penalty, solver)."""
    from sklearn.pipeline import Pipeline
    def __init__(self, test_size: float = 0.2, random_state: int = 27,
                 solver='lbfgs', penalty='l2', C=1.0, max_iter: int = 5000):
        self.preprocessor = LogisticRegressionPreprocessor()
        self.max_iter = max_iter
        self.C = C
        self.pipeline = None
        default_penalty = 'l2'
        default_solver = 'lbfgs'
        self.penalty = penalty or default_penalty
        self.solver = solver or default_solver
        self.feature_cols = None
        try:
            self.validate_penalty_solver(self.penalty, self.solver)
        except InvalidPenaltySolverCombination as e:
            print(f"[User Error]: {e}")
            print(f"[Default Parameters Selected: penalty='{default_penalty}', solver='{default_solver}']")
            self.penalty = default_penalty
            self.solver = default_solver
        self.model = LogisticRegression(solver=self.solver, penalty=self.penalty, C=C, max_iter=max_iter)
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
        self.feature_cols = X.columns
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.pipeline = Pipeline(steps=[
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', LogisticRegression(max_iter=self.max_iter, solver=self.solver, C=self.C, penalty=self.penalty))
        ])
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = self.pipeline.score(X_test, y_test)
        self.model = self.pipeline
        return {"model": self.model, "accuracy": acc, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        X_new = new_data.reindex(columns=self.feature_cols, fill_value=0)
        return self.model.predict(X_new)


class LogisticRegressionCVFacade:
    """Cross-validated logistic regression with automatic `C` selection."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27, solver='lbfgs',
                 penalty='l2', max_iter: int = 1000, cv: int = 5, scoring: str = 'accuracy'):
        self.preprocessor = LogisticRegressionPreprocessor()
        default_penalty = 'l2'; default_solver = 'lbfgs'
        self.penalty = penalty or default_penalty
        self.solver = solver or default_solver
        try:
            self.validate_penalty_solver(self.penalty, self.solver)
        except InvalidPenaltySolverCombination as e:
            print(f"[User Error]: {e}")
            print(f"[Default Parameters Selected: penalty='{default_penalty}', solver='{default_solver}']")
            self.penalty = default_penalty; self.solver = default_solver
        self.model = LogisticRegressionCV(
            solver=self.solver, penalty=self.penalty, max_iter=max_iter,
            cv=cv, scoring=scoring, random_state=random_state, Cs=10
        )
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {"model": self.model, "best_C": self.model.C_[0], "accuracy": accuracy, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        return self.model.predict(new_data)


class KNNFacade:
    """K-Nearest Neighbors classifier with Min-Max scaling."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27, n_neighbors: int = 3):
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.preprocessor = KNNPreprocessor()
        self.model = KNeighborsClassifier(n_neighbors)
        self.scaler = MinMaxScaler()
        self.fitted = False

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        self.fitted = True
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {"model": self.model, "n_neighbors": self.n_neighbors, "accuracy": accuracy, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise NotFittedError
        X, _ = self.preprocessor.process(new_data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class ANNFacade:
    """Dense feed-forward neural network for tabular classification."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27,
                 hidden_layers: list = [64, 32], epochs: int = 20, batch_size: int = 32):
        self.test_size = test_size
        self.random_state = random_state
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocessor = ANNCNNPreprocessor()
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.label_encoder: Optional[LabelEncoder] = None

    def build_model(self, input_dim: int, output_dim: int):
        model = Sequential()
        model.add(Dense(self.hidden_layers[0], activation='relu', input_dim=input_dim))
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
        self.label_encoder = self.preprocessor.label_encoder

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        output_dim = 1 if y.nunique() == 2 else y.nunique()
        self.model = self.build_model(input_dim=X.shape[1], output_dim=output_dim)

        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.fitted = True

        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)

        return {"model": self.model, "accuracy": float(test_acc), "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise NotFittedError

        X, _ = self.preprocessor.process(new_data)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled, verbose=0)

        if probabilities.shape[1] == 1:
            predicted_class = (probabilities > 0.5).astype(int).flatten()
            confidence = probabilities.flatten()
        else:
            predicted_class = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)

        if self.label_encoder:
            try:
                predicted_label = self.label_encoder.inverse_transform(predicted_class)
            except Exception:
                predicted_label = predicted_class
        else:
            predicted_label = predicted_class

        results_df = pd.DataFrame({"prediction": predicted_label, "confidence": confidence})
        prob_df = pd.DataFrame(probabilities, columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])])
        return pd.concat([results_df, prob_df], axis=1)


class CNNFacade:
    """1-D CNN for tabular features (after scaling), for classification."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27,
                 filters: int = 64, kernel_size: int = 3, epochs: int = 20, batch_size: int = 32):
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
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=input_shape))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        if output_dim == 1:
            model.add(Dense(1, activation='sigmoid')); loss = BinaryCrossentropy()
        else:
            model.add(Dense(output_dim, activation='softmax')); loss = SparseCategoricalCrossentropy()
        model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
        return model

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        output_dim = 1 if y.nunique() == 2 else y.nunique()
        self.model = self.build_model(input_shape=X_train.shape[1:], output_dim=output_dim)
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.fitted = True

        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)
        return {"model": self.model, "accuracy": float(test_acc), "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise NotFittedError
        X, _ = self.preprocessor.process(new_data, target_col=target_col)
        X_scaled = self.scaler.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        probabilities = self.model.predict(X_scaled, verbose=0)

        if probabilities.shape[1] == 1:
            predicted_class = (probabilities > 0.5).astype(int).flatten()
            confidence = probabilities.flatten()
        else:
            predicted_class = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)

        results_df = pd.DataFrame({"prediction": predicted_class, "confidence": confidence})
        prob_df = pd.DataFrame(probabilities, columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])])
        return pd.concat([results_df, prob_df], axis=1)


class SVMClassifierFacade:
    """Support Vector Machine classifier with scaling."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27,
                 C: float = 1.0, kernel: str = "rbf", gamma: str | float = "scale", degree: int = 3):
        self.preprocessor = LogisticRegressionPreprocessor()
        self.scaler = StandardScaler()
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree)
        self.test_size = test_size
        self.random_state = random_state
        self.fitted = False
        self.feature_cols: Optional[pd.Index] = None

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        self.feature_cols = X.columns
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        self.fitted = True
        return {"model": self.model, "accuracy": acc, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call train_and_evaluate first.")
        X_new = new_data.reindex(columns=self.feature_cols, fill_value=0)
        X_new = self.scaler.transform(X_new)
        return self.model.predict(X_new)


class SVMRegressorFacade:
    """Support Vector Regressor with scaling."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27,
                 C: float = 1.0, kernel: str = "rbf", gamma: str | float = "scale", degree: int = 3, epsilon: float = 0.1):
        self.preprocessor = LinearRegressionPreprocessor()
        self.scaler = StandardScaler()
        self.model = SVR(C=C, kernel=kernel, gamma=gamma, degree=degree, epsilon=epsilon)
        self.test_size = test_size
        self.random_state = random_state
        self.fitted = False
        self.feature_cols: Optional[pd.Index] = None

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        self.feature_cols = X.columns
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        self.fitted = True
        return {"model": self.model, "mse": mse, "rmse": rmse, "r2": r2, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call train_and_evaluate first.")
        X_new = new_data.reindex(columns=self.feature_cols, fill_value=0)
        X_new = self.scaler.transform(X_new)
        return self.model.predict(X_new)


class XGBClassifierFacade:
    """Gradient-boosted tree classifier (XGBoost) with LabelEncoder for text labels."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27,
                 n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1,
                 **kwargs: Any):
        self.preprocessor = LogisticRegressionPreprocessor()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            **kwargs
        )
        self.test_size = test_size
        self.random_state = random_state
        self.fitted = False
        self.feature_cols: Optional[list[str]] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        self.feature_cols = list(X.columns)

        # XGBoost expects y as ints 0..K-1; encode if needed
        self._label_encoder = None
        if y.dtype == object or isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            self._label_encoder = le
        else:
            y_enc = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=self.test_size, random_state=self.random_state, stratify=y_enc
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        self.fitted = True
        return {"model": self.model, "accuracy": acc, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call train_and_evaluate first.")
        X_new = new_data.reindex(columns=self.feature_cols, fill_value=0)
        y_pred_enc = self.model.predict(X_new)
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(y_pred_enc)
        return y_pred_enc


class XGBRegressorFacade:
    """Gradient-boosted tree regressor (XGBoost)."""
    def __init__(self, test_size: float = 0.2, random_state: int = 27,
                 n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1,
                 **kwargs: Any):
        self.preprocessor = LinearRegressionPreprocessor()
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
        self.test_size = test_size
        self.random_state = random_state
        self.fitted = False
        self.feature_cols: Optional[list[str]] = None

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str) -> dict:
        X, y = self.preprocessor.process(df, target_col)
        self.feature_cols = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        self.fitted = True
        return {"model": self.model, "mse": mse, "rmse": rmse, "r2": r2, "y_test": y_test, "prediction": y_pred}

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call train_and_evaluate first.")
        X_new = new_data.reindex(columns=self.feature_cols, fill_value=0)
        return self.model.predict(X_new)
