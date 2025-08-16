class TargetColumnNotFoundError(Exception):
    def __init__(self, column_name: str):
        super().__init__(f"Target column '{column_name}' not found in dataframe.")

class InvalidPenaltySolverCombination(Exception):
    def __init__(self, penalty: str, solver:str):
        """Raised when an invalid combination of penalty and solver is used in Logistic Regression"""
        super().__init__(f"Invalid combination of penalty: '{penalty}' and solver: {solver}.")

class NotFittedError(Exception):
    def __init__(self):
        """Raised when a clustered model is not fitted correctly before predicting new data"""
        super().__init__("Model must be fit before predicting.")

class MultipleFeaturesPolyError(Exception):
    def __init__(self):
        """Raised when a clustered model is not fitted correctly before predicting new data"""
        super().__init__("Cannot plot: Only works for 1 feature.")

class PolinomialMaxMinError(Exception):
    def __init__(self):
        """Raised when a clustered model is not fitted correctly before predicting new data"""
        super().__init__("PolynomialFacade.predict: expected 'max' or 'min' for optimization.")

class PolinomialNotDFError(Exception):
    def __init__(self):
        """Raised when a clustered model is not fitted correctly before predicting new data"""
        super().__init__("PolynomialFacade.predict: pass a DataFrame (row prediction) or 'max'/'min' (optimization).")

    # if isinstance(model, PolynomialFacade):
    #     prediction = model.predict_row(new_row)
    # else:
    #     prediction = model.predict(new_row)
    #
    # if isinstance(prediction, pd.DataFrame) and 'prediction' in prediction.columns:
    #     pred_value = prediction['prediction'].iloc[0]
    # elif isinstance(prediction, (pd.Series, np.ndarray)):
    #     pred_value = prediction[0]
    # else:
    #     pred_value = str(prediction)
