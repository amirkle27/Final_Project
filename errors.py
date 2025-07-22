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




