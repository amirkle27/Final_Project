class TargetColumnNotFoundError(Exception):
    def __init__(self, column_name: str):
        super().__init__(f"Target column '{column_name}' not found in dataframe.")

class InvalidPenaltySolverCombination(Exception):
    def __init__(self, penalty: str, solver:str):
        """Raised when an invalid combination of penalty and solver is used in Logistic Regression"""
        super().__init__(f"Invalid combination of penalty: '{penalty}' and solver: {solver}.")