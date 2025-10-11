from enum import Enum

class ModelName(str, Enum):
    """
    Canonical list of model identifiers exposed to the API.
    Using an Enum makes Swagger (OpenAPI) render a dropdown instead of a free-text input.
    """
    # Classification
    logreg = "logreg"
    dt     = "dt"
    rf     = "rf"
    knn    = "knn"
    ann    = "ann"
    svm    = "svm"
    xgb    = "xgb"

    # Regression
    linear = "linear"
    poly   = "poly"
    svr    = "svr"
    xgbr   = "xgbr"