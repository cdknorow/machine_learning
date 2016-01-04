import sklearn.base as  base
import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression



# To create your own estimator, simply create a subclass of BaseEstimator, RegressorMixin
# and implement the __init__, fit, and predict functions
# To learn more about rolling your own estimator, checkout
# http://scikit-learn.org/stable/developers/#rolling-your-own-estimator

class EnsembleRegressor(base.BaseEstimator, base.RegressorMixin):
    """Joins a linear, random forest, and nearest neighbors model."""
    def __init__(self):
        pass
    
    def fit(self, X, y):
        import pandas as pd
        self.logistic_regression = LogisticRegression().fit(X, y)
        y_err = y - self.logistic_regression.predict(X)
        self.linear_regression = LinearRegression().fit(X, y_err)

        X_ensemble = pd.DataFrame({
            "LOGISTIC": self.logistic_regression.predict(X),
            "LINEAR": self.linear_regression.predict(X),
        })
        self.ensemble_regression = LinearRegression().fit(X_ensemble, y)
        return self
    
    def predict(self, X):
        import pandas as pd
        X_ensemble = pd.DataFrame({
            "LOGISTIC": self.logistic_regression.predict(X),
            "LINEAR": self.linear_regression.predict(X),
        })
        return self.ensemble_regression.predict(X_ensemble)
