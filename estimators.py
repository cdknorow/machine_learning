import sklearn.base as  base
import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

class MeanCityEstimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self):
        self.cities= {}
        # initialization code

    def fit(self, X, y):
        # fit the model ...
        r = pd.DataFrame(X,columns=['city'])
        r = r.groupby('city')
        cities = r.groups.keys()
        for i in cities:
            index = r.get_group(i).index
            self.cities[i] = float(pd.DataFrame(y).loc[index].mean())
        return self

    def predict(self, X):
        p_city = []
        for city in X:
            try:
                p_city.append(self.cities[city])
            except:
                #print 'no city', city
                p_city.append(3.75)
        return p_city

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


class MonthHourEstimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self):
        self.city={}
        # initialization code

    def fit(self, X, y=None):
        # fit the model ...
        cities = X.groupby(['city'])
        for city in cities.groups.keys():
            MonthHours = cities.get_group(city).groupby(['month','hour'])
            self.city[city] = {}
            for key in MonthHours.groups.keys():
                self.city[city][key] = MonthHours.get_group(key).temp.mean()
        return self

    def predict(self, X):
        p_temp = []
        for index in range(len(X)):
            record = X.loc[index]
            p_temp.append(self.city[record.city][(int(record.month),int(record.hour))])
        return p_temp