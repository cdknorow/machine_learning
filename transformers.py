import sklearn.base as base

#Remove Specific columns from the Dataframe 
class ColumnRemoveTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, remove_cols = [0, 62, 122, 216, 238]):
        # initialization code
        self.remove_cols =  remove_cols

    def fit(self, X, y=None):
        # fit the transformation ...
        return self

    def transform(self, X):  
        n_cols = range(len(X.columns))
        for i in self.remove_cols:
            n_cols.remove(i)
        return X.icol(n_cols).fillna(0)

#Select Specific Columns from the DataFrame
class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, select_cols = [0, 62, 122, 216, 238]):
        # initialization code
        self.select_cols =  select_cols

    def fit(self, X, y=None):
        # fit the transformation ...
        return self

    def transform(self, X):  
        return X.icol(self.select_cols).fillna(0)

#Take a model and add a transform method that returns a prediction
class ConcantTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, num=3):
        self.num=num

    def fit(self, X, y=None):
        #fit the transformation
        return self

    def transform(self, X):
        #transform the data
        l = len(X)
        l_step = l/self.num
        A = []
        for i in range(l_step):
            B = []
            for j in range(self.num): 
                B.append(X[i+l_step*j])
            A.append(B)
        return A
    
#Take a list of columns and return a dictionary format cosumable by DictVectorizer
class DictMassagerTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        # initialization code
        pass

    def fit(self, X, y=None):
        # fit the transformation ...
        return self

    def transform(self, X):
        return X.T.to_dict().values()
    
#Take a model and add a transform method that returns a prediction
class ModelTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, model):
        # initialization code
        self.model = model

    def fit(self, X, y=None):
        # fit the transformation ...
        return self.model.fit(X,y)

    def transform(self, X, **transform_params):
        return self.model.predict(X)
    
    def score(self, X, y=None):
        # score the model ...
        return self.model.score(X,y)
