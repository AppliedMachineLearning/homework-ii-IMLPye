import numpy as np
from sklearn import datasets, linear_model

# Report the test error using R^2
def score_rent(a,b,c,d):
    regr = linear_model.LinearRegression()
    regr.fit(c, d)
    return np.mean((regr.predict(a) - b) ** 2)

# Returns test data, the true labels and predicted labels
def predict_rent(a,b,c,d):
    regr = linear_model.LinearRegression()
    regr.fit(c, d)
    return {'test data':a, 'true labels':b ,'predicted labels':regr.predict(a)}


