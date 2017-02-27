import numpy as np
from sklearn import datasets, linear_model
from homework2_rent import score_rent

# Load the diabetes dataset
testData = datasets.load_diabetes()

# Load the diabetes dataset
# testData = datasets.load_diabetes()


# Use only one feature
data_X = testData.data[:, np.newaxis, 2]

# Split the data into training/testing sets
X_train = data_X[:-20]
X_test = data_X[-20:]
y_train = testData.target[:-20]
y_test = testData.target[-20:]

print("Mean squared error is: %.2f" % score_rent(X_test,y_test,X_train,y_train))