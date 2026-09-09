# from the UCI dataset instructions:

from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
  
# fetch dataset 
real_estate_valuation = fetch_ucirepo(id=477) 
  
# data (as pandas dataframes) 
X = real_estate_valuation.data.features 
y = real_estate_valuation.data.targets 
  
# metadata 
print(real_estate_valuation.metadata) 
  
# variable information 
print(real_estate_valuation.variables) 

print(X)
print(y)

# preprocessing

# checking for missing feature values
print("Missing values in X:")
print(X.isnull().sum())

# checking for missing target values
print("\nMissing values in y:")
print(y.isnull().sum())

# checking for redundant values
data = pd.concat([X, y], axis=1)

# Check for duplicate rows
print("Number of duplicate rows:")
print(data.duplicated().sum())

# check for correlation
correlations = data.corr()["Y house price of unit area"]

print(correlations)


# splitting the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()

# Learn the mean/std from training data and standardize it
X_train_scaled = scaler.fit_transform(X_train)

# Standardize test data using the SAME mean/std learned from training data
X_test_scaled = scaler.transform(X_test)

print("\nOriginal training data:")
print(X_train.head())

print("\nScaled training data:")
print(X_train_scaled[:5])


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.mse_history = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.iterations):

            # make predictions using current weights
            predictions = np.dot(X, self.weights) + self.bias

            # calculate the errors
            errors = predictions - y

            # calculate + save mean squared error
            mse = np.mean(errors ** 2)
            self.mse_history.append(mse)

            # calculate gradients
            # dw represents how the weights should be changed, and db represents how the bias should be changed
            dw = (2 / num_samples) * np.dot(X.T, errors)
            db = (2 / num_samples) * np.sum(errors)

            # change the weights and bias using the gradients and learning rate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            

