# from the UCI dataset instructions:

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import matplotlib.pyplot as plt

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

# Take 20% of the training data for validation
# Final proportions: 64% train, 16% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()

# Learn the mean/std from training data and standardize it
X_train_scaled = scaler.fit_transform(X_train)

# Standardize test data using the SAME mean/std learned from training data
X_test_scaled = scaler.transform(X_test)

X_val_scaled = scaler.transform(X_val)

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
        self.mse_history = []

        for i in range(self.iterations):

            # make predictions using current weights
            predictions = np.dot(X, self.weights) + self.bias

            # calculate errors
            errors = predictions - y

            # calculate and save mean squared error
            mse = np.mean(errors ** 2)
            self.mse_history.append(mse)

            # calculate gradients
            dw = (2 / num_samples) * np.dot(X.T, errors)
            db = (2 / num_samples) * np.sum(errors)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        return predictions

# hyperparameters to test
learning_rates = [0.001, 0.01, 0.1, 0.2]
iterations_list = [100, 200, 300]

y_train = y_train.to_numpy().ravel()
y_val = y_val.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

plt.figure()

# configure log file
logging.basicConfig(
    filename="linear_regression_tuning.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w"
)

# variables for tracking the best model
best_mse = float("inf")
best_learning_rate = None
best_iterations = None

# try every combination
for learning_rate in learning_rates:
    for iterations in iterations_list:

        model = LinearRegression(
            learning_rate=learning_rate,
            iterations=iterations
        )

        # train model
        model.fit(X_train_scaled, y_train)

        # make predictions on validation set
        val_predictions = model.predict(X_val_scaled)

        # calculate validation MSE
        val_mse = np.mean((val_predictions - y_val) ** 2)
        train_predictions = model.predict(X_train_scaled)
        train_mse = np.mean((train_predictions - y_train) ** 2)

        # display result
        print(
            f"Learning Rate: {learning_rate}, "
            f"Iterations: {iterations}, "
            f"Training MSE: {train_mse:.4f}, "
            f"Validation MSE: {val_mse:.4f}"
        )


        plt.plot(
            range(len(model.mse_history)),
            model.mse_history,
            label=f"LR={learning_rate}, Iter={iterations}"
        )

        # save result to log file
        logging.info(
            f"Learning Rate: {learning_rate}, "
            f"Iterations: {iterations}, "
            f"Training MSE: {train_mse:.4f}, "
            f"Validation MSE: {val_mse:.4f}"
        )

        # update best hyperparameters
        if val_mse < best_mse:
            best_mse = val_mse
            best_learning_rate = learning_rate
            best_iterations = iterations


print("\nBest Hyperparameters:")
print("Learning Rate:", best_learning_rate)
print("Iterations:", best_iterations)
print("Validation MSE:", best_mse)

logging.info(
    f"BEST MODEL - Learning Rate: {best_learning_rate}, "
    f"Iterations: {best_iterations}, "
    f"Validation MSE: {best_mse:.4f}"
)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Effect of Learning Rate on Training")
plt.legend()
plt.show()

best_model = LinearRegression(
    learning_rate=best_learning_rate,
    iterations=best_iterations
)

best_model.fit(X_train_scaled, y_train)
plt.plot(range(len(best_model.mse_history)), best_model.mse_history)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Training MSE vs. Iteration")
plt.show()

print("\nFinal Model Weights:")
for feature, weight in zip(X.columns, best_model.weights):
    print(f"{feature}: {weight:.4f}")

print(f"Bias: {best_model.bias:.4f}")


test_predictions = best_model.predict(X_test_scaled)

test_mse = np.mean((test_predictions - y_test) ** 2)
# calculate R-squared
ss_res = np.sum((y_test - test_predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# calculate explained variance
explained_variance = 1 - (
    np.var(y_test - test_predictions) / np.var(y_test)
)

print(f"Final Test MSE: {test_mse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Explained Variance: {explained_variance:.4f}")

logging.info(
    f"Final Test MSE: {test_mse:.4f}, "
    f"R-squared: {r2:.4f}, "
    f"Explained Variance: {explained_variance:.4f}"
)

plt.scatter(y_test, test_predictions)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs. Predicted House Prices")

minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)

plt.show()
