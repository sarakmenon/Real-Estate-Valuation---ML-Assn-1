# from the UCI dataset instructions:

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
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


# check for duplicate rows
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


# taking 20% of the training data for validation
# final proportions: 64% train, 16% validation, 20% test
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

# standardizing the features using StandardScaler only from the training set
scaler = StandardScaler()
# learn the mean/std from training data and standardize it
X_train_scaled = scaler.fit_transform(X_train)


# mean/std learned from training data
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


print("\nOriginal training data:")
print(X_train.head())


print("\nScaled training data:")
print(X_train_scaled[:5])


# SGDRegressor expects a one-dimensional target
# .ravel() converts the target from a pandas table into a one-dimensional NumPy array
y_train = y_train.to_numpy().ravel()
y_val = y_val.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


# Same hyperparameters we tested in Part 1
learning_rates = [0.001, 0.01, 0.1, 0.2]
iterations_list = [100, 200, 300]


# configure log file
logging.basicConfig(
    filename="sgd_regression_tuning.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w"
)


# variables for tracking the best model
best_mse = float("inf")
best_learning_rate = None
best_iterations = None


plt.figure()


# try every combination we tried in part 1
for learning_rate in learning_rates:

    for iterations in iterations_list:

        # create the linear regression model using sklearn, not our own implementation
        model = SGDRegressor(
            eta0=learning_rate,
            max_iter=iterations,
            learning_rate="constant",
            penalty=None,
            tol=None,
            random_state=42
        )

        mse_history = []

        # train one epoch at a time
        for epoch in range(iterations):

            model.partial_fit(X_train_scaled, y_train)

            train_predictions = model.predict(X_train_scaled)

            epoch_mse = mean_squared_error(y_train, train_predictions)

            mse_history.append(epoch_mse)


        # make predictions on validation set
        val_predictions = model.predict(X_val_scaled)


        # calculate validation MSE
        val_mse = mean_squared_error(y_val, val_predictions)


        # calculate final training MSE
        train_predictions = model.predict(X_train_scaled)

        train_mse = mean_squared_error(y_train, train_predictions)

        print(
            f"Learning Rate: {learning_rate}, "
            f"Iterations: {iterations}, "
            f"Training MSE: {train_mse:.4f}, "
            f"Validation MSE: {val_mse:.4f}"
        )


        # plot training MSE
        plt.plot(
            range(len(mse_history)),
            mse_history,
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

print(f"Learning Rate: {best_learning_rate}")

print(f"Iterations: {best_iterations}")

print(f"Validation MSE: {best_mse:.4f}")


logging.info(
    f"BEST MODEL - Learning Rate: {best_learning_rate}, "
    f"Iterations: {best_iterations}, "
    f"Validation MSE: {best_mse:.4f}"
)


plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Effect of Learning Rate on SGDRegressor Training")
plt.legend()
plt.show()


best_model = SGDRegressor(
    eta0=best_learning_rate,
    max_iter=best_iterations,
    learning_rate="constant",
    penalty=None,
    tol=None,
    random_state=42
)

# Keep MSE history for best model
best_mse_history = []


for epoch in range(best_iterations):

    best_model.partial_fit(X_train_scaled, y_train)

    train_predictions = best_model.predict(X_train_scaled)

    train_mse = mean_squared_error(y_train, train_predictions)

    best_mse_history.append(train_mse)

#PLOTS

plt.figure()

plt.plot(
    range(len(best_mse_history)),
    best_mse_history
)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("SGDRegressor Training MSE vs. Iteration")

plt.show()

print("\nFinal Model Weights:")


# coef_ contains sklearn's learned weights
for feature, weight in zip(
    X.columns,
    best_model.coef_
):

    print(f"{feature}: {weight:.4f}")


# intercept_ contains sklearn's learned bias
print(f"Bias: {best_model.intercept_[0]:.4f}")

test_predictions = best_model.predict(X_test_scaled)


# calculate MSE using sklearn
test_mse = mean_squared_error(y_test, test_predictions)


# calculate R-squared using sklearn
r2 = r2_score(y_test,test_predictions)


# calculate explained variance using sklearn
explained_variance = explained_variance_score(
    y_test,
    test_predictions
)


print(f"Final Test MSE: {test_mse:.4f}")

print(f"R-squared: {r2:.4f}")

print(f"Explained Variance: {explained_variance:.4f}")


logging.info(
    f"Final Test MSE: {test_mse:.4f}, "
    f"R-squared: {r2:.4f}, "
    f"Explained Variance: {explained_variance:.4f}"
)


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure()

plt.scatter(
    y_test,
    test_predictions
)


plt.xlabel("Actual House Price")

plt.ylabel("Predicted House Price")

plt.title("SGDRegressor: Actual vs. Predicted House Prices")


# perfect prediction line
minimum = min(y_test.min(), test_predictions.min())

maximum = max(y_test.max(), test_predictions.max())


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)


plt.show()
