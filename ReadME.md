# Machine Learning Assignment 1 — Parts 1 and 2

* [Part 1: Multiple Linear Regression Using Gradient Descent](#part-1-multiple-linear-regression-using-gradient-descent) — `part1.py`
* [Part 2: Linear Regression Using SGDRegressor](#part-2-linear-regression-using-sgdregressor) — `part2.py`

## Part 1: Multiple Linear Regression Using Gradient Descent

### Overview

This program implements multiple linear regression using gradient descent from scratch. NumPy is used for numerical operations, but no machine learning library is used to implement linear regression or gradient descent.

The program loads and preprocesses the Real Estate Valuation dataset from the UCI Machine Learning Repository, trains multiple linear regression models using different hyperparameters, selects the best hyperparameters using a validation set, and evaluates the final model on a held-out test set.

---

## Dataset

**Dataset:** Real Estate Valuation
**Source:** UCI Machine Learning Repository
**UCI Dataset ID:** 477
**Instances:** 414
**Features:** 6
**Target:** House price of unit area

The dataset is retrieved directly from UCI using the `ucimlrepo` package, so no local dataset file is required.

The six input features are:

1. X1 — Transaction date
2. X2 — House age
3. X3 — Distance to the nearest MRT station
4. X4 — Number of convenience stores
5. X5 — Latitude
6. X6 — Longitude

The target variable is `Y house price of unit area`, measured in 10,000 New Taiwan Dollars per Ping.

---

## Preprocessing

The following preprocessing steps are performed:

* Checked all features and the target for missing values.
* Checked the complete dataset for duplicate rows.
* Examined Pearson correlations between each feature and the target.
* No categorical encoding was necessary because all six input features are numerical.
* All six features were retained for the regression model.
* The dataset was first divided into 80% training data and 20% test data.
* Twenty percent of the training portion was then reserved for validation, resulting in approximately:

  * 64% training
  * 16% validation
  * 20% testing
* Features were standardized using `StandardScaler`.
* The scaler was fitted only on the training data. The same training-set mean and standard deviation were then used to transform the validation and test sets to avoid data leakage.

---

## Linear Regression Implementation

The `LinearRegression` class implements multiple linear regression without using a machine learning regression implementation.

Predictions are calculated using:

`y_pred = Xw + b`

where:

* `X` contains the standardized input features
* `w` contains the model weights
* `b` is the bias/intercept

The weights are initialized to zero and are learned using gradient descent.

Mean Squared Error (MSE) is used as the error function:

`MSE = mean((y_pred - y)^2)`

The gradients are calculated in vectorized form using NumPy:

`dw = (2 / n) * X.T * errors`

`db = (2 / n) * sum(errors)`

The weights and bias are then updated using the selected learning rate.

The training MSE from every iteration is stored so that the convergence of gradient descent can be plotted.

---

## Hyperparameter Tuning

The following learning rates are tested:

* 0.001
* 0.01
* 0.1
* 0.2

The following numbers of iterations are tested:

* 100
* 200
* 500
* 1000

Every combination of learning rate and iteration count is evaluated.

For each combination, the program:

1. Trains a new linear regression model on the training set.
2. Calculates the training MSE.
3. Calculates the validation MSE.
4. Records the results in `linear_regression_tuning.log`.

The model configuration with the lowest validation MSE is selected as the best set of hyperparameters. The held-out test set is not used to select the hyperparameters.

---

## Final Evaluation

After selecting the best learning rate and number of iterations, a final model is trained using those hyperparameters and evaluated on the test set.

The program reports:

* Selected learning rate
* Selected number of iterations
* Validation MSE
* Final learned weights for each feature
* Final bias
* Test MSE
* R-squared (R²)
* Explained Variance

The final test set provides an estimate of how well the selected model generalizes to previously unseen observations.

---

## Plots

The program generates the following plots:

### 1. Training MSE During Hyperparameter Experiments

Training MSE is plotted across iterations for the tested hyperparameter configurations. This helps visualize how the learning rate and number of iterations affect gradient descent convergence.

### 2. Best Model — Training MSE vs. Iteration

The training MSE of the selected model is plotted against the number of gradient descent iterations. A decreasing MSE indicates that gradient descent is successfully minimizing the error.

### 3. Actual vs. Predicted House Prices

The final test predictions are plotted against the actual test values.

A diagonal reference line represents perfect predictions. Predictions closer to this line indicate smaller prediction errors.

---

## Log File

Running the program creates:

`linear_regression_tuning.log`

The log contains the learning rate, number of iterations, training MSE, and validation MSE for each hyperparameter combination. It also records the selected hyperparameters and final test evaluation results.

---

## Required Packages

The program requires:

* Python 3
* NumPy
* pandas
* matplotlib
* scikit-learn
* ucimlrepo

Install the required packages with:

`pip install numpy pandas matplotlib scikit-learn ucimlrepo`

Scikit-learn is used only for the train/test split and feature standardization in Part 1. It is not used to implement linear regression or gradient descent.

---

## Running the Program

From the directory containing `part1.py`, run:

`python part1.py`

The program will retrieve the dataset directly from UCI, perform preprocessing, train and tune the models, display evaluation results, create the plots, and write the tuning results to the log file.

---

## Best Solution Discussion

The program tests multiple combinations of learning rates and iteration counts and selects the combination producing the lowest validation MSE. Using a separate validation set allows the hyperparameters to be selected without using the final test set.

The selected model represents the best solution among the hyperparameter combinations tested. However, it cannot be guaranteed to be the absolute best possible solution because only a finite set of learning rates and iteration counts was evaluated. Additional hyperparameter values, feature engineering, or other preprocessing choices could potentially produce further improvements.

---

## Part 2: Linear Regression Using SGDRegressor

### Overview

`part2.py` implements linear regression using scikit-learn's `SGDRegressor`. It follows the same dataset preparation, validation-based hyperparameter selection, and held-out test evaluation workflow as Part 1, but uses stochastic gradient descent provided by scikit-learn instead of the custom gradient descent implementation.

### Dataset and Preprocessing

Part 2 retrieves the same Real Estate Valuation dataset from UCI using `fetch_ucirepo(id=477)` and retains all six numerical features to predict house price of unit area.

The program checks missing values, duplicate rows, and feature correlations. It splits the data into approximately 64% training, 16% validation, and 20% testing, using `random_state=42` for both splits. `StandardScaler` is fitted only on the training features and then applied to the validation and test features. Targets are flattened into one-dimensional arrays for `SGDRegressor`.

### Model and Hyperparameter Tuning

The program tests all 12 combinations of:

* Learning rate (`eta0`): `0.001`, `0.01`, `0.1`, `0.2`
* Training epochs: `100`, `200`, `300`

Each model uses the default squared-error loss with these settings:

* `learning_rate="constant"`: keeps the learning rate fixed during training.
* `penalty=None`: disables regularization.
* `tol=None`: disables tolerance-based stopping.
* `random_state=42`: makes the stochastic training reproducible.

Training calls `partial_fit` once per epoch and records the training MSE after each pass through the training data. Although `max_iter` is set when constructing the estimator, the explicit loop controls the number of epochs because each `partial_fit` call performs one epoch.

For every combination, the program prints and logs the final training and validation MSE. The combination with the lowest validation MSE is selected; the test set is not used for tuning.

### Final Evaluation

A new `SGDRegressor` is trained on the training set using the selected settings. The validation set is not combined with the training set for this final fit.

The program reports:

* Selected learning rate, epoch count, and validation MSE
* Learned feature weights (`coef_`) and bias (`intercept_`)
* Test MSE, R-squared, and explained variance, calculated using scikit-learn metrics

### Plots and Log File

Part 2 displays three plots:

1. Training MSE across epochs for every hyperparameter combination.
2. Training MSE across epochs for the selected model.
3. Actual versus predicted test house prices, with a diagonal reference line for perfect predictions.

Each run writes `sgd_regression_tuning.log` in the current working directory, replacing any previous contents. The log includes the training and validation MSE for each combination, the selected hyperparameters, and the final test metrics. Plots are displayed interactively rather than saved as image files.

### Required Packages and Running Part 2

Part 2 uses the same packages listed for Part 1. Install them with:

```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo
```

From the directory containing `part2.py`, run:

```bash
python part2.py
```

Internet access is required to retrieve the dataset from UCI. Close each plot window to allow the script to continue to the next stage when using a blocking interactive plotting backend.

### Comparison with Part 1

Both parts use the same features, split seed, standardization procedure, and validation MSE selection criterion. Part 1 computes full-training-set gradients using NumPy, while Part 2 uses scikit-learn's stochastic gradient descent, which updates parameters using individual training examples.

The learning rates tested are the same, but Part 2 tests 100, 200, and 300 epochs. Its epoch counts and update behavior differ from Part 1, so the same learning rate does not imply the same training behavior or results. Compare the reported test metrics to assess performance; the selected configuration is the best among those tested, not a guarantee of the best possible model.
