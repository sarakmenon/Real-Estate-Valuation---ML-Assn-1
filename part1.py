# from the UCI dataset instructions:

from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
  
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