# ================= IMPORT LIBRARIES =================
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ================= LOAD DATASET =================
df = pd.read_csv("Salary_Data.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ================= STATISTICAL ANALYSIS =================
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# ================= CHECK NULL VALUES =================
print("\n===== NULL VALUES =====")
print(df.isnull())

# ================= TOTAL NULL VALUES =================
print("\n===== TOTAL NULL VALUES =====")
print(df.isnull().sum())

# ================= CLEAN COLUMN NAMES =================
df.columns = df.columns.str.strip()

# ================= FEATURES & TARGET =================
X = df[["YearsExperience"]]
y = df["Salary"]

# ================= DISPLAY FEATURES & TARGET =================
print("\n===== FEATURES (X) =====")
print(X.head())

print("\n===== TARGET (y) =====")
print(y.head())

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================= DISPLAY DATA SHAPES =================
print("\n===== TRAIN TEST SHAPES =====")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ================= TRAIN MODEL =================
model = LinearRegression()
model.fit(X_train, y_train)

# ================= MAKE PREDICTIONS =================
y_pred = model.predict(X_test)

# ================= MODEL EVALUATION =================
print("\n===== MODEL PERFORMANCE =====")

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ================= TEST NEW DATA =================
new_data = [[5]]
prediction = model.predict(new_data)

print("\nPredicted Salary for 5 years experience:", prediction[0])

# ================= VISUALIZATION =================
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Salary Prediction (Linear Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# ================= SAVE MODEL =================
with open("salary_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as salary_model.pkl")