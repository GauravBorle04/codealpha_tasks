# CodeAlpha Internship
# Task 3: Car Price Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the dataset provided by CodeAlpha
df = pd.read_csv("car data.csv")


# 2. Clean column names
df.columns = df.columns.str.strip()


# 3. Display the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# 4. Prepare the data
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]


# 5. Identify categorical and numerical columns
categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_columns = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Owner"
]


# 6. Preprocess categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# 7. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 8. Transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 9. Train the Linear Regression model
model = LinearRegression()

model.fit(X_train_processed, y_train)

print("\nModel training completed!")


# 10. Make predictions
y_pred = model.predict(X_test_processed)


# 11. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# 12. Actual vs Predicted prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.close()


# 13. Distribution of selling prices
plt.figure(figsize=(8, 6))

sns.histplot(df["Selling_Price"], kde=True)

plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.title("Distribution of Car Selling Prices")

plt.tight_layout()
plt.savefig("selling_price_distribution.png", dpi=300)
plt.close()


print("\nAnalysis completed successfully!")