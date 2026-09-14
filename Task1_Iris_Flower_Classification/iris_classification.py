# CodeAlpha Internship
# Task 1: Iris Flower Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load the Iris dataset provided for the project
df = pd.read_csv("Iris.csv")

# Remove the Id column because it is not a flower measurement
df = df.drop("Id", axis=1)


# 2. Display the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSpecies distribution:")
print(df["Species"].value_counts())


# 3. Data Visualization
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.savefig("iris_scatter_plot.png", dpi=300, bbox_inches="tight")
plt.close()


# 4. Prepare data for Machine Learning
X = df.drop("Species", axis=1)
y = df["Species"]


# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 7. Train the Logistic Regression model
model = LogisticRegression(max_iter=200)

model.fit(X_train_scaled, y_train)

print("\nModel training completed!")


# 8. Make predictions
y_pred = model.predict(X_test_scaled)


# 9. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# 10. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# 12. Test a new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("\nNew Flower Prediction:")
print("Predicted species:", prediction[0])