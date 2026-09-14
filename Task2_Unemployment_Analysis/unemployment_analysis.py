# CodeAlpha Internship
# Task 2: Unemployment Analysis in India

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the dataset provided by CodeAlpha
df = pd.read_csv("Unemployment in India.csv")


# 2. Clean column names
df.columns = df.columns.str.strip()


# 3. Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values before cleaning:")
print(df.isnull().sum())


# 4. Clean missing values
df = df.dropna().copy()

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)


print("\nMissing values after cleaning:")
print(df.isnull().sum())


# 5. Basic statistics
print("\nBasic Statistics:")
print(df.describe())


# 6. Average unemployment rate by state
state_unemployment = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment Rate by State:")
print(state_unemployment)


# 7. Plot average unemployment rate by state
plt.figure(figsize=(12, 7))

state_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("average_unemployment_by_state.png", dpi=300)
plt.close()


# 8. Monthly unemployment trend
monthly_unemployment = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment.index,
    monthly_unemployment.values,
    marker="o"
)

plt.title("Monthly Unemployment Rate in India")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("monthly_unemployment_trend.png", dpi=300)
plt.close()


# 9. Rural vs Urban unemployment
area_unemployment = (
    df.groupby("Area")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\nAverage Unemployment Rate by Area:")
print(area_unemployment)


plt.figure(figsize=(7, 5))

area_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate: Rural vs Urban")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("rural_vs_urban_unemployment.png", dpi=300)
plt.close()


# 10. COVID-19 period analysis
covid_period = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-06-30")
]

print("\nCOVID-19 Period Analysis (March 2020 - June 2020):")

print(
    "Average unemployment rate:",
    f"{covid_period['Estimated Unemployment Rate (%)'].mean():.2f}%"
)


# 11. Highest unemployment month
highest_month = monthly_unemployment.idxmax()
highest_rate = monthly_unemployment.max()

print("\nHighest Average Unemployment Month:")
print(highest_month.strftime("%B %Y"))
print(f"Unemployment Rate: {highest_rate:.2f}%")


# 12. Lowest unemployment month
lowest_month = monthly_unemployment.idxmin()
lowest_rate = monthly_unemployment.min()

print("\nLowest Average Unemployment Month:")
print(lowest_month.strftime("%B %Y"))
print(f"Unemployment Rate: {lowest_rate:.2f}%")


print("\nAnalysis completed successfully!")