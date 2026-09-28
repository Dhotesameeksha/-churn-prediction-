"""
eda.py
------
Exploratory Data Analysis on the churn dataset.
Saves plots to outputs/ for inclusion in the project report / CV portfolio.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/telecom_churn.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().sum())
print("\nChurn distribution:\n", df["Churn"].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Churn distribution
sns.countplot(data=df, x="Churn", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Churn Distribution")

# 2. Churn by contract type
sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].tick_params(axis="x", rotation=15)

# 3. Monthly charges vs churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("Monthly Charges vs Churn")

# 4. Tenure vs churn
sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Tenure vs Churn")

plt.tight_layout()
plt.savefig("outputs/eda_summary.png", dpi=150)
print("\nSaved outputs/eda_summary.png")
