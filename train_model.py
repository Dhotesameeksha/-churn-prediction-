"""
train_model.py
---------------
Preprocesses data, trains Logistic Regression and Random Forest models,
evaluates them, and saves the best model + preprocessing pipeline to disk.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

df = pd.read_csv("data/telecom_churn.csv")
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    fitted_pipelines[name] = pipe
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

# Pick best model by F1 score
best_name = max(results, key=lambda k: results[k]["f1_score"])
best_pipe = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")
print(json.dumps(results, indent=2))

# Save metrics
with open("outputs/metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2)

# Confusion matrix plot for best model
preds_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, preds_best)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)

# ROC curve
fig, ax = plt.subplots(figsize=(5, 4))
RocCurveDisplay.from_estimator(best_pipe, X_test, y_test, ax=ax)
ax.set_title(f"ROC Curve - {best_name}")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)

# Feature importance (Random Forest only)
if best_name == "RandomForest":
    feat_names = (numeric_cols +
                  list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_cols)))
    importances = best_pipe.named_steps["model"].feature_importances_
    top_idx = np.argsort(importances)[-10:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feat_names[i] for i in top_idx], importances[top_idx], color="teal")
    ax.set_title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)

# Save the model pipeline
joblib.dump(best_pipe, "outputs/churn_model.pkl")
print("\nSaved: outputs/churn_model.pkl, metrics.json, confusion_matrix.png, roc_curve.png")
