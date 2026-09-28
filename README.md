# Customer Churn Prediction — End-to-End ML Project

A complete, portfolio-ready machine learning project that predicts whether a
telecom customer is likely to churn (leave the service), using classic
supervised learning techniques and an interactive web demo.

## 🎯 Problem Statement
Customer churn directly impacts revenue. This project builds a classification
model that flags high-risk customers early, so a business can target them
with retention offers.

## 🛠️ Tech Stack
- **Python**, **Pandas**, **NumPy** — data handling
- **Scikit-learn** — preprocessing (StandardScaler, OneHotEncoder), Logistic
  Regression, Random Forest
- **Matplotlib / Seaborn** — EDA and result visualization
- **Streamlit** — interactive prediction web app
- **Joblib** — model persistence

## 📂 Project Structure
```
churn_prediction/
├── data/
│   └── generate_data.py     # generates the dataset (swap for real data easily)
├── eda.py                   # exploratory data analysis + plots
├── train_model.py           # preprocessing, training, evaluation, model export
├── app.py                   # Streamlit demo app
├── outputs/                 # generated plots, metrics.json, saved model
└── requirements.txt
```

## 🔄 Pipeline
1. **Data preparation** — categorical encoding (One-Hot) + numeric scaling
   inside a single `sklearn.Pipeline` (no data leakage).
2. **EDA** — churn distribution, churn by contract type, monthly charges &
   tenure vs churn.
3. **Modeling** — trained and compared Logistic Regression and Random Forest.
4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
5. **Deployment** — Streamlit app for real-time, interactive predictions.

## 📊 Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.69 | 0.63 | 0.59 | 0.61 | **0.756** |
| Random Forest | 0.685 | 0.63 | 0.55 | 0.59 | 0.735 |

**Best model:** Logistic Regression (selected by F1-score)

Key churn drivers identified: **contract type** (month-to-month customers
churn far more), **payment method** (electronic check = higher risk),
**tenure**, and **internet service type** (fiber optic customers churn more).

## ▶️ How to Run
```bash
pip install -r requirements.txt
python data/generate_data.py     # generate dataset
python eda.py                    # run EDA, saves plots to outputs/
python train_model.py            # train models, saves best model to outputs/
streamlit run app.py             # launch the interactive demo
```

## 💡 What This Project Demonstrates (for recruiters)
- End-to-end ML workflow: data generation/cleaning → EDA → feature
  engineering → model training → evaluation → deployment
- Proper use of `sklearn.Pipeline` / `ColumnTransformer` to avoid data leakage
- Model comparison and selection based on business-relevant metrics
- Ability to ship a usable interactive product (Streamlit), not just a notebook

## 🚀 Possible Extensions
- Swap in the real [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Add XGBoost / LightGBM and hyperparameter tuning (GridSearchCV/Optuna)
- Add SHAP for model explainability
- Deploy on Streamlit Community Cloud / Render / AWS for a live public link
