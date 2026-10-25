# 📊 Telco Customer Churn Prediction

An end-to-end machine learning project predicting customer churn using the IBM Telco dataset (~7,043 rows, 21 features). Built with **Scikit-Learn**, **XGBoost**, **LightGBM**, and **SHAP**.

---

## 📁 Repository Structure & Dataset
- 💾 **Dataset:** [Download Telco Customer Churn Dataset / Notebook Link](https://colab.research.google.com/drive/1wNDdXggzccv_GpMSFmw1oXD8sCD--mZz)
- 📓 `Customer_Churn_Prediction.ipynb` — Main end-to-end ML training and evaluation notebook
- 📦 `telco_churn_pipeline.pkl` — Exported production model pipeline (`ColumnTransformer` + `XGBoost`)
- 📝 `README.md` — Project documentation and performance benchmarking

---

## 🛠️ Tech Stack & Workflow
- **Language & Core Libraries:** Python 3.10+, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, SHAP, Matplotlib, Seaborn, Joblib
- **Data Preprocessing:** Handled whitespace values in `TotalCharges`, binary target encoding (`Churn`), stratified 80/20 train-test split, and robust feature transformations via `ColumnTransformer` (`StandardScaler` for numerical, `OneHotEncoder` for categorical features).
- **Class Imbalance Management:** Algorithmic reweighting via `class_weight='balanced'` and `scale_pos_weight`.

---

## 📈 Model Performance Benchmarks

All models were evaluated side-by-side on an unseen 20% test set (1,409 customer records):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Primary Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Selected Winner)** | **80.41%** | **0.6655** | 0.5267 | 0.5881 | **0.8475** | Top overall ROC-AUC and highest Precision |
| **Random Forest** | 75.94% | 0.5318 | 0.7834 | **0.6335** | 0.8432 | Highest overall F1-Score |
| **LightGBM** | 74.80% | 0.5164 | 0.7995 | 0.6275 | 0.8405 | Fast training with high churner recall |
| **Logistic Regression** | 73.81% | 0.5043 | 0.7834 | 0.6136 | 0.8417 | Highly interpretable linear baseline |
| **Decision Tree** | 73.46% | 0.5000 | **0.8075** | 0.6176 | 0.8308 | Highest individual recall capture rate |
| **K-Nearest Neighbors** | 75.80% | 0.5447 | 0.5374 | 0.5410 | 0.7793 | Distance-based baseline model |

---

## 🔑 Key Insights & Business Recommendations

1. **Model Selection:** **XGBoost** is selected for deployment due to its superior **ROC-AUC (0.8475)** and **Precision (66.55%)**, which minimizes false-positive interventions and prevents unnecessary marketing expenditure on stable accounts.
2. **Key Risk Drivers (SHAP Analysis):**
   - **Tenure:** Shorter tenure is the single strongest indicator of customer churn risk.
   - **Contract Type:** Month-to-month contracts experience significantly higher churn rates compared to 1- or 2-year commitments.
   - **Internet Service:** Fiber optic subscribers churn at a higher rate compared to DSL users, highlighting potential service or price
