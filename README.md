# TelcoGuard AI — Customer Churn Intelligence Platform

TelcoGuard AI is an end-to-end machine learning application that predicts
whether a telecom customer is likely to churn (cancel their subscription).
It combines a trained XGBoost classifier with an interactive Streamlit
dashboard, giving support, retention, and management teams an instant,
explainable risk assessment for any customer profile.

---

## 1. What the Platform Does

Given a customer's profile — contract type, billing information, tenure,
and the services they're subscribed to — TelcoGuard AI returns:

- **Churn Probability** — the model's estimated likelihood the customer
  will leave.
- **Risk Level** — a simplified Low / Medium / High classification.
- **Confidence** — the model's confidence that the customer will stay.
- **Financial Risk** — the estimated revenue exposure if this customer
  churns.

Beyond the headline numbers, the app provides five dedicated analysis
views (Overview, Risk Analysis, AI Insights, Financial Impact, and
Explainable AI), a What-If scenario simulator to test retention actions
before committing to them, and a downloadable per-customer report.

---

## 2. Model Training Pipeline (`train.py`)

The prediction model behind TelcoGuard AI was built through a complete,
standard machine learning workflow. This is the most important part of
the project, so it's documented in detail below.

### 2.1 Dataset

- **Source:** the IBM Telco Customer Churn dataset
  (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), containing 7,043 customer
  records.
- **Target variable:** `Churn` (Yes / No).
- **Features:** 19 customer attributes covering demographics (gender,
  senior citizen, partner, dependents), account information (tenure,
  contract type, payment method, paperless billing), billing (monthly
  and total charges), and subscribed services (phone, multiple lines,
  internet service, online security, online backup, device protection,
  tech support, streaming TV, streaming movies).

### 2.2 Exploratory Data Analysis & Visualization

Before any modeling, the dataset was explored visually to understand
what actually drives churn:

- **Churn distribution** — a count plot of the overall Yes/No churn
  split, to check class balance.
- **Churn by contract type** — a grouped bar chart showing that
  month-to-month customers churn at a much higher rate than customers on
  one- or two-year contracts.
- **Tenure distribution (KDE plot)** — density curves comparing how long
  churned vs. retained customers had been subscribed, showing churn is
  concentrated in the first few months.
- **Monthly charges distribution (KDE plot)** — comparing the spread of
  monthly bills between churned and retained customers.
- **Monthly charges boxplot** — churned customers tend to have higher
  monthly charges on average.
- **Churn by internet service type** — fiber-optic customers show a
  noticeably higher churn rate than DSL or no-internet customers.
- **Churn by payment method** — customers paying by electronic check
  churn more than those on automatic payment methods.

These visualizations directly informed which features were expected to
carry the most predictive signal, which was later confirmed by the
model's own feature-importance rankings.

### 2.3 Data Cleaning & Preprocessing

- Dropped the `customerID` column (a unique identifier with no
  predictive value).
- `TotalCharges` was stored as text and contained blank entries for
  customers with zero tenure; these were converted to `0.0` and the
  column was cast to a numeric (float) type.
- Standardized categorical values by replacing `"No internet service"`
  and `"No phone service"` with a plain `"No"`, so each service column
  has a consistent Yes/No meaning.
- **Binary encoding:** columns such as `Partner`, `Dependents`,
  `PhoneService`, `OnlineSecurity`, `TechSupport`, `PaperlessBilling`,
  and `Churn` itself were mapped from Yes/No to 1/0.
- **One-hot encoding:** multi-category columns (`InternetService`,
  `Contract`, `PaymentMethod`) were expanded into binary indicator
  columns via `pd.get_dummies`.
- `gender` was mapped to a binary flag (Male = 1, Female = 0).
- **Train/test split:** 80/20 split, stratified on the target variable to
  preserve the churn ratio in both sets.
- **Feature scaling:** `tenure`, `MonthlyCharges`, and `TotalCharges`
  were standardized with `StandardScaler` (fit on the training set only,
  then applied to the test set — the fitted scaler is saved and reused
  at inference time).
- **Class imbalance handling:** the training set (about 27% churn) was
  rebalanced using **SMOTE** (Synthetic Minority Over-sampling
  Technique) before training the linear and boosted models, so the
  model doesn't just learn to predict the majority class.

### 2.4 Models Trained & Compared

Four classifiers were trained and evaluated on the same held-out test
set, using Accuracy, Precision, Recall, F1-Score, and ROC-AUC:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model, trained on SMOTE-balanced data |
| Decision Tree | `max_depth=5`, interpretable single-tree baseline |
| Random Forest | 200 trees, `max_depth=10`, for a stronger ensemble baseline |
| **XGBoost** | Gradient-boosted trees, trained on SMOTE-balanced data — **selected as the final model** |

For every model, the pipeline generated a confusion matrix, an ROC
curve, and a feature-importance chart. The four models were then
compared side-by-side on all five metrics, and **XGBoost was selected
as the production model** based on its overall F1-Score and ROC-AUC —
the best trade-off between correctly catching customers who will churn
and avoiding false alarms.

### 2.5 Saved Artifacts

The training script exports everything the web app needs to reproduce
predictions at inference time:

- `xgb_model.pkl` — the trained XGBoost classifier
- `scaler.pkl` — the `StandardScaler` fitted on the training data
- `model_columns.pkl` — the exact ordered list of feature columns the
  model expects, used to align any new customer input to the training
  schema

These three files live in the `models/` folder and are loaded by the
Streamlit app at startup.

### 2.6 Retraining

To retrain on the same (or an updated) dataset:
```
python train.py
```
This regenerates the three `.pkl` files above — copy them into
`models/` afterwards to update the live app.

---

## 3. Application Features

- **Instant prediction** from a simple input form covering all customer
  attributes, with a **🎲 Fill Random Values** button to instantly
  populate the form with a realistic random customer for quick testing.
- **Five analysis tabs** for each prediction:
  - *Overview* — a quick summary and a recommended next action.
  - *Risk Analysis* — gauge and probability charts, plus an overall
    customer health indicator.
  - *AI Insights* — a generated executive summary, recommended retention
    actions, and the detected risk factors behind the score.
  - *Financial Impact* — estimated customer lifetime value, expected
    loss if the customer churns, and the retention ROI.
  - *Explainable AI* — a plain-language explanation of what drove the
    model's decision.
- **What-If Scenario Simulator** — adjust a customer's contract, charges,
  or services and instantly see how their churn risk would change,
  before actually offering that change to a real customer.
- **Downloadable report** — a self-contained HTML report per customer
  that can be saved or shared.

---

## 4. Project Structure

```
telcoguard/
├── app.py                  # Main entry point — run this file
├── config.py                # Global settings and constants
├── styles.py                 # Custom CSS for the UI
├── loader.py                  # Loads the trained model / scaler / columns
├── preprocessing.py            # Prepares customer input before prediction
├── predictor.py                 # Prediction logic
├── charts.py                     # Interactive Plotly charts
├── components.py                  # Reusable UI components (cards, headers)
├── financial.py                    # Financial impact calculations
├── ai_engine.py                     # AI-generated analysis and recommendations
├── simulator.py                      # What-If scenario simulator
├── report.py                          # Generates a downloadable HTML customer report
├── html_utils.py                       # Helper for rendering HTML correctly in Streamlit
├── train.py                            # Full model training pipeline (see section 2)
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Training dataset
└── models/
    ├── xgb_model.pkl
    ├── scaler.pkl
    └── model_columns.pkl
```

---

## 5. Running Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the app from inside the `telcoguard` folder:
   ```
   streamlit run app.py
   ```
3. Your browser will open automatically at `http://localhost:8501`

---

## 6. Deploying to Streamlit Community Cloud

1. Push this repo's contents to a GitHub repository (`app.py` should sit
   at the repo root, not inside a subfolder).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **New app**.
3. Select your repository and branch, set **Main file path** to
   `app.py`, and click **Deploy**.

`requirements.txt` is pinned to package versions verified to load the
saved model files correctly, to avoid version-mismatch issues on
deployment.
