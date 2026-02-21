Good. This is a **serious MLOps project**, not just a notebook exercise.
Below is a **complete MVP + detailed SRS**, strictly using **Streamlit for deployment** and implementing Level 0 → Level 2 (with pipeline simulation in Python, not SageMaker).

---

# ✅ MINIMUM VIABLE PRODUCT (MVP)

## 🎯 MVP Goal

Deliver an end-to-end ML system that:

1. Preprocesses Online Retail II data (UK only)
2. Trains XGBoost classifier
3. Deploys via Streamlit
4. Simulates drift monitoring
5. Detects prediction change > 10%

---

## 🔹 MVP Scope

### Included

* Data preprocessing pipeline
* Feature engineering
* XGBoost model training
* Model evaluation
* Model saving (`xgb_model.pkl`)
* Streamlit app (prediction UI)
* Drift simulation script
* Monitoring report

### Excluded (For MVP simplicity)

* CI/CD automation
* Real-time production monitoring
* Cloud infra (AWS/GCP)
* Automated retraining

---

## 🏗 MVP Architecture

```
Kaggle Dataset
      ↓
Preprocessing Script
      ↓
train.csv / test.csv
      ↓
Model Training (XGBoost)
      ↓
xgb_model.pkl
      ↓
Streamlit App (Prediction UI)
      ↓
Drift Monitoring Script
```

---

# 📘 SOFTWARE REQUIREMENTS SPECIFICATION (SRS)

---

# 1️⃣ Introduction

## 1.1 Purpose

This system predicts whether a UK customer will make a purchase next month using transaction history. It demonstrates MLOps Level 0–2 implementation using Streamlit deployment.

## 1.2 Scope

The system includes:

* Data preprocessing
* Model training
* Local deployment using Streamlit
* Drift detection mechanism
* Monitoring evaluation

---

# 2️⃣ Overall Description

## 2.1 Product Perspective

Standalone ML system running locally.

Tech stack:

* Python 3.10+
* Pandas
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib (optional visualization)
* Joblib/Pickle

---

## 2.2 Product Functions

### Functional Modules

| Module              | Description                                |
| ------------------- | ------------------------------------------ |
| Data Preprocessing  | Clean and transform dataset                |
| Feature Engineering | Create TotalPrice, InvoiceHour, InvoiceDay |
| Model Training      | Train XGBoost                              |
| Evaluation          | Accuracy, Precision, Recall, F1            |
| Deployment          | Streamlit UI                               |
| Drift Monitoring    | Simulate and detect data drift             |

---

# 3️⃣ Functional Requirements

---

## FR1: Data Preprocessing

### Input

Online Retail II dataset (CSV)

### Processing Steps

1. Filter: `Country == "United Kingdom"`
2. Drop rows where:

   * CustomerID is null
   * Quantity <= 0
3. Convert InvoiceDate to datetime
4. Create features:

```
TotalPrice = Quantity * UnitPrice
InvoiceHour = hour(InvoiceDate)
InvoiceDay = day(InvoiceDate)
```

5. Target Creation:

For each customer:

* If purchase exists in next month → 1
* Else → 0

6. Drop:

* CustomerID
* InvoiceDate

7. Encode Country (Label Encoding)

8. Train-test split:

* 70% Train
* 30% Test
* random_state=42

### Output

* `train.csv`
* `test.csv`

---

## FR2: Model Development

### Model

XGBoostClassifier:

```
n_estimators=100
max_depth=5
learning_rate=0.1
random_state=42
```

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

### Output

* `xgb_model.pkl`
* Metrics table in notebook

---

## FR3: Deployment (Streamlit – Level 0)

### UI Requirements

Form Inputs:

* Quantity (number)
* UnitPrice (float)
* InvoiceHour (0–23)
* InvoiceDay (1–31)

### Output:

* “Purchase Likely”
* “Not Likely”

### Model Loading:

```
model = joblib.load("xgb_model.pkl")
```

---

## FR4: Pipeline Simulation (Level 1 Equivalent)

Pipeline script should:

1. Run preprocessing
2. Train model
3. Save model
4. Evaluate model
5. Log metrics

This acts as a reproducible pipeline.

---

## FR5: Drift Monitoring (Level 2)

### Drift Simulation

Modify test set:

```
Quantity = Quantity * 1.2
UnitPrice = UnitPrice * 0.8
```

### Detection Logic

1. Predict original test set
2. Predict drifted test set
3. Compute:

```
prediction_change % =
(number of different predictions / total predictions) * 100
```

### Alert Condition

If change > 10%:
→ Print "Drift Detected"

---

# 4️⃣ Non-Functional Requirements

## Performance

* Prediction time < 1 second

## Usability

* Streamlit UI must be simple
* Clear labels

## Reliability

* Model file must load safely
* Handle invalid inputs

## Maintainability

* Modular scripts:

  * preprocess.py
  * train.py
  * app.py
  * monitor.py

---

# 5️⃣ Project Folder Structure

```
mlops-retail-project/
│
├── data/
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── xgb_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── monitor.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 6️⃣ Streamlit Deployment Requirements

### Install

```
pip install streamlit xgboost pandas scikit-learn joblib
```

### Run

```
streamlit run app.py
```

---

# 7️⃣ Monitoring Logic Specification

### Drift Threshold

10%

### Example Output

```
Prediction Change: 14.3%
⚠ Drift Detected
```

---

# 8️⃣ Evaluation Criteria (Academic Alignment)

| Level   | Implementation                |
| ------- | ----------------------------- |
| Level 0 | Manual training + Streamlit   |
| Level 1 | Scripted pipeline             |
| Level 2 | Drift simulation + monitoring |

---

# 9️⃣ Risk Analysis

| Risk                 | Mitigation                |
| -------------------- | ------------------------- |
| Imbalanced data      | Use class_weight or SMOTE |
| Poor feature quality | Improve aggregation       |
| Overfitting          | Cross-validation          |
| Data leakage         | Careful target creation   |

---

# 🔥 Final Deliverables Checklist

✅ train.csv
✅ test.csv
✅ xgb_model.pkl
✅ Jupyter notebook with metrics
✅ Streamlit app
✅ Drift monitoring script
✅ Screenshot:

* 3 predictions
* Drift detection result

---

# 🎯 What Makes This Strong?

* Proper feature engineering
* Real drift detection logic
* Clear pipeline separation
* Production-style structure
* MLOps Level 0 → 2 coverage

---

help me to execute this plan
