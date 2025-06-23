# Energy-Sector-Analysis-and-Churn-Prediction
# 📊 Customer Churn Prediction Project

## 🧾 Overview

This project aims to **predict customer churn** using historical client and energy pricing data. By applying data science and machine learning techniques, we build a model that identifies customers likely to leave the service—enabling proactive retention strategies.

---

## 📁 Data Description

The project uses two main datasets:

- **Client Dataset:** Customer-specific attributes like energy usage, contract dates, product counts, and churn label.
- **Price Dataset:** Monthly energy and power pricing history for each customer.

---

## 🔍 Workflow Summary

### 1. Data Collection

- Loaded both datasets using `pandas`.
- Verified data types, missing values, and memory usage.

### 2. Exploratory Data Analysis (EDA)

- Plotted churn distribution using bar and pie charts.
- Analyzed `channel_sales` segments and their churn ratios.
- Explored distributions of key numerical features (e.g., `cons_12m`, `margin_net_pow_ele`, etc.).
- Highlighted high-risk segments based on visual trends.

### 3. Data Preprocessing

- Converted all date fields into proper datetime format.
- Merged client and price data using customer `id`.
- Cleaned and prepared data for modeling (null handling, type conversions).

---

## 🛠️ Feature Engineering

Created new features to boost predictive power:

- **Time Features:**
  - `tenure_days`, `days_since_modif`, `days_to_renewal`

- **Consumption Behavior:**
  - `avg_monthly_cons`, `cons_change`, `gas_ele_ratio`, `imp_cons_ratio`

- **Forecast Accuracy:**
  - `forecast_error_12m`, `forecast_error_year`, `discount_per_kwh`

- **Price Dynamics:**
  - Differences in peak/mid/off-peak energy & power prices

- **Profitability Metrics:**
  - `gross_net_margin_ratio`, `margin_per_product`

- **Behavioral Indicators:**
  - `has_multiple_products`, `pow_max_log`

- **Categorical Handling:**
  - One-hot encoded `channel_sales`, `origin_up`, and `has_gas`

- **Antiquity Buckets:**
  - `antig_category`: Binned into `new`, `recent`, `mid`, and `old`

- **Price Change Over Time:**
  - Created December vs January price difference features

---

## 🤖 Modeling

- **Model Used:** `RandomForestClassifier` (Scikit-learn)
- **Train/Test Split:** 80% training, 20% testing with stratification on churn
- **Evaluation Metrics:**
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC AUC Score
  - Confusion Matrix
  - Feature Importance Chart

The model achieved **perfect classification** with:
- Training Accuracy: `1.00`
- Testing Accuracy: `1.00`
- ROC AUC Score: `1.00`

---

## 💾 Model Deployment

The trained model was saved as a `.pkl` file:

```python
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
---

# 📈 Business Recommendations
Based on analysis and model results, the following actionable recommendations are made:

1. 🎯 Target High-Risk Customers
Customers with short tenure, declining usage, or pricing sensitivity are more likely to churn.

Use the model to flag them early for personalized outreach.

2. 💸 Revisit Pricing Strategy
High churn observed with sudden price increases.

Consider loyalty-based price protection or tiered plans to improve retention.

3. 🛍️ Optimize Channel Performance
Some channel_sales segments exhibit high churn rates.

Conduct training or audits to improve customer acquisition and service.

4. 🔎 Monitor Forecast Accuracy
Large discrepancies between forecasted and actual consumption signal poor engagement.

Use this as a trigger for customer support follow-ups.

5. 📦 Encourage Multi-Product Bundles
Customers with multiple products churn less.

Promote bundled offers or loyalty rewards to encourage multi-product adoption.

6. 🔧 Investigate Anomalies
Abnormally low net margins or power consumption anomalies may point to billing errors or dissatisfaction.

Use internal audits and alerts.

7. ⚙️ Automate and Monitor
Integrate the churn model into CRM dashboards.

Generate monthly churn-risk reports for the sales and support teams.

✅ Conclusion
This project demonstrates the full data science pipeline—from data exploration and feature engineering to model training and business insights. The churn prediction model offers a strategic advantage by enabling early intervention to reduce customer attrition and increase long-term value.
