# 📊 Customer Churn Prediction with Decision Trees
* This project focuses on building a machine learning model to predict customer churn based on client behavior and pricing data. It leverages feature engineering, data preprocessing, and resampling techniques to ensure balanced training and generalization of the model.
---
# 🧰 Tools & Libraries
* Python (pandas, NumPy)

* scikit-learn (DecisionTreeClassifier, preprocessing, evaluation metrics)

* imbalanced-learn (RandomOverSampler, RandomUnderSampler)

* Matplotlib & Seaborn (visualization)
---
# 📂 Dataset Overview
* Two datasets are used in this project:

* client_data.csv – Client-level information including churn labels and contract metadata.

* price_data.csv – Pricing history associated with each client over time.

**Dataset	Records	Features**
* client_data	14,606	26
* price_data	193,002	8

# 📊 Exploratory Data Analysis (EDA)
**Summary statistics**

* Data type checks and missing value handling

**Visualizations:**

* Distribution plots using histplot

* Bar and pie plots for churn distribution

**Channel sales breakdown**
---
# 🛠️ Feature Engineering
* New features were created to better capture client behavior and contract characteristics:
---
# 🔁 Time-based Features
* Tenure duration

* Days since last modification

* Days to renewal
---
# 🔌 Consumption & Forecast Features
* Average monthly consumption

* Change in consumption

* Gas/electricity ratio

* Forecast vs actual error

* Discount per kWh
---
# 💸 Price-Based Features
* Peak vs off-peak price differences

* Mid-peak comparisons

* January–December price deltas
---
# 📈 Margin & Product Features
* Gross/net margin ratios

* Products per client

* Power consumption log

* Antiquity bins
---
# 🧼 Data Preprocessing
* Converted date columns to datetime

* Encoded categorical variables using OrdinalEncoder

* Converted engineered features to integer type

* Binned antig_category into ordinal codes
---
# 🧪 Train/Test Split
* 80/20 stratified split:

* X_train: (140,119, 47)

* X_test: (35,030, 47)
---
# ⚖️ Handling Imbalanced Data
* To combat class imbalance (churn rate ~9.7%), both oversampling and undersampling techniques were applied:
---
# 🔁 RandomOverSampler
* Duplicates minority class to match majority

* Final shape: 252,882 samples
---
# ✂️ RandomUnderSampler
* Trims majority class down to match minority

* Final shape: 27,356 samples
---
# 🌲 Model Training – Decision Tree Classifier
* Three models were trained:

**Model Type	Train Accuracy	Test Accuracy**
* Original Data	1.00	0.9999
* Oversampled	1.00	0.9998
* Undersampled	0.9596	0.9550

**Cross-validation on original data (5 folds):**
* [0.9997, 0.9998, 0.9999, 0.9998, 0.9997]
* Mean Accuracy: 0.9998
* ✅ Indicates excellent generalization and low variance between training and testing.
---
# 📉 Evaluation Metrics
**Classification Report**

* Precision, Recall, F1-score = 1.00 (across both classes)

* Confusion Matrix

* Perfect classification observed

* Baseline Accuracy: 90.2%
---
# 🔍 Feature Importance (Top 10)
**Feature	Importance**
* forecast_meter_rent_12m	0.077
* margin_net_pow_ele	0.073
* offpeak_diff_dec_january_energy	0.072
* cons_change	0.069
* days_since_modif	0.062
* pow_max	0.055
* forecast_error_year	0.040
* tenure_days	0.040
* cons_last_month	0.040
* forecast_price_energy_off_peak	0.039

* Visualized with horizontal bar chart for easier interpretation.
