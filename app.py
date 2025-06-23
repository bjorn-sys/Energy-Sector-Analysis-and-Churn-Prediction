import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("🔍 Customer Churn Prediction App")
st.write("Upload customer data to predict churn likelihood.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_input.head())

    # Drop unnecessary columns (same as training)
    drop_cols = ['id', 'churn', 'date_activ', 'date_end', 'date_modif_prod', 'date_renewal', 'price_date']
    for col in drop_cols:
        if col in df_input.columns:
            df_input = df_input.drop(columns=[col])

    # Ensure all necessary features are present
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else model.feature_importances_.shape[0]
    missing_cols = set(model_features) - set(df_input.columns)
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Predict
        prediction = model.predict(df_input)
        proba = model.predict_proba(df_input)[:, 1]

        # Show results
        st.subheader("🔮 Prediction Results")
        df_result = pd.DataFrame({
            "Prediction": prediction,
            "Churn Probability": proba
        })

        df_result["Prediction"] = df_result["Prediction"].map({0: "No Churn", 1: "Churn"})
        st.dataframe(df_result)

        st.success("Prediction complete!")
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv, "churn_predictions.csv", "text/csv")

else:
    st.info("Please upload a CSV file to begin.")
