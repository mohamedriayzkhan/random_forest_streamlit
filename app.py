import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Status Prediction", layout="centered")

st.title("🏦 Loan Status Prediction (Random Forest)")

# Load model
model = joblib.load("model.pkl")

# Load dataset for feature reference
df = pd.read_csv("data/dataset.csv")
df.columns = df.columns.str.strip()
df.drop(columns=["loan_id", "loan_status"], inplace=True)

st.write("Enter applicant details:")

input_data = {}

for col in df.columns:
    if df[col].dtype == "object":
        input_data[col] = st.selectbox(col, sorted(df[col].unique()))
    else:
        input_data[col] = st.number_input(
            col,
            value=float(df[col].mean())
        )

input_df = pd.DataFrame([input_data])

if st.button("Predict Loan Status"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()

    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    st.subheader("Prediction Result")
    st.success(result)
    st.write(f"Confidence: {probability * 100:.2f}%")
    