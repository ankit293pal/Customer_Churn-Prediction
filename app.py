import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load saved model
# -----------------------------
model = joblib.load("churn_model.pkl")

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Enter customer details and see prediction + probability + insights.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Inputs")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
tenure = st.sidebar.slider("Tenure (Months)", 0, 120, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 10000000.0, 70.0)
total = st.sidebar.number_input("Total Charges", 0.0, 9000000000.0, 2000.0)

if gender=="Male":
    gender=1
else:
    gender=0

if phone=="Yes":
    phone=1 
else:
    phone=0
if internet=="No":
    internet=0
elif internet=="DSL":
    internet=1
else:
    internet=2
if contract=="Month-to-month":
    contract=0
elif contract=="One year":
    contract=1
else:
    contract=2

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "tenure": [tenure],
    "PhoneService": [phone],
    "InternetService": [internet],
    "Contract": [contract],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total]
})

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100  # Churn probability

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔮 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Customer Likely to CHURN ({prob:.2f}%)")
        else:
            st.success(f"✅ Customer Likely to STAY ({100-prob:.2f}%)")

    with col2:
        st.subheader("📈 Churn Probability")
        fig, ax = plt.subplots()
        ax.bar(["Stay","Churn"], [100-prob, prob])
        ax.set_ylabel("Probability (%)")
        st.pyplot(fig)

# -----------------------------
# Dashboard Charts
# -----------------------------
st.write("---")
st.subheader("📊 Data Insights (Dashboard)")

# Load dataset for visualization
data = pd.read_csv("Customer-Churn.csv")

c1, c2 = st.columns(2)

with c1:
    st.write("Churn by Contract Type")
    fig, ax = plt.subplots()
    data.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().plot(kind="bar", ax=ax)
    st.pyplot(fig)

with c2:
    st.write("Churn by Internet Service")
    fig, ax = plt.subplots()
    data.groupby("InternetService")["Churn"].value_counts(normalize=True).unstack().plot(kind="bar", ax=ax)
    st.pyplot(fig)

st.write("Built ❤️ by Ankit")
