import streamlit as st
import requests
import json

# Set the title
st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details below to check if it's fraudulent or not.")

# Define input fields
V1 = st.number_input("V1", value=0.0)
V2 = st.number_input("V2", value=0.0)
V3 = st.number_input("V3", value=0.0)
V4 = st.number_input("V4", value=0.0)
V5 = st.number_input("V5", value=0.0)
V6 = st.number_input("V6", value=0.0)
V7 = st.number_input("V7", value=0.0)
V8 = st.number_input("V8", value=0.0)
V9 = st.number_input("V9", value=0.0)
V10 = st.number_input("V10", value=0.0)
V11 = st.number_input("V11", value=0.0)
V12 = st.number_input("V12", value=0.0)
V13 = st.number_input("V13", value=0.0)
V14 = st.number_input("V14", value=0.0)
V15 = st.number_input("V15", value=0.0)
V16 = st.number_input("V16", value=0.0)
V17 = st.number_input("V17", value=0.0)
V18 = st.number_input("V18", value=0.0)
V19 = st.number_input("V19", value=0.0)
V20 = st.number_input("V20", value=0.0)
V21 = st.number_input("V21", value=0.0)
V22 = st.number_input("V22", value=0.0)
V23 = st.number_input("V23", value=0.0)
V24 = st.number_input("V24", value=0.0)
V25 = st.number_input("V25", value=0.0)
V26 = st.number_input("V26", value=0.0)
V27 = st.number_input("V27", value=0.0)
V28 = st.number_input("V28", value=0.0)
Amount = st.number_input("Transaction Amount", value=0.0)

# API URL (Replace this with your deployed API URL)
API_URL = "https://fraud-detection-api.onrender.com/predict"

# Button to predict fraud
if st.button("Predict Fraud"):
    # Prepare the input data
    data = {
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6, "V7": V7,
        "V8": V8, "V9": V9, "V10": V10, "V11": V11, "V12": V12, "V13": V13,
        "V14": V14, "V15": V15, "V16": V16, "V17": V17, "V18": V18, "V19": V19,
        "V20": V20, "V21": V21, "V22": V22, "V23": V23, "V24": V24, "V25": V25,
        "V26": V26, "V27": V27, "V28": V28, "Amount": Amount
    }

    # Send request to API
    response = requests.post("http://127.0.0.1:5000/predict", json=data)

    # Display result
    if response.status_code == 200:
        result = response.json()
        if result["fraud_prediction"] == 1:
            st.error("⚠️ Fraud Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")
    else:
        st.error("Error: Unable to get a prediction. Check API.")
