import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained RF model
# -----------------------------
model = joblib.load("rf_model.pkl")

# Must match your training features exactly and in the same order
FEATURES = [
    'Administrative', 'Administrative_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 
    'SpecialDay', 'Month_num', 'OperatingSystems',
    'Browser', 'Region', 'TrafficType',
    'VisitorType_num', 'Weekend_num'
]

st.title("🛒 Online Shopper Revenue Prediction (Random Forest)")
st.write("Fill in the session details to predict whether the visitor will generate revenue (purchase).")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Input Session / Visitor Information")

Administrative = st.number_input("Administrative (number of admin pages)", min_value=0, step=1, value=0)
Administrative_Duration = st.number_input("Administrative_Duration (seconds)", min_value=0.0, step=1.0, value=0.0)

ProductRelated = st.number_input("ProductRelated (number of product pages)", min_value=0, step=1, value=0)
ProductRelated_Duration = st.number_input("ProductRelated_Duration (seconds)", min_value=0.0, step=1.0, value=0.0)

BounceRates = st.number_input("BounceRates (0 - 1)", min_value=0.0, max_value=1.0, step=0.001, value=0.05)
ExitRates = st.number_input("ExitRates (0 - 1)", min_value=0.0, max_value=1.0, step=0.001, value=0.05)

SpecialDay = st.number_input("SpecialDay (0 - 1, closer to 1 near special day)", 
                             min_value=0.0, max_value=1.0, step=0.01, value=0.0)

# Month_num – assuming you encoded months as 1–12 (Jan=1, Feb=2, ..., Dec=12)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_choice = st.selectbox("Month", month_labels)
Month_num = month_labels.index(month_choice) + 1   # -> 1–12

OperatingSystems = st.selectbox("OperatingSystems (1–8)", options=list(range(1, 9)))
Browser = st.selectbox("Browser (1–13)", options=list(range(1, 14)))
Region = st.selectbox("Region (1–9)", options=list(range(1, 10)))
TrafficType = st.selectbox("TrafficType (1–20)", options=list(range(1, 21)))

# VisitorType_num – assuming: New Visitor = 0, Returning Visitor = 1
visitor_type_label = st.selectbox("Visitor Type", ["New Visitor", "Returning Visitor"])
VisitorType_num = 1 if visitor_type_label == "Returning Visitor" else 0

# Weekend_num – assuming: No = 0, Yes = 1
weekend_label = st.selectbox("Weekend?", ["No", "Yes"])
Weekend_num = 1 if weekend_label == "Yes" else 0

# -----------------------------
# Build input DataFrame
# -----------------------------
input_dict = {
    'Administrative': Administrative,
    'Administrative_Duration': Administrative_Duration,
    'ProductRelated': ProductRelated,
    'ProductRelated_Duration': ProductRelated_Duration,
    'BounceRates': BounceRates,
    'ExitRates': ExitRates,
    'SpecialDay': SpecialDay,
    'Month_num': Month_num,
    'OperatingSystems': OperatingSystems,
    'Browser': Browser,
    'Region': Region,
    'TrafficType': TrafficType,
    'VisitorType_num': VisitorType_num,
    'Weekend_num': Weekend_num
}

input_df = pd.DataFrame([input_dict], columns=FEATURES)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Revenue (Yes / No)"):
    pred_class = model.predict(input_df)[0]

    # Optional: probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
    else:
        prob = None

    if pred_class == 1:
        st.success("✅ Prediction: This session is **likely to generate revenue (purchase)**.")
    else:
        st.error("❌ Prediction: This session is **unlikely to generate revenue**.")

    if prob is not None:
        st.write(f"Estimated probability of revenue: **{prob:.2%}**")
