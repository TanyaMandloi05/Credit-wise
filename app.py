import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Loan Approval Model", layout="wide")

# ================= LOAD MODEL =================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body {
    background-color: #f5f6fa;
}

/* Center main container */
.block-container {
    max-width: 900px;
    margin: auto;
}

/* Section headings */
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #6c63ff;
    margin-top: 30px;
    margin-bottom: 10px;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #6c63ff, #8f7fff);
    color: white;
    border-radius: 10px;
    padding: 12px 30px;
    border: none;
    font-size: 16px;
}

/* Result box */
.result-approved {
    background: #50c878;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

.result-rejected {
    background: #fa003f;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

/* Progress bar */
.prob-wrap {
    margin-top: 15px;
}

.prob-track {
    width: 100%;
    height: 10px;
    background: #ddd;
    border-radius: 5px;
}

.prob-fill {
    height: 10px;
    background: linear-gradient(90deg, #6c63ff, #8f7fff);
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("<h1 style='text-align:center;'>Loan Approval model</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smarter Decisions for Better Finance</p>", unsafe_allow_html=True)

st.write("---")

# ================= FORM =================

# PERSONAL
st.markdown("<div class='section-title'>• PERSONAL INFORMATION</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 18, 70, 30)
with col2:
    dependents = st.number_input("Dependents", 0, 5, 0)
with col3:
    existing_loans = st.number_input("Existing Loans", 0, 5, 0)

# FINANCIAL
st.markdown("<div class='section-title'>• FINANCIAL INFORMATION</div>", unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    savings = st.number_input("Savings", 0, 500000, 200000)
    loan_amount = st.number_input("Loan Amount", 10000, 500000, 100000)
with col5:
    collateral = st.number_input("Collateral Value", 0, 1000000, 500000)
    loan_term = st.selectbox("Loan Term", [60, 120, 180, 240])

# EDUCATION
st.markdown("<div class='section-title'>• EMPLOYMENT & EDUCATION</div>", unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    education = st.selectbox("Education", ["Graduate", "Non Graduate"])
with col7:
    employment = st.selectbox("Employment", ["Salaried", "Unemployed"])

# ================= ENCODING =================
education_level = 1 if education == "Graduate" else 0
emp_sal = 1 if employment == "Salaried" else 0
emp_unemp = 1 if employment == "Unemployed" else 0

# ================= BUTTON =================
st.write("")
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    predict_btn = st.button("🚀 Check Loan Approval")

# ================= PREDICTION =================
if predict_btn:

    data = {
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_level,
        "DTI_Ratio_sq": 0.05,
        "Credit_Score_sq": 640000,
        "Applicant_Income_log": 9.5,
        "Employment_Status_Salaried": emp_sal,
        "Employment_Status_Unemployed": emp_unemp,
        "Marital_Status_Single": 1,
        "Loan_Purpose_Personal": 0,
        "Property_Area_Urban": 1,
        "Gender_Male": 1,
        "Employer_Category_Private": 1
    }

    df = pd.DataFrame([data])

    # Fix missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    # Scale
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    pct = int(probability * 100)

    # ================= RESULT =================
    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            <h3>✅ Loan Approved!</h3>
            <p>Congratulations! Your application meets our eligibility criteria.</p>
        </div>

        <div class="prob-wrap">
            <p><b>Approval Probability: {pct}%</b></p>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-rejected">
            <h3>❌ Loan Rejected</h3>
            <p>Your application does not meet our criteria.</p>
        </div>

        <div class="prob-wrap">
            <p><b>Approval Probability: {pct}%</b></p>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)