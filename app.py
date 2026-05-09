# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # ================= PAGE CONFIG =================
# st.set_page_config(page_title="Loan Approval Model", layout="wide")

# # ================= LOAD MODEL =================
# model = joblib.load("loan_model.pkl")
# scaler = joblib.load("scaler.pkl")
# feature_names = joblib.load("features.pkl")

# # ================= UI =================
# st.title("🏦 Loan Approval Model")
# st.write("Smart Loan Eligibility Prediction System")

# st.write("---")

# # ================= FORM =================

# # PERSONAL
# st.subheader("👤 Personal Information")
# col1, col2, col3 = st.columns(3)

# with col1:
#     age = st.number_input("Age", 18, 70, 30)

# with col2:
#     dependents = st.number_input("Dependents", 0, 5, 0)

# with col3:
#     existing_loans = st.number_input("Existing Loans", 0, 5, 0)

# # FINANCIAL
# st.subheader("💰 Financial Information")
# col4, col5 = st.columns(2)

# with col4:
#     savings = st.number_input("Savings", 0, 500000, 200000)
#     loan_amount = st.number_input("Loan Amount", 10000, 500000, 100000)
#     income = st.number_input("Applicant Income", 10000, 1000000, 50000)

# with col5:
#     collateral = st.number_input("Collateral Value", 0, 1000000, 500000)
#     loan_term = st.selectbox("Loan Term (months)", [60, 120, 180, 240])
#     credit_score = st.number_input("Credit Score", 300, 900, 750)
#     dti_ratio = st.number_input("DTI Ratio (0 - 1)", 0.0, 1.0, 0.3)

# # EDUCATION & EMPLOYMENT
# st.subheader("🎓 Employment & Education")
# col6, col7 = st.columns(2)

# with col6:
#     education = st.selectbox("Education", ["Graduate", "Non Graduate"])

# with col7:
#     employment = st.selectbox("Employment", ["Salaried", "Unemployed"])

# # ================= FEATURE ENGINEERING =================

# education_level = 1 if education == "Graduate" else 0
# emp_sal = 1 if employment == "Salaried" else 0
# emp_unemp = 1 if employment == "Unemployed" else 0

# # Transformations (VERY IMPORTANT)
# credit_score_sq = credit_score ** 2
# dti_ratio_sq = dti_ratio ** 2
# income_log = np.log(income)

# # ================= BUTTON =================
# st.write("")
# predict_btn = st.button("🚀 Check Loan Approval")

# # ================= PREDICTION =================
# if predict_btn:

#     data = {
#         "Age": age,
#         "Dependents": dependents,
#         "Existing_Loans": existing_loans,
#         "Savings": savings,
#         "Collateral_Value": collateral,
#         "Loan_Amount": loan_amount,
#         "Loan_Term": loan_term,
#         "Education_Level": education_level,
#         "DTI_Ratio_sq": dti_ratio_sq,
#         "Credit_Score_sq": credit_score_sq,
#         "Applicant_Income_log": income_log,
#         "Employment_Status_Salaried": emp_sal,
#         "Employment_Status_Unemployed": emp_unemp,
#         "Marital_Status_Single": 1,
#         "Loan_Purpose_Personal": 0,
#         "Property_Area_Urban": 1,
#         "Gender_Male": 1,
#         "Employer_Category_Private": 1
#     }

#     df = pd.DataFrame([data])

#     # Fix missing columns
#     for col in feature_names:
#         if col not in df.columns:
#             df[col] = 0

#     df = df[feature_names]

#     # Scale
#     scaled = scaler.transform(df)

#     # Predict
#     prediction = model.predict(scaled)[0]
#     probability = model.predict_proba(scaled)[0][1]
#     pct = int(probability * 100)

#     # ================= RESULT =================
#     if prediction == 1:
#         st.success(f"✅ Loan Approved! (Confidence: {pct}%)")
#     else:
#         st.error(f"❌ Loan Rejected (Confidence: {pct}%)")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(page_title="LoanWise", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu, footer, .stDeployButton,
.stApp > header { display: none !important; }
.block-container {
    padding-top: 0 !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

.stApp { background: #f5f6fa; color: #1a1a2e; }

/* ---- FIXED NAVBAR ---- */
.navbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    background: #fff; padding: 0.85rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #e5e7eb;
    box-shadow: 0 1px 12px rgba(0,0,0,0.06);
}
.nav-logo { display: flex; align-items: center; gap: 10px; }
.nav-logo-icon {
    width: 38px; height: 38px; background: #5b5ef4; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 1.05rem; font-weight: 800; flex-shrink: 0;
}
.nav-logo-text b { font-size: 1rem; font-weight: 700; color: #1a1a2e; display: block; line-height: 1.2; }
.nav-logo-text small { font-size: 0.52rem; letter-spacing: 1.8px; text-transform: uppercase; color: #9ca3af; }
.nav-badge {
    background: #ededfe; color: #5b5ef4; font-size: 0.7rem; font-weight: 600;
    padding: 6px 16px; border-radius: 20px; letter-spacing: 0.2px;
    border: 1px solid #d4d5fc; white-space: nowrap;
}

/* Push content below fixed navbar */
.page-content { margin-top: 66px; }

/* ---- HERO ---- */
.hero {
    text-align: center; padding: 3.5rem 1.5rem 2.5rem;
    background: #fff; border-bottom: 1px solid #ebebf0;
}
.hero h1 {
    font-size: clamp(1.6rem, 4vw, 2.8rem);
    font-weight: 900; color: #5b5ef4;
    line-height: 1.15; letter-spacing: -1px; margin: 0;
}

/* ---- VALUE CARDS ---- */
.values-row {
    display: flex; flex-wrap: wrap; gap: 0;
    background: #fff; border-bottom: 1px solid #ebebf0;
}
.value-card {
    flex: 1; min-width: 220px; padding: 2.5rem 2rem; text-align: center;
    border-right: 1px solid #ebebf0;
}
.value-card:last-child { border-right: none; }
.value-icon {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #5b5ef4, #8b8ef8);
    border-radius: 18px; display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin: 0 auto 1.25rem;
}
.value-title { font-size: 1.05rem; font-weight: 700; color: #1a1a2e; margin-bottom: 3px; }
.value-tag { font-size: 0.58rem; font-weight: 700; letter-spacing: 2.5px; color: #5b5ef4; text-transform: uppercase; margin-bottom: 0.9rem; }
.value-desc { font-size: 0.8rem; color: #6b7280; line-height: 1.7; max-width: 240px; margin: 0 auto; }

/* ---- FORM HEADER ---- */
.form-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 2rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid #f0f0f5;
}
.form-header-icon {
    width: 44px; height: 44px; background: #ededfe; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
}
.form-header h2 { font-size: 1.15rem; font-weight: 700; color: #1a1a2e; margin: 0; }
.form-header p { font-size: 0.78rem; color: #9ca3af; margin: 3px 0 0; }

/* ---- SECTION LABELS ---- */
.section-label {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.7rem; font-weight: 700; color: #5b5ef4;
    letter-spacing: 1px; margin: 1.75rem 0 1rem; text-transform: uppercase;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #f0f0f5; }
.section-label-icon {
    width: 28px; height: 28px; background: #ededfe; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; flex-shrink: 0;
}

/* ---- INPUT OVERRIDES ---- */
[data-testid="stNumberInput"] input {
    background: #fafafa !important; border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important; color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
    font-weight: 500 !important; padding: 0.6rem 0.9rem !important;
    transition: border-color 0.15s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #5b5ef4 !important; background: #fff !important;
    box-shadow: 0 0 0 3px rgba(91,94,244,0.1) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #fafafa !important; border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important; color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #4b5563 !important; font-size: 0.78rem !important;
    font-weight: 600 !important; margin-bottom: 4px !important;
}

/* Make Streamlit columns stack on small screens */
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .value-card { border-right: none !important; border-bottom: 1px solid #ebebf0; }
    .value-card:last-child { border-bottom: none; }
    .navbar { padding: 0.75rem 1rem; }
    .nav-badge { display: none; }
    .hero { padding: 2rem 1rem 1.5rem; }
    .steps-row { flex-direction: column; }
    .step { border-right: none !important; border-bottom: 1px solid #ebebf0; }
    .step:last-child { border-bottom: none; }
}

/* ---- BUTTON ---- */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #5b5ef4, #7c7ff6) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; padding: 0.8rem 2.5rem !important;
    box-shadow: 0 4px 20px rgba(91,94,244,0.4) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(91,94,244,0.5) !important;
}

/* ---- RESULT CARD ---- */
.result-card {
    border-radius: 16px; padding: 1.75rem 2rem; margin-top: 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    flex-wrap: wrap;
}
.result-approved { background: #f0fdf4; border: 1.5px solid #86efac; }
.result-rejected { background: #fff1f2; border: 1.5px solid #fca5a5; }
.result-icon { font-size: 2.8rem; }
.result-label { font-size: 0.58rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
.result-approved .result-label { color: #16a34a; }
.result-rejected .result-label { color: #dc2626; }
.result-title { font-size: clamp(1.2rem, 3vw, 1.6rem); font-weight: 800; color: #1a1a2e; }
.conf-bar-wrap { height: 5px; background: #e5e7eb; border-radius: 3px; width: min(200px, 100%); margin-top: 10px; }
.conf-bar { height: 100%; border-radius: 3px; }
.result-approved .conf-bar { background: #22c55e; }
.result-rejected .conf-bar { background: #ef4444; }
.conf-text { font-size: 0.72rem; color: #6b7280; margin-top: 5px; }

/* ---- FOOTER STEPS ---- */
.steps-row {
    display: flex; flex-wrap: wrap; gap: 0; background: #fff;
    border-top: 1px solid #ebebf0; border-bottom: 1px solid #ebebf0;
    margin-top: 2rem;
}
.step {
    display: flex; align-items: flex-start; gap: 14px;
    flex: 1; min-width: 160px;
    padding: 1.75rem 2rem; border-right: 1px solid #ebebf0;
}
.step:last-child { border-right: none; }
.step-num {
    width: 36px; height: 36px; flex-shrink: 0;
    background: linear-gradient(135deg, #5b5ef4, #8b8ef8);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 800; color: #fff;
}
.step-title { font-size: 0.85rem; font-weight: 700; color: #1a1a2e; margin-bottom: 3px; }
.step-desc { font-size: 0.73rem; color: #9ca3af; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# ================= FIXED NAVBAR =================
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-logo-icon">L</div>
        <div class="nav-logo-text">
            <b>LoanWise</b>
            <small>Smart Loans, Better Tomorrow</small>
        </div>
    </div>
    <div class="nav-badge">✦ AI Powered Loan Approval</div>
</div>
<div class="page-content">
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <h1>VALUES THAT POWER<br>BETTER FINANCE.</h1>
</div>
""", unsafe_allow_html=True)

# ================= VALUE CARDS =================
st.markdown("""
<div class="values-row">
    <div class="value-card">
        <div class="value-icon">⚡</div>
        <div class="value-title">Simplicity</div>
        <div class="value-tag">First</div>
        <div class="value-desc">We turn complex financial processes into effortless experiences. Because managing your money shouldn't feel like a chore.</div>
    </div>
    <div class="value-card">
        <div class="value-icon">🔒</div>
        <div class="value-title">Trust</div>
        <div class="value-tag">Through Transparency</div>
        <div class="value-desc">We put your privacy and security above everything else. Our platform uses bank-grade encryption and clear communication.</div>
    </div>
    <div class="value-card">
        <div class="value-icon">💡</div>
        <div class="value-title">Innovation</div>
        <div class="value-tag">With Purpose</div>
        <div class="value-desc">We don't innovate for the sake of it — we innovate to make your life easier. Every feature helps you make smarter decisions.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= FORM HEADER =================
st.markdown("""
<div class="form-header" style="margin-top:2rem;">
    <div class="form-header-icon">👤</div>
    <div>
        <h2>Applicant Information</h2>
        <p>Please provide the details below to check your loan eligibility</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= PERSONAL =================
st.markdown('<div class="section-label"><div class="section-label-icon">👤</div> Personal Information</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    age = st.number_input("Age", 18, 70, 30)
with col2:
    dependents = st.number_input("Dependents", 0, 5, 0)
with col3:
    existing_loans = st.number_input("Existing Loans", 0, 5, 0)

# ================= FINANCIAL =================
st.markdown('<div class="section-label"><div class="section-label-icon">💰</div> Financial Information</div>', unsafe_allow_html=True)
col4, col5 = st.columns([1, 1])
with col4:
    savings = st.number_input("Savings (₹)", 0, 500000, 200000)
    loan_amount = st.number_input("Loan Amount (₹)", 10000, 500000, 100000)
    income = st.number_input("Applicant Income (₹)", 10000, 1000000, 50000)
with col5:
    collateral = st.number_input("Collateral Value (₹)", 0, 1000000, 500000)
    loan_term = st.selectbox("Loan Term (Months)", [60, 120, 180, 240])
    credit_score = st.number_input("Credit Score", 300, 900, 750)
    dti_ratio = st.number_input("DTI Ratio (0 – 1)", 0.0, 1.0, 0.3)

# ================= EDUCATION & EMPLOYMENT =================
st.markdown('<div class="section-label"><div class="section-label-icon">🎓</div> Employment & Education</div>', unsafe_allow_html=True)
col6, col7 = st.columns([1, 1])
with col6:
    education = st.selectbox("Education", ["Graduate", "Non Graduate"])
with col7:
    employment = st.selectbox("Employment", ["Salaried", "Unemployed"])

# ================= FEATURE ENGINEERING =================
education_level = 1 if education == "Graduate" else 0
emp_sal = 1 if employment == "Salaried" else 0
emp_unemp = 1 if employment == "Unemployed" else 0

credit_score_sq = credit_score ** 2
dti_ratio_sq = dti_ratio ** 2
income_log = np.log(income)

# ================= BUTTON =================
st.write("")
col_btn, _ = st.columns([1, 3])
with col_btn:
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
        "DTI_Ratio_sq": dti_ratio_sq,
        "Credit_Score_sq": credit_score_sq,
        "Applicant_Income_log": income_log,
        "Employment_Status_Salaried": emp_sal,
        "Employment_Status_Unemployed": emp_unemp,
        "Marital_Status_Single": 1,
        "Loan_Purpose_Personal": 0,
        "Property_Area_Urban": 1,
        "Gender_Male": 1,
        "Employer_Category_Private": 1
    }

    df = pd.DataFrame([data])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    pct = int(probability * 100)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-approved">
            <div class="result-icon">✅</div>
            <div>
                <div class="result-label">Decision</div>
                <div class="result-title">Loan Approved</div>
                <div class="conf-bar-wrap"><div class="conf-bar" style="width:{pct}%"></div></div>
                <div class="conf-text">Model confidence: {pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-rejected">
            <div class="result-icon">❌</div>
            <div>
                <div class="result-label">Decision</div>
                <div class="result-title">Loan Rejected</div>
                <div class="conf-bar-wrap"><div class="conf-bar" style="width:{pct}%"></div></div>
                <div class="conf-text">Model confidence: {pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================= FOOTER STEPS =================
st.markdown("""
<div class="steps-row">
    <div class="step">
        <div class="step-num">1</div>
        <div>
            <div class="step-title">Fill Details</div>
            <div class="step-desc">Enter your information accurately</div>
        </div>
    </div>
    <div class="step">
        <div class="step-num">2</div>
        <div>
            <div class="step-title">AI Analysis</div>
            <div class="step-desc">Our AI analyses your eligibility</div>
        </div>
    </div>
    <div class="step">
        <div class="step-num">3</div>
        <div>
            <div class="step-title">Get Result</div>
            <div class="step-desc">Instant loan approval prediction</div>
        </div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)