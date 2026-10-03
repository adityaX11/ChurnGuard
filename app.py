import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="ChurnGuard AI | Customer Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Modern SaaS Theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 26px 32px;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(12px);
}

.hero-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25));
    color: #A5B4FC;
    border: 1px solid rgba(129, 140, 248, 0.35);
    margin-bottom: 12px;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    letter-spacing: -0.6px;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.02rem;
    margin: 0;
    max-width: 720px;
    line-height: 1.5;
}

/* Section Header Cards */
.section-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    padding-bottom: 8px;
}

/* Prediction Result Cards */
.result-card-safe {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(5, 150, 105, 0.06) 100%);
    border: 1.5px solid rgba(16, 185, 129, 0.35);
    border-radius: 20px;
    padding: 26px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.12);
}

.result-card-danger {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.06) 100%);
    border: 1.5px solid rgba(239, 68, 68, 0.4);
    border-radius: 20px;
    padding: 26px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.15);
}

.result-card-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(180, 83, 9, 0.06) 100%);
    border: 1.5px solid rgba(245, 158, 11, 0.35);
    border-radius: 20px;
    padding: 26px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.12);
}

.score-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}

.score-value {
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 6px 0;
}

.val-safe { color: #34D399; }
.val-warning { color: #FBBF24; }
.val-danger { color: #F87171; }

.status-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 9999px;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-top: 10px;
}

.badge-safe {
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.2);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-danger {
    background: rgba(239, 68, 68, 0.2);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.insight-box {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    margin-top: 16px;
    text-align: left;
}

.insight-header {
    font-size: 0.9rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 8px;
}

.insight-item {
    font-size: 0.86rem;
    color: #94A3B8;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_churn_model():
    return tf.keras.models.load_model('model.h5')

# Load encoders and scaler with caching
@st.cache_resource
def load_preprocessors():
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return label_encoder_gender, onehot_encoder_geo, scaler

model = load_churn_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_preprocessors()

# Sidebar: Details & Demo Presets
with st.sidebar:
    st.markdown("### 🛡️ ChurnGuard AI")
    st.caption("Deep Learning Customer Retention Engine")
    
    st.markdown("---")
    st.markdown("#### ⚡ Quick Test Presets")
    st.write("Test representative customer archetypes with a single click:")
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        if st.button("🟢 Loyal User", use_container_width=True, help="Load a high-retention customer profile"):
            st.session_state['credit_score'] = 740.0
            st.session_state['geography'] = 'France'
            st.session_state['gender'] = 'Male'
            st.session_state['age'] = 32
            st.session_state['tenure'] = 6
            st.session_state['balance'] = 52000.0
            st.session_state['num_of_products'] = 2
            st.session_state['has_cr_card'] = 1
            st.session_state['is_active_member'] = 1
            st.session_state['estimated_salary'] = 85000.0

    with col_pre2:
        if st.button("🔴 At-Risk User", use_container_width=True, help="Load a high-churn customer profile"):
            st.session_state['credit_score'] = 530.0
            st.session_state['geography'] = 'Germany'
            st.session_state['gender'] = 'Female'
            st.session_state['age'] = 54
            st.session_state['tenure'] = 1
            st.session_state['balance'] = 125000.0
            st.session_state['num_of_products'] = 1
            st.session_state['has_cr_card'] = 1
            st.session_state['is_active_member'] = 0
            st.session_state['estimated_salary'] = 42000.0

    st.markdown("---")
    st.markdown("#### 🧠 Model Specs")
    st.markdown("""
    - **Architecture**: Artificial Neural Network (ANN)
    - **Framework**: TensorFlow / Keras
    - **Optimization**: Adam Optimizer
    - **Task**: Binary Classification
    - **Inference Time**: < 50ms
    """)
    
    st.markdown("---")
    st.markdown("#### 📖 Feature Guide")
    st.caption("• **Active Membership** is a high-impact retention driver.")
    st.caption("• **Product Breadth (2 products)** yields the lowest historical churn.")
    st.caption("• **Older demographics** show higher sensitivity to churn.")

# Hero Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI Predictive Analytics</div>
    <h1 class="hero-title">ChurnGuard AI Intelligence</h1>
    <p class="hero-subtitle">Real-time Artificial Neural Network inference for proactive customer retention. Predict churn probability instantly and unlock automated customer success recommendations.</p>
</div>
""", unsafe_allow_html=True)

# Main Grid: Inputs on Left (60%), Prediction on Right (40%)
left_col, right_col = st.columns([1.4, 1.0], gap="large")

with left_col:
    # Card 1: Demographics
    st.markdown("""
    <div class="section-card">
        <div class="section-title">👤 Customer Demographics</div>
    </div>
    """, unsafe_allow_html=True)
    
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        geography = st.selectbox(
            'Geography',
            options=onehot_encoder_geo.categories_[0],
            key='geography',
            index=0 if 'geography' not in st.session_state else list(onehot_encoder_geo.categories_[0]).index(st.session_state['geography'])
        )
    with d_col2:
        gender = st.selectbox(
            'Gender',
            options=label_encoder_gender.classes_,
            key='gender',
            index=0 if 'gender' not in st.session_state else list(label_encoder_gender.classes_).index(st.session_state['gender'])
        )
    with d_col3:
        age = st.slider(
            'Age (Years)',
            min_value=18,
            max_value=92,
            value=st.session_state.get('age', 38),
            key='age'
        )

    # Card 2: Financial Overview
    st.markdown("""
    <div class="section-card">
        <div class="section-title">💳 Financial Metrics</div>
    </div>
    """, unsafe_allow_html=True)
    
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        credit_score = st.number_input(
            'Credit Score',
            min_value=300.0,
            max_value=850.0,
            value=st.session_state.get('credit_score', 650.0),
            step=5.0,
            key='credit_score'
        )
    with f_col2:
        balance = st.number_input(
            'Account Balance ($)',
            min_value=0.0,
            value=st.session_state.get('balance', 60000.0),
            step=1000.0,
            key='balance'
        )
    with f_col3:
        estimated_salary = st.number_input(
            'Estimated Salary ($)',
            min_value=0.0,
            value=st.session_state.get('estimated_salary', 50000.0),
            step=1000.0,
            key='estimated_salary'
        )

    # Card 3: Account Engagement
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📊 Account & Engagement Status</div>
    </div>
    """, unsafe_allow_html=True)
    
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        tenure = st.slider(
            'Tenure with Bank (Years)',
            min_value=0,
            max_value=10,
            value=st.session_state.get('tenure', 5),
            key='tenure'
        )
        num_of_products = st.slider(
            'Number of Products',
            min_value=1,
            max_value=4,
            value=st.session_state.get('num_of_products', 2),
            key='num_of_products'
        )
    with e_col2:
        has_cr_card = st.selectbox(
            'Has Credit Card?',
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key='has_cr_card',
            index=0 if st.session_state.get('has_cr_card', 1) == 1 else 1
        )
        is_active_member = st.selectbox(
            'Is Active Member?',
            options=[1, 0],
            format_func=lambda x: "Yes (Active)" if x == 1 else "No (Inactive)",
            key='is_active_member',
            index=0 if st.session_state.get('is_active_member', 1) == 1 else 1
        )

# Data Preparation & Pipeline
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography safely with feature names
geo_df = pd.DataFrame({'Geography': [geography]})
geo_encoded = onehot_encoder_geo.transform(geo_df)
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine and Scale
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# Neural Network Prediction
prediction = model.predict(input_data_scaled, verbose=0)
prediction_proba = float(prediction[0][0])
percentage = prediction_proba * 100

with right_col:
    st.markdown("### 🎯 Real-Time Assessment")
    
    # Determine risk category
    if prediction_proba < 0.35:
        card_class = "result-card-safe"
        val_class = "val-safe"
        badge_class = "badge-safe"
        status_label = "✅ LOW CHURN RISK - LOYAL"
        explanation = "Customer displays healthy engagement habits and high retention probability."
        action = "• Maintain satisfaction with standard loyalty rewards.<br>• Eligible for premium cross-sell offers."
    elif prediction_proba < 0.55:
        card_class = "result-card-warning"
        val_class = "val-warning"
        badge_class = "badge-warning"
        status_label = "⚠️ MODERATE RISK - WATCHLIST"
        explanation = "Customer is on the borderline. Early signs of reduced engagement detected."
        action = "• Trigger targeted satisfaction survey.<br>• Offer complimentary benefits or fee waivers."
    else:
        card_class = "result-card-danger"
        val_class = "val-danger"
        badge_class = "badge-danger"
        status_label = "🚨 HIGH CHURN RISK - AT RISK"
        explanation = "High likelihood of customer churn. Immediate retention intervention recommended."
        action = "• Assign dedicated account manager.<br>• Provide exclusive retention incentives & rate review."

    # Render Styled Card
    st.markdown(f"""
    <div class="{card_class}">
        <div class="score-title">Predicted Churn Probability</div>
        <div class="score-value {val_class}">{percentage:.1f}%</div>
        <div class="status-badge {badge_class}">{status_label}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Progress Meter
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    st.caption(f"Risk Meter: {percentage:.1f}%")
    st.progress(min(max(prediction_proba, 0.0), 1.0))

    # Actionable Insights Card
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-header">💡 Strategic Intelligence</div>
        <div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 10px;">{explanation}</div>
        <div class="insight-header" style="margin-top: 12px;">🎯 Recommended Actions:</div>
        <div style="font-size: 0.84rem; color: #94A3B8; line-height: 1.6;">{action}</div>
    </div>
    """, unsafe_allow_html=True)

    # Key Contributing Indicators
    with st.expander("🔍 Contributing Indicators for this Profile", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Activity Score", "Active" if is_active_member else "Inactive", delta="Positive" if is_active_member else "-Negative")
            st.metric("Credit Tier", f"{int(credit_score)} pts", delta="Healthy" if credit_score > 600 else "-Low")
        with c2:
            st.metric("Products", f"{num_of_products} items", delta="Optimal" if num_of_products == 2 else "Sub-optimal")
            st.metric("Balance/Salary", f"{(balance/(estimated_salary+1e-5)):.2f}x")