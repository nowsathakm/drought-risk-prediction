import streamlit as st

# ─── App configuration (MUST be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Dry Zone Drought Risk Forecaster",
    layout="centered",
    initial_sidebar_state="expanded"
)

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# ─── Load model and scaler ───────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load('drought_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_assets()

# ─── App title & intro ───────────────────────────────────────────────────────
st.title("🌾 Dry Zone Drought Risk Predictor")
st.markdown("""
Select a dry-zone city and enter recent weather values to forecast drought risk for the next 1–3 months.  
The model uses CatBoost trained on Sri Lanka weather data (2010–2023) with SHAP explanations.
""")

# ─── City & Season / Month ───────────────────────────────────────────────────
col_city, col_season = st.columns(2)
with col_city:
    city = st.selectbox("Dry Zone City", 
                        ["Puttalam", "Hambantota", "Trincomalee", "Mannar", "Jaffna", "Kurunegala"],
                        index=0)

with col_season:
    season = st.selectbox("Current Season", ["Maha", "Yala"])

month = st.slider("Current Month (1=Jan, 12=Dec)", 1, 12, 6)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# ─── Core inputs (only 5 visible) ────────────────────────────────────────────
st.subheader("Recent Weather (Last Month)")
col1, col2, col3 = st.columns(3)
with col1:
    precip_current = st.slider("Rainfall (mm)", 0, 500, 100, step=10)
with col2:
    temp_mean = st.slider("Avg Temperature (°C)", 22.0, 38.0, 30.0, step=0.5)
with col3:
    et0_current = st.slider("Evapotranspiration ET0 (mm)", 60, 220, 130, step=5)

# Most important lag (from SHAP)
precip_lag_3 = st.slider("3-Month Lagged Rainfall (mm)", 0, 1800, 400, step=50)

# ─── Hidden defaults (realistic medians from dry-zone data) ──────────────────
defaults = {
    'temperature_2m_max': 35.0,
    'shortwave_radiation_sum': 650.0,
    'windspeed_10m_max': 20.0,
    'precip_lag_1': 120.0,
    'precip_lag_6': 800.0,
    'et0_lag_1': 130.0,
    'et0_lag_3': 390.0,
    'et0_lag_6': 780.0,
    'temp_lag_1': 30.0,
    'temp_lag_3': 30.0,
    'temp_lag_6': 30.0
}

# ─── Prediction ──────────────────────────────────────────────────────────────
if st.button("Predict Drought Risk", type="primary"):
    input_dict = {
        'temperature_2m_mean': [temp_mean],
        'temperature_2m_max': [defaults['temperature_2m_max']],
        'precipitation_sum': [precip_current],
        'et0_fao_evapotranspiration': [et0_current],
        'shortwave_radiation_sum': [defaults['shortwave_radiation_sum']],
        'windspeed_10m_max': [defaults['windspeed_10m_max']],
        'month_sin': [month_sin],
        'month_cos': [month_cos],
        'season': [season],
        'precip_lag_1': [defaults['precip_lag_1']],
        'precip_lag_3': [precip_lag_3],
        'precip_lag_6': [defaults['precip_lag_6']],
        'et0_lag_1': [defaults['et0_lag_1']],
        'et0_lag_3': [defaults['et0_lag_3']],
        'et0_lag_6': [defaults['et0_lag_6']],
        'temp_lag_1': [defaults['temp_lag_1']],
        'temp_lag_3': [defaults['temp_lag_3']],
        'temp_lag_6': [defaults['temp_lag_6']]
    }

    input_df = pd.DataFrame(input_dict)
    num_cols = [col for col in input_df.columns if col != 'season']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prob = model.predict_proba(input_df)[0][1]
    risk = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"

    st.subheader(f"**{risk} Risk** ({prob:.1%} probability)")

    if risk == "High":
        st.error("**High Risk** – Prioritize irrigation, consider drought-tolerant varieties.")
    elif risk == "Moderate":
        st.warning("**Moderate Risk** – Monitor closely and prepare contingency plans.")
    else:
        st.success("**Low Risk** – Normal conditions expected.")

    st.subheader("Why this prediction? (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_df))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Note**: This is a prototype for educational purposes. Predictions based on trained CatBoost model using Sri Lanka Weather Dataset. For real use, consult local agricultural experts.")
st.markdown("Developed by A.K.M. Nowsath for AML Assignment – January 2026")