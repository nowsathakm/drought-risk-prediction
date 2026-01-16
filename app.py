import streamlit as st

# ─── App configuration (MUST be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Dry Zone Drought Risk Forecaster",
    layout="wide",
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

# ─── City & Season selection ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "Select Dry Zone City",
        ["Puttalam", "Hambantota", "Trincomalee", "Mannar", "Jaffna", "Kurunegala"],
        index=0  # default to Puttalam
    )

with col2:
    season = st.selectbox("Current Season", ["Maha", "Yala"])

# Month selection (affects cyclic features)
month = st.slider("Current Month (1=Jan, 12=Dec)", 1, 12, 1)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# ─── Core weather inputs (reduced to 4 key ones) ─────────────────────────────
st.subheader("Recent Monthly Weather")
col3, col4, col5 = st.columns(3)

with col3:
    precipitation_sum = st.slider("Current Month Rainfall (mm)", 0, 600, 150, step=10)

with col4:
    temperature_2m_mean = st.slider("Avg Temperature (°C)", 22.0, 38.0, 30.0, step=0.5)

with col5:
    et0_fao_evapotranspiration = st.slider("Evapotranspiration ET0 (mm)", 60, 220, 130, step=5)

# Most important lag (from SHAP)
precip_lag_3 = st.slider("3-Month Lagged Rainfall (mm)", 0, 1800, 450, step=50)

# ─── Hidden/default values for other lags (simplifies UI) ────────────────────
# Use realistic median defaults from your training data (adjust if you know exact medians)
default_lag_values = {
    'precip_lag_1': 150.0,
    'precip_lag_6': 900.0,
    'et0_lag_1': 130.0,
    'et0_lag_3': 390.0,
    'et0_lag_6': 780.0,
    'temp_lag_1': 30.0,
    'temp_lag_3': 30.0,
    'temp_lag_6': 30.0,
    'shortwave_radiation_sum': 650.0,
    'windspeed_10m_max': 20.0,
    'temperature_2m_max': 35.0
}

# ─── Predict button ──────────────────────────────────────────────────────────
if st.button("Predict Drought Risk", type="primary"):
    # Build input DataFrame with exact training column order
    input_dict = {
        'temperature_2m_mean': [temperature_2m_mean],
        'temperature_2m_max': [default_lag_values['temperature_2m_max']],
        'precipitation_sum': [precipitation_sum],
        'et0_fao_evapotranspiration': [et0_fao_evapotranspiration],
        'shortwave_radiation_sum': [default_lag_values['shortwave_radiation_sum']],
        'windspeed_10m_max': [default_lag_values['windspeed_10m_max']],
        'month_sin': [month_sin],
        'month_cos': [month_cos],
        'season': [season],
        'precip_lag_1': [default_lag_values['precip_lag_1']],
        'precip_lag_3': [precip_lag_3],
        'precip_lag_6': [default_lag_values['precip_lag_6']],
        'et0_lag_1': [default_lag_values['et0_lag_1']],
        'et0_lag_3': [default_lag_values['et0_lag_3']],
        'et0_lag_6': [default_lag_values['et0_lag_6']],
        'temp_lag_1': [default_lag_values['temp_lag_1']],
        'temp_lag_3': [default_lag_values['temp_lag_3']],
        'temp_lag_6': [default_lag_values['temp_lag_6']]
    }

    input_df = pd.DataFrame(input_dict)

    # Scale numerical columns
    numerical_cols = [col for col in input_df.columns if col != 'season']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    risk_level = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"

    st.subheader(f"Predicted Risk: **{risk_level}** ({prob:.1%} probability)")

    if risk_level == "High":
        st.error("**High Risk** – Prioritize irrigation, consider drought-tolerant varieties, monitor tanks closely.")
    elif risk_level == "Moderate":
        st.warning("**Moderate Risk** – Monitor closely and prepare contingency plans.")
    else:
        st.success("**Low Risk** – Normal conditions expected.")

    # SHAP explanation
    st.subheader("Why this prediction? (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Force plot
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_df))

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Note**: This is a prototype for educational purposes. Predictions based on trained CatBoost model using Sri Lanka Weather Dataset. For real use, consult local agricultural experts.")
st.markdown("Developed by A.K.M. Nowsath for AML Assignment – January 2026")