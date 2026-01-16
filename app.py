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

st.title("🌾 Dry Zone Agricultural Drought Risk Forecaster")
st.markdown("""
This app predicts drought risk for the next 1–3 months in Sri Lanka's dry zone (e.g., Puttalam, Hambantota, Trincomalee) based on recent weather data.  
Enter values below to simulate a scenario. The model uses a trained CatBoost classifier with features like lagged rainfall and evapotranspiration.
""")

# ─── Sidebar for user inputs ─────────────────────────────────────────────────
st.sidebar.header("Input Weather Parameters")
st.sidebar.markdown("Adjust sliders/dropdowns for a monthly snapshot.")

# Numerical inputs (sliders with realistic dry-zone ranges)
temperature_2m_mean = st.sidebar.slider("Average Temperature (°C)", 22.0, 38.0, 30.0, step=0.5)
temperature_2m_max = st.sidebar.slider("Max Temperature (°C)", 25.0, 42.0, 35.0, step=0.5)
precipitation_sum = st.sidebar.slider("Monthly Rainfall (mm)", 0.0, 600.0, 150.0, step=10.0)
et0_fao_evapotranspiration = st.sidebar.slider("Evapotranspiration ET0 (mm/month)", 60.0, 220.0, 130.0, step=5.0)
shortwave_radiation_sum = st.sidebar.slider("Shortwave Radiation Sum (MJ/m²)", 400.0, 900.0, 650.0, step=10.0)
windspeed_10m_max = st.sidebar.slider("Max Wind Speed (km/h)", 5.0, 40.0, 20.0, step=1.0)

# Cyclic month features (select month → compute sin/cos)
month = st.sidebar.slider("Month of Year", 1, 12, 1, step=1)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
st.sidebar.markdown(f"Computed: Month Sin = {month_sin:.2f}, Month Cos = {month_cos:.2f}")

# Categorical season
season = st.sidebar.selectbox("Season", ["Maha", "Yala"])

# Lagged features (sliders for 1/3/6 months)
st.sidebar.subheader("Lagged Values")
precip_lag_1 = st.sidebar.slider("1-Month Lagged Rainfall (mm)", 0.0, 600.0, 150.0, step=10.0)
precip_lag_3 = st.sidebar.slider("3-Month Lagged Rainfall (mm)", 0.0, 1800.0, 450.0, step=50.0)
precip_lag_6 = st.sidebar.slider("6-Month Lagged Rainfall (mm)", 0.0, 3600.0, 900.0, step=100.0)

et0_lag_1 = st.sidebar.slider("1-Month Lagged ET0 (mm)", 60.0, 220.0, 130.0, step=5.0)
et0_lag_3 = st.sidebar.slider("3-Month Lagged ET0 (mm)", 180.0, 660.0, 390.0, step=15.0)
et0_lag_6 = st.sidebar.slider("6-Month Lagged ET0 (mm)", 360.0, 1320.0, 780.0, step=30.0)

temp_lag_1 = st.sidebar.slider("1-Month Lagged Avg Temp (°C)", 22.0, 38.0, 30.0, step=0.5)
temp_lag_3 = st.sidebar.slider("3-Month Lagged Avg Temp (°C)", 22.0, 38.0, 30.0, step=0.5)
temp_lag_6 = st.sidebar.slider("6-Month Lagged Avg Temp (°C)", 22.0, 38.0, 30.0, step=0.5)

# ─── Main content: Prediction button ─────────────────────────────────────────
if st.button("Predict Drought Risk", type="primary", use_container_width=True):
    # Create input DataFrame matching exact training features and order
    input_data = pd.DataFrame({
        'temperature_2m_mean': [temperature_2m_mean],
        'temperature_2m_max': [temperature_2m_max],
        'precipitation_sum': [precipitation_sum],
        'et0_fao_evapotranspiration': [et0_fao_evapotranspiration],
        'shortwave_radiation_sum': [shortwave_radiation_sum],
        'windspeed_10m_max': [windspeed_10m_max],
        'month_sin': [month_sin],
        'month_cos': [month_cos],
        'season': [season],
        'precip_lag_1': [precip_lag_1],
        'precip_lag_3': [precip_lag_3],
        'precip_lag_6': [precip_lag_6],
        'et0_lag_1': [et0_lag_1],
        'et0_lag_3': [et0_lag_3],
        'et0_lag_6': [et0_lag_6],
        'temp_lag_1': [temp_lag_1],
        'temp_lag_3': [temp_lag_3],
        'temp_lag_6': [temp_lag_6]
    })

    # Scale numerical features (match training)
    numerical_features = [col for col in input_data.columns if col != 'season']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Predict
    prob = model.predict_proba(input_data)[0][1]  # Probability of drought (class 1)
    risk_level = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"

    st.header(f"Predicted Drought Risk: **{risk_level}** ({prob:.1%} probability)")
    
    if risk_level == "High":
        st.error("**High Risk Detected** – Recommend immediate actions: prioritize irrigation, use drought-tolerant seeds, monitor tank levels closely.")
    elif risk_level == "Moderate":
        st.warning("**Moderate Risk** – Stay vigilant: check forecasts weekly and prepare contingency plans for potential dry spells.")
    else:
        st.success("**Low Risk** – Normal conditions expected: continue standard farming practices.")

    # ─── SHAP Explanation ────────────────────────────────────────────────────
    st.subheader("Model Explanation (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Force plot for this prediction
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_data))

    # Summary bar plot for feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(fig)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Note**: This is a prototype for educational purposes. Predictions based on trained CatBoost model using Sri Lanka Weather Dataset. For real use, consult local agricultural experts.")
st.markdown("Developed by A.K.M. Nowsath for AML Assignment – January 2026")