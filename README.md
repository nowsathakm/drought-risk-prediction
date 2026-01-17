# Early Prediction of Agricultural Drought Risk in Sri Lanka's Dry Zone  
**Using CatBoost and SHAP Explainability**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B)
![SHAP](https://img.shields.io/badge/SHAP-0.4+-green)

**Author**: Nowsath A.K.M

**University**: University of Moratuwa

**Location Focus**: Dry Zone Districts (Puttalam, Hambantota, Trincomalee, Mannar, Jaffna, Kurunegala, Kalmunai)

## Overview

This project develops a machine learning model to **predict agricultural drought risk** 1–3 months ahead in Sri Lanka's dry zone — a critical issue for paddy farming, food security, and rural livelihoods. The model uses **CatBoost** (a gradient boosting algorithm not covered in lectures) trained on a filtered subset of the publicly available [Sri Lanka Weather Dataset](https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset) (2010–2023).

Due to instability in per-city SPEI computation, a domain-appropriate proxy target was adopted: **3-month cumulative rainfall < 150 mm** indicates drought risk. The model achieves strong test performance:  
- **F1-score**: 0.581  
- **AUC-ROC**: 0.932  
- **Accuracy**: 0.897  

**SHAP explainability** reveals lagged precipitation and evapotranspiration as dominant drivers — aligning with known dry-zone paddy vulnerability.  

For usability, the model is integrated into a **Streamlit web app** allowing users to input weather values and view predictions with interactive SHAP explanations.


## Features

- **Local real-world problem**: Agricultural drought prediction in Sri Lanka's dry zone
- **Novel algorithm**: CatBoost
- **Explainability**: SHAP global/local feature importance
- **Front-end integration**: Streamlit app with user inputs, predictions, and SHAP visualizations
- **Time-series awareness**: Chronological train/validation/test split
- **Ethical & reproducible**: Public dataset, transparent proxy target decision, full code & demo

## Project Structure
drought-risk-prediction/
├── app.py                      # Streamlit front-end application
├── drought_model.joblib        # Trained CatBoost model
├── scaler.joblib               # Fitted StandardScaler
├── requirements.txt            # Python dependencies
├── images/                     # Plots for report (ROC, confusion matrix, time-series, SHAP)
└── README.md              


## Installation & Setup

### Prerequisites
- Python 3.8+
- Git (to clone the repo)

### Clone the Repository
```bash
git clone https://github.com/nowsathakm/drought-risk-prediction.git
cd drought-risk-prediction
