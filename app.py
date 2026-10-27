import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. APPLICATION CONFIGURATION & MODEL LOAD
# ==========================================
st.set_page_config(page_title="House Price Predictor", layout="centered", page_icon="🏠")

# Model file load karein (.pkl file check karein aapke local folder mein ho)
@st.cache_resource
def load_model():
    # Kyunki pipeline .pkl format mein hai, joblib ya pickle dono se load ho sakti hai
    return joblib.load('house_price_xgb_model.pkl')

try:
    model_pipeline = load_model()
    st.success("🤖 Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: Ensure 'house_price_xgb_model.pkl' is in the same directory.")
    st.stop()

# ==========================================
# 2. WEB APPLICATION USER INTERFACE (UI)
# ==========================================
st.title("🏠 House Price Prediction App")
st.markdown("Enter the property details below to estimate the market valuation using our trained machine learning model.")

st.subheader("📋 Property Features Input")

# 2-column layout inputs ke liye
col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea) in sq ft:", min_value=300, max_value=6000, value=1500)
    overall_qual = st.slider("Overall Quality Rating (OverallQual) 1-10:", min_value=1, max_value=10, value=6)
    garage_cars = st.selectbox("Garage Car Capacity (GarageCars):", options=[0, 1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Total Basement Area (TotalBsmtSF) in sq ft:", min_value=0, max_value=4000, value=1000)

with col2:
    year_built = st.number_input("Year Built:", min_value=1800, max_value=2026, value=2000)
    lot_area = st.number_input("Lot Area in sq ft:", min_value=500, max_value=100000, value=10000)
# Default/Fallback columns handle karne ke liye jo Kaggle dataset mein hote hain par UI mein nahi hain
# Hamari input pipeline ko poore columns ki zaroorat hoti hai
all_columns = model_pipeline.named_steps['preprocessor'].transformers[0][2] + model_pipeline.named_steps['preprocessor'].transformers[1][2]

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if st.button("💰 Estimate House Price", type="primary"):
    # User inputs ko aik base dictionary mein dalna
    input_data = {
        'GrLivArea': [gr_liv_area],
        'OverallQual': [overall_qual],
        'GarageCars': [garage_cars],
        'TotalBsmtSF': [total_bsmt_sf],
        'YearBuilt': [year_built],
        'LotArea': [lot_area],
        'Neighborhood': [neighborhood],
        'MSZoning': [ms_zoning]
    }
    
    # Missing columns ko dummy variables (NaN/None) se fill karna taaki pipeline crash na ho
    for col in all_columns:
        if col not in input_data:
            input_data[col] = [np.nan]
            
    # Dataframe structure design karein
    input_df = pd.DataFrame(input_data)[all_columns]
    
    # Model se prediction nikalna
    predicted_log_price = model_pipeline.predict(input_df)
    
    # Kyunki model log scale par train hua tha, isko back-transform (exp) karna zaroori hai
    final_price = np.expm1(predicted_log_price)[0]
    
    # Valuations interface par show karna
    st.markdown("---")
    st.metric(label="Estimated Market Value", value=f"${final_price:,.2f}")
    st.info("💡 Note: This estimation is calculated dynamically by the pre-trained XGBoost Model Pipeline.")
