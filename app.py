import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# ==========================================
# 1. APPLICATION CONFIGURATION & MODEL LOAD
# ==========================================
st.set_page_config(page_title="House Price Predictor", layout="centered", page_icon="🏠")

@st.cache_resource
def load_model():
    try:
        with open('house_price_xgb_model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception:
        return joblib.load('house_price_xgb_model.pkl')

try:
    model_pipeline = load_model()
    st.success("🤖 Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: Ensure 'house_price_xgb_model.pkl' is correct.")
    st.stop()

# ==========================================
# 2. WEB APPLICATION USER INTERFACE (UI)
# ==========================================
st.title("🏠 House Price Prediction App")
st.markdown("Enter the property details below to estimate the market valuation using our trained machine learning model.")

st.subheader("📋 Property Features Input")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea) in sq ft:", min_value=300, max_value=6000, value=1500)
    overall_qual = st.slider("Overall Quality Rating (OverallQual) 1-10:", min_value=1, max_value=10, value=6)
    garage_cars = st.selectbox("Garage Car Capacity (GarageCars):", options=[0, 1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Total Basement Area (TotalBsmtSF) in sq ft:", min_value=0, max_value=4000, value=1000)

with col2:
    year_built = st.number_input("Year Built:", min_value=1800, max_value=2026, value=2000)
    lot_area = st.number_input("Lot Area in sq ft:", min_value=500, max_value=100000, value=10000)
    neighborhood = st.selectbox("Neighborhood:", options=['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NridgHt', 'OldTown', 'Edwards', 'Gilbert'])
    ms_zoning = st.selectbox("MS Zoning:", options=['RL', 'RM', 'C (all)', 'FV', 'RH'])

# ==========================================
# 3. ROBUST PREDICTION LOGIC WITH AUTO-COLUMNS
# ==========================================
if st.button("💰 Estimate House Price", type="primary"):
    # Pehle model pipeline ke original input columns dhoondein
    try:
        # Preprocessor ke underlying feature names list check karna
        original_features = model_pipeline.named_steps['preprocessor'].feature_names_in_
    except AttributeError:
        # Agar explicit features na milein toh default Kaggle train features ka fallback dictionary structure lagana
        original_features = [
            'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Neighborhood', 
            'OverallQual', 'YearBuilt', 'MasVnrType', 'TotalBsmtSF', 'GrLivArea', 
            'FullBath', 'GarageCars', 'GarageYrBlt', 'PoolQC'
        ]

    # Ek blank dictionary banayein jise default values se populate karenge
    full_input_dict = {}
    for col in original_features:
        full_input_dict[col] = [np.nan]  # Default setting missing parameters to NaN

    # User ke inputs ko dictionary mein update karein
    full_input_dict['GrLivArea'] = [gr_liv_area]
    full_input_dict['OverallQual'] = [overall_qual]
    full_input_dict['GarageCars'] = [garage_cars]
    full_input_dict['TotalBsmtSF'] = [total_bsmt_sf]
    full_input_dict['YearBuilt'] = [year_built]
    full_input_dict['LotArea'] = [lot_area]
    full_input_dict['Neighborhood'] = [neighborhood]
    full_input_dict['MSZoning'] = [ms_zoning]

    # DataFrame ko exactly usi order mein convert karna jo pipeline chahti hai
    input_df = pd.DataFrame(full_input_dict)[list(original_features)]
    
    try:
        # Prediction process run karein
        predicted_log_price = model_pipeline.predict(input_df)
        final_price = np.expm1(predicted_log_price)[0]
        
        st.markdown("---")
        st.balloons()  # Prediction success par celebration effect
        st.metric(label="📊 Estimated Market Value", value=f"${final_price:,.2f}")
        st.info("💡 Note: This estimation is calculated dynamically by the pre-trained XGBoost Model Pipeline.")
        
    except Exception as prediction_error:
        st.error(f"⚠️ Prediction Error: {str(prediction_error)}")
        st.warning("Please make sure your model pipeline was saved completely on Kaggle.")
