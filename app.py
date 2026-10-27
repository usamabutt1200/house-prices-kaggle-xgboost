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
    # Pehle pickle se load karne ki koshish karein (kyunki aapne .pkl banayi thi)
    try:
        with open('house_price_xgb_model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception:
        # Agar pickle fail ho toh joblib se backup load karein
        return joblib.load('house_price_xgb_model.pkl')

try:
    model_pipeline = load_model()
    st.success("🤖 Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: Ensure 'house_price_xgb_model.pkl' is correct. Technical Error: {str(e)}")
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
# 3. PREDICTION LOGIC
# ==========================================
if st.button("💰 Estimate House Price", type="primary"):
    # User inputs dictionary
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
    
    # DataFrame design
    input_df = pd.DataFrame(input_data)
    
    try:
        # Preprocessor se target columns extract karna dynamically
        feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        all_base_columns = [col.split('__')[-1] for col in feature_names]
        
        # Missing features ko handle karna taaki pipeline crash na ho
        for col in all_base_columns:
            if col not in input_df.columns:
                input_df[col] = np.nan
                
        # Sequence sahi karna jaisa model ko chahiye
        final_cols = [col for col in input_df.columns if col in all_base_columns]
        input_df = input_df[final_cols]
        
        # Prediction
        predicted_log_price = model_pipeline.predict(input_df)
        final_price = np.expm1(predicted_log_price)[0]
        
        st.markdown("---")
        st.metric(label="Estimated Market Value", value=f"${final_price:,.2f}")
        st.info("💡 Note: This estimation is calculated dynamically by the pre-trained XGBoost Model Pipeline.")
        
    except Exception as prediction_error:
        st.error(f"⚠️ App structure mismatch: {str(prediction_error)}")
        st.warning("Running automatic robust fallback prediction...")
        # Fallback raw features method
        try:
            predicted_log_price = model_pipeline.predict(pd.DataFrame(input_data))
            final_price = np.expm1(predicted_log_price)[0]
            st.metric(label="Estimated Market Value (Fallback Mode)", value=f"${final_price:,.2f}")
        except Exception as fallback_error:
            st.error("Could not run fallback prediction. Please re-train the model to align pipeline components.")
