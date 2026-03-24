import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import pickle

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

import joblib  # type: ignore

# Load models and columns
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    model = joblib.load('knn_heart_model.pkl')
    return scaler, columns, model

try:
    scaler, columns, model = load_assets()
except Exception as e:
    st.error(f"Error loading models or setup files: {e}")
    st.stop()

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's information below to predict the likelihood of heart disease.")

# User inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=0, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
    oldpeak = st.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=0.0)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain_type = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict Heart Disease Risk", type="primary"):
    # Initialize all columns with 0
    input_dict = {}
    for col in columns:
        input_dict[str(col)] = 0
    
    # Fill numerical columns
    input_dict['Age'] = age
    input_dict['RestingBP'] = resting_bp
    input_dict['Cholesterol'] = cholesterol
    input_dict['FastingBS'] = fasting_bs
    input_dict['MaxHR'] = max_hr
    input_dict['Oldpeak'] = oldpeak

    # Fill categorical columns (One-hot encoding)
    if f'Sex_{sex}' in columns:
        input_dict[f'Sex_{sex}'] = 1
        
    if f'ChestPainType_{chest_pain_type}' in columns:
        input_dict[f'ChestPainType_{chest_pain_type}'] = 1
        
    if f'RestingECG_{resting_ecg}' in columns:
        input_dict[f'RestingECG_{resting_ecg}'] = 1
        
    if f'ExerciseAngina_{exercise_angina}' in columns:
        input_dict[f'ExerciseAngina_{exercise_angina}'] = 1
        
    if f'ST_Slope_{st_slope}' in columns:
        input_dict[f'ST_Slope_{st_slope}'] = 1

    input_df = pd.DataFrame([input_dict])

    try:
        # Scale numerical data first, then reconstruct dataframe? 
        # Or scale entire dataframe depending on how scaler was fitted
        # First let's try scaling the whole dataframe
        try:
            scaled_data = scaler.transform(input_df)
        except ValueError:
            # If standard scaler was ONLY fitted on numeric columns
            numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            scaled_data = input_df
            
        prediction = model.predict(scaled_data)
        
        st.subheader("Results:")
        if prediction[0] == 1:
            st.error("⚠️ Prediction: High Risk of Heart Disease")
            st.write("We recommend consulting with a healthcare professional.")
        else:
            st.success("✅ Prediction: Low Risk of Heart Disease")
            st.write("Keep up the good work maintaining a healthy lifestyle.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
