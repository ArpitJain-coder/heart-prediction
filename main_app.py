import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Suppress scikit-learn version warnings for pickled models
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

# Set page config for a premium look
st.set_page_config(
    page_title="CardioGuard | Heart Prediction AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .input-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
    }
    
    .result-container {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-top: 2rem;
    }
    
    .risk-high {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    
    .risk-low {
        background-color: #dcfce7;
        border: 2px solid #22c55e;
        color: #166534;
    }
    
    .feature-label {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models safely
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        model = joblib.load('knn_heart_model.pkl')
        return scaler, columns, model
    except Exception as e:
        st.error(f"Failed to load AI models: {e}")
        return None, None, None

scaler, columns, model = load_assets()

# --- HEADER SECTION ---
st.markdown("<h1 class='main-header'>❤️ CardioGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Advanced heart disease prediction using machine learning.</p>", unsafe_allow_html=True)

# Add banner image
if os.path.exists("banner.png"):
    st.image("banner.png", use_container_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
    st.title("About the App")
    st.info("""
    This application uses a K-Nearest Neighbors (KNN) model to assess the risk of cardiovascular disease based on clinical parameters. 
    
    **Note:** This is for educational purposes and should not replace professional medical advice.
    """)
    st.divider()
    st.caption("Developed with ❤️ using Streamlit")

# --- INPUT SECTION ---
if model and columns and scaler:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.subheader("Patient Clinical Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Age of the patient in years.")
        sex = st.selectbox("Sex", ["M", "F"], help="Biological sex of the patient.")
        chest_pain_type = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"], help="ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina")

    with col2:
        resting_bp = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=300, value=120, help="Resting blood pressure.")
        cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=0, max_value=600, value=200, help="Serum cholesterol level.")
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["Low (< 120 mg/dl)", "High (> 120 mg/dl)"], help="Blood sugar level after fasting.")

    with col3:
        max_hr = st.number_input("Max Heart Rate", min_value=0, max_value=250, value=150, help="Maximum heart rate achieved during stress test.")
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"], help="Exercise-induced angina.")
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="The slope of the peak exercise ST segment.")

    # Additional parameters in a second row if needed or below
    c1, c2 = st.columns(2)
    with c1:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="Resting electrocardiogram results.")
    with c2:
        oldpeak = st.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=0.0, format="%.1f", help="ST depression induced by exercise relative to rest.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- PREDICTION LOGIC ---
    if st.button("Analyze Heart Risk Profile", type="primary"):
        # Map inputs back to numerical values for model
        bs_val = 1 if "High" in fasting_bs else 0
        
        # Initialize input dictionary based on columns.pkl
        input_dict = {str(col): 0 for col in columns}
        
        # Numerical fields
        input_dict['Age'] = age
        input_dict['RestingBP'] = resting_bp
        input_dict['Cholesterol'] = cholesterol
        input_dict['FastingBS'] = bs_val
        input_dict['MaxHR'] = max_hr
        input_dict['Oldpeak'] = oldpeak

        # Categorical OHE fields
        if f'Sex_{sex}' in columns: input_dict[f'Sex_{sex}'] = 1
        if f'ChestPainType_{chest_pain_type}' in columns: input_dict[f'ChestPainType_{chest_pain_type}'] = 1
        if f'RestingECG_{resting_ecg}' in columns: input_dict[f'RestingECG_{resting_ecg}'] = 1
        if f'ExerciseAngina_{exercise_angina}' in columns: input_dict[f'ExerciseAngina_{exercise_angina}'] = 1
        if f'ST_Slope_{st_slope}' in columns: input_dict[f'ST_Slope_{st_slope}'] = 1

        input_df = pd.DataFrame([input_dict])

        try:
            # Reorder columns to match model training exactly
            input_df = input_df[columns]
            
            # Predict
            try:
                # Some scalers only scale numeric, others scale OHE too
                # We'll assume the scaler matches the columns in columns.pkl if it's a standard pipeline
                scaled_data = scaler.transform(input_df)
            except:
                # Fallback: only scale numeric columns if OHE was done later
                numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                scaled_data = input_df
            
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)[0][1] if hasattr(model, 'predict_proba') else None

            # --- DISPLAY RESULTS ---
            st.divider()
            
            if prediction[0] == 1:
                st.markdown(f"""
                <div class='result-container risk-high'>
                    <h2>⚠️ HIGH RISK DETECTED</h2>
                    <p style='font-size: 1.1rem;'>The AI analysis indicates a high likelihood of heart disease.</p>
                    {f"<p>Confidence Score: {probability*100:.1f}%</p>" if probability else ""}
                    <p><b>Recommendation:</b> Please consult with a cardiologist for a thorough clinical examination.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-container risk-low'>
                    <h2>✅ LOW RISK DETECTED</h2>
                    <p style='font-size: 1.1rem;'>The AI analysis indicates a low likelihood of heart disease.</p>
                    {f"<p>Confidence Score: {(1-probability)*100:.1f}%</p>" if probability else ""}
                    <p><b>Recommendation:</b> Maintain your healthy lifestyle and continue regular check-ups.</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)
else:
    st.warning("Please ensure that model files (knn_heart_model.pkl, scaler.pkl, columns.pkl) are present in the directory.")
