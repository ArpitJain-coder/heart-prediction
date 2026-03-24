# ❤️ Heart Disease Prediction App

A machine learning web application built with Streamlit to predict the likelihood of heart disease based on patient health metrics. The app uses a K-Nearest Neighbors (KNN) model trained on medical data.

## Features
- **User-Friendly Interface**: Built with [Streamlit](https://streamlit.io/) for easy data input.
- **Instant Predictions**: Get immediate feedback on heart disease risk.
- **Pre-trained ML Model**: Uses a K-Nearest Neighbors model, standard scaler, and specialized column transformations.

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Create a virtual environment and activate it (optional but recommended):
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Deployment
This app can be easily deployed to Streamlit Community Cloud. 
Make sure your Github repository contains:
- `streamlit_app.py`
- `requirements.txt`
- `knn_heart_model.pkl`
- `scaler.pkl`
- `columns.pkl`
