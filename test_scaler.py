import pickle
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Features in:", getattr(scaler, 'n_features_in_', 'Unknown'))
    print("Feature names in:", getattr(scaler, 'feature_names_in_', 'Unknown'))
except Exception as e:
    print("Error:", e)
