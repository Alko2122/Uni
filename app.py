import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import requests
import io
import tempfile
import os
import gdown

# File URLs
MODEL_URL = "https://drive.google.com/uc?id=1NjXo2WpoSKCQqzQg_juYO83xf1Lhr-ks"
SCALER_URL = "https://raw.githubusercontent.com/Alko2122/Uni/756569d0500e6c5d5d6e6e1b5b949b423e3349d2/your_scaler.joblib"
CSV_URL = "https://raw.githubusercontent.com/Alko2122/Uni/756569d0500e6c5d5d6e6e1b5b949b423e3349d2/Airline%20Dataset%20-%20Cleaned%20(CSV)%20(Readjusted).csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    
    # Identify numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    
    # Handle infinite values in numeric columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
    # Impute numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    # Impute non-numeric columns
    non_numeric_imputer = SimpleImputer(strategy='most_frequent')
    df[non_numeric_columns] = non_numeric_imputer.fit_transform(df[non_numeric_columns])
    
    return df

@st.cache_resource
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        st.error(f"Failed to download file from {url}. Status code: {response.status_code}")
        return None

@st.cache_resource
def download_model(url):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            gdown.download(url, tmp_file.name, quiet=False)
            st.write(f"Model file size: {os.path.getsize(tmp_file.name)} bytes")
            model = joblib.load(tmp_file.name)
        return model
    except Exception as e:
        st.error(f"Error downloading or loading model: {str(e)}")
        return None

@st.cache_resource
def load_model_and_scaler():
    model = download_model(MODEL_URL)
    scaler = download_file(SCALER_URL)
    return model, scaler

# Load data, model, and scaler
df = load_data(CSV_URL)
model, scaler = load_model_and_scaler()

# Calculate model accuracy
def calculate_model_accuracy(model, scaler, df):
    X = df[['passengers', 'large_ms', 'nsmiles']]  # Adjust these features as per your model
    y = df['fare']
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return mse, r2

# Calculate accuracy metrics
mse, r2 = calculate_model_accuracy(model, scaler, df)

st.title("SkyFare Consultants")

if model is None:
    st.warning("The model file could not be loaded automatically. Please upload it manually.")
    uploaded_file = st.file_uploader("Choose the model file", type=["joblib", "pkl"])
    if uploaded_file is not None:
        model = joblib.load(uploaded_file)
        st.success("Model loaded successfully!")

if scaler is None:
    st.error("Failed to load the scaler. Please check the scaler URL and try again.")

# Display model accuracy metrics
st.markdown("## Model Accuracy Metrics")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"R-squared (R2) Score: {r2:.4f}")

# Rest of your Streamlit app code...
# (Include your input form, prediction logic, and result display here)

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
