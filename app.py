import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    return pd.read_csv(url)

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

st.title("SkyFare Consultants")

if model is None:
    st.warning("The model file could not be loaded automatically. Please upload it manually.")
    uploaded_file = st.file_uploader("Choose the model file", type=["joblib", "pkl"])
    if uploaded_file is not None:
        model = joblib.load(uploaded_file)
        st.success("Model loaded successfully!")

if scaler is None:
    st.error("Failed to load the scaler. Please check the scaler URL and try again.")
    
# Custom CSS for a cleaner look with white background
st.markdown("""
<style>
    .main {
        background-color: white;
        color: #333333;
    }
    .stButton>button {
        background-color: #0284c7;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0369a1;
    }
    .big-font {
        font-size: 36px !important;
        font-weight: bold;
        color: #0c4a6e;
    }
    .medium-font {
        font-size: 20px !important;
        color: #0c4a6e;
    }
    .small-font {
        font-size: 14px !important;
        color: #64748b;
    }
    .predictor {
        padding: 20px;
    }
    .stVideo {
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }
    .metrics {
        background-color: #f0f9ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .metric-value {
        font-size: 18px;
        font-weight: bold;
        color: #0284c7;
    }
</style>
""", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([6, 4])

with col1:
    st.markdown("<div class='predictor'>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Discover the fare charged for a specific number of passengers</p>", unsafe_allow_html=True)
    st.markdown("<p class='small-font'>SkyFare Predictor uses advanced machine learning to estimate flight costs. Enter your travel details below and let our AI do the rest.</p>", unsafe_allow_html=True)

    # Input form
    departure_airport = st.selectbox("Departure Airport", sorted(df['airport_1'].unique()))
    arrival_airport = st.selectbox("Arrival Airport", sorted(df['airport_1'].unique()))
    passenger_count = st.slider("Number of Passengers", 1, 100, 1)

    if st.button("Calculate Fare"):
        geolocator = Nominatim(user_agent="SkyFarePredictor")
        
        departure_location = geolocator.geocode(f"{departure_airport}, USA")
        arrival_location = geolocator.geocode(f"{arrival_airport}, USA")
        
        if departure_location and arrival_location:
            distance = geodesic(
                (departure_location.latitude, departure_location.longitude),
                (arrival_location.latitude, arrival_location.longitude)
            ).miles
            
            passenger_density = passenger_count / distance
            fare_per_mile = 0.1  # Placeholder value
            
            input_data = pd.DataFrame([[passenger_count, 0, distance, passenger_density, fare_per_mile]],
                                      columns=['passengers', 'large_ms', 'nsmiles', 'passenger_density', 'fare_per_mile'])
            
            input_data_log = np.log1p(input_data)
            input_data_scaled = scaler.transform(input_data_log)
            
            predicted_fare = model.predict(input_data_scaled)[0]
            
            st.markdown("---")
            st.markdown(f"<p class='medium-font'>Estimated Fare: ${predicted_fare:.2f}</p>", unsafe_allow_html=True)
            
            # Calculate performance metrics
            y_true = df['fare']
            y_pred = model.predict(scaler.transform(np.log1p(df[['passengers', 'large_ms', 'nsmiles', 'passenger_density', 'fare_per_mile']])))
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Display metrics in the right column
            with col2:
                st.markdown("<div class='metrics'>", unsafe_allow_html=True)
                st.markdown("<h3>Model Performance Metrics</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>Mean Squared Error: <span class='metric-value'>{mse:.4f}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p>Mean Absolute Error: <span class='metric-value'>{mae:.4f}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p>R-squared Score: <span class='metric-value'>{r2:.4f}</span></p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Not an operated route.")
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
