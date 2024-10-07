pip install streamlit
pip install pandas
numpy
scikit-learn
geopy
joblib

import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# File paths
CSV_PATH = r"C:\Users\alkoj\OneDrive\Uni\Business Intelligence\Data set\Airline Dataset - Cleaned (CSV) (Readjusted).csv"
MODEL_PATH = r"C:\Users\alkoj\OneDrive\Uni\Business Intelligence\your_trained_model.joblib"
SCALER_PATH = r"C:\Users\alkoj\OneDrive\Uni\Business Intelligence\your_scaler.joblib"
VIDEO_PATH = r"C:\Users\alkoj\Downloads\Airline Offer Your Story Gehra Neela par Halka Neela aur Halka Peela ke saath Playful Style.mp4"  # Replace with your local video file path

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
        border-radius: 5px;
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# Load data, model, and scaler
df = load_data(CSV_PATH)
model, scaler = load_model_and_scaler()

# App layout
st.markdown("<p class='big-font'>SkyFare Predictor</p>", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([6, 4])

with col1:
    st.markdown("<div class='predictor'>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Discover your next flight's price effortlessly</p>", unsafe_allow_html=True)
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
        else:
            st.error("Could not locate one or both airports. Please check your inputs.")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.video(VIDEO_PATH)

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
