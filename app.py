import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import requests
import io
import tempfile
import os
import gdown

# File URLs
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
def prepare_model_and_data(df):
    # Prepare features and target
    X = df[['passengers', 'large_ms', 'nsmiles']]
    y = df['fare']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions and evaluation
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred_rf)
    mse = mean_squared_error(y_test, y_pred_rf)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_rf)
    
    return rf_model, scaler, mae, mse, rmse, r2

# Load data and prepare model
df = load_data(CSV_URL)
model, scaler, mae, mse, rmse, r2 = prepare_model_and_data(df)

st.title("SkyFare Consultants")

# Display model accuracy metrics
st.markdown("## Model Accuracy Metrics")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R² Score: {r2:.4f}")

# Input form
st.markdown("## Fare Prediction")
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
        
        input_data = pd.DataFrame([[passenger_count, 0, distance]],
                                  columns=['passengers', 'large_ms', 'nsmiles'])
        
        input_data_scaled = scaler.transform(input_data)
        
        predicted_fare = model.predict(input_data_scaled)[0]
        
        st.markdown("---")
        st.markdown(f"<p class='medium-font'>Estimated Fare: ${predicted_fare:.2f}</p>", unsafe_allow_html=True)
    else:
        st.error("Not an operated route.")

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
