import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import requests
import io

# File URL
CSV_URL = "https://raw.githubusercontent.com/Alko2122/Uni/756569d0500e6c5d5d6e6e1b5b949b423e3349d2/Airline%20Dataset%20-%20Cleaned%20(CSV)%20(Readjusted).csv"

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

def remove_outliers(df, columns, lower_quantile=0.20, upper_quantile=0.90):
    df_clean = df.copy()
    for column in columns:
        Q1 = df_clean[column].quantile(lower_quantile)
        Q3 = df_clean[column].quantile(upper_quantile)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean

def apply_log_transform(df, columns):
    df_log = df.copy()
    for column in columns:
        df_log[column] = np.log(df_log[column] + 1e-6)  # Apply log transformation
    return df_log

@st.cache_resource
def prepare_model_and_data(df):
    # Remove outliers
    features = ['fare', 'large_ms', 'nsmiles', 'passengers']
    df_clean = remove_outliers(df, features)
    
    # Filter out rows with non-positive values before applying log transformation
    df_clean = df_clean[(df_clean['passengers'] > 0) & (df_clean['large_ms'] > 0) & (df_clean['nsmiles'] > 0)]
    
    # Feature engineering
    df_clean['passenger_density'] = df_clean['passengers'] / df_clean['nsmiles']
    df_clean['fare_per_mile'] = df_clean['fare'] / df_clean['nsmiles']
    
    # Prepare the data
    data = df_clean[['fare', 'large_ms', 'nsmiles', 'passengers', 'passenger_density', 'fare_per_mile']].dropna()
    
    # Apply log transformation to features (except target variable)
    X = data[['passengers', 'large_ms', 'nsmiles', 'passenger_density', 'fare_per_mile']]
    X_log = np.log1p(X)
    y = data['fare']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
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
    
    return rf_model, scaler, mae, mse, rmse, r2, df_clean

# Load data and prepare model
df = load_data(CSV_URL)
model, scaler, mae, mse, rmse, r2, df_clean = prepare_model_and_data(df)

st.title("SkyFare Consultants")

# Display model accuracy metrics
st.markdown("## Model Accuracy Metrics")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R² Score: {r2:.4f}")

# Input form
st.markdown("## Fare Prediction")
departure_airport = st.selectbox("Departure Airport", sorted(df_clean['airport_1'].unique()))
arrival_airport = st.selectbox("Arrival Airport", sorted(df_clean['airport_1'].unique()))
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
        fare_per_mile = df_clean['fare_per_mile'].mean()  # Use mean fare_per_mile as an estimate
        
        input_data = pd.DataFrame([[passenger_count, 0, distance, passenger_density, fare_per_mile]],
                                  columns=['passengers', 'large_ms', 'nsmiles', 'passenger_density', 'fare_per_mile'])
        
        input_data_log = np.log1p(input_data)
        input_data_scaled = scaler.transform(input_data_log)
        
        predicted_fare = model.predict(input_data_scaled)[0]
        
        st.markdown("---")
        st.markdown(f"<p class='medium-font'>Estimated Fare: ${predicted_fare:.2f}</p>", unsafe_allow_html=True)
    else:
        st.error("Not an operated route.")

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
