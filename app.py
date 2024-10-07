import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import requests
import io

# Custom CSS to create a background similar to the luxury line abstract design
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #ffffff 25%, #f0f0f0 25%, #f0f0f0 50%, #ffffff 50%, #ffffff 75%, #f0f0f0 75%, #f0f0f0 100%);
        background-size: 20px 20px;
        background-attachment: fixed;
    }
    .luxury-card {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sidebar .block-container {
        background-color: rgba(255, 255, 255, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# File URL
CSV_URL = "https://raw.githubusercontent.com/Alko2122/Uni/756569d0500e6c5d5d6e6e1b5b949b423e3349d2/Airline%20Dataset%20-%20Cleaned%20(CSV)%20(Readjusted).csv"

@st.cache_data
def load_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return pd.read_csv(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ... [rest of the function definitions remain the same]

st.title("SkyFare Consultants")

# Load data and prepare model
df = load_data(CSV_URL)

if df is not None:
    try:
        model, scaler, mae, mse, rmse, r2, df_clean, X_test_scaled, y_test, y_pred_rf = prepare_model_and_data(df)

        # Button to open model metrics panel
        if st.button("Show Model Metrics and Visualizations"):
            st.sidebar.title("Model Metrics and Visualizations")
            
            st.sidebar.markdown("## Model Accuracy Metrics")
            st.sidebar.markdown('<div class="luxury-card">', unsafe_allow_html=True)
            st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.sidebar.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.sidebar.write(f"R² Score: {r2:.4f}")
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            st.sidebar.markdown("## Correlation Matrix")
            st.sidebar.markdown('<div class="luxury-card">', unsafe_allow_html=True)
            corr_fig = plot_correlation_matrix(df_clean)
            st.sidebar.pyplot(corr_fig)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            st.sidebar.markdown("## Variance Inflation Factor")
            st.sidebar.markdown('<div class="luxury-card">', unsafe_allow_html=True)
            vif_fig, vif_data = plot_vif(df_clean)
            st.sidebar.pyplot(vif_fig)
            st.sidebar.dataframe(vif_data)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            st.sidebar.markdown("## Actual vs Predicted Fare")
            st.sidebar.markdown('<div class="luxury-card">', unsafe_allow_html=True)
            actual_vs_pred_fig = plot_actual_vs_predicted(y_test, y_pred_rf)
            st.sidebar.pyplot(actual_vs_pred_fig)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            st.sidebar.markdown("## Feature Importance")
            st.sidebar.markdown('<div class="luxury-card">', unsafe_allow_html=True)
            feature_imp_fig, feature_importance = plot_feature_importance(model, X_test_scaled)
            st.sidebar.pyplot(feature_imp_fig)
            st.sidebar.dataframe(feature_importance)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Input form with widgets
        st.markdown('<div class="luxury-card">', unsafe_allow_html=True)
        st.markdown("## Fare Prediction")

        # Create a list of unique airports
        unique_airports = list(set(df_clean['airport_1'].unique().tolist() + df_clean['airport_2'].unique().tolist()))
        unique_airports.sort()

        # Widgets for selecting airports and passengers
        departure_airport = st.selectbox("Departure Airport:", options=unique_airports)
        arrival_airport = st.selectbox("Arrival Airport:", options=unique_airports)
        passenger_count = st.slider("Number of Passengers:", min_value=1, max_value=1000, value=100, step=1)

        # Geopy to calculate distances between airports
        geolocator = Nominatim(user_agent="airport_selector")

        if st.button("Calculate Fare"):
            departure_location = geolocator.geocode(f"{departure_airport}, USA")
            arrival_location = geolocator.geocode(f"{arrival_airport}, USA")
            
            if departure_location and arrival_location:
                distance_value = calculate_distance(
                    (departure_location.latitude, departure_location.longitude),
                    (arrival_location.latitude, arrival_location.longitude)
                )
                
                # Prepare input for prediction
                passenger_density = passenger_count / distance_value
                fare_per_mile = df_clean['fare_per_mile'].mean()  # Use mean fare_per_mile as an estimate
                
                input_data = pd.DataFrame([[
                    passenger_count,
                    0,  # placeholder for large_ms
                    distance_value,
                    passenger_density,
                    fare_per_mile
                ]], columns=['passengers', 'large_ms', 'nsmiles', 'passenger_density', 'fare_per_mile'])
                
                input_data_log = np.log1p(input_data)
                input_data_scaled = pd.DataFrame(scaler.transform(input_data_log), columns=input_data.columns)
                
                # Predict fare
                pred_fare = model.predict(input_data_scaled)[0]
                
                st.markdown("---")
                st.markdown(f"<p class='medium-font'>Estimated Fare: ${pred_fare:.2f}</p>", unsafe_allow_html=True)
                st.write(f"Flight Distance: {distance_value:.2f} miles")
            else:
                st.error("Unable to geocode one or both of the airports. Please try different airports.")

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
else:
    st.error("Unable to load the dataset. Please check the URL and try again.")

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
