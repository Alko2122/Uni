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
    # Feature engineering
    df['passenger_density'] = df['passengers'] / df['nsmiles']
    df['fare_per_mile'] = df['fare'] / df['nsmiles']
    
    # Remove outliers
    features = ['fare', 'large_ms', 'nsmiles', 'passengers', 'passenger_density', 'fare_per_mile']
    df_clean = remove_outliers(df, features)
    
    # Filter out rows with non-positive values before applying log transformation
    df_clean = df_clean[(df_clean['passengers'] > 0) & (df_clean['large_ms'] > 0) & (df_clean['nsmiles'] > 0)]
    
    # Prepare the data
    data = df_clean[features].dropna()
    
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
    
    return rf_model, scaler, mae, mse, rmse, r2, df_clean, X_test_scaled, y_test, y_pred_rf

def plot_correlation_matrix(df):
    columns_to_include = ['fare', 'large_ms', 'nsmiles', 'passengers', 'passenger_density', 'fare_per_mile']
    data_for_correlation = df[columns_to_include].dropna()
    correlation_matrix = data_for_correlation.corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8}, ax=ax)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    return fig

def plot_vif(df):
    features = ['large_ms', 'nsmiles', 'passengers', 'passenger_density', 'fare_per_mile']
    X = df[features]
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(vif_data['Variable'], vif_data['VIF'])
    ax.set_title('Variance Inflation Factor for Each Feature')
    ax.set_xlabel('Features')
    ax.set_ylabel('VIF')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, vif_data

def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_title('Actual vs Predicted Fare')
    ax.set_xlabel('Actual Fare')
    ax.set_ylabel('Predicted Fare')
    ax.grid()
    return fig

def plot_feature_importance(model, X):
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(feature_importance['feature'], feature_importance['importance'])
    ax.set_title('Feature Importance')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, feature_importance

# Load data and prepare model
df = load_data(CSV_URL)
model, scaler, mae, mse, rmse, r2, df_clean, X_test_scaled, y_test, y_pred_rf = prepare_model_and_data(df)

st.title("SkyFare Consultants")

# Button to open model metrics panel
if st.button("Show Model Metrics and Visualizations"):
    st.sidebar.title("Model Metrics and Visualizations")
    
    st.sidebar.markdown("## Model Accuracy Metrics")
    st.sidebar.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.sidebar.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.sidebar.write(f"R² Score: {r2:.4f}")
    
    st.sidebar.markdown("## Correlation Matrix")
    corr_fig = plot_correlation_matrix(df_clean)
    st.sidebar.pyplot(corr_fig)
    
    st.sidebar.markdown("## Variance Inflation Factor")
    vif_fig, vif_data = plot_vif(df_clean)
    st.sidebar.pyplot(vif_fig)
    st.sidebar.dataframe(vif_data)
    
    st.sidebar.markdown("## Actual vs Predicted Fare")
    actual_vs_pred_fig = plot_actual_vs_predicted(y_test, y_pred_rf)
    st.sidebar.pyplot(actual_vs_pred_fig)
    
    st.sidebar.markdown("## Feature Importance")
    feature_imp_fig, feature_importance = plot_feature_importance(model, X_test_scaled)
    st.sidebar.pyplot(feature_imp_fig)
    st.sidebar.dataframe(feature_importance)

# Input form with widgets
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

def calculate_distance(airport1_coords, airport2_coords):
    return geodesic(airport1_coords, airport2_coords).miles

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
        st.error("The airport is not currently in operation")

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
