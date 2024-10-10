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

# ... [previous code remains unchanged] ...

# New functions for EDA visualizations
def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    return fig

def plot_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f'Boxplot of {column}')
    ax.set_xlabel(column)
    return fig

def plot_scatter(df, x, y):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f'{x} vs {y}')
    return fig

def plot_pairplot(df, vars):
    fig = sns.pairplot(df[vars], height=2.5)
    fig.fig.suptitle('Pairplot of Selected Features', y=1.02)
    return fig

# Load data and prepare model
df = load_data(CSV_URL)
model, scaler, mae, mse, rmse, r2, df_clean, X_test_scaled, y_test, y_pred_rf = prepare_model_and_data(df)

st.title("SkyFare Consultants")

# Button to open model metrics panel
if st.button("Show Model Metrics and Visualizations"):
    # ... [existing code for model metrics and visualizations] ...

# New button to open EDA panel
if st.button("Show Exploratory Data Analysis"):
    st.sidebar.title("Exploratory Data Analysis")
    
    st.sidebar.markdown("## Data Overview")
    st.sidebar.write(df_clean.describe())
    
    st.sidebar.markdown("## Distribution of Fare")
    fare_hist = plot_histogram(df_clean, 'fare')
    st.sidebar.pyplot(fare_hist)
    
    st.sidebar.markdown("## Boxplot of Passenger Count")
    passengers_box = plot_boxplot(df_clean, 'passengers')
    st.sidebar.pyplot(passengers_box)
    
    st.sidebar.markdown("## Fare vs Distance")
    fare_distance_scatter = plot_scatter(df_clean, 'nsmiles', 'fare')
    st.sidebar.pyplot(fare_distance_scatter)
    
    st.sidebar.markdown("## Pairplot of Key Features")
    pairplot_vars = ['fare', 'passengers', 'nsmiles', 'passenger_density']
    pairplot_fig = plot_pairplot(df_clean, pairplot_vars)
    st.sidebar.pyplot(pairplot_fig)

# ... [rest of the existing code remains unchanged] ...
