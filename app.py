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

# Logo URL - replace with your actual logo URL from GitHub
LOGO_URL = "https://github.com/Alko2122/Uni/blob/688014181dadcf69ccc216a6cb705aa11fe5fd0e/image-removebg-preview.png"

# Custom CSS to create a background and position the logo
st.markdown(f"""

    .logo-img {{
        position: fixed;
        top: 20px;
        right: 20px;
        width: 100px;  /* Adjust size as needed */
        z-index: 999;
    }}
    .luxury-card {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }}
    .sidebar .block-container {{
        background-color: rgba(255, 255, 255, 0.9);
    }}
</style>
<img src="{LOGO_URL}" class="logo-img">
""", unsafe_allow_html=True)

# File URL
CSV_URL = "https://raw.githubusercontent.com/Alko2122/Uni/756569d0500e6c5d5d6e6e1b5b949b423e3349d2/Airline%20Dataset%20-%20Cleaned%20(CSV)%20(Readjusted).csv"

# ... [rest of your existing code remains unchanged]

st.title("SkyFare Consultants")

# ... [rest of your existing code remains unchanged]

st.markdown("---")
st.markdown("<p class='small-font'>About | Contact | Terms of Service</p>", unsafe_allow_html=True)
