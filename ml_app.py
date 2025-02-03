import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import re

# Load data
st.title("House Price Prediction App")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    houses = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# Data Cleaning and Preprocessing
houses.dropna(axis=1, how='all', inplace=True)
houses.dropna(subset=['listed'], inplace=True)
value_counts = houses.count()
columns_to_drop = value_counts[value_counts < 10].index
houses.drop(columns=columns_to_drop, inplace=True)
numeric_cols = houses.select_dtypes(include=np.number).columns
object_cols = houses.select_dtypes(include=object).columns

# Convert numeric features
def calculate_house_age(year_string):
    if '-' in str(year_string):
        try:
            start, end = map(int, year_string.split('-'))
            return (start + end) / 2
        except ValueError:
            return None
    elif str(year_string).isdigit() and len(str(year_string)) == 4:
        return 2025 - int(year_string)
    else:
        return None

houses['house_year'] = houses['year_built'].fillna(houses['building_age'])
houses['house_age'] = houses['house_year'].apply(calculate_house_age)

# Show histogram of house ages
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(houses['house_age'].dropna(), bins=30, edgecolor='black')
ax.set_xlabel('House Age')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of House Ages')
st.pyplot(fig)

# ML Preparation
ml_houses = houses.select_dtypes(include=np.number)
ml_houses['listing_id'] = houses['listing_id']
ml_houses['price'] = houses['listed']
ml_houses['listing'] = houses['listing']
ml_houses['price'] = ml_houses['price'].astype(str).str.replace(r'[$,]', '', regex=True)
ml_houses['price'] = pd.to_numeric(ml_houses['price'], errors='coerce')
features = ml_houses.drop(columns=['listing_id', 'price', 'listing'])
price = ml_houses['price']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=0.2, random_state=100)

# Train Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Display Results
st.subheader("Model Evaluation")
st.write(f"Random Forest - MSE: {mse_rf}, R2: {r2_rf}")
st.write(f"Linear Regression - MSE: {mse_lr}, R2: {r2_lr}")

# Feature Importance
feature_names = X_test.columns.tolist()
feature_weights_rf = rf_model.feature_importances_

sorted_features_rf = sorted(zip(feature_names, feature_weights_rf), key=lambda x: x[1], reverse=True)
top_features_rf = sorted_features_rf[:20]
top_feature_names_rf, top_percentages_rf = zip(*top_features_rf)

fig, ax = plt.subplots()
ax.barh(top_feature_names_rf, top_percentages_rf, color='skyblue')
ax.set_xlabel("Contribution (%)")
ax.set_title("Top 20 Feature Contributions - Random Forest")
ax.invert_yaxis()
st.pyplot(fig)

# Single Data Point Prediction
single_data_point = X_test.iloc[[0]]
prediction = rf_model.predict(single_data_point)
st.subheader("Single Data Point Prediction")
st.write(f"Predicted Price: {prediction[0]}")
