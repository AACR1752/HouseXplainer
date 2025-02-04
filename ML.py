import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import bronze_to_silver_cleaning as btc
import feature_engineering as fe


# Load data
st.title("House Price Prediction App")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    houses = btc.clean_data(uploaded_file)
    st.write("Dataset Loaded Successfully!")
else:
    st.warning("Please upload a dataset to continue.")
    # st.stop()

# Feature Engineering

ml_houses = fe.feature_refining(houses)

# This is the final dataframe that will be used for ML
# features == X and price == y
features = ml_houses.drop(columns=['listing_id', 'price', 'listing'])
price = ml_houses['price']
 
# prompt: get a table of where correlating columns are put side by side along with their scores. And sort them in descending order.

# Calculate the correlation matrix
correlation_matrix = features.corr()

# Remove diagonal part
correlation_matrix = correlation_matrix.mask(np.equal(*np.indices(correlation_matrix.shape)))

# Stack the correlation matrix
corr_pairs = correlation_matrix.stack()

# Sort by absolute value
sorted_pairs = corr_pairs.abs().sort_values(ascending=False)

# Get the top n pairs (adjust n as needed)
top_n_pairs = sorted_pairs.head(20)  # Example: top 20 pairs

# Create a DataFrame for the results
correlation_table = pd.DataFrame({
    'Column1': top_n_pairs.index.get_level_values(0),
    'Column2': top_n_pairs.index.get_level_values(1),
    'CorrelationScore': top_n_pairs.values
})
# correlation_table

 
# prompt: Using dataframe correlation_table: take rows where the correlation score is greater than 0.8. then in that subset, take column 1 and all the unique names in that small subset. then drop these columns from the features table

# Filter the correlation table to include only rows where the correlation score is greater than 0.8
filtered_correlation = correlation_table[correlation_table['CorrelationScore'] > 0.8]

# Get the unique names from 'Column1' in the filtered table
col1_names = filtered_correlation['Column1'].unique()

# Assuming 'features' is another DataFrame where you want to drop the columns
# Replace 'features' with your actual DataFrame name if it's different
# Drop the specified columns from the features table
# Note that if any of the columns in col1_names do not exist in the features table,
# a KeyError will occur. You may consider adding a try-except block to handle the error gracefully
# or use the errors='ignore' parameter in the drop function.

try:
    features = features.drop(columns=col1_names)
    st.write("Columns dropped successfully.")
except KeyError as e:
    st.write(f"Error: Column(s) {e} not found in the features DataFrame.")

# Drop 'kitchens', 'rooms', and 'bathrooms' columns if they exist
columns_to_drop = ['kitchens', 'rooms', 'bathrooms', 'bedrooms']
for col in columns_to_drop:
    if col in features.columns:
        features = features.drop(columns=[col])
    else:
        st.write(f"Warning: Column '{col}' not found in features DataFrame.")



# Data Cleaning and Preprocessing

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=0.2, random_state=100)

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])

if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    feature_importance = model.feature_importances_
elif model_choice == "Linear Regression":
    model = Ridge() 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    feature_importance = np.abs(model.coef_)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
st.subheader("Model Evaluation")

results = [[mse, rmse, r2]]
results_df = pd.DataFrame(results, columns=['MSE','RMSE', 'R-squared'])
st.dataframe(results_df)

# Feature Importance
feature_names = X_test.columns.tolist()
sorted_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
top_features = sorted_features[:20]
top_feature_names, top_percentages = zip(*top_features)

# Streamlit bar chart

# Create a DataFrame for Streamlit's st.barchart
df = pd.DataFrame({"Feature": top_feature_names, "Contribution (%)": top_percentages})

# Plot using Streamlit's altair_chart

source = pd.DataFrame({"Feature": top_feature_names, "Contribution (%)": top_percentages})

# Create the Altair bar chart
bars = (
    alt.Chart(source)
    .mark_bar()
    .encode(
        x=alt.X("Contribution (%):Q", title="Contribution (%)"),
        y=alt.Y("Feature:N", title="Feature", sort="-x"),  # Sorting by Contribution
        color=alt.Color("Feature:N", legend=None),  # Optional: Color coding
    )
    .properties(width=700, height=500, title="Feature Importance")
)

# Display in Streamlit
st.altair_chart(bars, use_container_width=True)

# Single Data Point Prediction
joined_df = X_test.join(ml_houses[['listing_id', 'listing']], how='inner')
joined_df = joined_df.merge(houses[['listing_id', 'image-src']], on='listing_id', how='inner')

# Store the trained model and other variables in session state
st.session_state["trained_model"] = model
st.session_state["model_choice"] = model_choice
st.session_state["y_test"] = y_test.to_dict()

# Store the DataFrame values, columns, and index in session state
st.session_state["joined_df_values"] = joined_df.values
st.session_state["joined_df_columns"] = joined_df.columns.tolist()
st.session_state["joined_df_index"] = joined_df.index.tolist()

# Store X_test values, columns, and index in session state
st.session_state["X_test_values"] = X_test.values  # Store only values
st.session_state["X_test_columns"] = X_test.columns.tolist()  # Store columns
st.session_state["X_test_index"] = X_test.index.tolist()  # Store index

st.success("Model trained successfully! Go to 'Use Model' page to test it.")