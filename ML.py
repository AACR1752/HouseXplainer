import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import bronze_to_silver_cleaning as btc
import preprocessing as pp
import feature_engineering as fe
import os
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load data
st.title("House Price Prediction App")
st.sidebar.header("Upload Data")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

path = "data/housesigmadata"
files = os.listdir(path)
# drop .DS_Store and areas from files keeping only .csv as one list and geojson in another
csv_files = []
for file in files:
    if file.endswith('.csv'):
        csv_files.append(file)

all_dfs = []

# Loop through the CSV files
for file in csv_files:
    file_path = os.path.join(path, file)  # Construct the full file path
    try:
        # Read each CSV file into a DataFrame
        df = pd.read_csv(file_path)
        all_dfs.append(df)
    except pd.errors.ParserError:
        print(f"Skipping file {file} due to parsing error.")
    except Exception as e:
        print(f"An error occurred while processing {file}: {e}")

output = gpd.read_file('data/good_data/address_dictionary_neighbourhoods.geojson')
output = pd.DataFrame(output)

df_schools = pd.read_csv('data/good_data/schools.csv')
amenities = pd.read_csv('data/good_data/amenities.csv')

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.columns = map(str.lower, combined_df.columns)
combined_df = combined_df.drop_duplicates(subset=['listing_id'])
combined_df = combined_df[combined_df['city'].str.contains('Waterloo', case=False, na=False)]
combined_df['address'] = combined_df['address'].str.replace(' - Waterloo', '')
result_df = pp.process_housing(df_house_sigma=combined_df, output=output)

final_filled_df = pp.predict_missing_neighbourhoods(result_df)
final_filled_df = pp.add_school_details(final_filled_df, df_schools)
final_filled_df = pp.add_amenities_details(final_filled_df, amenities)
df_house_sigma = combined_df.drop(columns=['address'])
uploaded_file = pd.merge(df_house_sigma, final_filled_df, on='listing_id', how='inner')

if uploaded_file is not None and "houses" not in st.session_state:
    houses = btc.clean_data(uploaded_file)
    st.session_state['houses'] = houses.values
    st.session_state["houses_raw_columns"] = houses.columns.tolist()
    st.session_state["houses_raw_index"] = houses.index.tolist()
    st.write("Dataset Loaded Successfully!")
elif "houses" in st.session_state:
    houses = pd.DataFrame(st.session_state['houses'], 
                          columns=st.session_state["houses_raw_columns"], 
                          index=st.session_state["houses_raw_index"])
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

ml_houses = fe.feature_refining(houses)

# This is the final dataframe that will be used for ML
# features == X and price == y
features = ml_houses.drop(columns=['listing_id', 'price', 'listing'])
price = ml_houses['price']

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
    st.write("Correlated columns dropped successfully.")
except KeyError as e:
    st.write(f"Error: Column(s) {e} not found in the features DataFrame.")

# Drop 'kitchens', 'rooms', and 'bathrooms' columns if they exist
columns_to_drop = ['kitchens', 'rooms', 'bathrooms', 'bedrooms', 'depth']
for col in columns_to_drop:
    if col in features.columns:
        features = features.drop(columns=[col])
    else:
        st.write(f"Warning: Column '{col}' not found in features DataFrame.")

# Data Cleaning and Preprocessing

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=0.2, random_state=100)

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model", ["Ridge Regression", "Random Forest"])

if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    feature_importance = model.feature_importances_
elif model_choice == "Ridge Regression":
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
    .properties(width=800, height=600, title="Feature Importance")
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