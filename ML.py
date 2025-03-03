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
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load data
st.title("House Price Prediction App")
st.sidebar.header("Upload Data")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model", ["Ridge Regression", "Random Forest"])

path = "data/housesigmadata"
combined_df = pp.combine_dataframes(path)

combined_df = combined_df[combined_df['city'].str.contains('Waterloo', case=False, na=False)]
combined_df['address'] = combined_df['address'].str.replace(' - Waterloo', '')

output = gpd.read_file('data/good_data/address_dictionary_neighbourhoods.geojson')
output = pd.DataFrame(output)
df_schools = pd.read_csv('data/good_data/schools.csv')
amenities = pd.read_csv('data/good_data/amenities.csv')

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

if model_choice == "Ridge Regression":
    houses['neighbourhood_impact'] = pd.Categorical(houses['neighbourhood']).codes

ml_houses = fe.feature_refining(houses)

columns_to_encode = ['architecture_style','property_type',
                     'driveway_parking', 'frontage_type',
                     'sewer','basement_type','topography',
                     'bathrooms_detail', 
                     'lot_features',
                     'exterior_feature',
                     'roof', 
                     'waterfront_features', 
                     'appliances_included',
                     'laundry_features',
                     ]
split_exceptions = ['bathrooms_detail',]

if model_choice == "Ridge Regression":
    columns_to_encode += ['neighbourhood']

# TODO: Appliances Excluded has to be penalizing in giving value to the prices

for column in columns_to_encode:
    if column in houses.columns:
        encoded_df = fe.one_hot_encode_column(houses, column, split_exceptions=split_exceptions)
        ml_houses = pd.concat([ml_houses, encoded_df], axis=1)

# This is the final dataframe that will be used for ML
# features == X and price == y

features = ml_houses.drop(columns=['listing_id', 'listing'])
# price = ml_houses['price']

features = fe.correlation_analysis(features)

# Drop 'kitchens', 'rooms', and 'bathrooms' columns if they exist
columns_to_drop = ['kitchens', 'rooms', 
                   'latitude', 'longitude', 'year_built', 'building_age', 'house_year',
                   'distance_to_nearest_school',
                   'bathrooms', 
                   'bedrooms', 'depth',]
for col in columns_to_drop:
    if col in features.columns:
        features = features.drop(columns=[col])

# drop columns with nan values in features
features = features.dropna(axis=1)

# Separate data based on 'image-src' prefix
data_df = features[features['image-src'].str.startswith('data', na=False)]
http_df = features[features['image-src'].str.startswith('http', na=False)]

# Train/Test Split
seed = 1000
test_size = 0.2

# Split the 'http' data into training and testing sets
http_train, http_test = train_test_split(http_df, test_size=test_size, random_state=seed) # Adjust test_size as needed

# Combine the 'data' data with the training portion of 'http' data
X_train = pd.concat([data_df, http_train], ignore_index=True)

# The test set will consist only of 'http' data
X_test = http_test
y_train = X_train['price']
y_test = X_test['price']

# Drop 'price' column from X_train and X_test
X_train = X_train.drop(columns=['price', 'image-src'])
X_test = X_test.drop(columns=['price', 'image-src'])

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

st.write(X_train.info())
st.write(y_train.shape)

# Data Cleaning and Preprocessing

# X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=test_size, random_state=seed)

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