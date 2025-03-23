import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import modules as md
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import bronze_to_silver_cleaning as btc
import preprocessing as pp
import feature_engineering as fe
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor

 # Train/Test Split
seed = 100
test_size = 0.2


def split_dataset(features,price, images=True):
    if images:
        # Separate data based on 'image-src' prefix
        data_df = features[features['image-src'].str.startswith('data', na=False)]
        http_df = features[features['image-src'].str.startswith('http', na=False)]

        # # Split the 'http' data into training and testing sets
        http_train, http_test = train_test_split(http_df, test_size=test_size, random_state=seed) # Adjust test_size as needed

        # # Combine the 'data' data with the training portion of 'http' data
        X_train = pd.concat([data_df, http_train], ignore_index=True)

        # The test set will consist only of 'http' data
        X_test = http_test
        y_train = X_train['price']
        y_test = X_test['price']

        # Drop 'price' column from X_train and X_test
        X_train = X_train.drop(columns=['price', 'image-src'])
        X_test = X_test.drop(columns=['price', 'image-src'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test

@st.cache_data
def initial_data():
    path = "data/housesigmadata"
    combined_df = pp.combine_dataframes(path)
    combined_df = combined_df[combined_df['city'].str.contains('Waterloo', case=False, na=False)]
    combined_df['address'] = combined_df['address'].str.replace(' - Waterloo', '')
    output = gpd.read_file('data/good_data/address_dictionary_neighbourhoods.geojson')
    output = pd.DataFrame(output)
    df_schools = md.render_school()
    amenities = md.render_amenities()
    result_df = pp.process_housing(df_house_sigma=combined_df, output=output)
    final_filled_df = pp.predict_missing_neighbourhoods(result_df)
    final_filled_df = pp.add_school_details(final_filled_df, df_schools)
    final_filled_df = pp.add_amenities_details(final_filled_df, amenities)
    df_house_sigma = combined_df.drop(columns=['address'])
    uploaded_file = pd.merge(df_house_sigma, final_filled_df, on='listing_id', how='inner')
    houses = btc.clean_data(uploaded_file)

    return houses


def main(model_choice):
    
    houses = initial_data()

    if model_choice == "Random Forest":
        houses['neighbourhood_impact'] = pd.Categorical(houses['neighbourhood']).codes
        houses['roof_type'] = pd.Categorical(houses['roof']).codes
        houses['architecture_style_type'] = pd.Categorical(houses['architecture_style']).codes
        houses['frontage_type_code'] = pd.Categorical(houses['frontage_type']).codes
        houses['driveway_parking_type'] = pd.Categorical(houses['driveway_parking']).codes

    houses = houses.dropna(subset=['sold']) #these are removed events
    ml_houses = fe.feature_refining(houses)

    columns_to_encode = [        
                        # 'property_type',
                        'features',
                        # 'driveway_parking',
                        'basement_type',
                         'bathrooms_detail', 'sewer', 'topography',
                        'lot_features',
                        'exterior_feature',
                        'waterfront_features', 
                        'appliances_included',
                        'laundry_features',
                        ]
    split_exceptions = ['bathrooms_detail',]

    if model_choice == "Ridge Regression":
        columns_to_encode += ['neighbourhood', 'architecture_style', 'roof', 'frontage_type']

    # TODO: Appliances Excluded has to be penalizing in giving value to the prices

    for column in columns_to_encode:
        if column in houses.columns:
            encoded_df = fe.one_hot_encode_column(houses, column, split_exceptions=split_exceptions)
            ml_houses = pd.concat([ml_houses, encoded_df], axis=1)

    ml_houses['depth'].fillna(ml_houses['depth'].mean())
    ml_houses['frontage_length'].fillna(ml_houses['frontage_length'].mean())
    ml_houses = ml_houses.fillna(0)

    # This is the final dataframe that will be used for ML
    # features == X and price == y

    features = ml_houses.drop(columns=['listing_id', 'listing'])
    price = ml_houses['price']

    features = fe.correlation_analysis(features)

    # Drop 'kitchens', 'rooms', and 'bathrooms' columns if they exist
    columns_to_drop = ['kitchens', 'rooms', 
                    'latitude', 'longitude', 'year_built', 'building_age', 'house_year',
                    'distance_to_nearest_school',
                    'bathrooms',
                    # 'bedrooms_above_ground',
                    # 'garage', 'Airport_lot_features', 'Schools_lot_features',
                    # 'frontage_length',
                    'bedrooms', 'depth',]
    for col in columns_to_drop:
        if col in features.columns:
            features = features.drop(columns=[col])

    features = md.group_columns(features)
    
    X_train, X_test, y_train, y_test = split_dataset(features, price, images=True)

    if model_choice == "Random Forest":  ## This is now XGBOOST
        # model = RandomForestRegressor(n_estimators=200, random_state=seed)
        model = XGBRegressor(objective='reg:squarederror', random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        feature_importance = model.feature_importances_
    elif model_choice == "Ridge Regression":
        model = Ridge(random_state=seed, solver='lbfgs', positive=True) 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        absolute_coefficients = np.abs(model.coef_)
        feature_importance = (absolute_coefficients / np.sum(absolute_coefficients)) * 100

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Median Error Calculation
    percentage_errors = np.abs((y_test - y_pred) / y_test) * 100
    median_error = np.median(percentage_errors)

    results = [[mse, rmse, r2, median_error]]
    st.session_state["evaluation"] = results
    st.session_state["rmse"] = rmse
    st.session_state["median_error"] = median_error
    # md.display_df(results)

    # Feature Importance
    feature_names = X_test.columns.tolist()
    sorted_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

    # List of words to drop columns containing them
    words_to_drop = md.words_to_drop

    # Filter sorted_features to remove any feature names containing the words in words_to_drop
    filtered_sorted_features = [feature for feature in sorted_features if not md.should_drop(feature[0], words_to_drop)]

    top_features = filtered_sorted_features[:20]
    top_features = [(md.remove_suffixes(feature[0]), feature[1]) for feature in top_features]
    top_feature_names, top_percentages = zip(*top_features)

    st.session_state["top_feature_names"] = top_feature_names
    st.session_state["top_percentages"] = top_percentages

    # md.display_graph(top_feature_names, top_percentages)

    # Single Data Point Prediction
    joined_df = X_test.join(ml_houses[['listing_id', 'listing']], how='inner')
    joined_df = joined_df.merge(houses[['listing_id', 'image-src', 'neighbourhood', 'roof', 'frontage_type',
                                        'latitude','longitude', 'bedrooms', 'description', 'driveway_parking',
                                        'amenities_objectids_1km', 'nearest_school', 'architecture_style',
                                        'bathrooms', 'property_type']], on='listing_id', how='inner')

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
