from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import re
import streamlit as st

def calculate_house_age(year_string):
    if '-' in str(year_string):
        try:
            start, end = map(int, year_string.split('-'))
            return (start + end) / 2
        except ValueError:
            return None  # Handle cases where splitting fails
    else:
        try:
            return 2025 - int(year_string)
        except ValueError:
            return None

def one_hot_encode_column(houses, column, split_exceptions=[]):
    if column in houses.columns:
        if column not in split_exceptions:
            houses[f'{column}_arr'] = houses[column].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        else:
            houses[f'{column}_arr'] = houses[column].apply(lambda x: [x] if isinstance(x, str) else [])

        mlb = MultiLabelBinarizer()
        encoded_data = mlb.fit_transform(houses[f'{column}_arr'])
        encoded_df = pd.DataFrame(encoded_data, columns=mlb.classes_)

        # Add column variable as suffix to encoded column names
        encoded_df = encoded_df.add_suffix(f'_{column}')

        # Drop columns containing "none" case-insensitive
        encoded_df = encoded_df.drop(columns=[col for col in encoded_df.columns if re.search(r"none", col, re.IGNORECASE)])

        # Drop columns with '*' in their names
        encoded_df = encoded_df.drop(columns=[col for col in encoded_df.columns if '*' in col])
        
        encoded_df = encoded_df.fillna(0)

        return encoded_df

def feature_refining(houses):
    # prompt: st.write all the column names one by one
    houses.dropna(axis=1, how='all', inplace=True)

    #drop rows where listed is missing
    houses.dropna(subset=['listed'], inplace=True)

    # Identify columns with less than 10 non-NaN values
    # value_counts = houses.count()
    # columns_to_drop = value_counts[value_counts < 10].index #not needed
    # houses.drop(columns=columns_to_drop, inplace=True)

    # numeric_cols = houses.select_dtypes(include=np.number).columns
    # object_cols = houses.select_dtypes(include=object).columns
    
    numeric_features = ['rooms', 'bedrooms', 'bedrooms_above_ground',
                    'bedrooms_below_ground', 'bathrooms', '2_piece_bathrooms',
                    '3_piece_bathrooms', '4_piece_bathrooms', 'garage',
                    'frontage_length', 'depth']

    for col in numeric_features:
        if col in houses.columns:
            houses[col] = pd.to_numeric(houses[col], errors='coerce')
            houses[col] = houses[col].astype(float)
    
    
    houses['house_year'] = houses['year_built'].fillna(houses['building_age'])
    houses['house_age'] = houses['house_year'].apply(calculate_house_age)
    
    # Calculate the average house age, excluding NaN values
    average_house_age = houses['house_age'].mean()

    # Fill NaN values in 'house_age' with the average house age
    houses['house_age'].fillna(average_house_age, inplace=True)

    # ## Create ML Dataframes

    # Create a separate DataFrame with numeric columns and 'listing_id'
    ml_houses = houses.select_dtypes(include=np.number)
    ml_houses['listing_id'] = houses['listing_id']
    ml_houses['price'] = houses['sold']
    ml_houses['listing'] = houses['listing']
    ml_houses['image-src'] = houses['image-src']

    # Convert 'price' column to numeric, handling '$' and ','
    ml_houses['price'] = ml_houses['price'].astype(str).str.replace(r'[$,]', '', regex=True)
    ml_houses['price'] = pd.to_numeric(ml_houses['price'], errors='coerce')

    
    # place listing_id as the first column in the df
    cols = ml_houses.columns.tolist()
    cols.insert(0, cols.pop(cols.index('listing_id')))
    ml_houses = ml_houses.reindex(columns=cols)
    
    # Reset the index of the DataFrame
    ml_houses = ml_houses.reset_index(drop=True)

    return ml_houses

def correlation_analysis(features):
    # Calculate the correlation matrix
    # assuming img-src is not a feature
    img = features['image-src']
    price = features['price']
    features = features.select_dtypes(include=np.number)
    # features = features.drop(columns=['price'])
    correlation_matrix = features.corr()
    correlation_matrix = correlation_matrix.mask(np.equal(*np.indices(correlation_matrix.shape)))
    corr_pairs = correlation_matrix.stack() # Stack the correlation matrix
    sorted_pairs = corr_pairs.abs().sort_values(ascending=False) # Sort by absolute value
    top_n_pairs = sorted_pairs.head(20)  # Get the top 20 pairs

    # Create a DataFrame for the results
    correlation_table = pd.DataFrame({
        'Column1': top_n_pairs.index.get_level_values(0),
        'Column2': top_n_pairs.index.get_level_values(1),
        'CorrelationScore': top_n_pairs.values
    })

    # Filter the correlation table to include only rows where the correlation score is greater than 0.8
    filtered_correlation = correlation_table[correlation_table['CorrelationScore'] > 0.8]
    col1_names = filtered_correlation['Column1'].unique()
    try:
        features = features.drop(columns=col1_names)
    except KeyError as e:
        st.write(f"Error: Column(s) {e} not found in the features DataFrame.")

    features['image-src'] = img
    features['price'] = price
    return features