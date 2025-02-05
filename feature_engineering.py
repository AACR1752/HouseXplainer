from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import re

def calculate_house_age(year_string):
    if '-' in str(year_string):
        try:
            start, end = map(int, year_string.split('-'))
            return (start + end) / 2
        except ValueError:
            return None  # Handle cases where splitting fails
    elif str(year_string).isdigit() and len(str(year_string)) == 4:
        try:
            return 2025 - int(year_string)
        except ValueError:
            return None
    else:
        return None

def feature_refining(houses):
    # prompt: st.write all the column names one by one
    houses.dropna(axis=1, how='all', inplace=True)

    #drop rows where listed is missing
    houses.dropna(subset=['listed'], inplace=True)

    
    # Count non-NaN values in each column
    value_counts = houses.count()

    # Identify columns with less than 10 non-NaN values
    columns_to_drop = value_counts[value_counts < 10].index

    # Drop the identified columns
    houses.drop(columns=columns_to_drop, inplace=True)

    # numeric_cols = houses.select_dtypes(include=np.number).columns
    # object_cols = houses.select_dtypes(include=object).columns
    
    numeric_features = ['rooms', 'bedrooms', 'bedrooms_above_ground',
                    'bedrooms_below_ground', 'bathrooms', '2_piece_bathrooms',
                    '3_piece_bathrooms', '4_piece_bathrooms', 'garage',
                    'frontage_length', 'depth']

    for col in numeric_features:
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

    # Convert 'price' column to numeric, handling '$' and ','
    ml_houses['price'] = ml_houses['price'].astype(str).str.replace(r'[$,]', '', regex=True)
    ml_houses['price'] = pd.to_numeric(ml_houses['price'], errors='coerce')

    
    # place listing_id as the first column in the df

    # Get the current column order
    cols = ml_houses.columns.tolist()

    # Move 'listing_id' to the first position
    cols.insert(0, cols.pop(cols.index('listing_id')))

    # Reorder the DataFrame
    ml_houses = ml_houses.reindex(columns=cols)
    
    # Reset the index of the DataFrame
    ml_houses = ml_houses.reset_index(drop=True)

    
    columns_to_encode = ['architecture_style','property_type',
                        'driveway_parking', 'frontage_type',
                        'sewer', 
                        'bathrooms_detail', 'basement_type',
                        'lot_features',
                        'topography', 'exterior_feature', 
                        'roof', 'waterfront_features', 'appliances_included',
                        'laundry_features', 'topography',
                        ]
    split_exceptions = [
        'bathrooms_detail',
        ]

    
    def one_hot_encode_column(column):
        if column not in split_exceptions:
            houses[f'{column}_arr'] = houses[column].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        else:
            houses[f'{column}_arr'] = houses[column].apply(lambda x: [x] if isinstance(x, str) else [])

        mlb = MultiLabelBinarizer()
        encoded_data = mlb.fit_transform(houses[f'{column}_arr'])
        encoded_df = pd.DataFrame(encoded_data, columns=mlb.classes_)

        # Add column variable as suffix to encoded column names
        encoded_df = encoded_df.add_suffix(f'_{column}')

        return encoded_df

    
    # TODO: Appliances Excluded has to be penalizing in giving value to the prices

    
    for column in columns_to_encode:
        encoded_df = one_hot_encode_column(column)
        ml_houses = pd.concat([ml_houses, encoded_df], axis=1)

    # Drop columns containing "none" case-insensitive
    ml_houses = ml_houses.drop(columns=[col for col in ml_houses.columns if re.search(r"none", col, re.IGNORECASE)])

    # Drop columns with '*' in their names
    ml_houses = ml_houses.drop(columns=[col for col in ml_houses.columns if '*' in col])
    ml_houses = ml_houses.fillna(0)

    return ml_houses