import re
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, BallTree
import os

def combine_dataframes(path):
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

  combined_df = pd.concat(all_dfs, ignore_index=True)
  combined_df.columns = map(str.lower, combined_df.columns)
  combined_df = combined_df.drop_duplicates(subset=['listing_id'])
  
  return combined_df

def transform_address(address):
  """
  Transforms addresses in the following formats:
  - 'Unit A - 586 Mount Anne Drive' to '586A Mount Anne Drive'
  - '431-A Bairstow Crescent' to '431A Bairstow Crescent'
  - '424 a Tealby Cres' to '424a Tealby Cres'
  """

  if re.search(r'pinegrove', address, re.IGNORECASE):  # Check if 'pinegrove' exists
    address = re.sub(r'pinegrove', 'pine grove', address, flags=re.IGNORECASE)

  match1 = re.match(r"unit\s+([a-z])\s*-\s*(\d+)\s*(.*)", address, re.IGNORECASE)
  match2 = re.match(r"(\d+)\s*-\s*([a-z])\s*(.*)", address, re.IGNORECASE)
  match3 = re.match(r"(\d+)\s+([a-z])\s+(.*)", address, re.IGNORECASE)  # New pattern
  match4 = re.match(r"Unit\s+(\d+)\s*-\s*(\d+\s+.*)", address, re.IGNORECASE)

  if match1:
    return f"{match1.group(2)}{match1.group(1).upper()} {match1.group(3)}"
  elif match2:
    return f"{match2.group(1)}{match2.group(2).upper()} {match2.group(3)}"
  elif match3:
    return f"{match3.group(1)}{match3.group(2).upper()} {match3.group(3)}"  # New transformation
  elif match4:
    return f"{match4.group(2)} {match4.group(1)}"
  else:
    return address
  
def process_housing(df_house_sigma, output):
  # Convert the columns to lowercase
    df_house_sigma['address_lower'] = df_house_sigma['address'].str.lower().str.strip()
    output['civic_addr_lower'] = output['civic_addr'].str.lower().str.strip()

    # replace drive with dr and street with st on civic_addr_lower
    output['civic_addr_lower'] = output['civic_addr_lower'].str.replace('drive', 'dr')
    output['civic_addr_lower'] = output['civic_addr_lower'].str.replace('street', 'st')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('drive', 'dr')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('street', 'st')

    # only in df_house_sigma, crescent -> cres, avenue -> ave, boulevard -> blvd, road -> rd, place -> pl, court -> crt, circle -> cir
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('crescent', 'cres')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('avenue', 'ave')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('boulevard', 'blvd')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('road', 'rd')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('place', 'pl')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('court', 'crt')
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.replace('circle', 'cir')

    # Working on some edge cases
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].apply(transform_address)
    df_house_sigma['address_lower'] = df_house_sigma['address_lower'].str.lower().str.strip()

    # Perform the merge using the lowercase columns
    merged_df = pd.merge(df_house_sigma, output, left_on='address_lower', right_on='civic_addr_lower', how='left')

    # Select the specified columns
    result_df = merged_df[['listing_id', 'address', 'civic_addr', 'latitude', 'longitude', 'geometry', 'neighbourhood']]
    
    return result_df

def predict_missing_neighbourhoods(result_df):
    missing_neighborhoods = result_df[result_df['neighbourhood'].isna()]
    existing_neighborhoods = result_df.dropna(subset=['neighbourhood'])

    # Prepare the data for KNN
    X = existing_neighborhoods[['latitude', 'longitude']]  # Features
    y = existing_neighborhoods['neighbourhood']  # Target variable

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
    knn.fit(X, y)

    # Predict neighborhoods for missing values
    missing_neighborhoods_x = missing_neighborhoods[missing_neighborhoods['latitude'].notnull()]
    missing_X = missing_neighborhoods_x[['latitude', 'longitude']]
    predicted_neighborhoods = knn.predict(missing_X)

    # Add the predicted neighborhoods to the missing_neighborhoods DataFrame
    # WARNING: A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value instead
    missing_neighborhoods_x['neighbourhood'] = predicted_neighborhoods
    final_filled_df = pd.concat([existing_neighborhoods, missing_neighborhoods_x])
  
    return final_filled_df

def add_school_details(final_filled_df, df_schools):
    # Convert coordinates to NumPy arrays for faster processing
    house_coords = final_filled_df[['latitude', 'longitude']].to_numpy()
    school_coords = df_schools[['latitude', 'longitude']].to_numpy()

    # Fit NearestNeighbors model on school coordinates
    nn = NearestNeighbors(n_neighbors=1, metric='haversine')  # Haversine metric calculates geodesic distance
    nn.fit(np.radians(school_coords))  # Convert degrees to radians for Haversine metric

    # Find nearest school for each house
    distances, indices = nn.kneighbors(np.radians(house_coords))

    # Convert distances from radians to meters (Earth radius = 6371000 meters)
    distances = distances.flatten() * 6371000

    # Add distance column to housing DataFrame
    final_filled_df['distance_to_nearest_school'] = distances
    final_filled_df['log_distance_to_nearest_school'] = np.log1p(distances)
    final_filled_df['nearest_school'] = df_schools.iloc[indices.flatten()]['school name'].values

    # Map nearest school type to each house
    final_filled_df['nearest_school_type'] = df_schools.iloc[indices.flatten()]['school_type_label'].values
    
    return final_filled_df


def add_amenities_details(final_filled_df, amenities):
   # Convert coordinates to radians (required by BallTree for Haversine metric)
    house_coords = np.radians(final_filled_df[['latitude', 'longitude']].to_numpy())
    amenity_coords = np.radians(amenities[['latitude', 'longitude']].to_numpy())

    # Fit BallTree model on amenities
    tree = BallTree(amenity_coords, metric='haversine')  # Haversine metric for spherical distance

    # Query amenities within 1 km (convert to radians: 1 km / Earth radius in meters)
    radius = 1000 / 6371000  # 6371000 meters is Earth's radius
    indices = tree.query_radius(house_coords, r=radius)

    # Initialize empty lists for each output column
    amenity_counts = []
    amenity_types = []
    amenity_objectids = []
    amenity_type_codes = []

    # Iterate over the list of indices
    for idx in indices:
        nearby_amenities = amenities.iloc[idx] if len(idx) > 0 else pd.DataFrame()

        # Count number of amenities
        amenity_counts.append(len(nearby_amenities))

        # Get unique types, object IDs, and type codes
        amenity_types.append(','.join(nearby_amenities['type'].unique()) if not nearby_amenities.empty else '')
        amenity_objectids.append(','.join(map(str, nearby_amenities['type_objectid'].unique())) if not nearby_amenities.empty else '')
        amenity_type_codes.append(','.join(map(str, nearby_amenities['type_code'].unique())) if not nearby_amenities.empty else '')

    # Add the results to the final DataFrame
    final_filled_df['amenities_count_1km'] = amenity_counts
    final_filled_df['amenities_types_1km'] = amenity_types
    final_filled_df['amenities_objectids_1km'] = amenity_objectids
    final_filled_df['amenities_type_codes_1km'] = amenity_type_codes

    return final_filled_df