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



#######


# prompt: print all the column names one by one
houses.dropna(axis=1, how='all', inplace=True)

#drop rows where listed is missing
houses.dropna(subset=['listed'], inplace=True)

for col in houses.columns:
    print(col)
print(len(houses))
print(len(houses.columns))

 
# Count non-NaN values in each column
value_counts = houses.count()

# Identify columns with less than 10 non-NaN values
columns_to_drop = value_counts[value_counts < 10].index

# Drop the identified columns
houses.drop(columns=columns_to_drop, inplace=True)

# Display info to confirm changes
houses.info()

 
# prompt: separate out the columns which are numbers and the ones which are object (string)

numeric_cols = houses.select_dtypes(include=np.number).columns
object_cols = houses.select_dtypes(include=object).columns

 
numeric_features = ['rooms', 'bedrooms', 'bedrooms_above_ground',
                'bedrooms_below_ground', 'bathrooms', '2_piece_bathrooms',
                '3_piece_bathrooms', '4_piece_bathrooms', 'garage',
                'frontage_length', 'depth', 'fireplace_total']

 
for col in numeric_features:
    houses[col] = pd.to_numeric(houses[col], errors='coerce')
    houses[col] = houses[col].astype(float)

 
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

 
houses['house_year'] = houses['year_built'].fillna(houses['building_age'])
houses['house_age'] = houses['house_year'].apply(calculate_house_age)

 
# prompt: generate a histogram with house_age and plot it show. do a count for the ones with none

import matplotlib.pyplot as plt

# Calculate the count of None values in 'house_age'
none_count = houses['house_age'].isnull().sum()
print(f"Number of None values in 'house_age': {none_count}")

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(houses['house_age'].dropna(), bins=30, edgecolor='black')  # Drop NaN values for plotting
plt.xlabel('House Age')
plt.ylabel('Frequency')
plt.title('Distribution of House Ages')
plt.show()


 
# Calculate the average house age, excluding NaN values
average_house_age = houses['house_age'].mean()

# Fill NaN values in 'house_age' with the average house age
houses['house_age'].fillna(average_house_age, inplace=True)

# Verify the changes (optional)
print(houses['house_age'].isnull().sum()) # Should print 0


# ## Create ML Dataframes

 
# Create a separate DataFrame with numeric columns and 'listing_id'
ml_houses = houses.select_dtypes(include=np.number)
ml_houses['listing_id'] = houses['listing_id']
ml_houses['price'] = houses['listed']
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
ml_houses.head(5)

 
print(object_cols)

 
# prompt: show all the unique values in houses['lot_features']

# print(houses['appliances_included'].unique())

 
# quick plot and visualize a column that you would like to encode for its feasibility

column_name = 'topography'

plt.figure(figsize=(12, 6))
sns.countplot(data=houses, x=column_name)
plt.xticks(rotation=45, ha='right')
plt.title(f'Distribution of {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.show()

 
columns_to_encode = ['architecture_style','property_type',
                     'driveway_parking', 'frontage_type',
                     'sewer', 'bathrooms_detail', 'lot_features',
                     'topography', 'exterior_feature', 'basement_type',
                     'roof', 'waterfront_features', 'appliances_included',
                     'laundry_features', 'topography',
                     ]
split_exceptions = ['bathrooms_detail',]

 
from sklearn.preprocessing import MultiLabelBinarizer
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

 
print(len(ml_houses))


 
import re
# Drop columns containing "none" case-insensitive
ml_houses = ml_houses.drop(columns=[col for col in ml_houses.columns if re.search(r"none", col, re.IGNORECASE)])

# Drop columns with '*' in their names
ml_houses = ml_houses.drop(columns=[col for col in ml_houses.columns if '*' in col])
ml_houses = ml_houses.fillna(0)
print(len(ml_houses.columns))

 
features = ml_houses.drop(columns=['listing_id', 'price', 'listing'])
price = ml_houses['price']
print(len(price))
print(len(features))

 
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
    print("Columns dropped successfully.")
except KeyError as e:
    print(f"Error: Column(s) {e} not found in the features DataFrame.")

# Print the resulting features DataFrame (Optional)
# print(features.head())


# prompt: 'kitchens', 'rooms', 'bathrooms' drop these three columns from features dataframe

# Drop 'kitchens', 'rooms', and 'bathrooms' columns if they exist
columns_to_drop = ['kitchens', 'rooms', 'bathrooms', 'bedrooms']
for col in columns_to_drop:
    if col in features.columns:
        features = features.drop(columns=[col])
    else:
        print(f"Warning: Column '{col}' not found in features DataFrame.")

 
features.info()





########    




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

# Drop rows with NaN in any column to keep features and price aligned
ml_houses = ml_houses.dropna()

features = ml_houses.drop(columns=['listing_id', 'price', 'listing'])
price = ml_houses['price']

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
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    feature_importance = np.abs(model.coef_)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader("Model Evaluation")
st.write(f"{model_choice} - MSE: {mse}, R2: {r2}")

# Feature Importance
feature_names = X_test.columns.tolist()
sorted_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
top_features = sorted_features[:20]
top_feature_names, top_percentages = zip(*top_features)

fig, ax = plt.subplots()
ax.barh(top_feature_names, top_percentages, color='skyblue')
ax.set_xlabel("Contribution (%)")
ax.set_title(f"Top 20 Feature Contributions - {model_choice}")
ax.invert_yaxis()
st.pyplot(fig)

# Single Data Point Prediction
single_data_point = X_test.iloc[[0]]
prediction = model.predict(single_data_point)
st.subheader("Single Data Point Prediction")
st.write(f"Predicted Price: {prediction[0]}")
