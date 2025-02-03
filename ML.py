# %% [markdown]
# <a href="https://colab.research.google.com/github/SunithAreng/fydp_team14/blob/sunith/Machine_Learning_Pipeline_EDA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Import Libraries and CSV file

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

# %%
houses = pd.read_csv("house_sigma_raw_exploded.csv")

# %%
# prompt: print all the column names one by one
houses.dropna(axis=1, how='all', inplace=True)

#drop rows where listed is missing
houses.dropna(subset=['listed'], inplace=True)

# %%
print(len(houses))
print(len(houses.columns))

# %%
# Count non-NaN values in each column
value_counts = houses.count()

# Identify columns with less than 10 non-NaN values
columns_to_drop = value_counts[value_counts < 10].index

# Drop the identified columns
houses.drop(columns=columns_to_drop, inplace=True)

# Display info to confirm changes
houses.info()

# %%
# prompt: separate out the columns which are numbers and the ones which are object (string)

numeric_cols = houses.select_dtypes(include=np.number).columns
object_cols = houses.select_dtypes(include=object).columns

# %%
numeric_features = ['rooms', 'bedrooms', 'bedrooms_above_ground',
                'bedrooms_below_ground', 'bathrooms', '2_piece_bathrooms',
                '3_piece_bathrooms', '4_piece_bathrooms', 'garage',
                'frontage_length', 'depth', 'fireplace_total']

# %% [markdown]
# numeric_features convert numeric_features in the houses df into a float type

# %%
for col in numeric_features:
    houses[col] = pd.to_numeric(houses[col], errors='coerce')
    houses[col] = houses[col].astype(float)

# %% [markdown]
# create a new column called "house_year", where you will take the "year_built" and if "year_built" is none then take building_age
# 

# %%
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

# %%
houses['house_year'] = houses['year_built'].fillna(houses['building_age'])
houses['house_age'] = houses['house_year'].apply(calculate_house_age)

# %%
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


# %% [markdown]
# fill na in house_age with the average of the house_age column

# %%
# Calculate the average house age, excluding NaN values
average_house_age = houses['house_age'].mean()

# Fill NaN values in 'house_age' with the average house age
houses['house_age'].fillna(average_house_age, inplace=True)

# Verify the changes (optional)
print(houses['house_age'].isnull().sum()) # Should print 0

# %% [markdown]
# ## Create ML Dataframes

# %%
# Create a separate DataFrame with numeric columns and 'listing_id'
ml_houses = houses.select_dtypes(include=np.number)
ml_houses['listing_id'] = houses['listing_id']
ml_houses['price'] = houses['listed']
ml_houses['listing'] = houses['listing']

# Convert 'price' column to numeric, handling '$' and ','
ml_houses['price'] = ml_houses['price'].astype(str).str.replace(r'[$,]', '', regex=True)
ml_houses['price'] = pd.to_numeric(ml_houses['price'], errors='coerce')

# %%
# place listing_id as the first column in the df

# Get the current column order
cols = ml_houses.columns.tolist()

# Move 'listing_id' to the first position
cols.insert(0, cols.pop(cols.index('listing_id')))

# Reorder the DataFrame
ml_houses = ml_houses.reindex(columns=cols)

# %%
# Reset the index of the DataFrame
ml_houses = ml_houses.reset_index(drop=True)
ml_houses.head(5)

# %% [markdown]
# ## Feature Engineering
# We are going to columns which we think has relevant features and un-nest them if we can do a one-hot encoding. (Currently this is the best method, but we can work on this).

# %%
print(object_cols)

# %%
# prompt: show all the unique values in houses['lot_features']

# print(houses['appliances_included'].unique())

# %%
# quick plot and visualize a column that you would like to encode for its feasibility

column_name = 'topography'

plt.figure(figsize=(12, 6))
sns.countplot(data=houses, x=column_name)
plt.xticks(rotation=45, ha='right')
plt.title(f'Distribution of {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.show()

# %% [markdown]
# Based on the above EDA, we will now insert these into "columns_to_encode" array. These arrays will be spilt on commas in their values and will be one-hot encoded. The split exceptions are just labels which won't be split but will be encoded since each record is repeated (finite classes).  

# %% [markdown]
# ## Encoding for Training

# %%
columns_to_encode = ['architecture_style','property_type',
                     'driveway_parking', 'frontage_type',
                     'sewer', 'bathrooms_detail', 'lot_features',
                     'topography', 'exterior_feature', 'basement_type',
                     'roof', 'waterfront_features', 'appliances_included',
                     'laundry_features', 'topography',
                     ]
split_exceptions = ['bathrooms_detail',]

# %%
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

# %%
# TODO: Appliances Excluded has to be penalizing in giving value to the prices

# %%
for column in columns_to_encode:
    encoded_df = one_hot_encode_column(column)
    ml_houses = pd.concat([ml_houses, encoded_df], axis=1)

# %%
print(len(ml_houses))


# %%
import re
# Drop columns containing "none" case-insensitive
ml_houses = ml_houses.drop(columns=[col for col in ml_houses.columns if re.search(r"none", col, re.IGNORECASE)])

# Drop columns with '*' in their names
ml_houses = ml_houses.drop(columns=[col for col in ml_houses.columns if '*' in col])
ml_houses = ml_houses.fillna(0)
print(len(ml_houses.columns))

# %%
features = ml_houses.drop(columns=['listing_id', 'price', 'listing'])
price = ml_houses['price']
print(len(price))
print(len(features))

# %%
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

# %%
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

# %%
# prompt: 'kitchens', 'rooms', 'bathrooms' drop these three columns from features dataframe

# Drop 'kitchens', 'rooms', and 'bathrooms' columns if they exist
columns_to_drop = ['kitchens', 'rooms', 'bathrooms', 'bedrooms']
for col in columns_to_drop:
    if col in features.columns:
        features = features.drop(columns=[col])
    else:
        print(f"Warning: Column '{col}' not found in features DataFrame.")

# %%
features.info()

# %%
from sklearn.model_selection import train_test_split

# Set random state for reproducibility.
seed = 100
test_size = 0.2

# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=test_size, random_state=seed)

# %%
X_train.head()

# %%
# prompt: create a ml model with random forest and train on above sets and predict and evaluate

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=seed)  # You can adjust n_estimators
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")


# %%
# prompt: create a linear regression model and train on the above sets and predict

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# %%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# %%
# prompt: keep the predicted and price in one df and create a third column with the calculated error

# Create a DataFrame with predicted and actual prices
results_df = pd.DataFrame({'predicted_price': y_pred, 'actual_price': y_test})

# Calculate the error
results_df['error'] = results_df['predicted_price'] - results_df['actual_price']

# Display the DataFrame
results_df

# %%
feature_names = X_test.columns.tolist()
feature_weights = model.coef_
# for feature, weight in zip(feature_names, feature_weights):
#     print(f"{feature}: {weight}")


# %%
# prompt: get the coefficents for the weights in rf_model

feature_names = X_test.columns.tolist()
feature_weights = rf_model.feature_importances_
# for feature, weight in zip(feature_names, feature_weights):
#     print(f"{feature}: {weight}")


# %%
percentages = feature_weights * 100

# %% [markdown]
# The following applies only to linear regression

# %%
coefficients = model.coef_

# %%
# Convert coefficients to absolute values and normalize to percentages
absolute_coefficients = np.abs(coefficients)
percentages = (absolute_coefficients / np.sum(absolute_coefficients)) * 100

# %%
# Combine feature names and percentages, then sort by percentages in descending order
sorted_features = sorted(zip(feature_names, percentages), key=lambda x: x[1], reverse=True)

# Select the top 20 features
top_features = sorted_features[:20]
top_feature_names, top_percentages = zip(*top_features)


# %% [markdown]
# ## The following only shows overall for the market
# 
# This is why we are seeing frontage types several times
# 

# %%
# Visualize as a bar chart
plt.barh(top_feature_names, top_percentages, color='skyblue')
plt.xlabel("Contribution (%)")
plt.title("Top 20 Feature Contributions in Percentages")
plt.gca().invert_yaxis()  # Invert y-axis to show the highest contribution at the top
plt.show()

# %% [markdown]
# ## Implementing to predict

# %%
# prompt: take a single data point from test set and apply the model to give a prediction

# Assuming X_test and model are defined from the previous code

# Select a single data point from the test set (e.g., the first one)
single_data_point = X_test.iloc[[0]]

# Apply the model to the single data point
prediction = rf_model.predict(single_data_point)

print(f"Prediction for the selected data point: {prediction[0]}")


# %%
single_data_point

# %%
ml_houses.iloc[40]['listing']

# %%
y_test.iloc[[0]]


# %%
# Convert the single data point to an ndarray
single_data_point_array = single_data_point.values

# Maintain the order of columns
column_order = X_test.columns.tolist()

# %%
output = np.multiply(coefficients , single_data_point_array) # this is for linear regression

# %%
price_ndarray = ml_houses['price'].values

# %%
# Convert X_train to a NumPy ndarray
X_train_ndarray = X_train.values

# %%
# Calculate standard deviations
std_y = np.std(price_ndarray)
std_X = np.std(X_train_ndarray, axis=0)

importances = rf_model.feature_importances_

# Convert feature importances to weights
weights = importances * (std_y / std_X)

# print("Feature importances:", importances)
# print("Weights (scaled):", weights)

# %%
weights

# %%
output = np.multiply(weights , single_data_point_array) # this is for random forest

# %%
# prompt: in the output variable above cell replace na with 0

output = np.nan_to_num(output)

# %%
output

# %%
# prompt: lets map column_order with output, to show column_name with the results of output as weights. Store it as a df

# Create a DataFrame to store the column order and output weights
df_output = pd.DataFrame({'column_name': column_order, 'weights': output[0]})

# Display the DataFrame
df_output


# %%
absolute_coefficients_y = np.abs(output[0])
percentages_y = (absolute_coefficients_y / np.sum(absolute_coefficients_y)) * 100
# percentages_y = rf_model.feature_importances_ * 100

# %%
# Combine feature names and percentages, then sort by percentages in descending order
sorted_features_y = sorted(zip(column_order, percentages_y), key=lambda x: x[1], reverse=True)

# Select the top 20 features
top_features_y = sorted_features_y[:20]
top_feature_names_y, top_percentages_y = zip(*top_features_y)

# %%
# prompt: create a dataframe with top_feature_names_y, top_percentages_y

# Create a DataFrame with top_feature_names_y and top_percentages_y
top_features_df = pd.DataFrame({'feature': top_feature_names_y, 'percentage': top_percentages_y})
top_features_df


# %%
# Visualize as a bar chart
plt.barh(top_feature_names_y, top_percentages_y, color='skyblue')
plt.xlabel("Contribution (%)")
plt.title("Top 20 Feature Contributions in Percentages")
plt.gca().invert_yaxis()  # Invert y-axis to show the highest contribution at the top
plt.show()


