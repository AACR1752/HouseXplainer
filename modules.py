import streamlit as st
import altair as alt
import pandas as pd
import re
from nltk.stem import PorterStemmer
import random
import re
from collections import defaultdict

words_to_drop = ["schedule", "attachments", "airport", "bedrooms_above_ground",
                 'bathrooms_detail', 'sewer', 'topography',
                    "seller", "frontage", "microwave", 'garage',
                    "other", "locati", "multi", "is", "building",
                    "negoti", "condition"]

# def manual_adjustments():
#     return {"log_distance_to_nearest_school": "Proximity to School"}

# Function to remove suffixes from column names
def remove_suffixes(col_name):
    suffixes = [
                'basement_type', 'lot_features', 'exterior_feature',
                'waterfront_features', 'appliances_included']
    for suffix in suffixes:
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)-1]
    return col_name

def display_graph(top_feature_names, top_percentages):
    top_feature_names = [name.replace('_', ' ') for name in top_feature_names]
    source = pd.DataFrame({"Feature": top_feature_names, "Contribution (%)": top_percentages})

    # Create the Altair bar chart
    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x=alt.X("Contribution (%):Q", title="Contribution (%)"),
            y=alt.Y("Feature:N", title=None, sort="-x",
                    axis=alt.Axis(labelFontSize=18, titleFontSize=18, labelLimit=300)),  # Sorting by Contribution
            color=alt.Color("Feature:N", legend=None),  # Optional: Color coding
        )
        .properties(width=800, height=800, title="Top 20 Features as per Importance")
    )

    # Display in Streamlit
    st.altair_chart(bars, use_container_width=True)

def display_df(results):
    results_df = pd.DataFrame(results, columns=['MSE','RMSE', 'R-squared', 'median error (%)'])
    st.dataframe(results_df)

# Define a function to drop columns containing any of the specified words (case insensitive)
def drop_columns_containing_words(df, words):
    cols_to_drop = [col for col in df.columns if any(word.lower() in col.lower() for word in words)]
    df = df.drop(columns=cols_to_drop)
    return df

def should_drop(feature_name, words):
        return any(word.lower() in feature_name.lower() for word in words)

def remove_overlapping_features(features):
    # Convert all features to lowercase for case-insensitive comparison
    features_lower = [feature.lower() for feature in features]
    # Sort features by length in descending order to prioritize longer phrases
    features_lower = sorted(features_lower, key=len, reverse=True)
    unique_features = []

    for feature in features_lower:
        # Check if the feature is already part of any existing unique feature
        if not any(feature in uf for uf in unique_features):
            unique_features.append(feature)

    # Map back to original case-sensitive features
    return [feature for feature in features if feature.lower() in unique_features]

## This is where we merge

def clean_column_name(column):
    """Removes extra spaces, special characters, and extracts first three words for grouping.
    Leaves specified columns (like 'image-src') untouched."""
    
    # Normalize for grouping logic (but don't modify output format)
    normalized_col = column.strip()
    # normalized_col = re.sub(r'[^a-zA-Z0-9\s]', '', normalized_col)  # Remove punctuation, commenting works but no split
    words = re.split(r'\s+', normalized_col)  # Split by whitespace
    group_key = ' '.join(words[:4]).lower()  # Take first 3 words as key

    return group_key if column not in {"image-src"} else column

def group_columns(df):
    """Groups similar columns dynamically based on first three words and merges them."""
    grouped_columns = defaultdict(list)

    # Identify grouped column names
    for col in df.columns:
        key = clean_column_name(col)
        grouped_columns[key].append(col)

    # Create a new DataFrame with merged columns
    new_features = pd.DataFrame()
    
    for key, cols in grouped_columns.items():
        if len(cols) > 1:
            # If multiple columns match the same group, sum them
            new_features[key] = df[cols].sum(axis=1)
        else:
            # If only one column matches, keep it as is
            new_features[key] = df[cols[0]]

    return new_features


# Still caching school
@st.cache_data
def render_school():
    school_df = pd.read_csv('data/good_data/schools.csv')
    return school_df

@st.cache_data
def render_amenities():
    amenities = pd.read_csv('data/good_data/amenities.csv')
    return amenities

@st.cache_data
def render_features():
    features = pd.read_csv('data/features.csv')
    return features

@st.cache_data
def get_amenities():
    amenities = pd.read_csv('data/amenity_dict.csv')
    return amenities

def highlight_keywords(text, feature_df):
    """Highlights keywords in text based on DataFrame features with stemming and color-coded highlights."""
    highlighted_text = text
    found_features = []
    stemmer = PorterStemmer()

    for feature, feature_type in feature_df[['feature', 'type']].drop_duplicates().values:
        stemmed_feature = stemmer.stem(feature)

        def highlight_match(match):
            """Highlights matched words with color based on type."""
            if feature_type == 'exterior':
                color = 'lightblue'
            elif feature_type == 'interior':
                color = 'yellow'
            else:
                color = 'lightgreen'  # Default color if type is unknown

            return f'<span style="background-color: {color};">{match.group(0)}</span>'

        # Check for exact matches and stemmed matches
        for word in re.findall(r'\b\w+\b', text, re.IGNORECASE):
            stemmed_word = stemmer.stem(word)
            if stemmed_word == stemmed_feature:
                highlighted_text = re.sub(
                    r'\b' + re.escape(word) + r'\b',
                    highlight_match,
                    highlighted_text,
                    flags=re.IGNORECASE,
                )
                found_features.append(feature)
                break #prevent double highlighting if the original word and stemmed word exist.
            elif re.search(r'\b' + re.escape(feature) + r'\b', text, re.IGNORECASE):
                highlighted_text = re.sub(
                    r'\b' + re.escape(feature) + r'\b',
                    highlight_match,
                    highlighted_text,
                    flags=re.IGNORECASE,
                )
                found_features.append(feature)
                break #prevent double highlighting if the original word and stemmed word exist.

    return highlighted_text, list(set(found_features)) #remove duplicates from found_features.

def process_amenities(amenities, amenity_objectids):
    """
    Processes amenities data, randomly selects 10 object IDs,
    creates a subset of the amenities DataFrame, and prints names.
    """
    amenity_objectids = set(amenity_objectids)
    
    if len(amenity_objectids) <= 10:
        selected_objectids = list(amenity_objectids) # Convert back to list for random.sample
    else:
        random.seed(42)
        selected_objectids = random.sample(list(amenity_objectids), 10) # Convert back to list for random.sample

    subset_amenities = amenities[amenities['type_objectid'].astype(str).isin(selected_objectids)]

    return list(set(subset_amenities['name'].tolist())) #return unique names using set


def initialize_shared_state():
    if "styles" not in st.session_state:
        st.session_state["styles"] = {
            "nav": {
                "background-color": "#8BC34A",
                "justify-content": "left",
            },
            "div": {
                "max-width": "32rem",
            },
            "span": {
                "border-radius": "0.5rem",
                "color": "white",
                "margin": "0 0.125rem",
                "padding": "0.4375rem 0.625rem",
            },
            "active": {
                "background-color": "rgba(255, 255, 255, 0.25)",
            },
            "hover": {
                "background-color": "rgba(255, 255, 255, 0.35)",
            },
        }
    if "pgs" not in st.session_state:
        st.session_state["pgs"] = ["Home", "Explainer", "Compare", "FAQ", "Learn More"]

def apply_sidebar_minimization():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
