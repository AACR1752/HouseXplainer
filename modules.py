import streamlit as st
import altair as alt
import pandas as pd
import re
from nltk.stem import PorterStemmer
import random

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

words_to_drop = ["schedule", "attachments", "airport", "bedrooms_above_ground",
                 'bathrooms_detail', 'sewer', 'topography',
                    "seller", "garage", "frontage", "microwave",
                    "other", "locati", "multi", "is", "building",
                    'laundry room', "Wine Cooler", "Greenbelt",
                    "negoti", "condition"]

def display_graph(top_feature_names, top_percentages):
    top_feature_names = [name.replace('_', ' ') for name in top_feature_names]

    # Streamlit bar chart

    # Plot using Streamlit's altair_chart
    source = pd.DataFrame({"Feature": top_feature_names, "Contribution (%)": top_percentages})

    # Create the Altair bar chart
    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x=alt.X("Contribution (%):Q", title="Contribution (%)"),
            y=alt.Y("Feature:N", title=None, sort="-x",
                    axis=alt.Axis(labelFontSize=14, titleFontSize=16, labelLimit=300)),  # Sorting by Contribution
            color=alt.Color("Feature:N", legend=None),  # Optional: Color coding
        )
        .properties(width=800, height=800, title="Top 20 Features as per Importance")
    )

    # Display in Streamlit
    st.altair_chart(bars, use_container_width=True)

def display_df(results):
    results_df = pd.DataFrame(results, columns=['MSE','RMSE', 'R-squared'])
    st.dataframe(results_df)

# Define a function to drop columns containing any of the specified words (case insensitive)
def drop_columns_containing_words(df, words):
    cols_to_drop = [col for col in df.columns if any(word.lower() in col.lower() for word in words)]
    df = df.drop(columns=cols_to_drop)
    return df

def should_drop(feature_name, words):
        return any(word.lower() in feature_name.lower() for word in words)

# Function to remove suffixes from column names
def remove_suffixes(col_name, suffixes):
    for suffix in suffixes:
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)-1]
    return col_name

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
   