import streamlit as st
import modules as md
import pandas as pd
import model_training
import json
import pydeck as pdk
import os
from streamlit_navigation_bar import st_navbar
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Predict House Prices",
    page_icon="picture/HE_icon_B.png",
    layout="wide"
)

page_name = "Learn More"
md.initialize_shared_state()
page = st_navbar(
    st.session_state["pgs"], 
    styles=st.session_state["styles"], 
    logo_path="./picture/HE_icon_W.svg",
    options={"show_sidebar": False, 
             "hide_nav":True}, 
    selected=page_name)
md.apply_sidebar_minimization()
if page != "Home" and page!=page_name and page != 'Learn More':
    st.switch_page(f"./pages/{page}.py")
elif page == 'Learn More' and page != page_name:
    st.switch_page(f"./pages/learn_more.py")
elif page == "Home" and page != page_name:
    st.switch_page(f"./Home.py")

st.title("Learn about the market")

# Sidebar model selection
# model_choice = st.sidebar.selectbox("Select Model", ["Ridge Regression", "Random Forest"])
model_choice = "Random Forest" #We are fixing on random as the winner!

precheck = ["trained_model", "evaluation"]

if "trained_model" in st.session_state and model_choice == st.session_state["model_choice"]:
    model = st.session_state["trained_model"]
    st.subheader("Model Evaluation")
    md.display_df(st.session_state["evaluation"])
    top_feature_names = st.session_state["top_feature_names"] 
    top_percentages = st.session_state["top_percentages"]
    md.display_graph(top_feature_names, top_percentages)
else:
    # model_training.main(model_choice=model_choice)
    st.error("No trained model or test data found! Please train the model first.")

# Neighborhood map section
st.subheader("Neighborhood Boundaries")


current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)  
default_geojson_path = os.path.join(parent_dir, "data", "areas", "neighbourhoods.geojson")

try:
    
    with open(default_geojson_path, 'r') as f:
        geojson_data = json.load(f)
        
    center_lat, center_lon = 43.4747193, -80.5559122  # Defined Center Columbia Lake
    
    try:
        # Try to find a better center from the GeoJSON data
        if geojson_data.get("features"):
            # Find average of all feature coordinates
            all_lats = []
            all_lons = []
            
            for feature in geojson_data["features"]:
                if feature.get("geometry", {}).get("type") == "Polygon":
                    coords = feature["geometry"]["coordinates"][0]
                    for coord in coords:
                        all_lons.append(coord[0])
                        all_lats.append(coord[1])
                elif feature.get("geometry", {}).get("type") == "MultiPolygon":
                    for polygon in feature["geometry"]["coordinates"]:
                        coords = polygon[0]  # First ring of coordinates
                        for coord in coords:
                            all_lons.append(coord[0])
                            all_lats.append(coord[1])
            
            if all_lats and all_lons:
                center_lat = sum(all_lats) / len(all_lats)
                center_lon = sum(all_lons) / len(all_lons)
    except Exception:
        pass  # Use default center if something goes wrong

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=12,
        pitch=0,
    )

    # Preprocess GeoJSON data to add a tooltip property
    for feature in geojson_data["features"]:
        props = feature.get("properties", {})
        name = props.get("NAME", "Unknown")
        type_ = props.get("TYPE", "Unknown")
        props["tooltip"] = f"Name: {name}\nType: {type_}"

    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_data,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color=[255, 0, 0, 60], 
        get_line_color=[0, 0, 0, 255],
        get_line_width=2,
        pickable=True,
        getTooltip="tooltip"
    )

    deck = pdk.Deck(
        layers=[geojson_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/streets-v12",
        # map_style="mapbox://styles/mapbox/light-v10",
        tooltip={"text": "{tooltip}"}
    )
    
    st.pydeck_chart(deck)
    
    joined_df = pd.DataFrame(st.session_state["joined_df_values"], 
                             columns=st.session_state["joined_df_columns"], 
                             index=st.session_state["joined_df_index"])
    
    if 'neighbourhood' in joined_df.columns:
            neighborhood_counts = joined_df['neighbourhood'].value_counts().sort_index()

    count_df = pd.DataFrame({
                        'Neighborhood': neighborhood_counts.index,
                        'Number of Houses': neighborhood_counts.values
                    })

    count_df = count_df.sort_values(by='Number of Houses', ascending=False)
    
    # TODO: Hassan's Reporting

    st.subheader("Stats for Nerds - Model In-depth Analysis")

    with st.expander("Neighborhood Statistics"):
        if geojson_data.get("features"):
            num_neighborhoods = len(geojson_data["features"])
            st.write(f"Total neighborhoods: {num_neighborhoods}")

            st.dataframe(
                        count_df,
                        column_config={
                            "Neighborhood": st.column_config.TextColumn("Neighborhood"),
                            "Number of Houses": st.column_config.NumberColumn("Number of Houses", format="%d")
                        },
                        hide_index=True
                    )
            
            # neighborhood names
            neighborhood_names = []
            for feature in geojson_data["features"]:
                props = feature.get("properties", {})
                name = props.get("name") or props.get("neighborhood") or props.get("NEIGHBORHOOD")
                if name:
                    neighborhood_names.append(name)
            
            if neighborhood_names:
                st.write("Neighborhoods:")
                st.write(", ".join(sorted(neighborhood_names)))
except FileNotFoundError:
    st.error(f"File not found: {default_geojson_path}")
    st.info("The GeoJSON file was not found. Please ensure the file exists at the specified path.")
    
    st.write(f"Looking for file at: {default_geojson_path}")
    st.write(f"Current working directory: {os.getcwd()}")
    
except json.JSONDecodeError:
    st.error(f"Invalid GeoJSON format in file: {default_geojson_path}")
except Exception as e:
    st.error(f"Error loading or displaying the map: {e}")

with st.expander("Model Comparison"):
    evaluation_df = pd.read_csv("data/statsfornerds/results.csv")
    st.dataframe(evaluation_df)

with st.expander("Model Error Distribution Comparison"):
    all_percentage_errors = pd.read_csv("data/statsfornerds/percentage_errors.csv")
    # st.dataframe(all_percentage_errors)

    fig = go.Figure()

    # Group the DataFrame by the 'Model' column
    if 'Model' in all_percentage_errors.columns:
        grouped = all_percentage_errors.groupby('Model')

        for model_name, model_df in grouped:
            fig.add_trace(go.Box(
                y=model_df['Percentage Error'],
                name=model_name,
                boxpoints='outliers',  # Only show outliers
            ))

        fig.update_layout(
            title_text="Error Distribution Comparison for All Models",
            title_x=0.375,
            xaxis_title='Model',
            yaxis_title='Percentage Error',
            showlegend=False
        )
        st.plotly_chart(fig)
    else:
        st.error("The 'Model' column is missing in the percentage errors data.")