import streamlit as st
import numpy as np
import pandas as pd
import modules as md
import model_training
import plotly.express as px
import plotly.graph_objects as go
import shap
from plotly.subplots import make_subplots
from PIL import Image
import requests
from io import BytesIO
from streamlit_navigation_bar import st_navbar

# Set page configuration
st.set_page_config(
    page_title="Property Comparison",
    page_icon="picture/HE_icon_B.png",
    layout="wide"
)

page_name = "Compare"
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
elif page == 'Learn More':
    st.switch_page(f"./pages/learn_more.py")
elif page == "Home" and page != page_name:
    st.switch_page(f"./Home.py")

st.title("Property Comparison")
st.write("Compare different properties to understand value differences")

# Check if model is trained
if "trained_model" not in st.session_state:
    model_training.main(model_choice="Random Forest")

if "trained_model" in st.session_state:
    model = st.session_state["trained_model"]
    
    # Convert back to DataFrame
    X_test = pd.DataFrame(st.session_state["X_test_values"], 
                        columns=st.session_state["X_test_columns"], 
                        index=st.session_state["X_test_index"])
    
    joined_df = pd.DataFrame(st.session_state["joined_df_values"], 
                            columns=st.session_state["joined_df_columns"], 
                            index=st.session_state["joined_df_index"])
    
    y_test = pd.Series(st.session_state["y_test"], name="Price")  # Restores index
    
    # Filters for property selection
    st.subheader("Select Properties to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # First property selection
        neighbourhood_1 = st.selectbox("Select Neighbourhood for Property 1", 
                                      joined_df['neighbourhood'].unique().tolist(),
                                      key="neigh1")
        
        small_df_1 = joined_df[(joined_df['neighbourhood'] == neighbourhood_1)]
        property_type_1 = st.selectbox("Select Property Type for Property 1", 
                                     small_df_1['property_type'].unique().tolist(),
                                     key="prop_type1")
        
        filtered_df_1 = joined_df[
            (joined_df['neighbourhood'] == neighbourhood_1) &
            (joined_df['property_type'] == property_type_1)]
        
        try:
            property_1 = st.selectbox("Select Property 1", 
                                     filtered_df_1['listing'].tolist(),
                                     key="prop1")
            index_1 = filtered_df_1[filtered_df_1['listing'] == property_1].index.tolist()[0]
            data_point_1 = X_test.iloc[[index_1]]
            property_1_selected = True
        except:
            st.write("No available listings with current selection for Property 1!")
            property_1_selected = False
            
    with col2:
        # Second property selection
        neighbourhood_2 = st.selectbox("Select Neighbourhood for Property 2",
                                     joined_df['neighbourhood'].unique().tolist(),
                                     key="neigh2")
        
        small_df_2 = joined_df[(joined_df['neighbourhood'] == neighbourhood_2)]
        property_type_2 = st.selectbox("Select Property Type for Property 2",
                                     small_df_2['property_type'].unique().tolist(),
                                     key="prop_type2")
        
        filtered_df_2 = joined_df[
            (joined_df['neighbourhood'] == neighbourhood_2) &
            (joined_df['property_type'] == property_type_2)]
        
        try:
            property_2 = st.selectbox("Select Property 2",
                                     filtered_df_2['listing'].tolist(),
                                     key="prop2")
            index_2 = filtered_df_2[filtered_df_2['listing'] == property_2].index.tolist()[0]
            data_point_2 = X_test.iloc[[index_2]]
            property_2_selected = True
        except:
            st.write("No available listings with current selection for Property 2!")
            property_2_selected = False
    
        # Button styling with custom CSS
        st.markdown(
        """
        <style>
        div.stButton > button {
            padding: 0.5rem 2rem;
            display: block;
            margin: 0 auto;
            border: "black";
            border-radius: 5px;
            font-weight: bold;
            transition: box-shadow 0.3s ease;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Compare button
    compare = st.button("Compare Properties", use_container_width=False)


    def display_image_with_fixed_height(image_url, height):
        """Displays an image with a fixed height using HTML and CSS."""
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            img = Image.open(BytesIO(response.content))
            st.markdown(f"""
                <div style="text-align: center;"> 
                    <img src="{image_url}" style="height: {height}px; max-width: 100%; object-fit: contain;">
                </div>
            """, unsafe_allow_html=True)
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading image: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        
    if compare:
        # Check if both properties are selected
        if not property_1_selected or not property_2_selected:
            st.error("Please select valid properties for comparison!")
        # Check if the same property is selected twice
        elif property_1 == property_2 and neighbourhood_1 == neighbourhood_2 and property_type_1 == property_type_2:
            st.error("Please select different properties for comparison!")
        else:

            # Get predictions
            prediction_1 = model.predict(data_point_1)[0]
            prediction_2 = model.predict(data_point_2)[0]
            
            actual_1 = y_test.iloc[index_1]
            actual_2 = y_test.iloc[index_2]

            p1_col, p2_col = st.columns([1, 1])

            with p1_col:
                
                st.subheader(f"Property 1: {property_1}")
                
                image_container = st.container()
                with image_container:                
                    display_image_with_fixed_height(joined_df.loc[index_1, 'image-src'], 550)

            with p2_col:

                st.subheader(f"Property 2: {property_2}")

                image_container = st.container()
                with image_container:
                    display_image_with_fixed_height(joined_df.loc[index_2, 'image-src'], 550)

            # Property comparison section using native Streamlit components
            col1, col2 = st.columns(2, gap="large")

            with col1:                

                price_cols = st.columns(2)
                with price_cols[0]:
                    st.metric("Expectcted Market Value", f"${round(prediction_1):,}")
                

                with st.expander("Property Details", expanded=True):
                    detail_cols = st.columns(2)
                    
                    with detail_cols[0]:
                        st.markdown("**Bedrooms**")
                        st.markdown(f"{int(joined_df.loc[index_1, 'bedrooms'])}")
                        
                        st.markdown("**Property Type**")
                        st.markdown(f"{joined_df.loc[index_1, 'property_type']}")
                        
                        st.markdown("**Amenities within 1km**")
                        st.markdown(f"{joined_df.loc[index_1, 'amenities_count_1km']}")
                        
                    with detail_cols[1]:
                        st.markdown("**Bathrooms**")
                        st.markdown(f"{int(joined_df.loc[index_1, 'bathrooms'])}")

                        st.markdown("**Architectural Style**")
                        st.markdown(f"{joined_df.loc[index_1, 'architecture_style']}")

                        st.markdown("**Neighbourhood**")
                        st.markdown(f"{joined_df.loc[index_1, 'neighbourhood']}")

            with col2:
                
                price_cols = st.columns(2)
                with price_cols[0]:
                    st.metric("Expectcted Market Value", f"${round(prediction_2):,}")

                
                with st.expander("Property Details", expanded=True):
                    detail_cols = st.columns(2)
                    
                    with detail_cols[0]:
                        st.markdown("**Bedrooms**")
                        st.markdown(f"{int(joined_df.loc[index_2, 'bedrooms'])}")
                        
                        st.markdown("**Property Type**")
                        st.markdown(f"{joined_df.loc[index_2, 'property_type']}")
                        
                        st.markdown("**Amenities within 1km**")
                        st.markdown(f"{joined_df.loc[index_2, 'amenities_count_1km']}")
                        
                    with detail_cols[1]:
                        st.markdown("**Bathrooms**")
                        st.markdown(f"{int(joined_df.loc[index_2, 'bathrooms'])}")
                        
                        st.markdown("**Architectural Style**")
                        st.markdown(f"{joined_df.loc[index_2, 'architecture_style']}")
                        
                        st.markdown("**Neighbourhood**")
                        st.markdown(f"{joined_df.loc[index_2, 'neighbourhood']}")
            
            # Feature comparison section
            st.markdown("---")
            st.subheader("Feature Comparison")
            
            # Rename the column in X_test for display
            display_data_1 = data_point_1.copy()
            display_data_2 = data_point_2.copy()
            
            if 'log_distance_to_nearest_school' in display_data_1.columns:
                display_data_1 = display_data_1.rename(columns={'log_distance_to_nearest_school': 'School Proximity'})
                display_data_2 = display_data_2.rename(columns={'log_distance_to_nearest_school': 'School Proximity'})
            
            display_data_1.columns = [md.remove_suffixes(col) for col in display_data_1.columns]
            display_data_2.columns = [md.remove_suffixes(col) for col in display_data_2.columns]
            
            # Get model feature importances
            if "Random Forest" == st.session_state["model_choice"]:
                
                explainer = shap.TreeExplainer(model)
                shap_values_1 = explainer.shap_values(data_point_1)
                shap_values_2 = explainer.shap_values(data_point_2)
                
                # Calculate absolute SHAP values
                abs_shap_1 = np.abs(shap_values_1[0])
                abs_shap_2 = np.abs(shap_values_2[0])
                
                # Get top features by SHAP values
                feature_names = display_data_1.columns.tolist()
                
                # Combine features with their SHAP values
                feature_importance_1 = sorted(zip(feature_names, abs_shap_1), key=lambda x: x[1], reverse=True)
                feature_importance_2 = sorted(zip(feature_names, abs_shap_2), key=lambda x: x[1], reverse=True)
                
                # Filter out unwanted features
                filtered_importance_1 = [feat for feat in feature_importance_1 
                                        if not md.should_drop(feat[0], md.words_to_drop)]
                filtered_importance_2 = [feat for feat in feature_importance_2 
                                        if not md.should_drop(feat[0], md.words_to_drop)]
                
                # Get top 10 features for each property
                top_features_1 = filtered_importance_1[:10]
                top_features_2 = filtered_importance_2[:10]
                
                # Create a set of unique top features from both properties
                unique_top_features = set([f[0] for f in top_features_1]).union(set([f[0] for f in top_features_2]))
                
                # Create a bar chart to compare feature values
                processed_features = set()
                feature_data = []
                
                for feature in unique_top_features:
                    feature_display = feature.replace('_', ' ')
                    if feature_display.lower() in processed_features:
                        continue
                        
                    if feature in display_data_1.columns and feature in display_data_2.columns:
                        value_1 = float(display_data_1[feature].values.flatten()[0])
                        value_2 = float(display_data_2[feature].values.flatten()[0])
                        
                        # Only include if the feature has meaningful values (not near zero)
                        if abs(value_1) > 0.01 or abs(value_2) > 0.01:
                            feature_data.append({
                                "Feature": feature_display,
                                "Property 1": value_1,
                                "Property 2": value_2
                            })
                            processed_features.add(feature_display.lower())
                
                feature_df = pd.DataFrame(feature_data)
                # After creating feature_df
                feature_df['Total'] = feature_df['Property 1'] + feature_df['Property 2']
                feature_df['Difference'] = abs(feature_df['Property 1'] - feature_df['Property 2'])
                feature_df = feature_df.sort_values('Difference', ascending=False).drop(columns=['Total', 'Difference'])

                
                # Create comparison chart
                fig = px.bar(feature_df, x="Feature", y=["Property 1", "Property 2"], 
                            barmode="group", title="Feature Value Comparison",
                            labels={"value": "Feature Value", "variable": "Property"})
                
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
               
                # Create a price breakdown visualization
                st.subheader("Price Contribution Breakdown")
                
                # For Property 1
                top_5_features_1 = filtered_importance_1[:5]
                feature_names_1 = [f[0].replace('_', ' ') for f in top_5_features_1]
                shap_values_1_top5 = [f[1] for f in top_5_features_1]

                # For Property 2
                top_5_features_2 = filtered_importance_2[:5]
                feature_names_2 = [f[0].replace('_', ' ') for f in top_5_features_2]
                shap_values_2_top5 = [f[1] for f in top_5_features_2]

                # Create subplots with the correct specification for pie charts
                fig = make_subplots(rows=1, cols=2, 
                                specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                                subplot_titles=[f"Property 1: {property_1}", f"Property 2: {property_2}"])

                # Add pie charts
                fig.add_trace(
                    go.Pie(labels=feature_names_1, values=shap_values_1_top5, name="Property 1"),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Pie(labels=feature_names_2, values=shap_values_2_top5, name="Property 2"),
                    row=1, col=2
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Location Comparison")

                m1, m2 = st.columns(2)

                # Map for Property 1
                with m1:
                    st.subheader(property_1)
                    map_data_1 = pd.DataFrame({
                        'listing': [property_1],
                        'latitude': [joined_df.loc[index_1, 'latitude']],
                        'longitude': [joined_df.loc[index_1, 'longitude']]
                    })

                    fig1 = px.scatter_mapbox(map_data_1,
                                                lat="latitude",
                                                lon="longitude",
                                                hover_name="listing",
                                                zoom=16,
                                                height=400,
                                                color_discrete_sequence=["red"])

                    fig1.update_traces(marker=dict(size=18))
                    fig1.update_layout(mapbox_style="open-street-map")
                    fig1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

                    st.plotly_chart(fig1, use_container_width=True)

                # Map for Property 2
                with m2:
                    st.subheader(property_2)
                    map_data_2 = pd.DataFrame({
                        'listing': [property_2],
                        'latitude': [joined_df.loc[index_2, 'latitude']],
                        'longitude': [joined_df.loc[index_2, 'longitude']]
                    })

                    fig2 = px.scatter_mapbox(map_data_2,
                                                lat="latitude",
                                                lon="longitude",
                                                hover_name="listing",
                                                zoom=16,
                                                height=400,
                                                color_discrete_sequence=["red"])

                    fig2.update_traces(marker=dict(size=18))
                    fig2.update_layout(mapbox_style="open-street-map")
                    fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

                    st.plotly_chart(fig2, use_container_width=True)


else:
    st.error("No trained model or test data found! Please train the model first.")