import streamlit as st
import numpy as np
import pandas as pd
import modules as md
import model_training
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Property Comparison",
    page_icon="🔍",
    layout="wide"
)

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
        except:
            st.write("No available listings with current selection for Property 1!")
            
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
        except:
            st.write("No available listings with current selection for Property 2!")
    
    # Compare button
    compare = st.button("Compare Properties", use_container_width=True)
    
    if compare:
        try:
            # Get predictions
            prediction_1 = model.predict(data_point_1)[0]
            prediction_2 = model.predict(data_point_2)[0]
            
            actual_1 = y_test.iloc[index_1]
            actual_2 = y_test.iloc[index_2]
            
            # Display property images and predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Property 1: {property_1}")
                st.image(joined_df.loc[index_1, 'image-src'])
                
                # Property 1 details
                st.metric("Predicted Price", f"${round(prediction_1):,}")
                st.metric("Actual Price", f"${round(actual_1):,}")
                
                # Property details
                prop_details = {
                    "Bedrooms": joined_df.loc[index_1, 'bedrooms'],
                    "Bathrooms": joined_df.loc[index_1, 'bathrooms'],
                    "Property Type": joined_df.loc[index_1, 'property_type'],
                    "Neighbourhood": joined_df.loc[index_1, 'neighbourhood']
                }
                
                st.write("Property Details:")
                st.json(prop_details)
                
            with col2:
                st.subheader(f"Property 2: {property_2}")
                st.image(joined_df.loc[index_2, 'image-src'])
                
                # Property 2 details
                st.metric("Predicted Price", f"${round(prediction_2):,}")
                st.metric("Actual Price", f"${round(actual_2):,}")
                
                # Property details
                prop_details = {
                    "Bedrooms": joined_df.loc[index_2, 'bedrooms'],
                    "Bathrooms": joined_df.loc[index_2, 'bathrooms'],
                    "Property Type": joined_df.loc[index_2, 'property_type'],
                    "Neighbourhood": joined_df.loc[index_2, 'neighbourhood']
                }
                
                st.write("Property Details:")
                st.json(prop_details)
            
            # Feature comparison section
            st.markdown("---")
            st.subheader("Feature Comparison")
            
            # Rename the column in X_test for display
            display_data_1 = data_point_1.copy()
            display_data_2 = data_point_2.copy()
            
            if 'log_distance_to_nearest_school' in display_data_1.columns:
                display_data_1 = display_data_1.rename(columns={'log_distance_to_nearest_school': 'School Proximity'})
                display_data_2 = display_data_2.rename(columns={'log_distance_to_nearest_school': 'School Proximity'})
            
            # Remove suffixes for cleaner display
            suffixes_to_remove = ['driveway_parking', 'basement_type', 'lot_features', 
                                  'exterior_feature', 'waterfront_features', 'appliances_included']
            
            display_data_1.columns = [md.remove_suffixes(col, suffixes_to_remove) for col in display_data_1.columns]
            display_data_2.columns = [md.remove_suffixes(col, suffixes_to_remove) for col in display_data_2.columns]
            
            # Get model feature importances
            if "Random Forest" == st.session_state["model_choice"]:
                import shap
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
                feature_data = []
                
                for feature in unique_top_features:
                    if feature in display_data_1.columns and feature in display_data_2.columns:
                        feature_data.append({
                            "Feature": feature.replace('_', ' '),
                            "Property 1": float(display_data_1[feature].values[0]),
                            "Property 2": float(display_data_2[feature].values[0])
                        })
                
                feature_df = pd.DataFrame(feature_data)
                
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
                
                # Map view of both properties
                st.subheader("Location Comparison")
                
                map_data = pd.DataFrame({
                    'listing': [property_1, property_2],
                    'latitude': [joined_df.loc[index_1, 'latitude'], joined_df.loc[index_2, 'latitude']],
                    'longitude': [joined_df.loc[index_1, 'longitude'], joined_df.loc[index_2, 'longitude']],
                    'color': [10, 20]  # Different colors
                })
                
                fig = px.scatter_mapbox(map_data, 
                                       lat="latitude", 
                                       lon="longitude", 
                                       hover_name="listing",
                                       color="color",
                                       zoom=10,
                                       height=500)
                
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error comparing properties: {e}")
            st.error("Please make sure you've selected valid properties for comparison.")
else:
    st.error("No trained model or test data found! Please train the model first.")