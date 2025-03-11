import streamlit as st
import pandas as pd
import numpy as np
import shap
import modules as md
import plotly.graph_objects as go
import pydeck as pdk
import model_training

# Set the page configuration to wide mode
st.set_page_config(page_title="HouseXplainer - Home",
                   page_icon="🏠",
                   layout="wide")

# Load data
st.title("HouseXplainer - Xplain the whY behind a house")

model_choice = "Random Forest" #We are fixing on random as the winner!

if not "trained_model" in st.session_state:
    model_training.main(model_choice=model_choice)

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
    model_choice = st.session_state["model_choice"]

    # Create two columns with balanced ratio
    col1, col2 = st.columns([2, 3])  # Adjusted ratio to give more width to first column
    
    with col2:
        # Place the map in the second column
        try:
            st.subheader("Property Map")
            school_df = pd.read_csv('data/good_data/schools.csv')
            
            # Prepare data with icon column for houses
            filtered_df_with_icon = joined_df.copy()  # Use joined_df initially
            filtered_df_with_icon['icon_data'] = [{
                "url": "https://img.icons8.com/color/48/000000/home.png",
                "width": 128,
                "height": 128,
                "anchorY": 128
            } for _ in range(len(filtered_df_with_icon))]
            
            # Prepare data with icon column for schools
            school_df_with_icon = school_df.copy()
            school_df_with_icon['icon_data'] = [{
                "url": "https://img.icons8.com/color/48/000000/school.png",
                "width": 128,
                "height": 128,
                "anchorY": 128
            } for _ in range(len(school_df_with_icon))]
            
            # Define House Layer with icons - will be updated later
            house_layer = pdk.Layer(
                "IconLayer",
                data=filtered_df_with_icon,
                get_position=["longitude", "latitude"],
                get_icon="icon_data",
                get_size=4,
                size_scale=10,
                pickable=True,
            )
            
            # Define School Layer with icons
            school_layer = pdk.Layer(
                "IconLayer",
                data=school_df_with_icon,
                get_position=["longitude", "latitude"],
                get_icon="icon_data",
                get_size=4,
                size_scale=10,
                pickable=True,
            )
            
            # Add fallback layers in case the icons don't load
            house_fallback_layer = pdk.Layer(
                "ScatterplotLayer",
                data=joined_df,  # Use joined_df initially
                get_position=["longitude", "latitude"],
                get_radius=50,
                get_fill_color=[0, 0, 255, 180],  # Blue for houses
                pickable=True,
                opacity=0.8,
                visible=False,  # Only show if primary layer fails
            )
            
            school_fallback_layer = pdk.Layer(
                "ScatterplotLayer",
                data=school_df,
                get_position=["longitude", "latitude"],
                get_radius=80,
                get_fill_color=[255, 0, 0, 200],  # Red for schools
                pickable=True,
                opacity=0.9,
                visible=False,  # Only show if primary layer fails
            )
            
            # Initialize view with all data first
            view_state = pdk.ViewState(
                latitude=joined_df["latitude"].mean(),
                longitude=joined_df["longitude"].mean(),
                zoom=12,
                pitch=5,
            )
            
            # Display the Map with Mapbox Style
            r = pdk.Deck(
                layers=[house_layer, school_layer, house_fallback_layer, school_fallback_layer],
                initial_view_state=view_state,
                tooltip={"text": "{listing}\n{school_name}"},
                map_style="mapbox://styles/mapbox/satellite-streets-v12"
            )
            map_placeholder = st.pydeck_chart(r)
            
            # Add a small legend below the map
            legend_col1, legend_col2 = st.columns(2)
            with legend_col1:
                st.markdown("🏠 **Houses**")
            with legend_col2:
                st.markdown("🏫 **Schools**")
                
        except Exception as e:
            st.error(f"Map cannot be displayed: {e}")
    
    with col1:
        # Add vertical spacing to align with map
        st.write("")
        st.write("")
        st.write("")
        
        # Create a container with styling to center content
        with st.container():
            # Center the content with CSS
            st.markdown("""
                <style>
                    .centered-container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        padding: 20px;
                        background-color: #f8f9fa;
                        border-radius: 10px;
                        margin-top: 30px;
                    }
                    .selection-title {
                        text-align: center;
                        margin-bottom: 20px;
                    }
                </style>
                <div class="centered-container">
                    <h3 class="selection-title">Property Selection</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Place selectboxes in the first column with more space
            neighbourhood_name = st.selectbox("Select Neighbourhood", joined_df['neighbourhood'].unique().tolist())
            small_df = joined_df[(joined_df['neighbourhood'] == neighbourhood_name)]
            property_type_selection = st.selectbox("Select Property Type", small_df['property_type'].unique().tolist())
            
            # Filter data based on selections
            filtered_df = joined_df[
                (joined_df['neighbourhood'] == neighbourhood_name) &
                # (joined_df['bedrooms'] == bedroom_selection) &
                # (joined_df['bathrooms'] == bathroom_selection) &
                (joined_df['property_type'] == property_type_selection)]
            
            # Update map with filtered data
            try:
                # Update filtered data for house layer
                filtered_df_with_icon = filtered_df.copy()
                filtered_df_with_icon['icon_data'] = [{
                    "url": "https://img.icons8.com/color/48/000000/home.png",
                    "width": 128,
                    "height": 128,
                    "anchorY": 128
                } for _ in range(len(filtered_df_with_icon))]
                
                # Update layers with filtered data
                house_layer.data = filtered_df_with_icon
                house_fallback_layer.data = filtered_df
                
                # Update view state
                if not filtered_df.empty:
                    view_state.latitude = filtered_df["latitude"].mean()
                    view_state.longitude = filtered_df["longitude"].mean()
                    view_state.zoom = 14
                
                # Update the map
                r.initial_view_state = view_state
                r.layers[0].data = filtered_df_with_icon
                r.layers[2].data = filtered_df
                
                # Replace the map
                map_placeholder.pydeck_chart(r)
            except:
                # If update fails, continue with selection
                pass
            
            # Dropdown to select a value from X_test
            try: 
                datapoint = st.selectbox("Select House", filtered_df['listing'].tolist())
                
                # Get the index of the selected house
                index = filtered_df[filtered_df['listing'] == datapoint].index.tolist()
                single_data_point = X_test.iloc[[index[0]]]
                
                # Add some spacing
                st.write("")
                st.write("")
                
                # Add the Predict button in the first column with custom styling
                col1_but1, col1_but2, col1_but3 = st.columns([1, 2, 1])
                with col1_but2:
                    predict_button = st.button("Predict", use_container_width=True)
                
            except:
                st.error("There are no available listings with current selection!")
                predict_button = False

    # The rest of the code for prediction results (outside the columns)
    if 'predict_button' in locals() and predict_button:
        # Prediction results section after the columns
        st.markdown("---")  # Add a divider
        prediction = model.predict(single_data_point)
        st.subheader("Single Data Point Prediction")

        # Create two columns for image and prediction results
        img_col, pred_col = st.columns([1, 1])
        
        with img_col:
            try:
                st.image(joined_df.loc[index[0], 'image-src'], use_container_width=True)
            except:
                st.error("Image could not be loaded")
            
        with pred_col:
            final_output = [[round(prediction[0]), round(y_test.iloc[index[0]])]]
            single_point_df = pd.DataFrame(final_output, columns=['Predicted Price','Actual Price'])
            
            # Format the prices with dollar signs and commas
            formatted_df = single_point_df.copy()
            formatted_df['Predicted Price'] = formatted_df['Predicted Price'].apply(lambda x: f"${x:,}")
            formatted_df['Actual Price'] = formatted_df['Actual Price'].apply(lambda x: f"${x:,}")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Calculate difference and accuracy
            pred_price = final_output[0][0]
            actual_price = final_output[0][1]
            diff = abs(pred_price - actual_price)
            accuracy = (1 - (diff / actual_price)) * 100
            
            st.metric(
                label="Prediction Accuracy", 
                value=f"{accuracy:.1f}%",
                delta=f"${diff:,} difference"
            )

        # Predicted Range
        rmse = int(round(st.session_state["rmse"],0))
        predicted_price = final_output[0][0]
        min_price = predicted_price - rmse  
        max_price = predicted_price + rmse

        # Normalize the predicted price for plotting
        normalized_price = (predicted_price - min_price) / (max_price - min_price)

        # Create figure
        fig = go.Figure()

        # Add background price range bar (Fix width)
        fig.add_trace(go.Bar(
            x=[min_price, max_price],  # Correctly setting x values
            y=[1, 1],  # Keep y the same for horizontal alignment
            orientation="h",
            marker=dict(
                color=["#4285F4", "#34A853", "#EA4335"],  # Gradient color from blue to red
            ),
            width=0.2,  # Make the bar thick enough to be visible
            showlegend=False
        ))

        # Layout adjustments
        fig.update_layout(
            # title="Predicted Price Indicator",
            xaxis=dict(
                title="Price Range",
                range=[min_price - 100000, max_price + 100000],  # Extend range slightly
                tickvals=[min_price, predicted_price, max_price],
                ticktext=[f"${min_price:,}", f"${predicted_price:,}", f"${max_price:,}"],
                tickfont=dict(
                    size=14,  # Increase font size
                    color="black",  # Change font color if needed
                    family="Arial, sans-serif"  # Change font family if needed
        ),
            ),
            yaxis=dict(visible=False),
            width=700,
            height=150,
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # log_distance_to_nearest_school
        # Rename the column in X_test
        X_test = X_test.rename(columns={'log_distance_to_nearest_school': 'Proximity to School'})
        # Define the suffixes to remove
        suffixes_to_remove = ['driveway_parking',
                            'basement_type', 'lot_features', 'exterior_feature',
                            'waterfront_features', 'appliances_included']

        # # Rename columns in X_test
        # X_test.columns = [md.remove_suffixes(col, suffixes_to_remove) for col in X_test.columns]
        colors = ["gold", "silver", "#cd7f32", "#DAA520", "#B22222"]
        badge = ["🥇", "🥈", "🥉", "🏅", "🎖️"]
    
        # Maintain the order of columns
        column_order = [col.strip() for col in X_test.columns.tolist()]

        if model_choice == "Ridge Regression":
            feature_importance = model.coef_
            single_data_point_array = single_data_point.values
            output = np.multiply(feature_importance , single_data_point_array) # this is for linear regression

            absolute_coefficients_y = np.abs(output[0])
            percentages_y = (absolute_coefficients_y / np.sum(absolute_coefficients_y)) * 100

            # Combine feature names and percentages, then sort by percentages in descending order
            sorted_features = sorted(zip(column_order, percentages_y), key=lambda x: x[1], reverse=True)
        elif model_choice == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(single_data_point)

            # Calculate absolute SHAP values and convert to percentages
            absolute_shap_values = np.abs(shap_values[0]) # Extract SHAP values for this instance
            percentages = (absolute_shap_values / np.sum(absolute_shap_values)) * 100
  
            # Sort features by percentage contribution in descending order
            sorted_features = sorted(list(zip(column_order, percentages)), key=lambda x: x[1], reverse=True)

        # List of words to drop columns containing them
        words_to_drop = md.words_to_drop

        # Filter sorted_features to remove any feature names containing the words in words_to_drop
        filtered_sorted_features = [feature for feature in sorted_features if not md.should_drop(feature[0], words_to_drop)]

        # Select the top 20 features
        top_features_y = filtered_sorted_features[:20]
        top_feature_names_y, top_percentages_y = zip(*top_features_y)

        top_feature_names_y = [name.replace('_', ' ') for name in top_feature_names_y]

        # Convert tuple to list and extract strings
        top_names = [str(name) for name in top_feature_names_y]
        top_scores = [float(score) for score in top_percentages_y]

        # Title with divider
        st.markdown("---")
        st.title("🏆 Top 5 features")

        # Create a row of columns for the top 5 features
        feature_cols = st.columns(5)
        
        for i in range(0,5):
            # Display each feature in its own column
            with feature_cols[i]:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {colors[i]};">
                        <h2>{badge[i]} {top_names[i]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        md.display_graph(top_feature_names=top_feature_names_y,
                         top_percentages=top_percentages_y)

else:
    st.error("No trained model or test data found! Please train the model first.")