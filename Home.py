import streamlit as st
import pandas as pd
import numpy as np
import shap
import modules as md
import plotly.graph_objects as go
import pydeck as pdk
import model_training

# from mapie.regression import MapieRegressor

# Set the page configuration to wide mode
st.set_page_config(page_title="HouseXplainer - Home",
                   page_icon="🏠",
                   layout="wide")

# Load data
st.title("HouseXplainer - AI-powered Insights Tool")

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
    col1, col2 = st.columns([2, 3])

    with col1:
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
        neighbourhood_name = st.selectbox("Select Neighourhood", joined_df['neighbourhood'].unique().tolist())
        # bedroom_selection = st.selectbox("Select Bedroom", joined_df['bedrooms'].tolist())
        # bathroom_selection = st.selectbox("Select Bathroom", joined_df['bathrooms'].tolist())
        small_df = joined_df[(joined_df['neighbourhood'] == neighbourhood_name)]
        property_type_selection = st.selectbox("Select Property Type", small_df['property_type'].unique().tolist())

        filtered_df = joined_df[
        (joined_df['neighbourhood'] == neighbourhood_name) &
        # (joined_df['bedrooms'] == bedroom_selection) &
        # (joined_df['bathrooms'] == bathroom_selection) &
        (joined_df['property_type'] == property_type_selection)]

        
        # Update the selectbox for the house listing based on the filtered DataFrame
        try: 
            datapoint = st.selectbox("Select House", filtered_df['listing'].tolist())
            # Get the index of the selected house
            index = filtered_df[filtered_df['listing'] == datapoint].index.tolist()
            single_data_point = X_test.iloc[[index[0]]]
        except:
            st.write("There is no available listings with current selection!")

    with col2:
        st.markdown("<h3 style='text-align: center;'>Property Map</h3>", unsafe_allow_html=True)

        # Define House Layer (Blue Circles)
        house_layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df,
            get_position=["longitude", "latitude"],
            get_radius=25,  # Adjust size
            get_fill_color=[255, 0, 0, 255],  # Blue for houses [0, 0, 255, 180]
            pickable=True,
            opacity=0.9,
        )


        # Set the Map View
        view_state = pdk.ViewState(
            latitude=filtered_df["latitude"].mean(),
            longitude=filtered_df["longitude"].mean(),
            zoom=15,  # Adjust zoom for visibility
            pitch=5,  # Adds slight tilt for better visualization
        )

        # Display the Map with Mapbox Style
        r = pdk.Deck(
            layers=[house_layer],
            initial_view_state=view_state,
            tooltip={"text": "{listing}"},
            map_style="mapbox://styles/mapbox/streets-v12"

        )
        map_placeholder = st.pydeck_chart(r)


    with col1:
        st.markdown(
            """
            <style>
            div.stButton > button {
                display: block;
                margin: 0 auto;
                width: 50%;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        predict = st.button("Predict", use_container_width=True)

    if predict:
        # Celebration Effect (Optional)
        # st.balloons()  # Adds a fun animation effect!
        prediction = model.predict(single_data_point)
        st.subheader("Single Data Point Prediction")

        # Create two columns for image and prediction results
        img_col, pred_col = st.columns([1, 1])


        with img_col:
            st.image(joined_df.loc[index[0], 'image-src'])

        with pred_col:
            final_output = [[round(prediction[0]), round(y_test.iloc[index[0]])]]
            single_point_df = pd.DataFrame(final_output, columns=['Predicted Price','Actual Price'])
            
            # Format the prices with dollar signs and commas
            formatted_df = single_point_df.copy()
            formatted_df['Predicted Price'] = formatted_df['Predicted Price'].apply(lambda x: f"${x:,}")
            formatted_df['Actual Price'] = formatted_df['Actual Price'].apply(lambda x: f"${x:,}")
            
            st.dataframe(formatted_df, use_container_width=True)


            # Mapie CI
            # mapie = MapieRegressor(model, method="plus")  # Can also use 'cv' or 'naive'
            # mapie.fit(X_test, y_test)

            # y_pred, y_pis = mapie.predict(single_data_point, alpha=0.025)

            # y_pred_scalar = float(y_pred[0])
            # lower_bound = float(y_pis[0, 0])
            # lower_bound = max(lower_bound, y_pred_scalar * 0.8)
            # upper_bound = float(y_pis[0, 1])
            # upper_bound = min(upper_bound, y_pred_scalar * 1.2)

            # st.write(f"**Predicted Price:** ${y_pred[0]:,.0f}")
            # st.write(f"**95% Confidence Interval:** [${lower_bound:,.0f}, ${upper_bound:,.0f}]")


            
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

        # Rename the column in X_test
        X_test = X_test.rename(columns={'log_distance_to_nearest_school': 'Proximity to School'})
        
        suffixes_to_remove = ['driveway_parking',
                            'basement_type', 'lot_features', 'exterior_feature',
                            'waterfront_features', 'appliances_included']

        # # Rename columns in X_test
        X_test.columns = [md.remove_suffixes(col, suffixes_to_remove) for col in X_test.columns]
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
        
        st.write("")
        md.display_graph(top_feature_names=top_feature_names_y,
                         top_percentages=top_percentages_y)

else:
    st.error("No trained model or test data found! Please train the model first.")