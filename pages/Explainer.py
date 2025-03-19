import streamlit as st
import pandas as pd
import numpy as np
import shap
import modules as md
import plotly.graph_objects as go
import pydeck as pdk
# import model_training
from streamlit_navigation_bar import st_navbar
import copy

page_name = "Explainer"
st.set_page_config(page_title="HouseXplainer - Explains",
                   page_icon="picture/HE_icon_B.png",
                   layout="wide")

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
        
        
        # neighbourhood_name = st.selectbox("Select Neighourhood", joined_df['neighbourhood'].unique().tolist())
        neighbourhood_name = st.selectbox(
            "🏡 Search for neighbourhood",
            joined_df['neighbourhood'].unique().tolist(),
            index=joined_df['neighbourhood'].unique().tolist().index(st.session_state["neighbourhood"]) if st.session_state["neighbourhood"] in joined_df['neighbourhood'].unique().tolist() else 0,
        )

        # bathroom_selection = st.selectbox("Select Bathroom", joined_df['bathrooms'].tolist())
        small_df = joined_df[(joined_df['neighbourhood'] == neighbourhood_name)]
        # property_type_selection = st.selectbox("Select Property Type", small_df['property_type'].unique().tolist())
        property_type_selection = st.selectbox(
            "🏠 Select property type",
            small_df['property_type'].unique().tolist(),
            index=small_df['property_type'].unique().tolist().index(st.session_state["property_type"]) if st.session_state["property_type"] in small_df['property_type'].unique().tolist() else 0,
        )

        filtered_df = joined_df[
        (joined_df['neighbourhood'] == neighbourhood_name) &
        # (joined_df['bedrooms'] == bedroom_selection) &
        # (joined_df['bathrooms'] == bathroom_selection) &
        (joined_df['property_type'] == property_type_selection)]
        
        # Update the selectbox for the house listing based on the filtered DataFrame
        try: 
            # datapoint = st.selectbox("Select House", filtered_df['listing'].tolist())
            datapoint = st.selectbox(
                "🏡 Choose house",
                filtered_df['listing'].tolist(),
                index=filtered_df['listing'].tolist().index(st.session_state["house"]) if st.session_state["house"] in filtered_df['listing'].tolist() else 0,
            )
            # Get the index of the selected house
            index = filtered_df[filtered_df['listing'] == datapoint].index.tolist()
            single_data_point = X_test.iloc[[index[0]]]
        except:
            st.write("There is no available listings with current selection!")

    with col2:
        st.markdown("<h3 style='text-align: center;'>Property Map</h3>", unsafe_allow_html=True)
        selected_house = filtered_df[filtered_df['listing'] == datapoint]
        # Define House Layer (Blue Circles)
        house_layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df[filtered_df['listing'] != datapoint],
            get_position=["longitude", "latitude"],
            get_radius=25,  # Adjust size
            get_fill_color=[0, 0, 255, 255],  # Blue for houses [0, 0, 255, 180]
            pickable=True,
            opacity=0.9,
        )

        selected_house_layer = pdk.Layer(
            "ScatterplotLayer",
            data=selected_house,
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
            zoom=14.5,  # Adjust zoom for visibility
            pitch=5,  # Adds slight tilt for better visualization
        )

        # Display the Map with Mapbox Style
        r = pdk.Deck(
            layers=[house_layer, selected_house_layer],
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

            school_name = joined_df.loc[index[0], 'nearest_school']

            amenities = md.get_amenities()
            string_data = joined_df.loc[index[0], 'amenities_objectids_1km']
            amenity_objectids = string_data.split(",")
            amen_name = md.process_amenities(amenities, amenity_objectids)

            st.subheader("Key Property Details")
            st.write(f"🏠 Architechture Style: {joined_df.loc[index[0], 'architecture_style']}")
            st.write(f"🎓 Nearest School: {school_name}")
            def display_amenities_columns(amen_name):
                """Displays amenities in two columns."""
                col1, col2 = st.columns(2)  # Create two columns

                for i, name in enumerate(amen_name):
                    if "school" in name.lower():
                        display_text = f"🏫 {name} facilities"
                    elif "church" in name.lower():
                        display_text = f"⛪ {name}"
                    elif "park" in name.lower():
                        display_text = f"🍁 {name}"
                    else:
                        display_text = f"🏛️ {name}"

                    if i % 2 == 0:  # Even index, display in the first column
                        col1.write(display_text)
                    else:  # Odd index, display in the second column
                        col2.write(display_text)
            
            st.markdown("#### Amenities")
            display_amenities_columns(amen_name)
        
        # Predicted Range
        k = 0.5  # Adjust based on confidence needs
        rmse = int(round(st.session_state["rmse"],0))
        predicted_price = final_output[0][0]
        min_price = int(round(predicted_price - (k * rmse),0))
        max_price = int(round(predicted_price + (k * rmse),0))

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
                title="Expected Market Value",
                range=[min_price - 100000, max_price + 100000],  # Extend range slightly
                tickvals=[min_price, predicted_price, max_price],
                ticktext=[f"${min_price:,}", f"${predicted_price:,}", f"${max_price:,}"],
                tickfont=dict(
                    size=22,  # Increase font size
                    color="black",  # Change font color if needed
                    family="Arial, sans-serif"  # Change font family if needed
        ),
            ),
            yaxis=dict(visible=False),
            width=700,
            height=150,
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.subheader("Predicted Price Range:")

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

            #Create a copy of the sorted features
            # s_features = copy.deepcopy(sorted_features)

        words_to_drop = md.words_to_drop

        # Filter sorted_features to remove any feature names containing the words in words_to_drop
        filtered_sorted_features = [feature for feature in sorted_features if not md.should_drop(feature[0], words_to_drop)]

        #For the Micro/Macro Features
        s_features = copy.deepcopy(filtered_sorted_features)

        # Select the top 20 features
        top_features_y = filtered_sorted_features[:20]
        top_feature_names_y, top_percentages_y = zip(*top_features_y)

        top_feature_names_y = [name.replace('_', ' ') for name in top_feature_names_y]

        # Convert tuple to list and extract strings
        top_names = [str(name) for name in top_feature_names_y]
        top_scores = [float(score) for score in top_percentages_y]

        st.markdown("---")
        st.title("🏠 Property Description")

        description = joined_df.loc[index[0], 'description']

        feature_df = md.render_features()

        with st.spinner('Explaining Time.....'):
            highlighted_text, found_features = md.highlight_keywords(description, feature_df)

        st.markdown(highlighted_text, unsafe_allow_html=True)
        # if found_features:
        #     st.write("Highlighted Features:")
        #     for feature in found_features:
        #         st.write(feature)

        # Title with divider
        st.markdown("---")
        st.title("🏆 The five dominant factors")

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
        #Old bar graph
        #md.display_graph(top_feature_names=top_feature_names_y,
        #                 top_percentages=top_percentages_y)
        


        ### Testing Zone

        #For getting the feature catalogue
        import ast

        # Read the catalogue in
        def read_feature_list(file_path):
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    # Use ast.literal_eval() instead of eval() for safety
                    feature_list = ast.literal_eval(content.strip())
                return feature_list
            except FileNotFoundError:
                st.error("The file 'feature_catalog.txt' was not found!")
                return []
            except (SyntaxError, ValueError) as e:
                st.error(f"Error reading the file: {e}")
                return []

        # Path to the catalogue
        file_path = 'feature_catalog.txt'

        # Read the neigbourhood feature list from the backend file
        feature_list = read_feature_list(file_path)

        # Zip !
        top_feature_names_y, top_percentages_y = zip(*s_features)

        top_feature_names_y = [name.replace('_', ' ') for name in top_feature_names_y]

        # Convert the feature_list to a set for faster membership checking
        feature_set = set(feature_list)

        # Separate the top features into those within and not within feature_list
        features_within = [(feature, percentage) for feature, percentage in zip(top_feature_names_y, top_percentages_y) if feature in feature_set]
        # features_within = sorted(features_within, key=lambda x: x[1], reverse=True)

        features_not_within = [(feature, percentage) for feature, percentage in zip(top_feature_names_y, top_percentages_y) if feature not in feature_set]
        # features_not_within = sorted(features_not_within, key=lambda x: x[1], reverse=True)

        # Limit the lists to the top 20 features each
        features_within = features_within[:10]
        features_not_within = features_not_within[:10]

        # Sort the lists in ascending order based on the percentage (second element of the tuple)
        features_within.sort(key=lambda x: x[1])  # Sorting based on percentage
        features_not_within.sort(key=lambda x: x[1])  # Sorting based on percentage

        # Separate the names and percentages for features within the feature list
        top_feature_names_within, top_percentages_within = zip(*features_within) if features_within else ([], [])
        # Separate the names and percentages for features not within the feature list
        top_feature_names_not_within, top_percentages_not_within = zip(*features_not_within) if features_not_within else ([], [])

        # Create the alternating color scheme for the micro bars
        g_colors = ['#6AA84F', '#93C47D'] * (len(top_feature_names_not_within) // 2 + 1)
        g_colors = g_colors[:len(top_feature_names_not_within)]

        # Create the alternating color scheme for the macro bars
        o_colors = ['#FF9849', '#D8813E'] * (len(top_feature_names_within) // 2 + 1)
        o_colors = o_colors[:len(top_feature_names_within)]

        # Create the Micro graph (more detailed, smaller scale chart)
        def plot_micro():
            # Create the horizontal bar chart using Plotly
            fig = go.Figure(data=[
                go.Bar(
                    x=top_percentages_not_within,  # Use percentages for the horizontal bar length
                    y=top_feature_names_not_within,  # Feature names go on the Y-axis
                    orientation='h',  # Change the orientation to horizontal
                    marker_color=g_colors,  # Apply the alternating colors
                    width=0.8,  # Increase the width of the bars (this will make the bars bigger)
                    marker=dict(line=dict(width=0))  # Remove the border on the bars
                )
            ])

            # Update the layout to make the chart bigger and adjust bar spacing
            fig.update_layout(
                height=900,  # Increase the height of the chart
                width=900,   # Increase the width of the chart
                bargap=0.3,  # Decrease the gap between bars
                title="Top Internal Features",  # Set the title of the chart
                xaxis_title="Percentage Contribution (%)",  # X-axis label
                template="plotly_dark",  # Optional: Use a dark theme for the chart
                xaxis=dict(
                    showgrid=True,  # Show gridlines on the X-axis
                    minor=dict(
                        showgrid=True,  # Enable minor gridlines
                        gridwidth=0.5,  # Thinner minor gridlines
                    ),
                ),
                yaxis=dict(
                    showgrid=False,  # Hide gridlines on the Y-axis (optional)
                    title_font=dict(size=20),  # Increase the Y-axis title font size
                    tickfont=dict(size=16),  # Increase the font size of Y-axis ticks
                )
            )
            st.plotly_chart(fig, use_container_width=True, key="micro")

    # Create the Macro graph (more detailed, smaller scale chart)
        def plot_macro():
            # Create the horizontal bar chart using Plotly
            fig = go.Figure(data=[
                go.Bar(
                    x=top_percentages_within,  # Use percentages for the horizontal bar length
                    y=top_feature_names_within,  # Feature names go on the Y-axis
                    orientation='h',  # Change the orientation to horizontal
                    marker_color=o_colors,  # Apply the alternating colors
                    width=0.8,  # Increase the width of the bars (this will make the bars bigger)
                    marker=dict(line=dict(width=0))  # Remove the border on the bars
                )
            ])

            # Update the layout to make the chart bigger and adjust bar spacing
            fig.update_layout(
                height=900,  # Increase the height of the chart
                width=900,   # Increase the width of the chart
                bargap=0.3,  # Decrease the gap between bars
                title="Top External Features",  # Set the title of the chart
                xaxis_title="Percentage Contribution (%)",  # X-axis label
                template="plotly_dark",  # Optional: Use a dark theme for the chart
                xaxis=dict(
                    showgrid=True,  # Show gridlines on the X-axis
                    minor=dict(
                        showgrid=True,  # Enable minor gridlines
                        gridwidth=0.5,  # Thinner minor gridlines
                    ),
                ),
                yaxis=dict(
                    showgrid=False,  # Hide gridlines on the Y-axis (optional)
                    title_font=dict(size=20),  # Increase the Y-axis title font size
                    tickfont=dict(size=16),  # Increase the font size of Y-axis ticks
                )
            )
            st.plotly_chart(fig, use_container_width=True, key="macro")
        
        tab1, tab2 = st.tabs(["External Features", "Internal Features"])

        with tab1:
            st.subheader("External Features")
            plot_macro()
            
        with tab2:
            st.subheader("Internal Features")
            plot_micro()

else:
    st.error("No trained model or test data found! Please train the model first.")