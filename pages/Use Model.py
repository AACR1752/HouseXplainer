import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


st.title("Use the Trained Model")

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

    # TODO: Bring in the filters for neighbourhood, property_type, bedrooms, bathrooms
    # st.selectboxes 
    # neighbourhood_name = 
    # bedroom_selection =
    # bathroom_selection = 
    # property_type_selection =

    # TODO: joined_df will shrink based on the selection above
    # joined_df = (smaller set of listings based on the above filters).filter

    # Dropdown to select a value from X_test
    datapoint = st.selectbox("Select House", joined_df['listing'].tolist())

    index = joined_df[joined_df['listing'] == datapoint].index.tolist()
    single_data_point = X_test.iloc[[index[0]]]

    # pd.DataFrame(
    # np.random.randn(1000, 2) / [50, 50] + [43.4643, -80.5204],
    # columns=["lat", "lon"],
    # )

    st.map(joined_df[["latitude", "longitude"]])

    if st.button("Predict"):
        # Celebration Effect (Optional)
        st.balloons()  # Adds a fun animation effect!

        prediction = model.predict(single_data_point)
        st.subheader("Single Data Point Prediction")

        st.image(joined_df.loc[index[0], 'image-src'])

        # st.write(f"image url {joined_df.loc[index[0], 'image-src']}")

        final_output = [[round(prediction[0]), round(y_test.iloc[index[0]])]]
        single_point_df = pd.DataFrame(final_output, columns=['Predicted Price','Actual Price'])

        st.dataframe(single_point_df)

        if model_choice == "Ridge Regression":
            # Convert the single data point to an ndarray
            single_data_point_array = single_data_point.values

            # Maintain the order of columns
            column_order = X_test.columns.tolist()

            output = np.multiply(model.coef_ , single_data_point_array) # this is for linear regression

            absolute_coefficients_y = np.abs(output[0])
            percentages_y = (absolute_coefficients_y / np.sum(absolute_coefficients_y)) * 100

            # Combine feature names and percentages, then sort by percentages in descending order
            sorted_features_y = sorted(zip(column_order, percentages_y), key=lambda x: x[1], reverse=True)

            # Select the top 20 features
            top_features_y = sorted_features_y[:20]
            top_feature_names_y, top_percentages_y = zip(*top_features_y)

            # Convert tuple to list and extract strings
            top_names = [str(name) for name in top_feature_names_y]
            top_scores = [float(score) for score in top_percentages_y]

            # Sample top features and their contributions
            top_features = [
                {"name": top_names, "score": top_scores}
            ]


            # Title
            st.title("🏆 Feature Importance Leaderboard")

            # 1st Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: gold;">
                        <h2>🥇 {top_names[0]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 2nd Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: silver;">
                        <h2>🥈 {top_names[1]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 3rd Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #cd7f32;">
                        <h2>🥉 {top_names[2]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 4th Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #DAA520;">
                        <h2>🏅 {top_names[3]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 5th Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #B22222;">
                        <h2>🎖️ {top_names[4]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)


            # Create the plot
            fig, ax = plt.subplots()
            ax.barh(top_feature_names_y, top_percentages_y, color='skyblue')
            ax.set_xlabel("Contribution (%)")
            ax.set_title("Top 20 Feature Contributions in Percentages")
            ax.invert_yaxis()  # Invert y-axis to show the highest contribution at the top

            # Display in Streamlit
            st.pyplot(fig)


        elif model_choice == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(single_data_point)

            # Get feature names
            feature_names = single_data_point.columns.tolist()

            # Convert SHAP values into a structured array
            shap_values_single = shap_values[0]  # Extract SHAP values for this instance

            # Calculate absolute SHAP values and convert to percentages
            absolute_shap_values = np.abs(shap_values_single)
            percentages = (absolute_shap_values / np.sum(absolute_shap_values)) * 100

            # Combine feature names and percentages
            feature_importance = list(zip(feature_names, percentages))

            # Sort features by percentage contribution in descending order
            sorted_feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

            # Select the top 20 features
            top_20_features = sorted_feature_importance[:20]
            top_20_feature_names, top_20_percentages = zip(*top_20_features)



            # Convert tuple to list and extract strings
            top_fnames = [str(name) for name in top_20_feature_names]
            top_fscores = [float(score) for score in top_20_percentages]

            # Sample top features and their contributions
            top_forest_features = [
                {"name": top_fnames, "score": top_fscores}
            ]

            # Title
            st.title("🏆 Feature Importance Leaderboard")

            # 1st Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: gold;">
                        <h2>🥇 {top_fnames[0]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 2nd Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: silver;">
                        <h2>🥈 {top_fnames[1]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 3rd Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #cd7f32;">
                        <h2>🥉 {top_fnames[2]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 4th Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #DAA520;">
                        <h2>🏅 {top_fnames[3]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)

            # 5th Place
            with st.container():
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #B22222;">
                        <h2>🎖️ {top_fnames[4]}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
             
                st.markdown("<br>", unsafe_allow_html=True)


            # Define colors based on SHAP values
            # colors = ['red' if shap_values_single[feature_names.index(name)] < 0 else 'green' for name in top_20_feature_names]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot horizontal bar chart
            ax.barh(top_20_feature_names, top_20_percentages, color="skyblue")

            # Set labels and title
            ax.set_xlabel("Contribution (%)")
            ax.set_ylabel("Feature")
            ax.set_title("Top 20 Features Impacting House Price Prediction")

            # Invert y-axis to show the highest contribution at the top
            ax.invert_yaxis()

            # Display plot in Streamlit
            st.pyplot(fig)

else:
    st.error("No trained model or test data found! Please train the model first.")
