import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    # Dropdown to select a value from X_test
    datapoint = st.selectbox("Select House", joined_df['listing'].tolist())

    index = joined_df[joined_df['listing'] == datapoint].index.tolist()
    single_data_point = X_test.iloc[[index[0]]]

    if st.button("Predict"):
        prediction = model.predict(single_data_point)
        st.subheader("Single Data Point Prediction")

        st.image(joined_df.loc[index[0], 'image-src'])

        # st.write(f"image url {joined_df.loc[index[0], 'image-src']}")

        final_output = [[round(prediction[0]), round(y_test.iloc[index[0]])]]
        single_point_df = pd.DataFrame(final_output, columns=['Predicted Price','Actual Price'])

        st.dataframe(single_point_df)

        if model_choice == "Linear Regression":
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

            # Create the plot
            fig, ax = plt.subplots()
            ax.barh(top_feature_names_y, top_percentages_y, color='skyblue')
            ax.set_xlabel("Contribution (%)")
            ax.set_title("Top 20 Feature Contributions in Percentages")
            ax.invert_yaxis()  # Invert y-axis to show the highest contribution at the top

            # Display in Streamlit
            st.pyplot(fig)


else:
    st.error("No trained model or test data found! Please train the model first.")
