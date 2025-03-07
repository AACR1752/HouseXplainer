import streamlit as st
import altair as alt
import pandas as pd

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