import streamlit as st
import modules as md
import model_training

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Load data
st.title("HouseXplainer - A House Price Prediction App")
# st.sidebar.header("Selection Panel")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

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
    model_training.main(model_choice=model_choice)

st.success("Model trained successfully! Go to 'Use Model' page to test it.")