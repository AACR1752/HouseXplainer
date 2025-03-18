import streamlit as st
import modules as md
import model_training
from streamlit_navigation_bar import st_navbar

# Set page configuration
st.set_page_config(
    page_title="Predict House Prices",
    page_icon="📊",
    layout="wide"
)

page_name = "Learn More"
# md.initialize_shared_state()
# page = st_navbar(st.session_state["pgs"], styles=st.session_state["styles"])
# if page != 'Home' and page != 'Learn More':
#     st.switch_page(f"./pages/{page}.py")
# elif page == 'Learn More':
#     st.switch_page(f"./pages/learn_more.py")
# md.apply_sidebar_minimization()

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

