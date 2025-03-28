import streamlit as st
import modules as md
from streamlit_navigation_bar import st_navbar

# Set page configuration
st.set_page_config(
    page_title="Q&A",
    page_icon="picture/HouseXPlainer_B.png",
    layout="centered"
)

page_name = "FAQ"
md.initialize_shared_state()
page = st_navbar(
    st.session_state["pgs"], 
    styles=st.session_state["styles"], 
    logo_path="./picture/HouseXPlainer_W.svg",
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
md.apply_sidebar_minimization()
    
# App title and description
st.title("Interactive Q&A Page")
st.markdown("Click on any question to reveal its answer.")
    
# Add some space
st.write("")


qa_pairs = [
    {
        "question": "What is HouseXplainer?",
        "answer": "HouseExplainer is an online platform to understand house prices through its features, both internal and external."
    },
    {
        "question": "Who is it for?",
        "answer": "HouseXplainer is for anyone who wants to look at houses and know more about them.\n\n It provides a simple and easy to digest way to look at houses and their features."
    },
    {
        "question": "Where does HouseXplainer get the data from?",
        "answer": "HouseExplainer scrapes the publicly available house listings from realtor platform HouseSigma®️."
    },
    {
        "question": "What does these features mean?",
        "answer": "Features are directly scraped from Housesigma and how it is stored in MLS.\n\n These features are directly related to the house listing.\n\n Except for some neighbourhood features that we gather open data and calculate ourselves."
    },
    {
        "question": "Does HouseExplainer factor in market changes?",
        "answer": "Yes, HouseXplainer runs on overall listings provided as more house are sold,\n\n HouseXplainer periodically reanalyzes market to provide upto date information."
    },
    {
        "question": "Explain the Xplainer",
        "answer": "HouseXplainer uses xgboost model to predict house prices based on features.\n\n The features weights and explanations utilizes Shapley Additive exPlanations (SHAP)."
    },
    {
        "question": "Is the website cross-platform compatible?",
        "answer": "Yes, you can access through any device with a web browser.\n\n Please ensure you turn your phone to landscape mode for proper usability."
    },
    {
        "question": "Can I trust the information for real estate decisions?",
        "answer": "Any information provided is for educational purposes only.\n\n Please contact a real estate professional for any real estate decisions."
    }
]
    
# Sample Q&A data - replace with your own questions and answers
# qa_pairs = [
#     {
#         "question": "What is Streamlit?",
#         "answer": "Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science. It allows you to turn data scripts into shareable web apps in minutes, not weeks."
#     },
#     {
#         "question": "How do I install Streamlit?",
#         "answer": "You can install Streamlit using pip:\n```\npip install streamlit\n```\nAfter installation, you can run your app with the command:\n```\nstreamlit run your_app.py\n```"
#     },
#     {
#         "question": "Can I customize the appearance of my Streamlit app?",
#         "answer": "Yes, you can customize your Streamlit app using themes, custom CSS, and layout options. Streamlit provides configuration options to control the page layout, sidebar behavior, and color theme. For more advanced customization, you can use st.markdown with HTML and CSS."
#     },
#     {
#         "question": "Is Streamlit suitable for production applications?",
#         "answer": "While Streamlit was initially designed for prototyping and internal tools, it's increasingly being used for production applications. For production use cases, consider using Streamlit sharing, Streamlit Cloud, or deploying your app on platforms like Heroku, AWS, or GCP with proper authentication and scaling configurations."
#     },
#     {
#         "question": "How can I save user inputs in a Streamlit app?",
#         "answer": "Streamlit provides several options for persisting data:\n\n1. Session State: For storing data within a user session\n2. File-based storage: Writing to CSV, JSON, or other file formats\n3. Databases: Connecting to SQL or NoSQL databases\n4. Caching: Using @st.cache_data or @st.cache_resource decorators to store computed results"
#     }
# ]
    
# Create expandable sections for each Q&A pair
for i, qa in enumerate(qa_pairs):
    with st.expander(qa["question"]):
        st.markdown(qa["answer"])
            
# Add an optional feedback section
st.write("")
st.write("---")
st.subheader("Did you find this Q&A helpful?")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("👍 Yes"):
        st.success("Thanks for your feedback!")
with col2:
    if st.button("👎 No"):
        st.info("We'll work on improving our content!")
with col3:
    if st.button("💡 Suggest a question"):
        st.text_area("What question would you like to see answered?")
        if st.button("Submit"):
            st.success("Thank you for your suggestion!")

