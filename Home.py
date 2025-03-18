import streamlit as st
import pandas as pd
import model_training
import os
import base64
from streamlit_navigation_bar import st_navbar
import modules as md


page_name = "Home"
# Set the page configuration to wide mode
st.set_page_config(page_title="HouseXplainer",
                   page_icon="picture/HE_icon_B.png",
                   layout="wide")

md.initialize_shared_state()

page = st_navbar(st.session_state["pgs"], styles=st.session_state["styles"])

if page != page_name and page != 'Learn More':
    st.switch_page(f"./pages/{page}.py")
elif page == 'Learn More':
    st.switch_page(f"./pages/learn_more.py")

md.apply_sidebar_minimization()

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
    
    y_test = pd.Series(st.session_state["y_test"], name="Price")

def main():
    # Create a local path for images folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "picture", "home_page.png")
    
    # Convert the image to base64 for CSS background
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    # Get the base64 string
    bin_str = get_base64_of_bin_file(image_path)
    
    # CSS to create a top section with image background and white bottom section
    st.markdown(f"""
    <style>
    /* Reset default padding */
    .stApp {{
      padding: 0 !important;
    }}
    
    /* Create a hero section with background image */
    .hero-section {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
      height: 60vh;
      width: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 20px;
      margin-bottom: 20px;
    }}
    
    /* Container for search elements */
    .search-container {{
      background-color: rgba(255, 255, 255, 0.1);
      padding: 20px;
      border-radius: 10px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    
    /* Title and subtitle styles */
    .hero-title {{
      text-align: center; 
      color: white; 
      font-size: 3.5rem; 
      font-weight: bold;
      text-shadow: 2px 2px 4px #000000;
      margin-bottom: 10px;
    }}
    
    .hero-subtitle {{
      text-align: center; 
      color: white; 
      font-size: 1.8rem; 
      text-shadow: 2px 2px 4px #000000;
      margin-bottom: 30px;
    }}
    
    /* Content section */
    .content-section {{
      background-color: white;
      min-height: 40vh;
      padding: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Hero section with background image
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">AI-Powered Home Value Insights tool</h1>
        <h2 class="hero-subtitle">Know the Factors That Shape Home Prices</h2>
        <div class="search-container">
            <!-- Search inputs will be placed here by Streamlit -->
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for the search boxes
    # These will appear inside the hero section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        neighborhood = st.selectbox("🏡 Search for neighbourhood", 
                                     joined_df['neighbourhood'].unique().tolist())
    
    with col2:
        small_df = joined_df[(joined_df['neighbourhood'] == neighborhood)]
        property_type = st.selectbox("🏠 Select property type", 
                                small_df['property_type'].unique().tolist())
    
    with col3:
        filtered_df = joined_df[
            (joined_df['neighbourhood'] == neighborhood) &
            # (joined_df['bedrooms'] == bedroom_selection) &
            # (joined_df['bathrooms'] == bathroom_selection) &
            (joined_df['property_type'] == property_type)]
        house = st.selectbox("🏡 Choose house", 
                            filtered_df['listing'].tolist())
        
    st.session_state["neighbourhood"] = neighborhood
    st.session_state["property_type"] = property_type
    st.session_state["house"] = house
    
    # Button styling with custom CSS
    st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #75c44b;
        color: black;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        display: block;
        margin: 0 auto;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: box-shadow 0.3s ease;
    }

    div.stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        switch_page = st.button("See houses!", use_container_width=True)
        if switch_page:
            # Switch to the selected page
            # page_file = pages[selected_page]
            st.switch_page("./pages/Explainer.py")
    
    # Content section (white background)
    # st.markdown("""
    # <div class="content-section">
    #     <!-- This area will be white and ready for additional content -->
    # </div>
    # """, unsafe_allow_html=True)
    
    # Additional content in the white section
    # This will appear in the white background section
    st.markdown("### Future content will go here")
    st.write("This section has a white background and can be populated with charts, tables, or other information.")

if __name__ == "__main__":
    main()