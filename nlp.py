import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy import displacy
import plotly.express as px
import html
import re

# Load the transformer model
@st.cache_resource
def load_nlp_model():
    nlp = spacy.load("en_core_web_trf")  # More accurate transformer-based model
    
    # Add custom lemmatization component
    if "custom_lemma_component" not in nlp.pipe_names:
        nlp.add_pipe('custom_lemma_component', after='tagger')
    
    return nlp

# Define custom lemmatization component
@Language.component("custom_lemma_component")
def custom_lemma_component(doc):
    custom_lemmas = {
        "br": "bedroom",
        "apt": "apartment",
        "st": "street",
        "min": "minute",
        "w/": "with",
    }
    
    for token in doc:
        lower_text = token.text.lower()
        if lower_text in custom_lemmas:
            token.lemma_ = custom_lemmas[lower_text]
    
    return doc

def extract_tokens(text):
    """Extract lemmatized tokens excluding stopwords"""
    nlp = load_nlp_model()
    doc = nlp(text)
    tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop and token.is_ascii]
    return tokens

def extract_keywords(text, max_keywords=10):
    """Extract keywords from text using spaCy patterns and matchers"""
    nlp = load_nlp_model()
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    
    # Noun and Noun Phrases
    noun_phrases_patterns = [
        [{'POS': 'NUM'}, {'POS': 'NOUN'}],  # example: 2 bedrooms
        [{'POS': 'ADJ', 'OP': '*'}, {'POS': 'NOUN'}],  # example: beautiful house
        [{'POS': 'NOUN', 'OP': '+'}],  # example: house
    ]
    
    # Geo-political entity
    gpe_patterns = [
        [{'ENT_TYPE': 'GPE'}],  # example: Tokyo
    ]
    
    # Location
    loc_patterns = [
        [{'ENT_TYPE': 'LOC'}],  # example: downtown
    ]
    
    # Facility
    fac_patterns = [
        [{'ENT_TYPE': 'FAC'}],  # example: airport
    ]
    
    # Proximity
    proximity_patterns = [
        [{'POS': 'ADJ'}, {'POS': 'ADP'}, {'POS': 'NOUN', 'ENT_TYPE': 'FAC', 'OP': '?'}],  # example: near airport
        [{'POS': 'ADJ'}, {'POS': 'ADP'}, {'POS': 'PROPN', 'ENT_TYPE': 'FAC', 'OP': '?'}]  # example: near to Narita
    ]
    
    for entity, patterns in zip(['NOUN_PHRASE', 'GPE', 'LOC', 'FAC', "PROXIMITY"], 
                                [noun_phrases_patterns, gpe_patterns, loc_patterns, fac_patterns, proximity_patterns]):
        matcher.add(entity, patterns)
    
    matches = matcher(doc)
    keywords = []
    for match_id, start, end in matches:
        span = doc[start:end]
        match_label = nlp.vocab.strings[match_id]
        keywords.append((match_label, span.text.strip().lower()))
    
    keyword_freq = {}
    for keyword in keywords:
        keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
    
    keywords = sorted(keyword_freq, key=keyword_freq.get, reverse=True)
    return keywords[:max_keywords]

def render_entities_html(text):
    """Render named entities from text as HTML"""
    nlp = load_nlp_model()
    doc = nlp(text)
    
    # Get HTML representation with displacy
    html_content = displacy.render(doc, style="ent")
    
    # Clean up the HTML for streamlit
    html_content = re.sub(r'<div class="entities" style=".*?">', 
                          '<div class="entities" style="line-height: 2.5; direction: ltr">', 
                          html_content)
    
    return html_content

def plot_keyword_distribution(keywords):
    """Create a horizontal bar chart of keyword frequencies"""
    if not keywords:
        return None
    
    # Extract labels and counts
    labels = []
    counts = []
    categories = []
    
    for (category, text), count in keywords:
        labels.append(text)
        counts.append(count)
        categories.append(category)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Keyword': labels,
        'Frequency': counts,
        'Category': categories
    })
    
    # Create color mapping for categories
    color_map = {
        'NOUN_PHRASE': '#1f77b4',
        'GPE': '#ff7f0e',
        'LOC': '#2ca02c',
        'FAC': '#d62728',
        'PROXIMITY': '#9467bd'
    }
    
    # Create figure
    fig = px.bar(
        df, 
        y='Keyword', 
        x='Frequency', 
        color='Category',
        color_discrete_map=color_map,
        orientation='h',
        title='Top Keywords in Property Description',
        labels={'Frequency': 'Occurrence', 'Keyword': ''},
        height=400
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=10, r=10, t=40, b=20)
    )
    
    return fig

def main():
    st.title("Property Description Analysis")
    
    # Check if description exists in session state
    if "single_data_point" not in st.session_state:
        st.warning("Please select a property from the Home page first.")
        return
    
    try:
        # Get the description from single_data_point
        description = st.session_state.single_data_point['description'].values[0]
        
        if not description or pd.isna(description):
            st.warning("This property doesn't have a description.")
            return
    
        # Display the original description
        with st.expander("Original Property Description", expanded=True):
            st.write(description)
        
        # Process the description
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Named Entity Recognition")
            html_content = render_entities_html(description)
            st.markdown(html_content, unsafe_allow_html=True)
            
            # Add legend
            st.markdown("""
            <style>
            .entity-box {
                display: inline-block;
                padding: 2px 8px;
                margin-right: 10px;
                border-radius: 5px;
                font-size: 0.8em;
            }
            </style>
            <div style="margin-top: 20px;">
                <div class="entity-box" style="background-color: #ddd; border: 1px solid #999;">Legend:</div>
                <div class="entity-box" style="background-color: #7aecec; border: 1px solid #6fb7b7;">PERSON</div>
                <div class="entity-box" style="background-color: #feca74; border: 1px solid #c49c5d;">GPE</div>
                <div class="entity-box" style="background-color: #ff9561; border: 1px solid #cc774e;">LOC</div>
                <div class="entity-box" style="background-color: #9cc9cc; border: 1px solid #7d9fa3;">FAC</div>
                <div class="entity-box" style="background-color: #aa9cfc; border: 1px solid #887cc9;">ORG</div>
                <div class="entity-box" style="background-color: #ffeb80; border: 1px solid #ccbc66;">DATE</div>
                <div class="entity-box" style="background-color: #bfe1d9; border: 1px solid #98b4ad;">MONEY</div>
                <div class="entity-box" style="background-color: #e4e7d2; border: 1px solid #b6b9a8;">QUANTITY</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Keyword Analysis")
            
            # Extract and display keywords
            keywords = extract_keywords(description, max_keywords=15)
            
            # Plot keywords
            fig = plot_keyword_distribution(keywords)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No keywords extracted from the description.")
            
            # Display raw tokens for debugging
            with st.expander("Raw Tokens", expanded=False):
                tokens = extract_tokens(description)
                st.write(tokens)

    except KeyError:
        st.error("Error: 'single_data_point' is missing or improperly formatted.")
        return
            