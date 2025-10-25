"""
Modela - Lean AutoML MVP
Main Streamlit application entry point.
"""

import streamlit as st
import os
from utils.firebase_client import get_firebase_client

# Page configuration
st.set_page_config(
    page_title="Modela - AutoML Made Simple",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-text {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 3rem;
        color: #666;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "local_user"
    if 'dataset_saved' not in st.session_state:
        st.session_state.dataset_saved = False
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Modela</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-text">Upload your dataset and train an AI model in minutes — no coding required.</p>', unsafe_allow_html=True)
    
    # Show main content
    show_main_content()

def show_main_content():
    """Show main content."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Navigation buttons
        nav_options = {
            "📊 Upload Dataset": "pages/1_upload.py",
            "🤖 Train Model": "pages/2_train.py", 
            "📈 View Results": "pages/3_results.py",
            "📁 My Projects": "pages/4_projects.py"
        }
        
        for label, page in nav_options.items():
            if st.button(label, use_container_width=True):
                st.switch_page(page)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Modela** is an open-source AutoML platform that makes machine learning accessible to everyone.
        
        **Features:**
        - Upload CSV/Excel files
        - Automatic model training
        - Rich visualizations
        - Download trained models
        - Completely free & local
        """)
    
    # Main content - Upload page content
    st.markdown("### 🚀 Get Started")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Upload Data</h4>
            <p>Upload CSV or Excel files up to 50MB. We'll automatically detect your data structure.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Train Models</h4>
            <p>Our AI automatically trains multiple models and picks the best one for your data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 View Results</h4>
            <p>Get detailed insights, feature importance, and download your trained model.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start button
    st.markdown("---")
    if st.button("🚀 Start Training Your First Model", type="primary", use_container_width=True):
        st.switch_page("pages/1_upload.py")

if __name__ == "__main__":
    main()
