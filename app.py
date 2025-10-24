"""
Modela - Lean AutoML MVP
Main Streamlit application entry point.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from utils.firebase_client import get_firebase_client

# Load environment variables
load_dotenv()

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
    
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'is_guest' not in st.session_state:
        st.session_state.is_guest = False
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Modela</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-text">Upload your dataset and train an AI model in minutes — no coding required.</p>', unsafe_allow_html=True)
    
    # Check if user is authenticated
    if st.session_state.user_id is None:
        show_auth_section()
    else:
        show_main_content()

def show_auth_section():
    """Show authentication section."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        # Guest mode button
        st.markdown("### 🚀 Get Started")
        st.markdown("**Try Modela instantly with guest mode:**")
        
        if st.button("🎯 Start as Guest", type="primary", use_container_width=True):
            firebase_client = get_firebase_client()
            guest_id = firebase_client.create_guest_user()
            st.session_state.user_id = guest_id
            st.session_state.is_guest = True
            st.session_state.user_email = "Guest User"
            st.rerun()
        
        st.markdown("---")
        
        # Sign in section
        st.markdown("### 🔐 Sign In")
        st.markdown("**Or sign in to save your models:**")
        
        with st.form("auth_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                sign_in = st.form_submit_button("Sign In", use_container_width=True)
            with col2:
                sign_up = st.form_submit_button("Sign Up", use_container_width=True)
            
            if sign_in:
                if email and password:
                    firebase_client = get_firebase_client()
                    user_id = firebase_client.authenticate_user(email, password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.user_email = email
                        st.session_state.is_guest = False
                        st.success("Successfully signed in!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")
                else:
                    st.error("Please enter both email and password.")
            
            if sign_up:
                st.info("Sign up functionality will be added in future versions. Please use guest mode for now.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("---")
    st.markdown("### ✨ What You Can Do")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Upload Data</h4>
            <p>Upload CSV or Excel files up to 50MB. We'll automatically detect your data structure.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Train Models</h4>
            <p>Our AI automatically trains multiple models and picks the best one for your data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 View Results</h4>
            <p>Get detailed insights, feature importance, and download your trained model.</p>
        </div>
        """, unsafe_allow_html=True)

def show_main_content():
    """Show main content for authenticated users."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Info")
        if st.session_state.is_guest:
            st.info(f"**Guest User**\n{st.session_state.user_id}")
        else:
            st.success(f"**Signed in as**\n{st.session_state.user_email}")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.rerun()
        
        if st.button("📊 Upload Dataset", use_container_width=True):
            st.switch_page("pages/1_upload.py")
        
        if st.button("🤖 Train Model", use_container_width=True):
            st.switch_page("pages/2_train.py")
        
        if st.button("📈 View Results", use_container_width=True):
            st.switch_page("pages/3_results.py")
        
        if st.button("📁 My Projects", use_container_width=True):
            st.switch_page("pages/4_projects.py")
        
        st.markdown("---")
        
        # Sign out
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.is_guest = False
            st.rerun()
    
    # Main content
    st.markdown("### 🎯 Welcome to Modela!")
    
    if st.session_state.is_guest:
        st.info("👋 You're using guest mode. Your models will be saved temporarily. Sign up to save them permanently!")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Upload Your Data**
        - Click "Upload Dataset" in the sidebar
        - Choose a CSV or Excel file (max 50MB)
        - We'll show you a preview of your data
        """)
        
        if st.button("📊 Upload Dataset", type="primary", use_container_width=True):
            st.switch_page("pages/1_upload.py")
    
    with col2:
        st.markdown("""
        **Step 2: Train Your Model**
        - Select your target column
        - Choose classification or regression
        - Click "Start Training" and wait 1-3 minutes
        """)
        
        if st.button("🤖 Train Model", type="primary", use_container_width=True):
            st.switch_page("pages/2_train.py")
    
    # Recent projects (if any)
    if not st.session_state.is_guest:
        st.markdown("### 📁 Recent Projects")
        
        try:
            firebase_client = get_firebase_client()
            projects = firebase_client.list_user_projects(st.session_state.user_id)
            
            if projects:
                for project in projects[:3]:  # Show only recent 3
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"**{project['name']}** - {project['model_type']}")
                            st.write(f"Dataset: {project['dataset_name']} | Accuracy: {project['accuracy']:.1%}")
                        with col2:
                            if st.button("View", key=f"view_{project['name']}"):
                                st.switch_page("pages/3_results.py")
                        with col3:
                            if st.button("Download", key=f"download_{project['name']}"):
                                st.download_button(
                                    label="Download Model",
                                    data=firebase_client.download_model(st.session_state.user_id, project['name']),
                                    file_name=f"{project['name']}.pkl",
                                    mime="application/octet-stream"
                                )
            else:
                st.info("No projects yet. Upload your first dataset to get started!")
        except Exception as e:
            st.warning("Could not load recent projects. Please try again later.")

if __name__ == "__main__":
    main()
