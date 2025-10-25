"""
Shared navigation component for all pages.
"""

import streamlit as st

def show_sidebar():
    """Show consistent sidebar navigation on all pages."""
    
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Navigation buttons
        nav_options = {
            "🏠 Home": "app.py",
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
        
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("""
        - [GitHub Repository](https://github.com/youssof20/modela)
        - [Documentation](https://github.com/youssof20/modela#readme)
        - [Report Issues](https://github.com/youssof20/modela/issues)
        """)
