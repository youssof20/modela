"""
Projects Dashboard Page
Lists all user's trained models and projects.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils.firebase_client import get_firebase_client

def main():
    """Main projects page function."""
    
    # Check authentication
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        st.error("Please sign in first.")
        st.stop()
    
    st.title("📁 My Projects")
    st.markdown("View and manage all your trained models.")
    
    # Load projects
    try:
        firebase_client = get_firebase_client()
        projects = firebase_client.list_user_projects(st.session_state.user_id)
        
        if not projects:
            # No projects found
            st.markdown("### 📭 No Projects Yet")
            st.info("You haven't trained any models yet. Get started by uploading a dataset!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Upload Dataset", type="primary", use_container_width=True):
                    st.switch_page("pages/1_upload.py")
            with col2:
                if st.button("🤖 Train Model", use_container_width=True):
                    st.switch_page("pages/2_train.py")
            
            # Show sample projects for inspiration
            st.markdown("---")
            st.markdown("### 💡 Get Inspired")
            st.markdown("Here are some example projects you could create:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **📊 Classification Projects:**
                - Customer churn prediction
                - Email spam detection
                - Image classification
                - Sentiment analysis
                """)
            
            with col2:
                st.markdown("""
                **📈 Regression Projects:**
                - House price prediction
                - Sales forecasting
                - Stock price prediction
                - Energy consumption
                """)
            
            with col3:
                st.markdown("""
                **🎯 Business Applications:**
                - Risk assessment
                - Quality control
                - Demand forecasting
                - Fraud detection
                """)
        
        else:
            # Display projects
            st.markdown(f"### 📊 Your Projects ({len(projects)})")
            
            # Project statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_projects = len(projects)
            classification_projects = len([p for p in projects if p['model_type'] == 'classification'])
            regression_projects = len([p for p in projects if p['model_type'] == 'regression'])
            avg_accuracy = sum(p['accuracy'] for p in projects) / len(projects) if projects else 0
            
            with col1:
                st.metric("Total Projects", total_projects)
            with col2:
                st.metric("Classification", classification_projects)
            with col3:
                st.metric("Regression", regression_projects)
            with col4:
                st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
            
            # Projects table
            st.markdown("### 📋 Project List")
            
            # Create projects dataframe
            projects_data = []
            for i, project in enumerate(projects):
                projects_data.append({
                    'Name': project['name'],
                    'Type': project['model_type'].title(),
                    'Dataset': project['dataset_name'],
                    'Accuracy': f"{project['accuracy']:.1%}",
                    'Created': project['created_at'][:10] if project['created_at'] else 'Unknown',
                    'Actions': i  # Use index for action buttons
                })
            
            projects_df = pd.DataFrame(projects_data)
            
            # Display projects table
            st.dataframe(projects_df, use_container_width=True, hide_index=True)
            
            # Project details and actions
            st.markdown("### 🔍 Project Details")
            
            for i, project in enumerate(projects):
                with st.expander(f"📊 {project['name']} - {project['model_type'].title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Model Information:**
                        - **Name**: {project['name']}
                        - **Type**: {project['model_type'].title()}
                        - **Dataset**: {project['dataset_name']}
                        - **Accuracy**: {project['accuracy']:.1%}
                        - **Created**: {project['created_at'][:19] if project['created_at'] else 'Unknown'}
                        """)
                    
                    with col2:
                        st.markdown("**Actions:**")
                        
                        # Action buttons
                        action_col1, action_col2, action_col3 = st.columns(3)
                        
                        with action_col1:
                            if st.button("📈 View", key=f"view_{i}", use_container_width=True):
                                # Store project data for results page
                                st.session_state.current_project = project
                                st.switch_page("pages/3_results.py")
                        
                        with action_col2:
                            if st.button("📥 Download", key=f"download_{i}", use_container_width=True):
                                # Download model
                                model_data = firebase_client.download_model(
                                    st.session_state.user_id, 
                                    project['name']
                                )
                                if model_data:
                                    st.download_button(
                                        label="Download Model",
                                        data=model_data,
                                        file_name=f"{project['name']}.pkl",
                                        mime="application/octet-stream",
                                        key=f"download_btn_{i}"
                                    )
                                else:
                                    st.error("Could not download model")
                        
                        with action_col3:
                            if st.button("🗑️ Delete", key=f"delete_{i}", use_container_width=True):
                                st.warning("Delete functionality will be available soon!")
            
            # Quick actions
            st.markdown("---")
            st.markdown("### ⚡ Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Upload New Dataset", type="primary", use_container_width=True):
                    st.switch_page("pages/1_upload.py")
            
            with col2:
                if st.button("🤖 Train New Model", use_container_width=True):
                    st.switch_page("pages/2_train.py")
            
            with col3:
                if st.button("🏠 Back to Home", use_container_width=True):
                    st.switch_page("app.py")
            
            # Export projects
            if st.button("📤 Export Projects List"):
                # Create CSV export
                export_df = pd.DataFrame(projects_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Projects CSV",
                    data=csv,
                    file_name="my_projects.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Could not load projects: {str(e)}")
        st.info("Please check your internet connection and try again.")
        
        # Fallback navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🏠 Back to Home", use_container_width=True):
                st.switch_page("app.py")

if __name__ == "__main__":
    main()
