"""
Upload Dataset Page
Handles dataset upload, validation, and preview.
"""

import streamlit as st
import pandas as pd
import io
from utils.firebase_client import get_firebase_client
from utils.preprocessing import validate_dataset

def main():
    """Main upload page function."""
    
    st.title("📊 Upload Dataset")
    st.markdown("Upload your CSV or Excel file to get started with AutoML.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls). Maximum file size: 50MB"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate dataset
            validation_result = validate_dataset(df)
            
            if validation_result['valid']:
                st.success("✅ Dataset uploaded successfully!")
                
                # Auto-save dataset
                firebase_client = get_firebase_client()
                user_id = st.session_state.get('user_id', 'local_user')
                
                # Save dataset
                dataset_name = uploaded_file.name.split('.')[0]
                file_data = uploaded_file.getvalue()
                
                file_path = firebase_client.upload_dataset(user_id, dataset_name, file_data)
                
                if file_path:
                    st.session_state.dataset_saved = True
                    st.session_state.current_dataset = df
                    st.session_state.dataset_name = dataset_name
                    st.session_state.dataset_path = file_path
                    
                    st.success(f"💾 Dataset '{dataset_name}' saved successfully!")
                    
                    # Show dataset info
                    show_dataset_info(df, dataset_name)
                    
                    # Auto-proceed to training
                    st.markdown("---")
                    st.markdown("### 🚀 Ready to Train!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🤖 Start Training", type="primary", use_container_width=True):
                            st.switch_page("pages/2_train.py")
                    
                    with col2:
                        if st.button("📊 View Dataset Details", use_container_width=True):
                            st.switch_page("pages/2_train.py")
                
                else:
                    st.error("Failed to save dataset. Please try again.")
            
            else:
                st.error("❌ Dataset validation failed:")
                for error in validation_result['errors']:
                    st.error(f"• {error}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        # Show sample datasets
        show_sample_datasets()

def show_dataset_info(df, dataset_name):
    """Show dataset information and preview."""
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    with col4:
        st.metric("File Size", f"{len(df) * len(df.columns) / 1000:.1f}K cells")
    
    # Data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    st.markdown("### 📊 Column Information")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Missing': df[col].isnull().sum(),
            'Unique': df[col].nunique(),
            'Sample': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, use_container_width=True)
    
    # Suggest target column
    suggest_target_column(df)

def suggest_target_column(df):
    """Suggest the best target column based on data analysis."""
    
    st.markdown("### 🎯 Target Column Suggestion")
    
    # Analyze columns for target suggestions
    suggestions = []
    
    for col in df.columns:
        col_data = df[col]
        
        # Skip if too many missing values
        if col_data.isnull().sum() > len(df) * 0.5:
            continue
        
        # Check if it's a good target for classification
        if col_data.dtype == 'object' or col_data.nunique() <= 20:
            suggestions.append({
                'column': col,
                'type': 'Classification',
                'reason': f'{col_data.nunique()} unique values',
                'score': 1.0 if col_data.nunique() <= 10 else 0.7
            })
        
        # Check if it's a good target for regression
        elif col_data.dtype in ['int64', 'float64']:
            suggestions.append({
                'column': col,
                'type': 'Regression',
                'reason': f'Numeric with {col_data.nunique()} unique values',
                'score': 0.8 if col_data.nunique() > 20 else 0.5
            })
    
    if suggestions:
        # Sort by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        best_suggestion = suggestions[0]
        
        st.success(f"💡 **We suggest using '{best_suggestion['column']}' as your target column.**")
        st.info(f"**Type:** {best_suggestion['type']} | **Reason:** {best_suggestion['reason']}")
        
        # Store suggestion in session state
        st.session_state.suggested_target = best_suggestion['column']
        st.session_state.suggested_type = best_suggestion['type']
    
    else:
        st.warning("⚠️ Could not suggest a target column. Please review your data manually.")

def show_sample_datasets():
    """Show sample datasets for testing."""
    
    st.markdown("---")
    st.markdown("### 🧪 Try Sample Datasets")
    st.markdown("Don't have a dataset? Try one of our sample datasets:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Iris Dataset**
        - 150 rows, 5 columns
        - Classification problem
        - Predict flower species
        """)
        if st.button("Use Iris Dataset", use_container_width=True):
            load_sample_dataset("iris")
    
    with col2:
        st.markdown("""
        **📈 Synthetic Classification**
        - 1000 rows, 11 columns
        - Classification problem
        - Predict categories
        """)
        if st.button("Use Classification Dataset", use_container_width=True):
            load_sample_dataset("synthetic_classification")
    
    with col3:
        st.markdown("""
        **📊 Synthetic Regression**
        - 1000 rows, 11 columns
        - Regression problem
        - Predict continuous values
        """)
        if st.button("Use Regression Dataset", use_container_width=True):
            load_sample_dataset("synthetic_regression")

def load_sample_dataset(dataset_name):
    """Load a sample dataset."""
    
    try:
        import os
        sample_path = f"sample_data/{dataset_name}.csv"
        
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            
            # Auto-save dataset
            firebase_client = get_firebase_client()
            user_id = st.session_state.get('user_id', 'local_user')
            
            with open(sample_path, 'rb') as f:
                file_data = f.read()
            
            file_path = firebase_client.upload_dataset(user_id, dataset_name, file_data)
            
            if file_path:
                st.session_state.dataset_saved = True
                st.session_state.current_dataset = df
                st.session_state.dataset_name = dataset_name
                st.session_state.dataset_path = file_path
                
                st.success(f"✅ Loaded {dataset_name} dataset!")
                st.rerun()
            else:
                st.error("Failed to load sample dataset.")
        else:
            st.error("Sample dataset not found. Please run 'python create_demo_simple.py' first.")
    
    except Exception as e:
        st.error(f"Error loading sample dataset: {str(e)}")

if __name__ == "__main__":
    main()