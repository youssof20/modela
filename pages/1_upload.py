"""
Upload Dataset Page
Allows users to upload CSV or Excel files and preview the data.
"""

import streamlit as st
import pandas as pd
import io
from utils.firebase_client import get_firebase_client
from utils.preprocessing import validate_dataset, get_dataset_summary, suggest_target_column

def main():
    """Main upload page function."""
    
    # Check authentication
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        st.error("Please sign in first.")
        st.stop()
    
    st.title("📊 Upload Dataset")
    st.markdown("Upload your CSV or Excel file to get started with AutoML training.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls). Maximum file size: 50MB"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        if file_size > 50:
            st.error("File size exceeds 50MB limit. Please upload a smaller file.")
            st.stop()
        
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate dataset
            is_valid, warnings = validate_dataset(df)
            
            if not is_valid:
                st.error("❌ Dataset validation failed:")
                for warning in warnings:
                    st.error(f"- {warning}")
                st.stop()
            
            # Show warnings if any
            if warnings:
                st.warning("⚠️ Dataset warnings:")
                for warning in warnings:
                    st.warning(f"- {warning}")
            
            # Store dataset in session state
            st.session_state.uploaded_dataset = df
            st.session_state.dataset_name = uploaded_file.name
            
            # Show success message
            st.success(f"✅ Successfully uploaded {uploaded_file.name} ({file_size:.1f}MB)")
            
            # Dataset overview
            st.markdown("### 📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Size", f"{file_size:.1f}MB")
            with col4:
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Dataset summary
            with st.expander("📊 Detailed Dataset Summary"):
                summary = get_dataset_summary(df)
                
                st.markdown("**Data Types:**")
                dtype_df = pd.DataFrame(list(summary['dtypes'].items()), columns=['Column', 'Data Type'])
                st.dataframe(dtype_df, use_container_width=True)
                
                if summary['missing_values']:
                    st.markdown("**Missing Values:**")
                    missing_df = pd.DataFrame(list(summary['missing_values'].items()), columns=['Column', 'Missing Count'])
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    if not missing_df.empty:
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.info("No missing values found!")
                
                if summary['numeric_columns']:
                    st.markdown("**Numeric Columns Statistics:**")
                    st.dataframe(pd.DataFrame(summary['numeric_stats']), use_container_width=True)
                
                if summary['categorical_columns']:
                    st.markdown("**Categorical Columns:**")
                    for col in summary['categorical_columns']:
                        st.write(f"**{col}:** {df[col].nunique()} unique values")
                        if col in summary['categorical_stats']:
                            st.write(f"Top values: {list(summary['categorical_stats'][col].keys())[:5]}")
            
            # Target column suggestion
            st.markdown("### 🎯 Target Column Suggestion")
            suggested_target = suggest_target_column(df)
            st.info(f"💡 We suggest using **'{suggested_target}'** as your target column.")
            st.write("This column will be used to train your model. You can change this in the next step.")
            
            # Store suggested target
            st.session_state.suggested_target = suggested_target
            
            # Upload to Firebase Storage
            if st.button("💾 Save Dataset", type="primary"):
                with st.spinner("Saving dataset to cloud storage..."):
                    firebase_client = get_firebase_client()
                    
                    # Convert dataframe to CSV bytes
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue().encode('utf-8')
                    
                    # Upload to Firebase
                    download_url = firebase_client.upload_dataset(
                        st.session_state.user_id,
                        uploaded_file.name,
                        csv_bytes
                    )
                    
                    if download_url:
                        st.success("✅ Dataset saved to cloud storage!")
                        st.session_state.dataset_saved = True
                        st.session_state.dataset_url = download_url
                    else:
                        st.error("❌ Failed to save dataset. Please try again.")
            
            # Navigation
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⬅️ Back to Home", use_container_width=True):
                    st.switch_page("app.py")
            
            with col2:
                if st.button("➡️ Configure Training", type="primary", use_container_width=True):
                    if 'dataset_saved' in st.session_state and st.session_state.dataset_saved:
                        st.switch_page("pages/2_train.py")
                    else:
                        st.warning("Please save the dataset first before proceeding.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV or Excel file.")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("### 📝 Instructions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📋 Supported Formats:**
            - CSV files (.csv)
            - Excel files (.xlsx, .xls)
            
            **📏 File Requirements:**
            - Maximum size: 50MB
            - Minimum rows: 10
            - Minimum columns: 2
            """)
        
        with col2:
            st.markdown("""
            **💡 Tips for Best Results:**
            - Clean your data before uploading
            - Remove unnecessary columns
            - Ensure your target column is clear
            - Check for missing values
            """)
        
        # Sample datasets
        st.markdown("### 🎯 Sample Datasets")
        st.markdown("Don't have a dataset? Try these examples:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Iris Dataset", use_container_width=True):
                # Load iris dataset
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['target'] = iris.target
                st.session_state.uploaded_dataset = df
                st.session_state.dataset_name = "iris_sample.csv"
                st.session_state.suggested_target = "target"
                st.success("✅ Loaded Iris dataset!")
                st.rerun()
        
        with col2:
            if st.button("🚢 Titanic Dataset", use_container_width=True):
                # Load titanic dataset
                try:
                    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
                    st.session_state.uploaded_dataset = df
                    st.session_state.dataset_name = "titanic_sample.csv"
                    st.session_state.suggested_target = "Survived"
                    st.success("✅ Loaded Titanic dataset!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load Titanic dataset: {str(e)}")
        
        with col3:
            if st.button("🏠 Boston Housing", use_container_width=True):
                # Load boston housing dataset
                try:
                    from sklearn.datasets import load_boston
                    boston = load_boston()
                    df = pd.DataFrame(boston.data, columns=boston.feature_names)
                    df['target'] = boston.target
                    st.session_state.uploaded_dataset = df
                    st.session_state.dataset_name = "boston_housing_sample.csv"
                    st.session_state.suggested_target = "target"
                    st.success("✅ Loaded Boston Housing dataset!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load Boston Housing dataset: {str(e)}")

if __name__ == "__main__":
    main()
