"""
Model Training Page
Allows users to configure and train AutoML models.
"""

import streamlit as st
import pandas as pd
import time
from utils.automl import train_model, detect_problem_type
from utils.preprocessing import validate_target_column, prepare_for_training
from utils.firebase_client import get_firebase_client
from utils.visualization import plot_training_progress

def main():
    """Main training page function."""
    
    # Check authentication
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        st.error("Please sign in first.")
        st.stop()
    
    # Check if dataset is uploaded
    if 'uploaded_dataset' not in st.session_state:
        st.error("Please upload a dataset first.")
        if st.button("📊 Upload Dataset"):
            st.switch_page("pages/1_upload.py")
        st.stop()
    
    st.title("🤖 Train Your Model")
    st.markdown("Configure your training parameters and let our AI find the best model for your data.")
    
    df = st.session_state.uploaded_dataset
    
    # Dataset info
    st.markdown("### 📊 Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset", st.session_state.dataset_name)
    with col2:
        st.metric("Rows", f"{len(df):,}")
    with col3:
        st.metric("Columns", len(df.columns))
    
    # Target column selection
    st.markdown("### 🎯 Select Target Column")
    st.markdown("Choose the column you want to predict:")
    
    # Get suggested target
    suggested_target = st.session_state.get('suggested_target', df.columns[0])
    
    target_col = st.selectbox(
        "Target Column",
        options=df.columns,
        index=df.columns.get_loc(suggested_target) if suggested_target in df.columns else 0,
        help="This is the column your model will learn to predict"
    )
    
    # Validate target column
    is_valid, error_msg = validate_target_column(df, target_col)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        st.stop()
    
    # Problem type detection
    st.markdown("### 🔍 Problem Type")
    detected_type = detect_problem_type(df, target_col)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🤖 **Auto-detected**: {detected_type.title()}")
    with col2:
        problem_type = st.selectbox(
            "Problem Type",
            options=['classification', 'regression'],
            index=0 if detected_type == 'classification' else 1,
            help="Classification: predicting categories. Regression: predicting numbers."
        )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            test_split = st.slider(
                "Test Split Ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Percentage of data used for testing (20% recommended)"
            )
        
        with col2:
            if problem_type == 'classification':
                metric_priority = st.selectbox(
                    "Metric Priority",
                    options=['accuracy', 'precision', 'recall', 'f1'],
                    index=0,
                    help="Primary metric for model selection"
                )
            else:
                metric_priority = st.selectbox(
                    "Metric Priority",
                    options=['rmse', 'mae', 'r2'],
                    index=0,
                    help="Primary metric for model selection"
                )
    
    # Training summary
    st.markdown("### 📋 Training Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown(f"""
        **Dataset**: {st.session_state.dataset_name}  
        **Target**: {target_col}  
        **Problem Type**: {problem_type.title()}  
        **Training Data**: {(1-test_split)*100:.0f}% ({len(df)*(1-test_split):.0f} rows)
        """)
    
    with summary_col2:
        st.markdown(f"""
        **Test Data**: {test_split*100:.0f}% ({len(df)*test_split:.0f} rows)  
        **Metric**: {metric_priority.upper()}  
        **Models**: 5 algorithms  
        **Expected Time**: 1-3 minutes
        """)
    
    # Start training button
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Store training parameters
        st.session_state.training_params = {
            'target_col': target_col,
            'problem_type': problem_type,
            'test_split': test_split,
            'metric_priority': metric_priority
        }
        
        # Start training process
        start_training(df, target_col, problem_type, test_split)

def start_training(df, target_col, problem_type, test_split):
    """Start the model training process."""
    
    # Create progress containers
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        progress_text = st.empty()
    
    with status_container:
        status_text = st.empty()
    
    # Training stages
    stages = [
        {"stage": "Data Preprocessing", "progress": 0, "message": "Cleaning and preparing your data..."},
        {"stage": "Model Training", "progress": 0, "message": "Training multiple algorithms..."},
        {"stage": "Model Evaluation", "progress": 0, "message": "Evaluating and comparing models..."},
        {"stage": "Saving Results", "progress": 0, "message": "Saving your trained model..."}
    ]
    
    try:
        # Stage 1: Data Preprocessing
        progress_text.text("Stage 1/4: Data Preprocessing")
        status_text.text("Cleaning and preparing your data...")
        progress_bar.progress(10)
        time.sleep(1)
        
        # Prepare data for training
        df_clean, detected_type = prepare_for_training(df, target_col)
        
        progress_bar.progress(25)
        time.sleep(0.5)
        
        # Stage 2: Model Training
        progress_text.text("Stage 2/4: Model Training")
        status_text.text("Training multiple algorithms...")
        progress_bar.progress(30)
        time.sleep(1)
        
        # Train the model
        training_result = train_model(df_clean, target_col, problem_type, test_split)
        
        if training_result is None:
            st.error("❌ Training failed. Please check your data and try again.")
            return
        
        progress_bar.progress(70)
        time.sleep(0.5)
        
        # Stage 3: Model Evaluation
        progress_text.text("Stage 3/4: Model Evaluation")
        status_text.text("Evaluating and comparing models...")
        progress_bar.progress(80)
        time.sleep(1)
        
        # Store results in session state
        st.session_state.training_results = training_result
        st.session_state.training_completed = True
        
        progress_bar.progress(90)
        time.sleep(0.5)
        
        # Stage 4: Saving Results
        progress_text.text("Stage 4/4: Saving Results")
        status_text.text("Saving your trained model...")
        
        # Save model to Firebase
        firebase_client = get_firebase_client()
        
        # Serialize model
        from utils.automl import serialize_model
        model_bytes = serialize_model(training_result['model'])
        
        # Prepare metadata
        metadata = {
            'model_name': training_result['model_name'],
            'model_type': problem_type,
            'dataset_name': st.session_state.dataset_name,
            'target_column': target_col,
            'accuracy': training_result['metrics'].get('accuracy', training_result['metrics'].get('r2', 0)),
            'created_at': pd.Timestamp.now().isoformat(),
            'training_params': st.session_state.training_params
        }
        
        # Save to Firebase
        model_url = firebase_client.save_model(
            st.session_state.user_id,
            training_result['model_name'],
            model_bytes,
            metadata
        )
        
        if model_url:
            st.session_state.model_url = model_url
            st.session_state.model_saved = True
        
        progress_bar.progress(100)
        progress_text.text("✅ Training Complete!")
        status_text.text("Your model is ready!")
        
        time.sleep(1)
        
        # Show results
        with results_container:
            st.success("🎉 Model training completed successfully!")
            
            # Show model performance
            st.markdown("### 📊 Model Performance")
            
            metrics = training_result['metrics']
            if problem_type == 'classification':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
                with col4:
                    st.metric("F1-Score", f"{metrics.get('f1', 0):.1%}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
                with col2:
                    st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
                with col3:
                    st.metric("R²", f"{metrics.get('r2', 0):.1%}")
                with col4:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.1%}")
            
            # Show feature importance
            if training_result['feature_importance']:
                st.markdown("### 🔍 Top Features")
                feature_df = pd.DataFrame(
                    list(training_result['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                )
                st.dataframe(feature_df.head(10), use_container_width=True)
            
            # Navigation buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⬅️ Back to Upload", use_container_width=True):
                    st.switch_page("pages/1_upload.py")
            
            with col2:
                if st.button("📈 View Detailed Results", type="primary", use_container_width=True):
                    st.switch_page("pages/3_results.py")
    
    except Exception as e:
        st.error(f"❌ Training failed with error: {str(e)}")
        st.info("Please check your data and try again. If the problem persists, try with a smaller dataset.")

if __name__ == "__main__":
    main()
