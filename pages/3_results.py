"""
Results Dashboard Page
Displays detailed model results, metrics, and visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.visualization import (
    plot_feature_importance, 
    plot_confusion_matrix, 
    plot_regression_scatter,
    plot_metrics_comparison,
    create_metrics_summary_card
)
from utils.automl import generate_predictions_sample, serialize_model
from utils.firebase_client import get_firebase_client
from utils.navigation import show_sidebar

def main():
    """Main results page function."""
    
    # Show sidebar navigation
    show_sidebar()
    
    # Check if training is completed
    if 'training_completed' not in st.session_state or not st.session_state.training_completed:
        st.error("Please complete model training first.")
        if st.button("🤖 Train Model"):
            st.switch_page("pages/2_train.py")
        st.stop()
    
    st.title("📈 Model Results")
    st.markdown("Detailed analysis of your trained model performance and insights.")
    
    training_results = st.session_state.training_results
    training_params = st.session_state.training_params
    
    # Model overview
    st.markdown("### 🤖 Model Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", training_results['model_name'])
    with col2:
        st.metric("Type", training_params['problem_type'].title())
    with col3:
        st.metric("Target", training_params['target_col'])
    with col4:
        st.metric("Dataset", st.session_state.dataset_name)
    
    # Performance metrics
    st.markdown("### 📊 Performance Metrics")
    
    metrics = training_results['metrics']
    problem_type = training_params['problem_type']
    
    # Show metrics in a clear way
    if problem_type == 'classification':
        show_classification_metrics(metrics)
    else:
        show_regression_metrics(metrics)
    
    # Feature importance
    if training_results['feature_importance']:
        st.markdown("### 🔍 Feature Importance")
        st.markdown("The most important features that influence your model's predictions:")
        
        feature_fig = plot_feature_importance(training_results['feature_importance'])
        st.plotly_chart(feature_fig, use_container_width=True)
        
        # Feature importance table
        with st.expander("📋 Feature Importance Details"):
            feature_df = pd.DataFrame(
                list(training_results['feature_importance'].items()),
                columns=['Feature', 'Importance Score']
            )
            feature_df['Rank'] = range(1, len(feature_df) + 1)
            st.dataframe(feature_df, use_container_width=True)
    
    # Model predictions visualization
    st.markdown("### 📈 Model Predictions")
    
    try:
        # Get test data for visualization
        df = st.session_state.current_dataset
        target_col = training_params['target_col']
        test_split = training_params['test_split']
        
        # Split data for visualization
        test_size = int(len(df) * test_split)
        test_data = df.tail(test_size)
        
        # Generate predictions
        predictions_df = generate_predictions_sample(
            training_results['model'], 
            training_results['scaler'],
            training_results['label_encoders'],
            test_data, 
            target_col
        )
        
        if not predictions_df.empty:
            if problem_type == 'classification':
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(predictions_df['actual'], predictions_df['predicted'])
                
                # Get unique classes
                classes = sorted(predictions_df['actual'].unique())
                
                confusion_fig = plot_confusion_matrix(
                    predictions_df['actual'].tolist(),
                    predictions_df['predicted'].tolist(),
                    [str(c) for c in classes]
                )
                st.plotly_chart(confusion_fig, use_container_width=True)
                
                # Predictions table
                with st.expander("📋 Sample Predictions"):
                    st.dataframe(predictions_df, use_container_width=True)
            
            else:
                # Regression scatter plot
                scatter_fig = plot_regression_scatter(
                    predictions_df['actual'].tolist(),
                    predictions_df['predicted'].tolist()
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Predictions table
                with st.expander("📋 Sample Predictions"):
                    st.dataframe(predictions_df, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate predictions visualization: {str(e)}")
    
    # Model insights
    st.markdown("### 💡 Model Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**🎯 What this model does:**")
        if problem_type == 'classification':
            st.markdown(f"""
            - **Predicts categories** for the **{training_params['target_col']}** column
            - Uses **{len(df.columns)-1} features** to make predictions
            - Achieves **{metrics.get('accuracy', 0):.1%} accuracy** on test data
            - **Precision:** {metrics.get('precision', 0):.1%} | **Recall:** {metrics.get('recall', 0):.1%}
            """)
        else:
            st.markdown(f"""
            - **Predicts numerical values** for the **{training_params['target_col']}** column
            - Uses **{len(df.columns)-1} features** to make predictions
            - Achieves **{metrics.get('r2', 0):.1%} R² score** on test data
            - **RMSE:** {metrics.get('rmse', 0):.2f} | **MAE:** {metrics.get('mae', 0):.2f}
            """)
    
    with insights_col2:
        st.markdown("**🔧 How to use this model:**")
        st.markdown("""
        - Download the model file (.pkl)
        - Load it in Python with pickle
        - Use model.predict(new_data) for predictions
        - Ensure new data has the same features as training data
        """)
    
    # Download and actions
    st.markdown("### 💾 Download & Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download model
        if st.button("📥 Download Model", type="primary", use_container_width=True):
            model_bytes = serialize_model(training_results['model'])
            st.download_button(
                label="Download Model (.pkl)",
                data=model_bytes,
                file_name=f"{training_results['model_name']}_model.pkl",
                mime="application/octet-stream"
            )
    
    with col2:
        # Train another model
        if st.button("🔄 Train Another Model", use_container_width=True):
            st.switch_page("pages/2_train.py")
    
    with col3:
        # View projects
        if st.button("📁 My Projects", use_container_width=True):
            st.switch_page("pages/4_projects.py")
    
    # Model details
    with st.expander("🔧 Technical Details"):
        st.markdown("**Model Information:**")
        st.json({
            "Model Name": training_results['model_name'],
            "Problem Type": problem_type,
            "Target Column": training_params['target_col'],
            "Training Data Size": f"{len(df):,} rows",
            "Test Split": f"{training_params['test_split']*100:.0f}%",
            "Features Used": len(df.columns) - 1,
            "Cross-Validation Folds": 3
        })
        
        st.markdown("**Performance Metrics:**")
        st.json(metrics)

def show_classification_metrics(metrics):
    """Show classification metrics in a clear way."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy", 
            f"{metrics.get('accuracy', 0):.1%}",
            help="Percentage of correct predictions"
        )
    
    with col2:
        st.metric(
            "Precision", 
            f"{metrics.get('precision', 0):.1%}",
            help="Percentage of positive predictions that were correct"
        )
    
    with col3:
        st.metric(
            "Recall", 
            f"{metrics.get('recall', 0):.1%}",
            help="Percentage of actual positives that were correctly identified"
        )
    
    with col4:
        st.metric(
            "F1 Score", 
            f"{metrics.get('f1', 0):.1%}",
            help="Harmonic mean of precision and recall"
        )
    
    # Interpretation
    accuracy = metrics.get('accuracy', 0)
    if accuracy >= 0.9:
        st.success("🎉 Excellent! Your model has very high accuracy.")
    elif accuracy >= 0.8:
        st.success("✅ Great! Your model has good accuracy.")
    elif accuracy >= 0.7:
        st.info("👍 Good! Your model has decent accuracy.")
    else:
        st.warning("⚠️ Consider improving your data or trying different models.")

def show_regression_metrics(metrics):
    """Show regression metrics in a clear way."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "R² Score", 
            f"{metrics.get('r2', 0):.1%}",
            help="Percentage of variance explained by the model"
        )
    
    with col2:
        st.metric(
            "RMSE", 
            f"{metrics.get('rmse', 0):.2f}",
            help="Root Mean Square Error - lower is better"
        )
    
    with col3:
        st.metric(
            "MAE", 
            f"{metrics.get('mae', 0):.2f}",
            help="Mean Absolute Error - lower is better"
        )
    
    with col4:
        # Calculate MAPE if possible
        mape = metrics.get('mape', 0)
        if mape > 0:
            st.metric(
                "MAPE", 
                f"{mape:.1%}",
                help="Mean Absolute Percentage Error - lower is better"
            )
        else:
            st.metric("MAPE", "N/A", help="Not calculated")
    
    # Interpretation
    r2 = metrics.get('r2', 0)
    if r2 >= 0.8:
        st.success("🎉 Excellent! Your model explains most of the variance.")
    elif r2 >= 0.6:
        st.success("✅ Great! Your model has good predictive power.")
    elif r2 >= 0.4:
        st.info("👍 Good! Your model has decent predictive power.")
    else:
        st.warning("⚠️ Consider improving your data or trying different models.")

if __name__ == "__main__":
    main()