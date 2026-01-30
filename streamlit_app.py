import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import mlflow
from zenml.client import Client
from pipelines.training_pipeline import train_pipeline


st.set_page_config(
    page_title="MLOps Customer Review Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Review Score Prediction")
st.markdown("---")


def get_latest_model():
    """Load the latest trained model from MLflow"""
    try:
        client = Client()
        tracking_uri = client.active_stack.experiment_tracker.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        
        # Search all runs across all experiments, sorted by start time
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        
        if not runs.empty:
            run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{run_id}/model"
            
            # Load the model
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get metrics
            metrics = {
                "r2_score": runs.iloc[0].get("metrics.r2_score", runs.iloc[0].get("metrics.r2", "N/A")),
                "mse": runs.iloc[0].get("metrics.mse", "N/A"),
                "rmse": runs.iloc[0].get("metrics.rmse", "N/A")
            }
            
            return model, metrics, run_id
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
    return None, None, None


# Sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Action", ["Prediction", "Batch Prediction", "Train New Model", "Model Info"])

# Add refresh button
if st.sidebar.button("🔄 Refresh Model"):
    st.rerun()

if app_mode == "Model Info":
    st.header("📊 Model Information")
    
    model, metrics, run_id = get_latest_model()
    
    if model and metrics:
        st.success(f"✅ Model loaded successfully! (Run ID: {run_id[:8]}...)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['r2_score']:.4f}" if isinstance(metrics['r2_score'], float) else metrics['r2_score'])
        with col2:
            st.metric("MSE", f"{metrics['mse']:.4f}" if isinstance(metrics['mse'], float) else metrics['mse'])
        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.4f}" if isinstance(metrics['rmse'], float) else metrics['rmse'])
        
        st.info("This model predicts customer review scores based on order and product features.")
    else:
        st.warning("⚠️ No trained model found. Please train a model first.")

elif app_mode == "Prediction":
    st.header("🎯 Single Prediction")
    
    model, metrics, run_id = get_latest_model()
    
    if model:
        st.success(f"Model loaded (Run: {run_id[:8]}...)")
        
        st.subheader("Enter Feature Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            payment_sequential = st.number_input("Payment Sequential", min_value=1, value=1)
            payment_installments = st.number_input("Payment Installments", min_value=1, value=1)
            payment_value = st.number_input("Payment Value", min_value=0.0, value=100.0)
            price = st.number_input("Price", min_value=0.0, value=50.0)
            freight_value = st.number_input("Freight Value", min_value=0.0, value=10.0)
        
        with col2:
            product_name_length = st.number_input("Product Name Length", min_value=1, value=50)
            product_description_length = st.number_input("Product Description Length", min_value=0, value=500)
            product_photos_qty = st.number_input("Product Photos Quantity", min_value=0, value=3)
            product_weight_g = st.number_input("Product Weight (g)", min_value=0.0, value=500.0)
            product_length_cm = st.number_input("Product Length (cm)", min_value=0.0, value=20.0)
        
        col3, col4 = st.columns(2)
        with col3:
            product_height_cm = st.number_input("Product Height (cm)", min_value=0.0, value=15.0)
        with col4:
            product_width_cm = st.number_input("Product Width (cm)", min_value=0.0, value=10.0)
        
        if st.button("🔮 Predict Review Score", type="primary"):
            # Create feature array in the correct order (matching training data)
            features = pd.DataFrame([{
                'payment_sequential': payment_sequential,
                'payment_installments': payment_installments,
                'payment_value': payment_value,
                'price': price,
                'freight_value': freight_value,
                'product_name_lenght': product_name_length,
                'product_description_lenght': product_description_length,
                'product_photos_qty': product_photos_qty,
                'product_weight_g': product_weight_g,
                'product_length_cm': product_length_cm,
                'product_height_cm': product_height_cm,
                'product_width_cm': product_width_cm
            }])
            
            try:
                prediction = model.predict(features)[0]
                
                st.success("### 🎉 Prediction Result")
                st.metric("Predicted Review Score", f"{prediction:.2f} / 5.0")
                
                # Show interpretation
                if prediction >= 4.5:
                    st.success("😊 Excellent! Customer likely very satisfied")
                elif prediction >= 3.5:
                    st.info("🙂 Good! Customer likely satisfied")
                elif prediction >= 2.5:
                    st.warning("😐 Average! Room for improvement")
                else:
                    st.error("😞 Poor! Customer likely unsatisfied")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("⚠️ No model available. Please train a model first.")

elif app_mode == "Batch Prediction":
    st.header("📁 Batch Prediction")
    
    model, metrics, run_id = get_latest_model()
    
    if model:
        st.success(f"Model loaded (Run: {run_id[:8]}...)")
        
        uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                try:
                    # Ensure columns match training data
                    predictions = model.predict(df)
                    df['predicted_review_score'] = predictions
                    
                    st.success("✅ Predictions completed!")
                    st.write("### Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.write("### Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Score", f"{predictions.mean():.2f}")
                    with col2:
                        st.metric("Min Score", f"{predictions.min():.2f}")
                    with col3:
                        st.metric("Max Score", f"{predictions.max():.2f}")
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    else:
        st.warning("⚠️ No model available. Please train a model first.")

elif app_mode == "Train New Model":
    st.header("🎓 Train New Model")
    
    st.info("Train a new model using the training pipeline")
    
    data_path = st.text_input("Data Path", value="./data/olist_customers_dataset.csv")
    
    if st.button("🚂 Start Training", type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                run = train_pipeline(data_path=data_path)
                st.success(f"✅ Training completed! Run name: {run.name}")
                st.info("Refresh the page to see updated model metrics.")
                st.balloons()
            except Exception as e:
                st.error(f"Training failed: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Built with ZenML, MLflow, and Streamlit")
