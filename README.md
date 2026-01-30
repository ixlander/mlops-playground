# MLOps Pipeline

A simple MLOps project with training, evaluation, deployment, and inference pipelines.

## Project Structure
```
MLOps/
├── src/                          # Core business logic
│   ├── data_cleaning.py         # Data preprocessing strategies
│   ├── evaluation.py            # Model evaluation metrics
│   └── model_dev.py             # Model training classes
├── steps/                        # ZenML pipeline steps
│   ├── clean_data.py            # Data cleaning step
│   ├── config.py                # Configuration classes
│   ├── evaluation.py            # Model evaluation step
│   ├── inference.py             # Prediction step
│   ├── ingest_data.py           # Data ingestion step
│   ├── model_train.py           # Model training step
│   └── preprocess_inference.py  # Inference preprocessing step
├── pipelines/                    # ZenML pipelines
│   ├── training_pipeline.py     # Training pipeline
│   ├── deployment_pipeline.py   # Deployment pipeline
│   └── inference_pipeline.py    # Inference pipeline
├── run_pipeline.py              # Training script
├── run_deployment.py            # Deployment/inference script
└── streamlit_app.py             # Streamlit web interface
```

## Installation
```bash
# Install dependencies
pip install zenml mlflow scikit-learn pandas numpy click rich

# Initialize ZenML
zenml init

# Set up MLflow integration
zenml integration install mlflow
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -e mlflow_tracker -d mlflow_deployer
zenml stack set mlflow_stack
```

## Usage

### 1. Training Pipeline
Run the training pipeline with your data:
```bash
python run_pipeline.py -d ./data/olist_customers_dataset.csv
```

This will:
- Load and clean the data
- Train a Linear Regression model
- Evaluate with R2, MSE, and RMSE metrics
- Log results to MLflow

### 2. Deployment Pipeline (Linux/WSL only)
Deploy the model (only if R2 >= min_r2):
```bash
python run_deployment.py --config deploy --data-path ./data/olist_customers_dataset.csv --min-r2 0.0
```

**Windows Limitation**: MLflow model deployment requires daemon processes which are not supported on Windows. Use WSL or Linux for deployment features.

### 3. Manual Model Serving (Windows Alternative)
After training, serve the model manually with MLflow:
```bash
# View MLflow tracking UI to find run ID
mlflow ui --backend-store-uri "file:C:\Users\Admin\AppData\Roaming\zenml\local_stores\<store-id>\mlruns"

# Serve the model manually (replace <run-id> with actual run ID from MLflow UI)
mlflow models serve -m runs:/<run-id>/model -p 5000 --no-conda
```

### 4. Inference Pipeline (Linux/WSL only)
Run predictions on new data:
```bash
python run_deployment.py --config predict --data-path ./data/olist_customers_dataset.csv
```

### 5. View MLflow UI
Track experiments and models:
```bash
mlflow ui --backend-store-uri "file:C:\Users\Admin\AppData\Roaming\zenml\local_stores\<store-id>\mlruns"
```

Access at: http://localhost:5000

### 6. Streamlit Web Interface
Launch the interactive web application:
```bash
streamlit run streamlit_app.py
```

Features:
- Model Information: View current model metrics (R2, MSE, RMSE)
- Single Prediction: Enter feature values manually for predictions
- Batch Prediction: Upload CSV files for bulk predictions with download
- Train New Model: Trigger training pipeline from the web interface

Access at: http://localhost:8501

## CLI Options

### run_pipeline.py
- `--data-path`, `-d`: Path to training data CSV (required)

### run_deployment.py
- `--config`, `-c`: Mode - `deploy`, `predict`, or `deploy_and_predict` (default: `deploy_and_predict`)
- `--data-path`, `-d`: Path to data CSV (required)
- `--min-r2`: Minimum R2 score to deploy model (default: 0.2)

## Platform Notes

### Windows
- Training pipeline works fully
- Model evaluation and MLflow tracking work
- Streamlit web interface works fully
- Automatic deployment service not supported (daemon limitation)
- Manual model serving with `mlflow models serve` works as alternative

### Linux/WSL/Mac
- All features supported including automatic deployment