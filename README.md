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
└── run_deployment.py            # Deployment/inference script
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
zenml stack register mlflow_stack -a default -o default -e mlflow_tracker
zenml stack set mlflow_stack
```

## Usage

### 1. Training Pipeline
Run the training pipeline with your data:
```bash
python run_pipeline.py --data-path /path/to/your/data.csv
```

Example:
```bash
python run_pipeline.py -d ./data/olist_customers_dataset.csv
```

### 2. Deployment Pipeline
Deploy the model (only if R2 >= min_r2):
```bash
python run_deployment.py --config deploy --data-path /path/to/data.csv --min-r2 0.2
```

### 3. Inference Pipeline
Run predictions on new data:
```bash
python run_deployment.py --config predict --data-path /path/to/new_data.csv
```

### 4. Deploy and Predict
Do both in one command:
```bash
python run_deployment.py --config deploy_and_predict --data-path /path/to/data.csv --min-r2 0.2
```

## CLI Options

### run_pipeline.py
- `--data-path`, `-d`: Path to training data CSV (required)

### run_deployment.py
- `--config`, `-c`: Mode - `deploy`, `predict`, or `deploy_and_predict` (default: `deploy_and_predict`)
- `--data-path`, `-d`: Path to data CSV (required)
- `--min-r2`: Minimum R2 score to deploy model (default: 0.2)