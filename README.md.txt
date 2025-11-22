# Tourism Package Purchase Prediction MLOps Project

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting tourism package purchases. It covers data preparation, model training, model deployment to Hugging Face Hub, a Streamlit-based front-end for inference, and a CI/CD pipeline using GitHub Actions for automated Docker image builds and pushes.

## Project Structure

```
project-root/
│
├── data/
│   ├── train.csv         # Processed training data
│   └── test.csv          # Processed testing data
├── Dockerfile            # Dockerfile for containerizing the Streamlit app
├── requirements.txt      # Python dependencies
├── app.py                # Streamlit front-end application
├── src/
│   ├── data_preparation.py # Script for data cleaning and dataset registration
│   ├── model_training.py   # Script for model training and model registration
│   ├── deployment_utils.py # Utility functions for model inference
│   └── config.py           # Configuration variables
├── .github/workflows/
│   └── pipeline.yml      # GitHub Actions CI/CD pipeline
├── README.md             # This file
└── [project_notebook].ipynb # Original development notebook (e.g., this Colab notebook)
```

## Project Components

### 1. `src/data_preparation.py`

This script handles the initial data processing. It performs:
- Data cleaning (dropping irrelevant columns, filling missing values, encoding categorical features).
- Splits the data into training and testing sets.
- Saves the processed `train.csv` and `test.csv` locally in the `data/` directory.
- Registers (pushes) the processed datasets to Hugging Face Datasets under `pallabbh/tourism-dataset-train` and `pallabbh/tourism-dataset-test`.

### 2. `src/model_training.py`

This script is responsible for training the machine learning model:
- Loads the pre-processed datasets from Hugging Face Datasets.
- Trains an XGBoost classifier.
- Logs model parameters and metrics (e.g., accuracy) using MLflow.
- Saves the trained model locally (`model_dir/best_xgb_model.json`).
- Pushes the trained model to Hugging Face Model Hub under `pallabbh/tourism-model`.

### 3. `src/deployment_utils.py`

This module provides utility functions for model inference:
- `load_model_from_hf()`: Downloads the `best_xgb_model.json` from `pallabbh/tourism-model` on Hugging Face Hub and loads it into an XGBoost classifier object.
- `predict(input_dict)`: Takes a dictionary of input features, converts it into a pandas DataFrame, and uses the loaded model to make a prediction.

### 4. `src/config.py`

A simple configuration file to store important paths and settings, such as Hugging Face dataset and model repository IDs, and the target column name.

### 5. `app.py` (Streamlit Front-end)

This is a Streamlit application that provides an interactive web interface for users to get predictions from the deployed model. It collects user inputs for various features and calls the `predict` function from `src/deployment_utils.py` to display the prediction (Purchase/Not Purchase).

### 6. `Dockerfile`

This file containerizes the Streamlit application and its dependencies. It sets up a Python 3.9 environment, installs all packages from `requirements.txt`, copies the application code, exposes port `8501`, and defines the entry point to run the Streamlit app.

### 7. `.github/workflows/pipeline.yml` (GitHub Actions CI/CD)

This GitHub Actions workflow automates the Continuous Integration/Continuous Deployment process:
- **Trigger:** Initiates on every push to the `main` branch.
- **Steps:**
    - Checks out the repository code.
    - Logs into Docker Hub using `DOCKER_USERNAME` and `DOCKER_PASSWORD` GitHub secrets.
    - Builds the Docker image based on the `Dockerfile`.
    - Pushes the built Docker image to Docker Hub, tagging it with `pallabbh/tourism-app:latest` and a unique `pallabbh/tourism-app:${{ github.sha }}` tag.

## Setup and Usage

### Prerequisites

- Python 3.9+
- Git
- Docker Desktop (for local Docker builds/runs)
- Hugging Face Account and API Token (with write access to create/push datasets and models)
- Docker Hub Account

### 1. Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bpallab/walsh-tourism.git
    cd walsh-tourism
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Hugging Face Token (Environment Variable):**
    Ensure your Hugging Face API token is set as an environment variable named `HF_TOKEN`. For example:
    ```bash
    export HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
    ```

4.  **Run Data Preparation (if data is not already on Hugging Face Hub):**
    To create and push the datasets to Hugging Face, you'll need your `tourism.csv` file. You can adapt `src/data_preparation.py` to read from a local path.
    ```bash
    python src/data_preparation.py
    ```
    *Note: The current `data_preparation.py` is configured for Google Colab with Google Drive mounting. You might need to modify the data loading line `df = pd.read_csv('/content/drive/My Drive/Walsh DBA/tourism.csv')` to a local path or a different data source if running outside Colab.* For initial setup, running it in the Colab notebook is recommended to push the datasets.

5.  **Run Model Training (if model is not already on Hugging Face Hub):**
    ```bash
    python src/model_training.py
    ```
    This will train the model and push it to `pallabbh/tourism-model` on Hugging Face.

6.  **Run Streamlit Application Locally:**
    ```bash
    streamlit run app.py
    ```
    Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### 2. Docker Deployment

1.  **Build the Docker Image:**
    Navigate to the project root directory and run:
    ```bash
    docker build -t pallabbh/tourism-app:latest .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -p 8501:8501 pallabbh/tourism-app:latest
    ```
    The Streamlit app will be accessible at `http://localhost:8501` in your web browser.

### 3. CI/CD with GitHub Actions

1.  **Configure GitHub Secrets:**
    In your GitHub repository settings (`Settings` -> `Secrets and variables` -> `Actions`), add the following repository secrets:
    - `DOCKER_USERNAME`: Your Docker Hub username.
    - `DOCKER_PASSWORD`: Your Docker Hub password or an access token.

2.  **Push to `main` branch:**
    Every push to the `main` branch of your GitHub repository will automatically trigger the CI/CD pipeline defined in `.github/workflows/pipeline.yml`. This pipeline will build the Docker image and push it to your Docker Hub repository (`pallabbh/tourism-app`).

3.  **Monitor Workflow:**
    Check the `Actions` tab in your GitHub repository to monitor the status and logs of the CI/CD pipeline runs.

## Hugging Face Resources

- **Datasets:**
  - `pallabbh/tourism-dataset-train`
  - `pallabbh/tourism-dataset-test`
- **Model:**
  - `pallabbh/tourism-model`

## Next Steps

- Deploy the Docker image to a cloud platform (e.g., AWS EC2, Google Cloud Run, Azure Container Instances).
- Set up monitoring for the deployed model's performance and data drift.
- Implement more sophisticated experiment tracking and model registry features using MLflow.
- Enhance the Streamlit UI with more visualizations or input validation.
