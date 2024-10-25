# IBM Telco Customer Churn Prediction Project

This project predicts customer churn using machine learning models, with a focus on the **IBM Telco Customer Churn Dataset** from Kaggle. We use **XGBoost** as the primary model for prediction, and deploy a **Streamlit app** to interactively predict customer churn based on user input.

## Dataset Overview

- **Source**: [Kaggle - IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Size**: 7,043 rows, 21 columns
- **Target Variable**: `Churn` (binary: Yes/No) – whether the customer has churned or not.

## Project Structure


## Overview of Steps

1. **Data Exploration**: In the Jupyter notebook, we perform Exploratory Data Analysis (EDA) to understand the dataset and identify patterns related to customer churn.
2. **Feature Engineering**: We preprocess the data, create new features (e.g., `MonthlyCostPerYear`), and handle categorical data using encoding.
3. **Model Selection**: The XGBoost model is used due to its performance on structured data.
4. **Model Evaluation**: We evaluate the model using metrics like AUC-ROC, accuracy, precision, and recall, focusing on maximizing recall to capture as many churn cases as possible.
5. **Deployment**: The model is deployed using a Streamlit application for easy, interactive predictions.

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/ChurnPredictionProject.git
    cd ChurnPredictionProject
    ```

2. **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run Jupyter Notebook**:
   - Open `notebooks/churn_prediction_notebook.ipynb` to view or execute data exploration and model training steps.
   - Ensure you have the dataset in the `data/churn/` directory.

5. **Run Streamlit App**:
    ```bash
    streamlit run app/app.py
    ```

## Files and Directories

### 1. `data/`
- **Purpose**: Contains the dataset used for analysis and model training.
- **Data**: Place the dataset file (e.g., `Telco-Customer-Churn.csv`) here.

### 2. `notebooks/`
- **Purpose**: Contains Jupyter notebooks used for Exploratory Data Analysis (EDA), feature engineering, and model building.
- **File**: `churn_prediction_notebook.ipynb` - Main notebook for all analysis and model training steps.

### 3. `models/`
- **Purpose**: Stores the trained XGBoost model used in the Streamlit app.
- **File**: `final_xgboost_churn_model.pkl` - Trained model for making predictions in the app.

### 4. `app/`
- **Purpose**: Contains the Streamlit app for real-time churn prediction.
- **File**: `app.py` - Main application file that uses the trained model to make predictions based on user input.

### 5. `requirements.txt`
- **Purpose**: Lists all required dependencies for the project, such as Streamlit, XGBoost, pandas, and scikit-learn.

### 6. `.gitignore`
- **Purpose**: Specifies files and directories to be ignored by Git. This includes:
    - `venv/` - Virtual environment folder
    - `.DS_Store` - macOS system files
    - `__pycache__/` - Python cache files
    - `models/*.pkl` - Large model files if applicable

## Model Performance

After training, the **XGBoost model** achieved the following results:

- **AUC-ROC**: 0.840
- **Accuracy**: 0.700
- **Precision**: 0.465
- **Recall**: 0.858

These results indicate a strong ability to capture churn cases (high recall), which is ideal for a customer churn prediction task.

## Usage

The Streamlit app allows users to input customer details and predict the likelihood of churn. It’s useful for business teams or customer service departments to identify high-risk customers and take proactive measures.


## Acknowledgments

Special thanks to IBM and Kaggle for providing the Telco Customer Churn dataset.

