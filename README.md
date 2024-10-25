# Customer Churn Prediction Project

This project predicts customer churn using machine learning models. The trained model can be deployed with a Streamlit app for easy access.

## Project Structure
- `data/`: Contains the churn dataset used for training and evaluation.
- `notebooks/`: Jupyter notebook used for data exploration, feature engineering, and model training.
- `models/`: Stores the trained model file (`final_xgboost_churn_model.pkl`).
- `app/`: Contains the Streamlit app (`app.py`) for churn prediction.
- `requirements.txt`: Dependencies required to run the project.

## Setup Instructions

1. **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Jupyter Notebook**:
   Open the notebook in `notebooks/churn_prediction_notebook.ipynb` for analysis and model training details.

4. **Run Streamlit App**:
    ```bash
    streamlit run app/app.py
    ```

## Usage
Run the Streamlit app to interactively predict customer churn by entering customer details.

## License
Specify the license if applicable.
