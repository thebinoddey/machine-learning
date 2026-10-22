# Customer Churn Prediction using Logistic Regression

A machine learning project that predicts whether a telecom customer is likely to leave a service using **Logistic Regression**, Python, Pandas, and Scikit-learn.

## Overview

Customer churn is the rate at which customers stop using a company's products or services. Predicting churn helps businesses identify customers who may leave and take proactive retention measures.

This project implements an end-to-end machine learning workflow, including data loading, preprocessing, model training, evaluation, and customer-level predictions.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Dataset:** Telco Customer Churn (Kaggle)

## Project Structure

```text
Project3_CustomerChurn/
├── main.py
├── data_loader.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── predict.py
├── churn.csv
└── models/
    └── churn_model.pkl
```

### File Responsibilities

| File | Purpose |
|---|---|
| `main.py` | Orchestrates the complete ML workflow |
| `data_loader.py` | Loads and validates the dataset |
| `preprocessing.py` | Handles missing values, scaling, and categorical encoding |
| `train.py` | Splits the data, trains Logistic Regression, and saves the pipeline |
| `evaluate.py` | Calculates classification metrics |
| `predict.py` | Loads the saved model and predicts customer churn |

## Dataset

The project uses the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The dataset contains customer information such as tenure, contract type, monthly charges, and service subscriptions.

- **Target variable:** `Churn`
- `Yes` is encoded as `1`.
- `No` is encoded as `0`.
- `customerID` is excluded from the model features.

Download the dataset and place the CSV file in the project root, renamed to `churn.csv`.

The dataset represents a fictional telecom company and is intended for learning and experimentation.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Project3_CustomerChurn
```

Replace `<your-repository-url>` with your GitHub repository URL. Alternatively, open your existing project folder directly in VS Code.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn joblib
```

## Run the Project

Ensure that `churn.csv` is in the project root and run:

```bash
python main.py
```

The program will:

1. Load the dataset.
2. Separate input features from the target.
3. Define preprocessing transformations.
4. Split the data into training and testing sets.
5. Train the Logistic Regression model.
6. Save the fitted pipeline to `models/churn_model.pkl`.
7. Evaluate the model on held-out test data.
8. Generate a churn prediction for a sample customer.

The `models` directory is created automatically during training.

## Machine Learning Workflow

### 1. Data preprocessing

- Converts `TotalCharges` to numeric values.
- Separates the target variable from the features.
- Handles missing numerical values using median imputation.
- Handles missing categorical values using most-frequent imputation.
- Scales numerical features using `StandardScaler`.
- Encodes categorical features using `OneHotEncoder`.

### 2. Model training

The project uses Logistic Regression with a maximum of 1,000 iterations.

The dataset is split into:

- **80% training data**
- **20% testing data**

A fixed random seed and stratified splitting are used to make the experiment reproducible.

Preprocessing and classification are combined into a Scikit-learn `Pipeline`, ensuring that the same transformations are applied during training and prediction.

### 3. Model evaluation

The project reports:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churners who actually churned.
- **Recall:** Proportion of actual churners correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Breakdown of correct and incorrect predictions.
- **ROC-AUC:** Ability to rank churners above non-churners across probability thresholds.

No particular performance score is guaranteed; results depend on the data and implementation.

### 4. Prediction

The trained pipeline is saved using Joblib. The prediction module loads the saved pipeline and returns:

- Predicted class: `Churn` or `No Churn`
- Estimated churn probability

The current implementation demonstrates prediction using a held-out customer record.

## Future Improvements

- Compare Logistic Regression against a `DummyClassifier` baseline.
- Experiment with decision thresholds and class weights.
- Add cross-validation and hyperparameter tuning.
- Visualize churn patterns and model performance.
- Build an interactive Streamlit dashboard.
- Deploy a prediction API using FastAPI.
- Add automated tests and data validation.

## Limitations

- The dataset is sample data, not live telecom customer data.
- Model performance on this dataset does not guarantee performance in a real business environment.
- A predicted churn probability is a model estimate, not a certainty.
- The current project is a learning prototype and has not been validated for production use.

## Learning Objectives

This project demonstrates practical understanding of:

- Supervised machine learning
- Binary classification
- Data preprocessing
- Feature encoding and scaling
- Train-test splitting
- Scikit-learn pipelines
- Model evaluation
- Model serialization and inference

## License

Check the dataset's licensing and usage terms before redistributing it. Add a project license, such as MIT, if you want to make your own source code openly reusable.

---

**Author:** Binod Dey  
**Project:** Machine Learning — Customer Churn Prediction
