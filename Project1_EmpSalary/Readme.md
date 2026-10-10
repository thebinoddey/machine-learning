# Indian Tech Salary Prediction using Linear Regression

A beginner-friendly Machine Learning project that explores Indian tech job listings and predicts salary midpoints using Linear Regression.

## Project Overview

The goal of this project is to understand how job requirements and company-related factors relate to advertised salaries in India's technology job market.

The project covers data cleaning, Exploratory Data Analysis (EDA), feature selection, model training, and evaluation using regression metrics.

## Dataset

The dataset contains approximately 23,201 job listings.

| Column | Description |
|---|---|
| `job_id` | Unique job listing identifier |
| `company_rating` | Company rating |
| `experience_min_yrs` | Minimum required experience in years |
| `experience_max_yrs` | Maximum required experience in years |
| `salary_min_lpa` | Minimum advertised salary in LPA |
| `salary_max_lpa` | Maximum advertised salary in LPA |
| `skills_count` | Number of skills associated with the listing |
| `salary_midpoint_lpa` | Midpoint of the advertised salary range |
| `days_since_posted` | Age of the job posting in days |

**Source** Indian Tech Job Market 2026 | 23K+ Records by Shreyas Gade on Kaggle (URL: https://www.kaggle.com/datasets/shree0910/india-tech-job-market-2026-23k-records)

**Note:** Verify the dataset's provenance and whether zero salary values represent undisclosed compensation before modeling.


## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

### 1. Data Loading and Inspection

- Load the CSV dataset using Pandas.
- Inspect the dataset's shape, columns, data types, and descriptive statistics.
- Check for missing values and duplicate records.

### 2. Exploratory Data Analysis (EDA)

Visualizations include:

- Histograms for numeric variables.
- Scatterplots to explore experience and salary relationships.
- Regression plots using Seaborn.
- Boxplots to inspect potential outliers.
- Correlation heatmaps to examine numeric relationships.
- Pairplots to compare selected numeric features.

### 3. Data Cleaning

- Investigate missing and zero-coded salary values.
- Check salary ranges for inconsistencies.
- Examine extreme values and duplicates.
- Remove identifier columns from model features.

Undisclosed salaries should not be treated as genuine zero salaries.

### 4. Feature Selection

**Target variable (`y`):**
- `salary_midpoint_lpa`

**Initial features (`X`):**
- `experience_min_yrs`
- `experience_max_yrs`
- `company_rating`
- `skills_count`

The features can be expanded after evaluating their usefulness.

`job_id` is excluded because it is an identifier. `salary_min_lpa` and `salary_max_lpa` are excluded because the target midpoint is derived from those values. Including them would cause target leakage.

### 5. Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

A fixed `random_state` ensures reproducibility.

### 6. Model Training

The initial model uses Scikit-learn's `LinearRegression`.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### 7. Model Evaluation

The model is evaluated using:

| Metric | Purpose |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Average squared prediction error |
| RMSE | Error magnitude in the target's units |
| R² | Performance relative to a mean-prediction baseline |
| Pearson correlation (r) | Strength of linear association between actual and predicted salaries |

Actual-versus-predicted plots and residual plots are also used to inspect model performance.

A baseline model that always predicts the mean training salary provides a useful comparison.

## Installation and Usage

### Prerequisites

Install Python 3.10 or later.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

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
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open your project notebook, update the CSV path, and run the cells in order.

## Limitations

- Job listings represent advertised compensation, not necessarily actual employee salaries.
- Some salary values may be undisclosed or encoded as zero.
- Job requirements are not equivalent to an employee's actual experience.
- Linear Regression may not capture nonlinear salary relationships.
- Synthetic or biased data can produce misleading evaluation results.
- Model performance on this dataset does not guarantee accuracy on future job listings.

## Future Improvements

- Add job role and location using appropriate categorical encoding.
- Investigate additional features such as company size and employment type.
- Use cross-validation for more robust evaluation.
- Compare Linear Regression with Ridge Regression and tree-based models.
- Evaluate performance across different job roles and experience ranges.
- Test on newer job listings to assess generalization.

## Learning Outcomes

Through this project, I aim to understand:

- Exploratory Data Analysis.
- Feature selection and data preprocessing.
- Training and testing Machine Learning models.
- Regression evaluation metrics.
- Residual analysis and model limitations.
- The importance of preventing target leakage.

## Disclaimer

This project is for educational purposes. Predictions are estimates and should not be treated as authoritative salary benchmarks or career advice.

Dataset attribution and licensing should be added after verifying the original source.
