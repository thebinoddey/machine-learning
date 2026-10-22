import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def prepare_data(data):
    df = data.copy()
    
    #convert numeric-looking text to numbers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    #Separate target from feature
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(
        columns = ["Churn", "customerID"],
        errors="ignore"
    )
    
    #Identify numerical and categorical
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    #Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    #categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    #Apply the correct transformation 
    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_cols
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    ])
    
    #Validate target
    if y.isna().any():
        raise ValueError("The churn column contains unexpected or missing values")
    
    print("\nNumerical features:", numerical_cols)
    print("Categorical features:", categorical_cols)

    return X, y, preprocessor