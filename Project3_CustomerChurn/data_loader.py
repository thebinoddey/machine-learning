import pandas as pd
from pathlib import Path

DATA_PATH = Path("churn.csv")

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please ensure the file exists.")
    
    data = pd.read_csv(DATA_PATH)
    if data.empty:
        raise ValueError("The data file is empty. Please provide a valid dataset.")
    
    print("Dataset loaded successfully.")
    print("Dataset shape: ", data.shape)
    print("First 5 rows: /n")
    print(data.head(5))
    data.info()
    
    return data