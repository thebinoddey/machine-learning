from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/churn_model.pkl")


def train_model(X, y, preprocessor):
    # Split data before fitting preprocessing steps
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Combine preprocessing and model into one pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ])

    # Learn from training data
    model.fit(X_train, y_train)

    # Create folder and save the complete fitted pipeline
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("\nModel trained successfully.")
    print("Saved model to:", MODEL_PATH)

    return model, X_test, y_test