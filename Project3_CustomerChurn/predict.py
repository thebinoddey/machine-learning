from pathlib import Path
import joblib


MODEL_PATH = Path("models/churn_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run the training step first."
        )

    return joblib.load(MODEL_PATH)


def predict_customer(model, customer):
    # customer must be a DataFrame with the original
    # feature columns expected by the pipeline.
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0, 1]

    result = {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4)
    }

    print("\n--- Customer Prediction ---")
    print("Prediction:", result["prediction"])
    print(
        f"Churn probability: {probability:.2%}"
    )

    return result