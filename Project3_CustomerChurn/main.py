from data_loader import load_data
from preprocessing import prepare_data
from train import train_model
from evaluate import evaluate_model
from predict import load_model, predict_customer


def main():
    # Step 1: Load dataset
    df = load_data()

    # Step 2: Prepare features, labels, and preprocessing
    X, y, preprocessor = prepare_data(df)

    # Step 3: Train model and obtain test data
    _, X_test, y_test = train_model(
        X,
        y,
        preprocessor
    )

    # Step 4: Evaluate the saved model
    model = load_model()
    evaluate_model(model, X_test, y_test)

    # Step 5: Demonstrate a prediction
    # Use one held-out customer as an inference example
    customer = X_test.iloc[[0]]
    predict_customer(model, customer)


if __name__ == "__main__":
    main()