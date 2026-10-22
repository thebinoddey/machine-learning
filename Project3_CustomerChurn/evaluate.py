from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    # Estimated probability that each customer will churn
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print(
        "\nROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 4)
    )