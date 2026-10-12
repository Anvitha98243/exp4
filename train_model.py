
import os
import random
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_demo_dataset():
    random.seed(42)

    crops = {
        "rice": [80, 40, 40, 25, 85, 6.2, 200],
        "maize": [70, 50, 30, 23, 65, 6.5, 100],
        "chickpea": [40, 65, 80, 20, 50, 7.0, 80],
        "cotton": [100, 45, 50, 28, 60, 6.0, 90],
    }

    rows = []
    for crop, values in crops.items():
        for _ in range(50):
            row = [
                max(0, value + random.uniform(-5, 5))
                for value in values
            ]
            rows.append(row + [crop])

    columns = [
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall", "label"
    ]

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(
        "data/crop_raw.csv", index=False
    )


def main():
    dataset_path = "data/crop_raw.csv"

    if not os.path.exists(dataset_path):
        create_demo_dataset()

    data = pd.read_csv(dataset_path)

    features = [
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ]

    X = data[features]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/crop_model.joblib")

    print("Model training completed.")
    print("Test accuracy:", round(accuracy, 4))
    print("Artifact saved: artifacts/crop_model.joblib")


if __name__ == "__main__":
    main()
