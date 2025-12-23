import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # ===============================
    # LOAD CLEAN DATASET
    # ===============================
    df = pd.read_csv("housedata-preprocessing.csv")

    TARGET = "price_category"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===============================
    # MLFLOW SETUP
    # ===============================
    mlflow.set_experiment("Tora-Margaretha")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest-CI"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Training finished | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
