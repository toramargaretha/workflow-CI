import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")


# =====================================================
# LOAD DATA
# =====================================================
def load_preprocessed_data(filepath):
    df = pd.read_csv(filepath)

    print("COLUMNS FOUND:", df.columns.tolist())

    TARGET = "price_category"

    # Buat target jika belum ada
    if TARGET not in df.columns:
        print("Target 'price_category' not found. Creating from 'price' column...")
        df[TARGET] = pd.qcut(
            df["price"],
            q=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )

    # Encode target ke numerik (WAJIB untuk sklearn)
    df[TARGET] = df[TARGET].astype("category").cat.codes

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Drop kolom non-numerik
    non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric_cols:
        print("Dropping non-numeric columns:", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    return X, y


# =====================================================
# MAIN
# =====================================================
def main():
    mlflow.set_experiment("Tora-Margaretha")

    with mlflow.start_run():

        print("Starting MLflow training run")

        # Load data
        X, y = load_preprocessed_data("housedata-preprocessing.csv")
        print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        models = {
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
        }

        for name, model in models.items():
            print(f"Training {name}...")

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            # LOG ke MLflow (INI YANG DINILAI)
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            print(f"{name} | acc={acc:.4f} f1={f1:.4f}")

        print("Training finished successfully")


if __name__ == "__main__":
    main()
