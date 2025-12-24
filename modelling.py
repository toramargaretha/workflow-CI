import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings("ignore")


# =====================================================
# LOAD DATA
# =====================================================
def load_preprocessed_data(filepath):
    df = pd.read_csv(filepath)

    print("COLUMNS FOUND:", df.columns.tolist())

    TARGET = "price_category"

    if TARGET not in df.columns:
        print("Creating target from price...")
        df[TARGET] = pd.qcut(
            df["price"], q=3, labels=["Rendah", "Sedang", "Tinggi"]
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y)

    non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric_cols:
        print("Dropping non-numeric columns:", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    return X, y


# =====================================================
# MAIN
# =====================================================
def main():
    print("Starting ML training...")

    X, y = load_preprocessed_data("housedata-preprocessing.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GaussianNB": GaussianNB()
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # ✅ NESTED RUN (INI KUNCI-NYA)
        with mlflow.start_run(run_name=name, nested=True):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    print("\nTraining finished successfully!")


if __name__ == "__main__":
    main()
