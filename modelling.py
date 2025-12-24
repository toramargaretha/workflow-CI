import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")


# =====================================================
# LOAD DATA (CI-SAFE)
# =====================================================
def load_preprocessed_data(filepath):
    df = pd.read_csv(filepath)

    print("COLUMNS FOUND:", df.columns.tolist())

    TARGET = "price_category"

    # ===============================
    # CREATE TARGET IF NOT EXISTS
    # ===============================
    if TARGET not in df.columns:
        if "price" not in df.columns:
            raise ValueError(
                "Target 'price_category' not found AND no 'price' column to derive it."
            )

        print("Target 'price_category' not found. Creating from 'price' column...")

        df[TARGET] = pd.qcut(
            df["price"],
            q=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )

    # ===============================
    # SPLIT X & y
    # ===============================
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    # ===============================
    # DROP NON-NUMERIC FEATURES (CRITICAL FIX)
    # ===============================
    non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if non_numeric_cols:
        print("Dropping non-numeric columns:", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    return X, y


# =====================================================
# TRAIN MODELS WITH MLFLOW AUTOLOG
# =====================================================
def train_models_with_autolog(X_train, X_test, y_train, y_test):

    models = {
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(kernel="rbf", random_state=42),
        "GaussianNB": GaussianNB()
    }

    results = []

    mlflow.sklearn.autolog(log_models=True)

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

            print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

    mlflow.sklearn.autolog(disable=True)
    return pd.DataFrame(results)


# =====================================================
# MAIN
# =====================================================
def main():

    # ===============================
    # CI-FRIENDLY MLFLOW SETUP
    # ===============================
    mlflow.set_experiment("Tora-Margaretha")

    print("Starting model training with MLflow...")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {mlflow.get_experiment_by_name('Tora-Margaretha').name}\n")

    filepath = "housedata-preprocessing.csv"
    X, y = load_preprocessed_data(filepath)

    print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results_df = train_models_with_autolog(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_model = results_df.loc[results_df["Accuracy"].idxmax()]
    print("\nBest Model:", best_model["Model"])
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()

