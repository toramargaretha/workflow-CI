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

    TARGET = "price_category"

    if TARGET not in df.columns:
        df[TARGET] = pd.qcut(
            df["price"], q=3, labels=["Rendah", "Sedang", "Tinggi"]
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    non_numeric_cols = X.select_dtypes(include=["object"]).columns
    X = X.drop(columns=non_numeric_cols)

    return X, y


# =====================================================
# TRAIN MODELS WITH MLFLOW
# =====================================================
def train_models(X_train, X_test, y_train, y_test):

    models = {
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVC": SVC(),
        "GaussianNB": GaussianNB()
    }

    results = []

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("model_type", model_name)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            results.append({
                "Model": model_name,
                "Accuracy": acc,
                "F1": f1
            })

    return pd.DataFrame(results)


# =====================================================
# MAIN
# =====================================================
def main():
    mlflow.set_experiment("Tora-Margaretha")

    X, y = load_preprocessed_data("housedata-preprocessing.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = train_models(X_train, X_test, y_train, y_test)

    print("\nTraining completed!")
    print(results)


if __name__ == "__main__":
    main()
