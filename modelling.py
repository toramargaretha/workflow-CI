import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


def load_preprocessed_data(filepath):
    df = pd.read_csv(filepath)

    TARGET = "price_category"

    if TARGET not in df.columns:
        df[TARGET] = pd.qcut(
            df["price"],
            q=3,
            labels=["Rendah", "Sedang", "Tinggi"]
        )

    X = df.drop(columns=[TARGET])

    y = df[TARGET].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.select_dtypes(exclude=["object"])

    return X, y


def main():
    mlflow.set_experiment("Tora-Margaretha")

    # ✅ AUTolog aktif (WAJIB Kriteria 2)
    mlflow.sklearn.autolog(log_models=True)

    X, y = load_preprocessed_data("housedata-preprocessing.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        GaussianNB()
    ]

    # ❗ TIDAK pakai start_run
    for model in models:
        model.fit(X_train, y_train)
        model.score(X_test, y_test)


if __name__ == "__main__":
    main()
