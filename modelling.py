import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
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
    y = LabelEncoder().fit_transform(df[TARGET].astype(str))

    X = X.select_dtypes(exclude=["object"])

    return X, y


def main():
    mlflow.set_experiment("Tora-Margaretha")

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
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))


if __name__ == "__main__":
    main()
