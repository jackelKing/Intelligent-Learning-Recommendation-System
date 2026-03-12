import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib


def train_xgboost():

    df = pd.read_csv("data/processed/feature_engineered_data.csv")

    label_map = {
        "beginner": 0,
        "intermediate": 1,
        "advanced": 2
    }

    y = df["knowledge_level"].map(label_map)


    # remove target from features
    X = df.drop(["knowledge_level"], axis=1)
    X = pd.get_dummies(X)

    X.columns = X.columns.str.replace('[', '', regex=False)
    X.columns = X.columns.str.replace(']', '', regex=False)
    X.columns = X.columns.str.replace('<', '', regex=False)
    X.columns = X.columns.str.replace('>', '', regex=False)
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average="weighted"))
    print("Recall:", recall_score(y_test, preds, average="weighted"))
    print("F1 Score:", f1_score(y_test, preds, average="weighted"))

    joblib.dump(model, "models_saved/xgboost_model.pkl")

    print("XGBoost model saved!")


if __name__ == "__main__":
    train_xgboost()