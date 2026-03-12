import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/feature_engineered_data.csv")

X = df.drop("knowledge_level", axis=1)
y = df["knowledge_level"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "RandomForest": RandomForestClassifier(),
    "Logistic": LogisticRegression(max_iter=500),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, accuracy_score(y_test, pred))