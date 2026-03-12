import shap
import joblib
import pandas as pd

model = joblib.load("models_saved/xgboost_model.pkl")

df = pd.read_csv("data/processed/feature_engineered_data.csv").head(50)

X = df.drop("knowledge_level", axis=1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)