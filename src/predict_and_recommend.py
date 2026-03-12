import joblib
import pandas as pd
from src.recommender.recommendation_engine import generate_recommendations

model = joblib.load("models_saved/xgboost_model.pkl")

# sample input
student_data = pd.read_csv("data/processed/feature_engineered_data.csv").head(1)

student_id = student_data["id_student"].iloc[0]

X = student_data.drop(["knowledge_level"], axis=1)
pred = model.predict(X)

print("Predicted Knowledge Level:", pred)

recs = generate_recommendations(student_id)

print("Recommended Resources:", recs)
