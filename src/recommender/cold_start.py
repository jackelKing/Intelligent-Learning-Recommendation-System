import pandas as pd

def cold_start_recommend(student_profile, top_k=5):

    df = pd.read_csv("data/processed/clustered_students.csv")

    similar = df[df["cluster"] == student_profile["cluster"]]

    recs = similar["id_student"].head(top_k).tolist()

    return recs