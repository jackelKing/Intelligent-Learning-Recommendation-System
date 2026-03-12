import joblib
import pandas as pd
import numpy as np

def hybrid_recommend(student_id, top_k=5):

    matrix = pd.read_csv("data/processed/interaction_matrix.csv", index_col=0)
    svd = joblib.load("models_saved/svd_model.pkl")

    if student_id not in matrix.index:
        return []

    student_vec = matrix.loc[student_id].values.reshape(1, -1)

    latent = svd.transform(student_vec)
    scores = (latent @ svd.components_).flatten()

    scores = (scores - scores.min()) / (scores.max() - scores.min())

    top_items = matrix.columns[np.argsort(scores)[::-1][:top_k]]

    return list(top_items)