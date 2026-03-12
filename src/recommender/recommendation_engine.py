import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_recommendations(student_id, train_data, top_n=5):

    # Load models
    svd_model = joblib.load("models_saved/svd_model.pkl")
    tfidf_vectorizer = joblib.load("models_saved/tfidf_vectorizer.pkl")

    # Load datasets
    student_vle = pd.read_csv("data/raw/studentVle.csv")
    vle = pd.read_csv("data/raw/vle.csv")

    # Resources used by student
    student_resources = train_data["id_site"].unique()

    if len(student_resources) == 0:
        return None



    interaction_matrix = pd.read_csv(
        "data/processed/interaction_matrix.csv", index_col=0
    )

    if student_id not in interaction_matrix.index:
        return None

    student_vector = interaction_matrix.loc[student_id].values.reshape(1, -1)

    student_embedding = svd_model.transform(student_vector)

    resource_embeddings = svd_model.components_.T

    cf_scores = cosine_similarity(student_embedding, resource_embeddings)[0]


    text_data = vle["activity_type"].astype(str)

    tfidf_matrix = tfidf_vectorizer.transform(text_data)
    mean_vector = np.asarray(tfidf_matrix.mean(axis=0))

    content_scores = cosine_similarity(
        mean_vector,
        tfidf_matrix
    ).flatten()



    popularity = (
        student_vle.groupby("id_site")["sum_click"]
        .sum()
        .rank(pct=True)
    )


    scores = {}

    for i, resource in enumerate(vle["id_site"]):

        cf = cf_scores[i] if i < len(cf_scores) else 0
        content = content_scores[i] if i < len(content_scores) else 0
        pop = popularity.get(resource, 0)

        final_score = (
            0.4 * cf +
            0.3 * content +
            0.3 * pop
        )

        scores[resource] = final_score

    # --------------------------------
    # Rank resources
    # --------------------------------

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    recommendations = [
        r[0] for r in ranked if r[0] not in student_resources
    ]

    return recommendations[:top_n]