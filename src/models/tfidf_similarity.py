import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def train_tfidf():

    vle = pd.read_csv("data/raw/vle.csv")

    text_data = vle["activity_type"].astype(str)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(text_data)

    similarity = cosine_similarity(tfidf_matrix)

    joblib.dump(vectorizer, "models_saved/tfidf_vectorizer.pkl")

    print("TF-IDF model created!")


if __name__ == "__main__":
    train_tfidf()