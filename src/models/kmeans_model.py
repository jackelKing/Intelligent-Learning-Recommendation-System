import pandas as pd
from sklearn.cluster import KMeans
import joblib


def train_kmeans():

    df = pd.read_csv("data/processed/feature_engineered_data.csv")

    # Select numerical features only
    X = df.select_dtypes(include=["int64", "float64"])

    model = KMeans(n_clusters=4, random_state=42)

    clusters = model.fit_predict(X)

    df["cluster"] = clusters

    df.to_csv("data/processed/clustered_students.csv", index=False)

    joblib.dump(model, "models_saved/kmeans_model.pkl")

    print("KMeans clustering completed!")


if __name__ == "__main__":
    train_kmeans()