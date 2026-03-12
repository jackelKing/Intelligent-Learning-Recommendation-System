import pandas as pd
from sklearn.decomposition import TruncatedSVD
import joblib


def train_svd():

    print("Loading interaction matrix...")

    matrix = pd.read_csv("data/processed/interaction_matrix.csv", index_col=0)

    print("Training SVD model...")

    svd = TruncatedSVD(n_components=50, random_state=42)

    svd.fit(matrix)

    joblib.dump(svd, "models_saved/svd_model.pkl")

    print("SVD model trained and saved!")


if __name__ == "__main__":
    train_svd()