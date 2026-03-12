# from src.data_preprocessing.clean_data import clean_data
# from src.data_preprocessing.feature_engineering import create_features
# from src.models.interaction_matrix import create_interaction_matrix

# def run_pipeline():

#     print("Cleaning dataset...")
#     clean_data()

#     print("Feature engineering...")
#     create_features()

#     print("Creating interaction matrix...")
#     create_interaction_matrix()

#     print("Pipeline completed!")

# if __name__ == "__main__":
#     run_pipeline()

from src.models.xgboost_model import train_xgboost
from src.models.kmeans_model import train_kmeans
from src.models.svd_recommender import train_svd
from src.models.tfidf_similarity import train_tfidf


def run_models():

    print("Training XGBoost...")
    train_xgboost()

    print("Running KMeans clustering...")
    train_kmeans()

    print("Training SVD recommender...")
    train_svd()

    print("Training TF-IDF model...")
    train_tfidf()

    print("All models trained!")


if __name__ == "__main__":
    run_models()