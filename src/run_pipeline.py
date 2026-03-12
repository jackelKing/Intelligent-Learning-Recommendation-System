from src.data_preprocessing.clean_data import clean_data
from src.features.feature_engineering import create_features
from src.models.xgboost_model import train_xgboost
from src.recommender.interaction_matrix import create_interaction_matrix
from src.recommender.svd_recommender import train_svd

print("Running full pipeline...")

clean_data()
create_features()
train_xgboost()
create_interaction_matrix()
train_svd()

print("Pipeline completed!")